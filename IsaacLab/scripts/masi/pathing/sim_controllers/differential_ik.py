# Updated to keep the simulation IK controller aligned with the IRL implementation (modules/diff_ik_V2.py):
# - Frame transform into base frame to decouple slew yaw from pitch commands
# - Reduced Jacobian support (drop uncontrollable roll) with controllable_dofs mask
# - Adaptive damping, velocity limiting, joint-limit avoidance, and optional anti-windup
# - Position/rotation weighting applied to Jacobian rows plus ignore_axes filtering on errors and Jacobian rows
# - Joint weighting uses W (multipliers) consistently across all IK methods

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

        # Extended settings to mirror IRL controller
        self.use_reduced_jacobian = bool(getattr(self.cfg, "use_reduced_jacobian", False))
        self.use_ignore_axes_in_jacobian = bool(getattr(self.cfg, "use_ignore_axes_in_jacobian", True))
        self.enable_frame_transform = bool(getattr(self.cfg, "enable_frame_transform", True))
        self.enable_velocity_limiting = bool(getattr(self.cfg, "enable_velocity_limiting", True))
        self.enable_adaptive_damping = bool(getattr(self.cfg, "enable_adaptive_damping", True))
        self.enable_joint_limit_avoidance = bool(getattr(self.cfg, "enable_joint_limit_avoidance", True))
        self.enable_anti_windup = bool(getattr(self.cfg, "enable_anti_windup", False))

        # Controllable DOFs (default to excavator: position + pitch + yaw)
        controllable = getattr(self.cfg, "controllable_dofs", None)
        if controllable is None and self.use_reduced_jacobian:
            controllable = [0, 1, 2, 4, 5]
        self.controllable_dofs = controllable
        self._controllable_idx = (
            torch.tensor(self.controllable_dofs, dtype=torch.long, device=self._device)
            if self.controllable_dofs is not None
            else None
        )

        # Velocity and limit handling
        self._max_joint_velocities = None  # filled on first compute() when joint count is known
        limits = getattr(self.cfg, "joint_limits", None)
        if limits is not None:
            limits_t = torch.tensor(limits, dtype=torch.float32, device=self._device)
            self._joint_limits = (limits_t[:, 0], limits_t[:, 1])
        else:
            self._joint_limits = None

        # Anti-windup state
        self.prev_error_norm = torch.full((self.num_envs,), float("nan"), device=self._device)
        self.windup_counter = torch.zeros(self.num_envs, device=self._device, dtype=torch.int64)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        self.ee_pos_des[env_ids] = 0.0
        self.ee_quat_des[env_ids] = 0.0
        self._command[env_ids] = 0.0
        self.prev_error_norm[env_ids] = float("nan")
        self.windup_counter[env_ids] = 0

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                self.ee_pos_des = self._command[:, 0:3]
                self.ee_quat_des = self._command[:, 3:7]

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
        desired_ee_velocity: torch.Tensor | None = None,
        dt: float = 0.02,
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose."""
        ik_params = self.cfg.ik_params or {}
        batch_size, num_joints = joint_pos.shape
        dt = float(dt)
        dt = max(1e-4, min(dt, 1.0))

        # Lazily build velocity limits on the correct device/dtype
        if self._max_joint_velocities is None or self._max_joint_velocities.numel() != num_joints:
            if self.cfg.max_joint_velocities is None:
                max_vel = torch.full((num_joints,), 0.035, device=joint_pos.device, dtype=joint_pos.dtype)
            else:
                max_vel = torch.tensor(self.cfg.max_joint_velocities, device=joint_pos.device, dtype=joint_pos.dtype)
            self._max_joint_velocities = max_vel

        # Ensure controllable DOF index tensor lives with the current tensors
        if self._controllable_idx is not None and self._controllable_idx.device != jacobian.device:
            self._controllable_idx = self._controllable_idx.to(jacobian.device)

        # Move joint limits if necessary
        if self._joint_limits is not None and self._joint_limits[0].device != joint_pos.device:
            self._joint_limits = (self._joint_limits[0].to(joint_pos.device), self._joint_limits[1].to(joint_pos.device))

        # Optional frame transform into base frame to decouple slew yaw from pitch control
        current_pos = ee_pos
        target_pos = self.ee_pos_des
        jacobian_full = jacobian
        if self.enable_frame_transform and jacobian.size(1) >= 3:
            base_rot = self._rotation_matrix_from_yaw(-joint_pos[:, 0])
            current_pos = torch.bmm(base_rot, ee_pos.unsqueeze(-1)).squeeze(-1)
            target_pos = torch.bmm(base_rot, self.ee_pos_des.unsqueeze(-1)).squeeze(-1)

            j_pos = torch.bmm(base_rot, jacobian[:, 0:3, :])
            if jacobian.size(1) >= 6:
                j_rot = torch.bmm(base_rot, jacobian[:, 3:6, :])
                jacobian_full = torch.cat([j_pos, j_rot], dim=1)
            else:
                jacobian_full = j_pos

        # Optionally reduce Jacobian rows (remove uncontrollable roll)
        jacobian_use = jacobian_full
        if (
            self.use_reduced_jacobian
            and self._controllable_idx is not None
            and jacobian_full.size(1) >= int(self._controllable_idx.max()) + 1
        ):
            jacobian_use = torch.index_select(jacobian_full, 1, self._controllable_idx)

        # Adaptive damping based on condition number (mirrors IRL)
        adaptive_lambda = self._compute_adaptive_damping(jacobian_use, ik_params)

        axis_angle_error = None
        ignore_axes = getattr(self.cfg, "ignore_axes", [])

        # Compute pose error
        if "position" in self.cfg.command_type:
            position_error = target_pos - current_pos
            pose_error = position_error

            pos_weight = ik_params.get("position_weight", 1.0)
            jacobian_weighted = pos_weight * jacobian_use[:, 0:3, :]  # Weight position rows only
        else:
            if self.enable_frame_transform:
                ee_quat_local = self._transform_to_local_frame(ee_quat, joint_pos[:, 0])
                target_quat_local = self._transform_to_local_frame(self.ee_quat_des, joint_pos[:, 0])
            else:
                ee_quat_local = ee_quat
                target_quat_local = self.ee_quat_des

            position_error, axis_angle_error = compute_pose_error(
                current_pos, ee_quat_local, target_pos, target_quat_local, rot_error_type="axis_angle"
            )

            axis_angle_error = self._apply_rotation_filter(axis_angle_error)

            pos_weight = ik_params.get("position_weight", 1.0)
            rot_weight = ik_params.get("rotation_weight", 0.1)

            full_pose_error = torch.cat([position_error, axis_angle_error], dim=1)

            if self.use_reduced_jacobian and self._controllable_idx is not None:
                pose_error = torch.index_select(full_pose_error, 1, self._controllable_idx)

                weighted_rows = []
                for i, dof in enumerate(self.controllable_dofs):
                    row = jacobian_use[:, i, :]
                    if dof < 3:
                        row = pos_weight * row
                    else:
                        row = rot_weight * row
                        if self.use_ignore_axes_in_jacobian and ignore_axes:
                            if (dof == 3 and "roll" in ignore_axes) or (dof == 4 and "pitch" in ignore_axes) or (
                                dof == 5 and "yaw" in ignore_axes
                            ):
                                row = torch.zeros_like(row)
                    weighted_rows.append(row.unsqueeze(1))
                jacobian_weighted = torch.cat(weighted_rows, dim=1)
            else:
                pose_error = full_pose_error
                jacobian_weighted = torch.cat(
                    [pos_weight * jacobian_use[:, 0:3, :], rot_weight * jacobian_use[:, 3:6, :]], dim=1
                )
                if self.use_ignore_axes_in_jacobian and ignore_axes and jacobian_weighted.size(1) >= 6:
                    if "roll" in ignore_axes:
                        jacobian_weighted[:, 3, :] = 0.0
                    if "pitch" in ignore_axes:
                        jacobian_weighted[:, 4, :] = 0.0
                    if "yaw" in ignore_axes:
                        jacobian_weighted[:, 5, :] = 0.0


        # Compute delta joint positions with weighted Jacobian
        if getattr(self.cfg, "velocity_mode", False):
            desired_vec = self._prepare_desired_velocity(
                desired_ee_velocity, pose_error.shape[1], reference=pose_error
            )
            task_vec = desired_vec + float(self.cfg.velocity_error_gain) * pose_error
            delta_joint_pos = self._compute_delta_joint_pos(
                delta_pose=task_vec, jacobian=jacobian_weighted, adaptive_lambda=adaptive_lambda
            )
            delta_joint_pos = delta_joint_pos * dt
        else:
            delta_joint_pos = self._compute_delta_joint_pos(
                delta_pose=pose_error, jacobian=jacobian_weighted, adaptive_lambda=adaptive_lambda
            )

        # Post-processing: velocity limit, joint limit avoidance, anti-windup
        if self.enable_velocity_limiting:
            delta_joint_pos = self._apply_velocity_limits(delta_joint_pos)

        if self.enable_joint_limit_avoidance:
            delta_joint_pos = self._add_joint_limit_avoidance(delta_joint_pos, joint_pos)

        if self.enable_anti_windup:
            delta_joint_pos = self._apply_anti_windup(
                delta_joint_pos, position_error, axis_angle_error if "pose" in self.cfg.command_type else None
            )

        return joint_pos + delta_joint_pos

    def _prepare_desired_velocity(
        self, desired_ee_velocity: torch.Tensor | None, required_size: int, reference: torch.Tensor
    ) -> torch.Tensor:
        """Align desired EE velocity with the active task dimension (handles reduced Jacobian)."""
        if desired_ee_velocity is None:
            return torch.zeros(
                self.num_envs, required_size, device=reference.device, dtype=reference.dtype
            )

        desired_vec = desired_ee_velocity.to(reference.device)

        # Expand position-only inputs when rotation components are expected
        if desired_vec.shape[1] == 3 and required_size > 3:
            pad = torch.zeros(self.num_envs, required_size - 3, device=desired_vec.device, dtype=desired_vec.dtype)
            desired_vec = torch.cat([desired_vec, pad], dim=1)

        # Map full 6D vectors into reduced task space (drops uncontrollable axes like roll)
        if self.use_reduced_jacobian and self._controllable_idx is not None and required_size != 6:
            if desired_vec.shape[1] == 6:
                desired_vec = torch.index_select(desired_vec, 1, self._controllable_idx)

        # Position-only mode: trim any extra components
        if required_size == 3 and desired_vec.shape[1] > 3:
            desired_vec = desired_vec[:, :3]

        if desired_vec.shape[1] != required_size:
            raise ValueError(
                f"desired_ee_velocity size {desired_vec.shape[1]} does not match required size {required_size}"
            )

        return desired_vec

    def _compute_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor, adaptive_lambda: float | torch.Tensor
    ) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose."""
        method = self.cfg.ik_method
        ik_params = self.cfg.ik_params or {}

        joint_weights = ik_params.get("joint_weights", None)
        num_joints = jacobian.size(-1)
        batch_size = jacobian.size(0)

        # Default to uniform weights if none are given
        if joint_weights is None:
            joint_weights = [1.0] * num_joints

        # Build weight matrix W (higher = more preferred/more movement)
        W = self._build_W(joint_weights, batch_size)
        k_val = ik_params.get("k_val", 1.0)

        if method == "pinv":
            jac_w = jacobian @ W
            jacobian_pinv = torch.linalg.pinv(jac_w)
            dq_prime = jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = k_val * (W @ dq_prime)

        elif method == "svd":
            min_singular = ik_params.get("min_singular_value", 1e-4)

            jac_weighted = jacobian @ W

            U, S, Vh = torch.linalg.svd(jac_weighted, full_matrices=False)

            ut_delta = U.transpose(-2, -1) @ delta_pose.unsqueeze(-1)

            result = torch.zeros(batch_size, num_joints, 1, device=self._device)
            for i in range(S.size(-1)):
                mask = S[:, i] > min_singular
                result[mask] += Vh[mask, i:i+1, :].transpose(-2, -1) * (
                    ut_delta[mask, i:i+1, :] / S[mask, i:i+1, None]
                )

            delta_joint_pos = k_val * (W @ result)

        elif method == "trans":
            jac_w = jacobian @ W
            delta_joint_pos = k_val * (jac_w.transpose(1, 2) @ delta_pose.unsqueeze(-1))

        elif method == "dls":
            lambda_val = adaptive_lambda if adaptive_lambda is not None else ik_params.get("lambda_val", 0.1)

            jac_weighted = jacobian @ W

            jjt = jac_weighted @ jac_weighted.transpose(1, 2)
            I = torch.eye(jjt.shape[-1], device=self._device).unsqueeze(0).expand(batch_size, -1, -1)
            if isinstance(lambda_val, torch.Tensor):
                lam_sq = (lambda_val ** 2).view(batch_size, 1, 1)
            else:
                lam_sq = (lambda_val ** 2)
            damped_term = jjt + lam_sq * I

            try:
                inv_term = torch.linalg.inv(damped_term)
                delta_joint_pos = W @ jac_weighted.transpose(1, 2) @ inv_term @ delta_pose.unsqueeze(-1)
            except torch.linalg.LinAlgError:
                jacobian_pinv = torch.linalg.pinv(jacobian)
                delta_joint_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)

        else:
            raise ValueError(f"Unsupported IK method: {method}. Supported methods: 'pinv', 'svd', 'trans', 'dls'")

        return delta_joint_pos.squeeze(-1)

    def _compute_adaptive_damping(self, jacobian: torch.Tensor, ik_params: dict) -> torch.Tensor:
        """Compute adaptive damping factor based on Jacobian conditioning (matches IRL logic)."""
        base_lambda = ik_params.get("lambda_val", 0.1)
        if not self.enable_adaptive_damping:
            return torch.tensor(base_lambda, device=self._device)

        # Compute condition number per batch element
        _, S, _ = torch.linalg.svd(jacobian, full_matrices=False)
        cond = S[:, 0] / (S[:, -1] + 1e-12)
        adaptive = base_lambda * (1.0 + 0.5 * torch.log1p(cond))
        lam = torch.clamp(adaptive, min=base_lambda, max=base_lambda * 10.0)

        return lam

    def _apply_velocity_limits(self, delta_joint_pos: torch.Tensor) -> torch.Tensor:
        """Limit joint velocities to prevent large jumps."""
        max_vel = self._max_joint_velocities
        if max_vel is None:
            return delta_joint_pos
        return torch.clamp(delta_joint_pos, -max_vel.unsqueeze(0), max_vel.unsqueeze(0))

    def _add_joint_limit_avoidance(self, delta_joint_pos: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        """Add repulsion forces to avoid joint limits."""
        if self._joint_limits is None:
            return delta_joint_pos

        q_min, q_max = self._joint_limits
        repulsion_strength = 0.1
        margin_fraction = 0.15

        q_range = q_max - q_min
        margin = margin_fraction * q_range

        below_mask = joint_pos < (q_min + margin)
        above_mask = joint_pos > (q_max - margin)

        repulsion = torch.zeros_like(delta_joint_pos)

        if below_mask.any():
            distance_ratio = (joint_pos - q_min) / margin
            repulsion += torch.where(
                below_mask, repulsion_strength * (1.0 - distance_ratio).clamp(min=0.0) ** 2, torch.zeros_like(repulsion)
            )

        if above_mask.any():
            distance_ratio = (q_max - joint_pos) / margin
            repulsion -= torch.where(
                above_mask, repulsion_strength * (1.0 - distance_ratio).clamp(min=0.0) ** 2, torch.zeros_like(repulsion)
            )

        return delta_joint_pos + repulsion

    def _apply_anti_windup(
        self, delta_joint_pos: torch.Tensor, position_error: torch.Tensor, rotation_error: torch.Tensor | None
    ) -> torch.Tensor:
        """Prevent error accumulation for unreachable targets (per-env, matches IRL behavior)."""
        if rotation_error is not None:
            error_norm = torch.norm(position_error, dim=1) + torch.norm(rotation_error, dim=1)
        else:
            error_norm = torch.norm(position_error, dim=1)

        if self.prev_error_norm.numel() != error_norm.numel():
            self.prev_error_norm = torch.full_like(error_norm, float("nan"))
            self.windup_counter = torch.zeros_like(error_norm, dtype=torch.int64)

        not_decreasing = error_norm > 0.95 * self.prev_error_norm

        self.windup_counter[not_decreasing] += 1
        self.windup_counter[~not_decreasing] = torch.clamp(self.windup_counter[~not_decreasing] - 1, min=0)

        scale_mask = self.windup_counter > 5
        if scale_mask.any():
            scale = 0.5 ** (self.windup_counter - 5)
            scale = torch.clamp(scale, min=0.1)
            delta_joint_pos = torch.where(scale_mask.unsqueeze(-1), delta_joint_pos * scale.unsqueeze(-1), delta_joint_pos)

        self.prev_error_norm = error_norm.detach()
        return delta_joint_pos

    def _apply_rotation_filter(self, rotation_error: torch.Tensor) -> torch.Tensor:
        """Zero out ignored rotation axes from the error vector."""
        ignore_axes = getattr(self.cfg, "ignore_axes", [])
        if not ignore_axes:
            return rotation_error

        filtered = rotation_error.clone()
        for axis in ignore_axes:
            if axis == "roll":
                filtered[:, 0] = 0.0
            elif axis == "pitch":
                filtered[:, 1] = 0.0
            elif axis == "yaw":
                filtered[:, 2] = 0.0
        return filtered

    def _rotation_matrix_from_yaw(self, yaw: torch.Tensor) -> torch.Tensor:
        """Create batch rotation matrix about Z from yaw angles."""
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        return torch.stack(
            [
                torch.stack([c, -s, zeros], dim=-1),
                torch.stack([s, c, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=1,
        )

    def _quat_conjugate(self, quat: torch.Tensor) -> torch.Tensor:
        """Return quaternion conjugate (wxyz)."""
        wxyz = quat.clone()
        wxyz[..., 1:] = -wxyz[..., 1:]
        return wxyz

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply q1 * q2 for batched wxyz tensors."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion (safe clamp)."""
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-9)
        return quat / norm

    def _transform_to_local_frame(self, quat: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
        """Remove base yaw from quaternion so pitch/yaw are expressed in robot frame."""
        half_yaw = base_yaw * 0.5
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        base_quat = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)
        base_quat_inv = self._quat_conjugate(base_quat)
        quat_local = self._quat_multiply(base_quat_inv, quat)
        return self._normalize_quat(quat_local)

    def _build_W(self, joint_weights: list[float], batch_size: int) -> torch.Tensor:
        """Build weight matrix for joint weighting (matches IRL implementation)."""
        w = torch.tensor(joint_weights, device=self._device, dtype=torch.float32)
        w = torch.clamp(w, min=1e-6)
        return torch.diag(w).unsqueeze(0).expand(batch_size, -1, -1)
