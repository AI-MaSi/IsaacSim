# weighting added + small usage fix to SVD method

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
        pass

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
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            # compute targets
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
            self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: Current end-effector position (N, 3)
            ee_quat: Current end-effector quaternion (N, 4)
            jacobian: Jacobian matrix (N, 6, num_joints) or (N, 3, num_joints)
            joint_pos: Current joint positions (N, num_joints)

        Returns:
            torch.Tensor: (N, num_joints) - desired joint positions
        """
        # Get IK parameters with defaults
        ik_params = self.cfg.ik_params or {}
        k_val = ik_params.get("k_val", 1.0)

        # Compute pose error
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            pose_error = position_error
            jacobian_used = jacobian[:, 0:3]  # Use only position part of Jacobian
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )

            # Apply weighting to position and rotation errors
            pos_weight = ik_params.get("position_weight", 1.0)
            rot_weight = ik_params.get("rotation_weight", 0.1)

            # Combine weighted errors for IK solving
            pose_error = torch.cat((pos_weight * position_error, rot_weight * axis_angle_error), dim=1)
            jacobian_used = jacobian  # Use full 6DOF Jacobian

        # Compute delta joint positions
        delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian_used)

        # Apply scaling factor
        delta_joint_pos = k_val * delta_joint_pos

        # Compute desired joint positions
        q_desired = joint_pos + delta_joint_pos

        return q_desired

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        Args:
            delta_pose: Desired change in pose (N, 3) or (N, 6)
            jacobian: Jacobian matrix (N, 3, num_joints) or (N, 6, num_joints)

        Returns:
            torch.Tensor: Change in joint positions (N, num_joints)
        """
        method = self.cfg.ik_method
        ik_params = self.cfg.ik_params or {}

        joint_weights = ik_params.get("joint_weights", None)
        num_joints = jacobian.size(-1)
        batch_size = jacobian.size(0)

        # Default to uniform weights if none are given
        if joint_weights is None:
            joint_weights = [1.0] * num_joints

        W_inv = self._build_W_inv(joint_weights, batch_size)

        if method == "pinv":
            # Moore-Penrose pseudo-inverse
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)

        if method == "svd":

            min_singular = ik_params.get("min_singular_value", 1e-4)

            J_weighted = jacobian @ W_inv

            U, S, Vh = torch.linalg.svd(J_weighted, full_matrices=False)

            S_inv = torch.where(S > min_singular, 1.0 / S, torch.zeros_like(S))

            S_inv_mat = torch.diag_embed(S_inv)

            jacobian_pinv = Vh.transpose(-2, -1) @ S_inv_mat @ U.transpose(-2, -1)

            delta_joint_pos = W_inv @ jacobian_pinv @ delta_pose.unsqueeze(-1)



        elif method == "trans":
            # Simple transpose (fast but less accurate)
            jacobian_T = torch.transpose(jacobian, 1, 2)
            delta_joint_pos = jacobian_T @ delta_pose.unsqueeze(-1)



        elif method == "dls":

            lambda_val = ik_params.get("lambda_val", 0.1)

            JW_inv = jacobian @ W_inv

            JJT = JW_inv @ jacobian.transpose(1, 2)

            I = torch.eye(JJT.shape[-1], device=self._device).unsqueeze(0).expand(batch_size, -1, -1)

            damped_term = JJT + (lambda_val ** 2) * I

            try:

                inv_term = torch.linalg.inv(damped_term)

                delta_joint_pos = W_inv @ jacobian.transpose(1, 2) @ inv_term @ delta_pose.unsqueeze(-1)

            except torch.linalg.LinAlgError:

                jacobian_pinv = torch.linalg.pinv(jacobian)

                delta_joint_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)



        else:
            raise ValueError(f"Unsupported IK method: {method}. Supported methods: 'pinv', 'svd', 'trans', 'dls'")

        return delta_joint_pos.squeeze(-1)

    def _build_W_inv(self, joint_weights: list[float], batch_size: int) -> torch.Tensor:
        """Build inverse weight matrix for joint weighting.

        Args:
            joint_weights: List of weights for each joint (higher = more preferred)
            batch_size: Number of environments

        Returns:
            Inverse weight matrix W^{-1} of shape [batch_size, num_joints, num_joints]
        """
        w = torch.tensor(joint_weights, device=self._device)

        # Validation: prevent division by zero
        w = torch.clamp(w, min=1e-6)  # Ensure all weights are at least 1e-6

        # Optional: normalize weights
        # w = w / w.mean()

        # Create diagonal matrix with inverse weights
        W_inv = torch.diag(1.0 / w).unsqueeze(0).expand(batch_size, -1, -1)
        return W_inv