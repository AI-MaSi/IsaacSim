from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .differential_ik import DifferentialIKController


@configclass
class DifferentialIKControllerCfg:
    """Configuration for differential inverse kinematics controller."""

    class_type: type = DifferentialIKController
    """The associated controller class."""

    command_type: Literal["position", "pose"] = MISSING
    """Type of task-space command to control the articulation's body.

    If "position", then the controller only controls the position of the articulation's body.
    Otherwise, the controller controls the pose of the articulation's body.
    """

    use_relative_mode: bool = False
    """Whether to use relative mode for the controller. Defaults to False.

    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """

    ik_method: Literal["pinv", "svd", "trans", "dls"] = MISSING
    """Method for computing inverse of Jacobian."""

    ik_params: dict[str, float | list[float]] | None = None
    """Parameters for the inverse-kinematics method.

    Common parameters for all methods:
    - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - "position_weight": Weight for position errors (default: 1.0).
    - "rotation_weight": Weight for rotation errors (default: 0.1).
    - "joint_weights": List of per-joint weights (e.g., [1.0, 0.5, 0.1, ..., 1.0]). Works with SVD and DLS methods.

    Method-specific parameters:
    - SVD ("svd"):
        - "min_singular_value": Values below are suppressed to zero (default: 1e-5).
    - Damped least squares ("dls"):
        - "lambda_val": Damping coefficient (default: 0.01).
    """

    # Axes to ignore in orientation error when solving IK (e.g., ["roll", "yaw"]).
    # If None or empty, all orientation components are used.
    ignore_axes: list[str] | None = None

    # Whether to zero out ignored rotational DOFs in the Jacobian as well (matches IRL default).
    use_ignore_axes_in_jacobian: bool = True

    # Velocity-mode toggle (matches IRL diff IK behaviour)
    velocity_mode: bool = False
    velocity_error_gain: float = 1.0
    use_rotational_velocity: bool = True

    # Transform pose errors into the robot base frame (removes slew yaw coupling).
    enable_frame_transform: bool = True

    # Reduced Jacobian handling (drop uncontrollable roll for excavator).
    use_reduced_jacobian: bool = True
    controllable_dofs: list[int] | None = None  # defaults to [0,1,2,4,5] in controller when reduced is enabled

    # Velocity limiting
    enable_velocity_limiting: bool = True
    max_joint_velocities: list[float] | None = None  # radians per control step; defaults to 0.035 rad if None

    # Joint limits and avoidance
    joint_limits: list[tuple[float, float]] | None = None
    enable_joint_limit_avoidance: bool = True

    # Adaptive damping (DLS) and anti-windup
    enable_adaptive_damping: bool = True
    enable_anti_windup: bool = False # TODO: test this at some point

    def __post_init__(self):
        # Validate command and method
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.command_type}.")
        if self.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}.")

        # Default parameters for each method
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"k_val": 1.0, "lambda_val": 0.01},
        }

        # Default weighting parameters (apply to all methods)
        weighting_defaults = {
            "position_weight": 1.0,
            "rotation_weight": 0.1,
        }

        # Start with method-specific defaults
        ik_params = default_ik_params[self.ik_method].copy()

        # Add weighting defaults
        ik_params.update(weighting_defaults)

        # Override with user-provided parameters
        if self.ik_params is not None:
            ik_params.update(self.ik_params)

        self.ik_params = ik_params

        # Validate ignore_axes, if provided
        if self.ignore_axes is None:
            self.ignore_axes = []
        else:
            allowed = {"roll", "pitch", "yaw"}
            normalized: list[str] = []
            for axis in self.ignore_axes:
                a = axis.lower()
                if a not in allowed:
                    raise ValueError(f"Invalid axis in ignore_axes: {axis}. Must be one of {allowed}")
                normalized.append(a)
            self.ignore_axes = normalized

        # Validate joint limits if provided
        if self.joint_limits is not None:
            if not isinstance(self.joint_limits, (list, tuple)):
                raise ValueError("joint_limits must be a list/tuple of (min, max) pairs when provided.")
            self.joint_limits = [(float(a), float(b)) for (a, b) in self.joint_limits]

        # Validate max_joint_velocities if provided
        if self.max_joint_velocities is not None:
            if not isinstance(self.max_joint_velocities, (list, tuple)):
                raise ValueError("max_joint_velocities must be a list/tuple when provided.")
            self.max_joint_velocities = [float(v) for v in self.max_joint_velocities]

        # Validate controllable_dofs if provided
        if self.controllable_dofs is not None:
            self.controllable_dofs = [int(v) for v in self.controllable_dofs]
