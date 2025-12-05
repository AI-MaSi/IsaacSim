"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

from typing import Tuple, List, Union, Dict, Literal
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Environment setup configuration for both sim and real systems."""

    # Point A configuration
    point_a_pos: Tuple[float, float, float] = (0.43, 0.1, -0.20)
    point_a_rotation_deg: float = 0.0 # y axis rotation. 0 = horizontal
    
    # Point B configuration  
    point_b_pos: Tuple[float, float, float] = (0.63, -0.1, -0.20)
    point_b_rotation_deg: float = 0.0
    
    # Single wall configuration (matches real hardware format)
    wall_size: Tuple[float, float, float] = (0.03, 0.50, 0.60)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (0.55, 0.0, -0.45)     # Wall center position
    wall_rot: Tuple[float, float, float, float] = (0.985, 0.0, 0.0, -0.174)  # Quaternion rotation


@dataclass
class PathExecutionConfig:
    """Unified configuration for path execution in both sim and real systems."""

    # Motion parameters ------------------------------
    speed_mps: float = 0.020  # Target constant speed for standardized execution (m/s)
    update_frequency: float = 100.0  # Hz - target harware loop frequency / simulation update rate
    # TODO: ass name, change someday haha
    dt: float = 0.02          # 50Hz Pathing Execution sample period for standardized paths (s). NOT the hw/sim_dt time!


    # TODO: very experimental
    enable_jerk: bool = False  # Enable jerk-limited motion smoothing (S-curve)
    # S-curve velocity profile parameters (jerk-limited motion)
    max_jerk_mps3: float = 2.0  # Maximum jerk (rate of change of acceleration) in m/s^3
    max_accel_mps2: float = 0.5  # Maximum acceleration in m/s^2
    max_decel_mps2: float = 0.5  # Maximum deceleration in m/s^2

    # Normalization / trajectory representation options
    # These are forwarded into NormalizerParams for all planners.
    normalizer_return_poses: bool = True  # Whether normalized planners return poses alongside positions (7D).
    # TODO: returns identity quaternions at the moment!


    normalizer_force_goal: bool = True    # Force exact goal as final waypoint when collision-free. (instead of the nearest planned point)



    # Path planning general parameters (apply to all algorithms)
    grid_resolution: float = 0.020  # 20mm. Grid cell size used for A* (and bounds for others)
    safety_margin: float = 0.06    # Obstacle safety margin in meters.

    # Algorithm dimensionality
    # Use full 3D planning vs X-Z plane only
    use_3d: bool = True



    # Inverse-kinematics controller
    ik_command_type: Literal["position", "pose"] = "pose"
    ik_use_relative_mode: bool = True
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    ik_velocity_mode: bool = True
    ik_velocity_error_gain: float = 350.0 #  Note: hw/sim_dt dependent! 100.0 irl, maybe try a bit larger?
    ik_use_rotational_velocity: bool = True

    # Method/weighting parameters for IK
    ik_params: Dict[str, Union[float, List[float]]] = field(default_factory=lambda: {
        "k_val": 1.15,
        # For SVD
        "min_singular_value": 1e-5,
        # for DLS
        "lambda_val": 0.01, # base (min) value if adaptive damping is used
        "position_weight": 1.0,
        "rotation_weight": 1.1,  # 1.2 for delta position control
        "joint_weights" : [1.0, 1.0, 0.8, 1.0], # [1.0, 1.0, 1.0, 0.8] for delta position control
    })

    # Relative-mode gains (applied to per-step delta pose when relative mode is enabled)
    relative_pos_gain: float = 1.0  # 1.0/3.0 irl/sim. Position delta control

    relative_rot_gain: float = 1.0

    # Axes to ignore in orientation error during IK solving.
    # Any combination of ["roll", "pitch", "yaw"].
    # For excavator: roll is locked (hardware), yaw follows slew automatically.
    # User controls position (X,Y,Z) and pitch only.
    # TODO: CheckDOF for automatic correct settings
    ignore_axes: List[str] = field(default_factory=lambda: ["roll", "yaw"])

    # Use reduced Jacobian (removes uncontrollable DOFs like roll)
    # Recommended for cleaner excavator control
    use_reduced_jacobian: bool = True

    # Relative joint limits (degrees or radians) for [slew, boom, arm, bucket]
    # Provide as list of (min, max). If magnitudes > pi they are treated as degrees and converted.
    # joint_limits_relative: List[Tuple[float, float]] = [(-90.0, 90.0), (-45.0, 60.0), (-90.0, 90.0), (-100.0, 100.0)]
    joint_limits_relative: List[Tuple[float, float]] = field(default_factory=lambda: [])


    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.010  # Final target tolerance in meters (10 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)

    # Optional progress feedback
    progress_update_interval: int = 1  # How often to print progress (seconds)


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()
