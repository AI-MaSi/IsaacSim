"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment setup configuration for both sim and real systems."""

    # Point A configuration
    point_a_pos: Tuple[float, float, float] = (0.43, 0.075, -0.20)
    point_a_rotation_deg: float = 0.0 # y axis rotation. 0 = horizontal

    # Point B configuration
    point_b_pos: Tuple[float, float, float] = (0.63, -0.075, -0.20)
    point_b_rotation_deg: float = 0.0

    # Single wall configuration (matches real hardware format)
    wall_size: Tuple[float, float, float] = (0.03, 0.50, 0.60)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (0.55, 0.0, -0.45)     # Wall center position
    wall_rot: Tuple[float, float, float, float] = (0.985, 0.0, 0.0, -0.174)  # Quaternion rotation


@dataclass
class PathExecutionConfig:
    """Unified configuration for path execution in both sim and real systems."""

    # Motion parameters ------------------------------
    speed_mps: float = 0.030  #(0.02...70worked good) Target constant speed for standardized execution (m/s)
    update_frequency: float = 100.0  # Hz - target harware loop frequency / simulation update rate
    # TODO: bad name, change someday
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

    # NOTE: IK configuration has been moved to configuration_files/control_config.yaml
    # See the 'ik' section in that file for: command_type, method, velocity_mode,
    # params, relative gains, ignore_axes, use_reduced_jacobian, joint_limits_relative

    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.010  # Final target tolerance in meters (10 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)

    # Optional progress feedback
    progress_update_interval: int = 1  # How often to print progress (seconds)


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()
