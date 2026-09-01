"""Path planning algorithms and utilities."""

from .path_planning_algorithms import (
    # Exceptions
    PathPlanningError,
    NoPathFoundError,
    CollisionError,
    TimeoutError,
    InvalidInputError,
    # Parameter dataclasses
    AStarParams,
    AStarPlanarParams,
    RRTParams,
    RRTStarParams,
    PRMParams,
    # Algorithm classes
    AStar3D,
    RRT,
    RRTStar,
    PRM,
    # High-level wrapper functions
    create_astar_3d_trajectory,
    create_astar_plane_trajectory,
    create_rrt_plane_trajectory,
    create_rrt_star_plane_trajectory,
    create_rrt_trajectory,
    create_rrt_star_trajectory,
    create_prm_trajectory,
    create_prm_plane_trajectory,
    # Helper functions
    setup_planner_environment,
)

from .path_utils import (
    # Path manipulation
    interpolate_path,
    calculate_path_length,
    interpolate_along_path,
    precompute_cumulative_distances,
    interpolate_at_s,
    resample_constant_speed,
    resample_constant_circum_speed,
    downsample_by_points,
    standardize_path,
    project_point_onto_path,
    calculate_execution_time,
    is_target_reached,
    print_path_info,
    build_poses_xyz_quat,
    # Quaternion utilities
    normalize_vector,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    basis_start_goal_plane,
    # Workspace utilities
    calculate_workspace_bounds,
    GridConfig,
    ObstacleChecker,
)

__all__ = [
    # Exceptions
    "PathPlanningError",
    "NoPathFoundError",
    "CollisionError",
    "TimeoutError",
    "InvalidInputError",
    # Parameter dataclasses
    "AStarParams",
    "AStarPlanarParams",
    "RRTParams",
    "RRTStarParams",
    "PRMParams",
    # Algorithm classes
    "AStar3D",
    "RRT",
    "RRTStar",
    "PRM",
    # High-level wrapper functions
    "create_astar_3d_trajectory",
    "create_astar_plane_trajectory",
    "create_rrt_plane_trajectory",
    "create_rrt_star_plane_trajectory",
    "create_rrt_trajectory",
    "create_rrt_star_trajectory",
    "create_prm_trajectory",
    "create_prm_plane_trajectory",
    "setup_planner_environment",
    # Path utilities
    "interpolate_path",
    "calculate_path_length",
    "interpolate_along_path",
    "precompute_cumulative_distances",
    "interpolate_at_s",
    "resample_constant_speed",
    "resample_constant_circum_speed",
    "downsample_by_points",
    "standardize_path",
    "project_point_onto_path",
    "calculate_execution_time",
    "is_target_reached",
    "print_path_info",
    "build_poses_xyz_quat",
    # Quaternion utilities
    "normalize_vector",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_conjugate",
    "quaternion_multiply",
    "basis_start_goal_plane",
    # Workspace utilities
    "calculate_workspace_bounds",
    "GridConfig",
    "ObstacleChecker",
]
