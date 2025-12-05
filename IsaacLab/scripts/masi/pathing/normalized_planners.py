"""
Normalized (constant-speed) path planning wrappers.
===================================================

Public-facing helpers that:
- Call the low-level planners from path_planning_algorithms.
- Run the resulting paths through ``standardize_path`` to enforce a
  constant-speed, fixed-dt representation.

This keeps the core algorithms focused on geometry/search, and
collects the execution-shaping logic in one place.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import logging
import numpy as np

from .path_utils import standardize_path, ObstacleChecker
from .path_planning_algorithms import (
    AStarParams,
    RRTParams,
    RRTStarParams,
    PRMParams,
    create_astar_3d_trajectory,
    create_astar_plane_trajectory,
    create_rrt_star_trajectory,
    create_rrt_trajectory,
    create_prm_trajectory,
)


logger = logging.getLogger(__name__)


@dataclass
class NormalizerParams:
    """Configuration for converting raw paths into normalized trajectories.

    All fields must be provided explicitly by higher-level config (no internal defaults).
    """

    speed_mps: float  # Target constant speed (m/s)
    dt: float         # Sample period (s)
    # Whether to return full poses (4x4 matrices / 7D pose vectors) or just positions.
    return_poses: bool
    # When True, the normalized trajectory is forced to end exactly at the requested goal
    # (if a straight-line segment from the last planner waypoint to the goal is collision-free).
    force_goal: bool


def force_goal_as_final_waypoint(
    path: np.ndarray,
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    safety_margin: float,
    force_goal: bool,
) -> np.ndarray:
    """Optionally append the exact goal as final waypoint, in world coordinates.

    This is done before normalization so constant-speed resampling remains valid.
    A straight segment from the last waypoint to the goal is only added if it is
    collision-free with respect to the (already inflated) obstacles.
    """
    if not force_goal:
        return path

    if path is None:
        return path

    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return pts

    goal = np.asarray(goal_pos, dtype=np.float32)
    last = pts[-1]

    # If we're already effectively at the goal, do nothing.
    if np.linalg.norm(last - goal) < 1e-4:
        return pts

    checker = ObstacleChecker(obstacle_data or [], safety_margin=safety_margin)
    if checker.is_line_collision_free(tuple(last.tolist()), tuple(goal.tolist())):
        pts = np.vstack([pts, goal.astype(np.float32)])

    return pts


def create_astar_3d_trajectory_normalized(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    *,
    astar_params: AStarParams,
    normalizer_params: NormalizerParams,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Create an A* path and normalize it to constant-speed execution."""

    raw = create_astar_3d_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        max_iterations=astar_params.max_iterations,
        verbose=verbose,
    )

    raw = force_goal_as_final_waypoint(
        path=raw,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        force_goal=normalizer_params.force_goal,
    )

    return standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )


def create_astar_plane_trajectory_normalized(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    *,
    astar_params: AStarParams,
    normalizer_params: NormalizerParams,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Planar (X-Z) A* normalized to constant-speed execution."""

    raw = create_astar_plane_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        max_iterations=astar_params.max_iterations,
        verbose=verbose,
    )

    raw = force_goal_as_final_waypoint(
        path=raw,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        force_goal=normalizer_params.force_goal,
    )

    return standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )


def create_rrt_star_trajectory_normalized(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    use_3d: bool = True,
    *,
    rrt_params: RRTStarParams,
    normalizer_params: NormalizerParams,
) -> Dict[str, np.ndarray]:
    """Create an RRT* path and normalize it to constant-speed execution."""

    raw = create_rrt_star_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        max_iterations=rrt_params.max_iterations,
        max_acceptable_cost=rrt_params.max_acceptable_cost,
        max_step_size=rrt_params.max_step_size,
        goal_bias=rrt_params.goal_bias,
        rewire_radius=rrt_params.rewire_radius,
        goal_tolerance=rrt_params.goal_tolerance,
        minimum_iterations=rrt_params.minimum_iterations,
        cost_improvement_patience=rrt_params.cost_improvement_patience,
    )

    raw = force_goal_as_final_waypoint(
        path=raw,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        force_goal=normalizer_params.force_goal,
    )

    return standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )


def create_rrt_trajectory_normalized(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    use_3d: bool = True,
    *,
    rrt_params: RRTParams,
    normalizer_params: NormalizerParams,
) -> Dict[str, np.ndarray]:
    """Create an RRT path and normalize it to constant-speed execution."""

    raw = create_rrt_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        max_iterations=rrt_params.max_iterations,
        max_acceptable_cost=rrt_params.max_acceptable_cost,
        max_step_size=rrt_params.max_step_size,
        goal_bias=rrt_params.goal_bias,
        goal_tolerance=rrt_params.goal_tolerance,
    )

    raw = force_goal_as_final_waypoint(
        path=raw,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        force_goal=normalizer_params.force_goal,
    )

    return standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )


def create_prm_trajectory_normalized(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    use_3d: bool = True,
    *,
    prm_params: PRMParams,
    normalizer_params: NormalizerParams,
) -> Dict[str, np.ndarray]:
    """Create a PRM path and normalize it to constant-speed execution."""

    raw = create_prm_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        num_samples=prm_params.num_samples,
        connection_radius=prm_params.connection_radius,
        max_connections_per_node=prm_params.max_connections_per_node,
    )

    raw = force_goal_as_final_waypoint(
        path=raw,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        force_goal=normalizer_params.force_goal,
    )

    return standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )


def plan_to_target(
    start_pos_world: Tuple[float, float, float],
    target_pos_world: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    *,
    algorithm: str,
    grid_resolution: float,
    safety_margin: float,
    normalizer_params: NormalizerParams,
    astar_params: Optional[AStarParams] = None,
    rrt_params: Optional[RRTParams] = None,
    rrt_star_params: Optional[RRTStarParams] = None,
    prm_params: Optional[PRMParams] = None,
    use_3d: bool = True,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Plan a normalized path from an arbitrary start pose to a target.

    This is intended for feeding the *current* robot EE position directly
    into the planners (e.g. ee→A, ee→B instead of A→B, B→A).

    The underlying planners operate in position space only. Orientation in
    the returned trajectory is currently set to identity (no rotation).
    This matches the current demos where pitch is kept at 0 and other
    orientation axes are ignored / not actuated.
    """

    if algorithm == "a_star":
        if astar_params is None:
            astar_params = AStarParams()
        return create_astar_3d_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacle_data,
            grid_resolution=grid_resolution,
            safety_margin=safety_margin,
            astar_params=astar_params,
            normalizer_params=normalizer_params,
            verbose=verbose,
        )

    if algorithm == "rrt":
        if rrt_params is None:
            rrt_params = RRTParams()
        return create_rrt_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacle_data,
            grid_resolution=grid_resolution,
            safety_margin=safety_margin,
            use_3d=use_3d,
            rrt_params=rrt_params,
            normalizer_params=normalizer_params,
        )

    if algorithm == "rrt_star":
        if rrt_star_params is None:
            rrt_star_params = RRTStarParams()
        return create_rrt_star_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacle_data,
            grid_resolution=grid_resolution,
            safety_margin=safety_margin,
            use_3d=use_3d,
            rrt_params=rrt_star_params,
            normalizer_params=normalizer_params,
        )

    if algorithm == "prm":
        if prm_params is None:
            prm_params = PRMParams()
        return create_prm_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacle_data,
            grid_resolution=grid_resolution,
            safety_margin=safety_margin,
            use_3d=use_3d,
            prm_params=prm_params,
            normalizer_params=normalizer_params,
        )

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. Expected one of: 'a_star', 'rrt', 'rrt_star', 'prm'."
    )
