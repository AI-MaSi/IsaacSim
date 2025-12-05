"""
Quick sanity check for A* planners (3D and planar).

Runs both planners against the default wall / start / goal configuration and
prints timing + waypoint counts. Intended for local validation of planar A*
after safety-margin changes.
"""

import time
from pathlib import Path
import sys

import numpy as np

# Make repository root importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration_files.pathing_config import DEFAULT_ENV_CONFIG
from pathing.path_planning_algorithms import (
    create_astar_3d_trajectory,
    create_astar_plane_trajectory,
)


def build_obstacle(env_cfg):
    """Construct a single-wall obstacle list from the default env config."""
    return [
        {
            "size": np.asarray(env_cfg.wall_size, dtype=np.float32),
            "pos": np.asarray(env_cfg.wall_pos, dtype=np.float32),
            "rot": np.asarray(env_cfg.wall_rot, dtype=np.float32),
        }
    ]


def run_planner(name, fn, *, start_pos, goal_pos, obstacles, grid_resolution, safety_margin):
    t0 = time.time()
    path = fn(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacles,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
    )
    dt = time.time() - t0
    length = float(np.linalg.norm(path[-1] - path[0])) if len(path) > 1 else 0.0
    print(f"[{name}] waypoints={len(path)}, calc_time={dt:.3f}s, start->goal straight={length:.3f} m")
    return path


def main() -> None:
    env = DEFAULT_ENV_CONFIG
    obstacles = build_obstacle(env)
    start = tuple(env.point_a_pos)
    goal = tuple(env.point_b_pos)

    grid_resolution = 0.02
    safety_margin = 0.02  # non-zero margin to validate planar inflation fix

    print("Running A* planners with safety_margin =", safety_margin)
    path_3d = run_planner(
        "A* 3D",
        create_astar_3d_trajectory,
        start_pos=start,
        goal_pos=goal,
        obstacles=obstacles,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
    )
    path_plane = run_planner(
        "A* plane",
        create_astar_plane_trajectory,
        start_pos=start,
        goal_pos=goal,
        obstacles=obstacles,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
    )

    assert len(path_3d) > 1, "3D A* returned empty path"
    assert len(path_plane) > 1, "Planar A* returned empty path"
    print("[OK] Both planners produced valid paths.")


if __name__ == "__main__":
    main()
