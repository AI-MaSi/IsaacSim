from __future__ import annotations

import math
import torch

from isaaclab.managers import SceneEntityCfg


def reset_log_radial(
    env,
    distance_range: tuple[float, float] = (0.4, 0.6),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    z_height: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("log"),
) -> None:
    """Reset the log object around the robot in a 360-degree ring with random yaw.

    - Samples radius uniformly in [r_min, r_max] and angle uniformly in [-pi, pi].
    - Places the log center at that XY relative to the robot root, with fixed Z height.
    - Sets roll=pitch=0, yaw=random, i.e., the log lies on the ground with Z-up.
    """
    scene = env.scene
    robot = scene["robot"]
    log = scene[asset_cfg.name]

    num_envs = scene.num_envs
    device = robot.device

    # Robot roots in world
    root_w = robot.data.root_state_w[:, :7]

    r_min, r_max = distance_range
    th_min, th_max = yaw_range

    # Sample polar coordinates
    radii = torch.empty((num_envs,), device=device).uniform_(r_min, r_max)
    thetas = torch.empty((num_envs,), device=device).uniform_(th_min, th_max)

    # Positions relative to robot base (in its local XY) then map to world
    # Compute world basis from base orientation
    R_bw = env.scene["robot"].data.root_quat_w
    # Use matrix conversion
    from isaaclab.utils.math import matrix_from_quat

    R_bw_m = matrix_from_quat(root_w[:, 3:7])
    xy_local = torch.stack([torch.cos(thetas) * radii, torch.sin(thetas) * radii, torch.zeros_like(radii)], dim=1)
    xy_world = torch.bmm(R_bw_m, xy_local.unsqueeze(-1)).squeeze(-1)

    # Final world positions
    pos_w = root_w[:, :3] + xy_world
    pos_w[:, 2] = z_height

    # Orientations: Z-up with random yaw around Z
    # quaternion from yaw
    half = thetas * 0.5
    yaw_quat = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)

    # Write root state
    log.write_root_pose_to_sim(pos_w, yaw_quat)

