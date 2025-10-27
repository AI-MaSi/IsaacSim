from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat


def ee_to_log_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    log_cfg: SceneEntityCfg = SceneEntityCfg("log"),
    ee_body_name: str = "gripper_frame",
    ee_offset_local: tuple[float, float, float] = (0.0, 0.0, -0.115),
    use_l2: bool = True,
) -> torch.Tensor:
    """Negative distance between EE point and the log object in world frame.

    The EE point is defined as a fixed local offset from the given body name
    (default: gripper_frame, with Z = -115 mm).
    Returns a per-env tensor of shape (num_envs,) to be used directly as a reward.
    """
    robot = env.scene[robot_cfg.name]
    log = env.scene[log_cfg.name]

    # Resolve body id once (assumes constant) and cache in env for speed
    cache_key = "_masi_gripper_body_id"
    if not hasattr(env, cache_key):
        body_ids, _ = robot.find_bodies([ee_body_name])
        if len(body_ids) != 1:
            raise RuntimeError(f"EE body '{ee_body_name}' not found on robot")
        setattr(env, cache_key, int(body_ids[0]))
    body_id = getattr(env, cache_key)

    # Poses
    body_w = robot.data.body_state_w[:, body_id, :7]
    # Rotate local offset into world and add to body position
    offset_local = torch.tensor(ee_offset_local, device=robot.device).unsqueeze(0).expand(env.scene.num_envs, 3)
    R_w_ref = matrix_from_quat(body_w[:, 3:7])
    ee_pos_w = body_w[:, :3] + torch.bmm(R_w_ref, offset_local.unsqueeze(-1)).squeeze(-1)

    log_pos_w = log.data.root_pos_w[:, :3]

    d = ee_pos_w - log_pos_w
    if use_l2:
        dist = torch.linalg.norm(d, dim=1)
    else:
        dist = torch.sum(torch.abs(d), dim=1)
    # Reward: negative distance
    return -dist

