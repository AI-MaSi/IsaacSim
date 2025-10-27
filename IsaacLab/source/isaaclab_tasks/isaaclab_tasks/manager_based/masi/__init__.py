"""
MASI manager-based task package.

Provides a ManagerBasedRLEnv configuration for the MASI robot, including
an IK-based action with a simple reaching task towards a "log" object.
Also exposes a script-friendly velocity-integrated actuator.
"""

import gymnasium as gym

# Register a small, ready-to-train MASI IK task (32 envs variant included)

gym.register(
    id="Isaac-Masi-LogIK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.ik_rel_log_env_cfg:MasiIkReachLogEnvCfg",
        # optional agent configs (rl_games used most commonly)
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Masi-LogIK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.ik_rel_log_env_cfg:MasiIkReachLogEnvCfg_PLAY",
        # use a play-specific rl_games config tuned for 32 envs
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg_play.yaml",
    },
)
