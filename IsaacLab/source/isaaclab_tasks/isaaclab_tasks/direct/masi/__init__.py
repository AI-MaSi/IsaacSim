# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
masi excavator project environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Masi-v0",
    entry_point=f"{__name__}.excv_env:ExcavatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.excv_env:ExcavatorEnvCfg",

        # direct copy from other examples:
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:excvPPORunnerCfg",
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        #"sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# add more environments here