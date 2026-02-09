"""Configuration for the MASI excavator robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


##
# Configuration
##

MASI_PATHING_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # where to find the robot prim
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/excv_pathing.usd",
        copy_from_source=True,             # copy physical properties and joints from USD
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), # raise Z 0.09 to match lower carriage height
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "main_joints": ImplicitActuatorCfg(
            joint_names_expr=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop"],

            stiffness={
                "revolute_cabin": 600.0,
                "revolute_lift": 600.0,
                "revolute_tilt": 600.0,
                "revolute_scoop": 600.0,
            },
            damping={
                "revolute_cabin": 40.0,
                "revolute_lift": 40.0,
                "revolute_tilt": 40.0,
                "revolute_scoop": 40.0,
            },

            velocity_limit={  # in sim deg/s, here rad/s haha
                "revolute_cabin": 1.5,
                "revolute_lift": 1.0,
                "revolute_tilt": 1.0,
                "revolute_scoop": 1.0,
            },
            effort_limit={
                "revolute_cabin": 200.0,
                "revolute_lift": 500.0,
                "revolute_tilt": 500.0,
                "revolute_scoop": 500.0,
            },

            friction={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },

            armature={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },
        ),
    },
)

MASIV2_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # where to find the robot prim
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model2.usd",
        copy_from_source=True,             # copy physical properties and joints from USD
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.09), # Z raised to match lower carriage height
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "main_joints": ImplicitActuatorCfg(
            joint_names_expr=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop"],

            stiffness={
                "revolute_cabin": 600.0,
                "revolute_lift": 600.0,
                "revolute_tilt": 600.0,
                "revolute_scoop": 600.0,
            },
            damping={
                "revolute_cabin": 40.0,
                "revolute_lift": 40.0,
                "revolute_tilt": 40.0,
                "revolute_scoop": 40.0,
            },

            velocity_limit={  # in sim deg/s, here rad/s haha
                "revolute_cabin": 10.0, # 1.7
                "revolute_lift": 10.0,
                "revolute_tilt": 10.0,
                "revolute_scoop": 10.0,
            },
            effort_limit={
                "revolute_cabin": 500.0,
                "revolute_lift": 1000.0,
                "revolute_tilt": 1000.0,
                "revolute_scoop": 1000.0,
            },

            friction={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },

            armature={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },
        ),

        "tool": ImplicitActuatorCfg(
            joint_names_expr=["revolute_gripper", "revolute_claw_1", "revolute_claw_2"],

            stiffness={
                "revolute_gripper": 600.0,
                "revolute_claw_1": 600.0,
                "revolute_claw_2": 600.0,

            },
            damping={
                "revolute_gripper": 40.0,
                "revolute_claw_1": 40.0,
                "revolute_claw_2": 40.0,

            },
            velocity_limit={
                "revolute_gripper": 1.7,
                "revolute_claw_1": 2.0,
                "revolute_claw_2": 2.0,

            },
            effort_limit={
                "revolute_gripper": 50.0, # 20
                "revolute_claw_1": 50.0,
                "revolute_claw_2": 50.0,

            },
            friction={
                "revolute_gripper": 0.0,
                "revolute_claw_1": 0.0,
                "revolute_claw_2": 0.0,

            },
            armature={
                "revolute_gripper": 0.0,
                "revolute_claw_1": 0.0,
                "revolute_claw_2": 0.0,

            },
        ),
    },
)

MASI_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # where to find the robot prim
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/excavator.usd",
        copy_from_source=True,             # copy physical properties and joints from USD
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.09), # Z raised to match lower carriage height
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "main_joints": ImplicitActuatorCfg(
            joint_names_expr=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop"],

            stiffness={
                "revolute_cabin": 600.0,
                "revolute_lift": 600.0,
                "revolute_tilt": 600.0,
                "revolute_scoop": 600.0,
            },
            damping={
                "revolute_cabin": 40.0,
                "revolute_lift": 40.0,
                "revolute_tilt": 40.0,
                "revolute_scoop": 40.0,
            },

            velocity_limit={  # in sim deg/s, here rad/s haha
                "revolute_cabin": 10.0, # 1.7
                "revolute_lift": 10.0,
                "revolute_tilt": 10.0,
                "revolute_scoop": 10.0,
            },
            effort_limit={
                "revolute_cabin": 200.0,
                "revolute_lift": 500.0,
                "revolute_tilt": 500.0,
                "revolute_scoop": 500.0,
            },

            friction={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },

            armature={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
            },
        ),

        "tool": ImplicitActuatorCfg(
            joint_names_expr=["revolute_gripper", "revolute_claw_1", "revolute_claw_2"],

            stiffness={
                "revolute_gripper": 600.0,
                "revolute_claw_1": 600.0,
                "revolute_claw_2": 600.0,

            },
            damping={
                "revolute_gripper": 40.0,
                "revolute_claw_1": 40.0,
                "revolute_claw_2": 40.0,

            },
            velocity_limit={
                "revolute_gripper": 1.7,
                "revolute_claw_1": 2.0,
                "revolute_claw_2": 2.0,

            },
            effort_limit={
                "revolute_gripper": 20.0,
                "revolute_claw_1": 20.0,
                "revolute_claw_2": 20.0,

            },
            friction={
                "revolute_gripper": 0.0,
                "revolute_claw_1": 0.0,
                "revolute_claw_2": 0.0,

            },
            armature={
                "revolute_gripper": 0.0,
                "revolute_claw_1": 0.0,
                "revolute_claw_2": 0.0,

            },
        ),
    },
)