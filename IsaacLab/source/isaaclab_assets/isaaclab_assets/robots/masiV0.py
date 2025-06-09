"""Configuration for the MASI excavator robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


##
# Configuration
##

# Model0, lift, tilt, scoop
MASIV0_CFG = ArticulationCfg(
    # alot of example configs, but not all are needed
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Hardcoded path to the .usd file!
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model0.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Required properties
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,  # Increased from 1.0 to help with collision resolution

            # Additional properties from examples
            enable_gyroscopic_forces=True,
            max_contact_impulse=1e4,  # Maximum contact force allowed (limit impacts)
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Keep false for efficiency if no self-collisions expected
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,

        ),


        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,  # Distance to start collision detection
            rest_offset=0.0,  # Rest distance between colliding objects
        ),


        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="position",  # Can be "force" or "position"
        ),

        copy_from_source=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        #pos=(0.0, 0.0, 0.0),

        # initial joint angles in radians
        joint_pos={
            "revolute_lift": 0.0,
            "revolute_tilt": 0.0,
            "revolute_scoop": 0.0,
        },
        joint_vel={".*": 0.0},  # Set all joint velocities to zero initially
    ),

    # Soft joint limits to prevent hitting hard stops
    soft_joint_pos_limit_factor=1.00,

    actuators={
        "excavator": ImplicitActuatorCfg(
            joint_names_expr=["revolute_lift", "revolute_tilt", "revolute_scoop"],
            stiffness={
                "revolute_lift": 1000,
                "revolute_tilt": 1000,
                "revolute_scoop": 1000,
            },
            damping={
                "revolute_lift": 1.5,
                "revolute_tilt": 1.5,
                "revolute_scoop": 1.5,
            },
            velocity_limit={ # in sim deg/s, here rad/s haha
                "revolute_lift": 0.6,
                "revolute_tilt": 0.6,
                "revolute_scoop": 0.6,
            },

            friction={
                "revolute_lift": 0.1,
                "revolute_tilt": 0.1,
                "revolute_scoop": 0.1,
            },
            # armature = "shaft inertia"
            armature={
                "revolute_lift": 0.01,
                "revolute_tilt": 0.01,
                "revolute_scoop": 0.01,
            },
        ),
    },
)


# Model2, slew, lift, tilt, scoop, gripper rotator, gripper
MASIV2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Hardcoded path to the .usd file!
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model2.usd",
        #usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model2_mimic.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Required properties
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,  # Increased from 1.0 to help with collision resolution

            # Additional properties from examples
            enable_gyroscopic_forces=True,
            max_contact_impulse=1e4,  # Maximum contact force allowed (limit impacts)
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Keep false for efficiency if no self-collisions expected
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,

        ),


        collision_props=sim_utils.CollisionPropertiesCfg(
            #contact_offset=0.002,  # Distance to start collision detection  # default -inf?
            rest_offset=0.0,  # Rest distance between colliding objects
        ),
    copy_from_source=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),

        # initial joint angles in radians
        joint_pos={
            "revolute_cabin": 0.0,
            "revolute_lift": 0.0,
            "revolute_tilt": 0.0,
            "revolute_scoop": 0.0,
            "revolute_gripper": 0.0,
            "revolute_claw_1": 0.0,
            "revolute_claw_2": 0.0,
        },
        joint_vel={".*": 0.0},  # Set all joint velocities to zero initially
    ),

    # joint limits?


    soft_joint_pos_limit_factor=0.95,


    actuators={
        # main movements


        "main_joints": ImplicitActuatorCfg(
            joint_names_expr=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop", "revolute_gripper"],    # "revolute_claw"

            #min_delay=5,  # timesteps minimum delay
            #max_delay=5,  # timesteps maximum delay

            stiffness={
                "revolute_cabin": 500.0,
                "revolute_lift": 800.0,
                "revolute_tilt": 600.0,
                "revolute_scoop": 500.0,
                "revolute_gripper": 100.0,
                #"revolute_claw": 600.0,
            },
            damping={
                "revolute_cabin": 5.0,
                "revolute_lift": 40.0,
                "revolute_tilt": 40.0,
                "revolute_scoop": 40.0,
                "revolute_gripper": 25.0,
                #"revolute_claw": 25.0,
            },

            velocity_limit={ # in sim deg/s, here rad/s haha
                "revolute_cabin": 0.7,       # 0.7 = ~40 deg/s
                "revolute_lift": 0.7,
                "revolute_tilt": 0.7,
                "revolute_scoop": 0.7,
                "revolute_gripper": 1.05,   # 1.05 = ~60 deg/s
                #"revolute_claw": 1.3,       # 1.3 = ~75 deg/s
            },
            effort_limit={
                "revolute_cabin": 200.0,
                "revolute_lift": 400.0,
                "revolute_tilt": 300.0,
                "revolute_scoop": 350.0,
                "revolute_gripper": 200.0,
                #"revolute_claw": 200.0,
            },

            friction={
                "revolute_cabin": 0.1,
                "revolute_lift": 0.1,
                "revolute_tilt": 0.1,
                "revolute_scoop": 0.1,
                "revolute_gripper": 0.1,
                #"revolute_claw": 0.1,
            },

            armature={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
                "revolute_gripper": 0.0,
                #"revolute_claw": 1.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["revolute_claw_1", "revolute_claw_2"], # only one joint with mimic joint, but its causing issues

            #min_delay=5,  # timesteps minimum delay
            #max_delay=5,  # timesteps maximum delay

            stiffness={
                "revolute_claw_1": 600.0,
                "revolute_claw_2": 600.0,
            },
            damping={
                "revolute_claw_1": 30.0,
                "revolute_claw_2": 30.0,
            },
            velocity_limit={
                "revolute_claw_1": 1.4,
                "revolute_claw_2": 1.4,
            },
            effort_limit={
                "revolute_claw_1": 500.0,
                "revolute_claw_2": 500.0,
            },
            friction={
                "revolute_claw_1": 0.1,
                "revolute_claw_2": 0.1,
            },

            # armature = "shaft inertia".
            armature={
                "revolute_claw_1": 0.0,
                "revolute_claw_2": 0.0,
            },
        ),
    },
)


# Model2, slew, lift, tilt, scoop, gripper rotator, gripper
# Mimic joint causing issues atm, but once fixed this should be good
MASIV2MIMIC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Hardcoded path to the .usd file!
        #usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model2.usd",
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Excavator_model/model2_mimic.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Required properties
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,  # Increased from 1.0 to help with collision resolution

            # Additional properties from examples
            enable_gyroscopic_forces=True,
            max_contact_impulse=1e4,  # Maximum contact force allowed (limit impacts)
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Keep false for efficiency if no self-collisions expected
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,

        ),


        collision_props=sim_utils.CollisionPropertiesCfg(
            #contact_offset=0.002,  # Distance to start collision detection  # default -inf?
            rest_offset=0.0,  # Rest distance between colliding objects
        ),
    copy_from_source=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),

        # initial joint angles in radians
        joint_pos={
            "revolute_cabin": 0.0,
            "revolute_lift": 0.0,
            "revolute_tilt": 0.0,
            "revolute_scoop": 0.0,
            "revolute_gripper": 0.0,
            "revolute_claw": 0.0,
            "revolute_mimic": 0.0,
        },
        joint_vel={".*": 0.0},  # Set all joint velocities to zero initially
    ),

    # joint limits?


    soft_joint_pos_limit_factor=0.95,


    actuators={
        # main movements

        "main_joints": ImplicitActuatorCfg(
            joint_names_expr=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop", "revolute_gripper"],    # "revolute_claw"

            stiffness={
                "revolute_cabin": 500.0,
                "revolute_lift": 800.0,
                "revolute_tilt": 600.0,
                "revolute_scoop": 500.0,
                "revolute_gripper": 100.0,
                #"revolute_claw": 600.0,
            },
            damping={
                "revolute_cabin": 5.0,
                "revolute_lift": 40.0,
                "revolute_tilt": 40.0,
                "revolute_scoop": 40.0,
                "revolute_gripper": 25.0,
                #"revolute_claw": 25.0,
            },

            velocity_limit={ # in sim deg/s, here rad/s haha
                "revolute_cabin": 0.7,       # 0.7 = ~40 deg/s
                "revolute_lift": 0.7,
                "revolute_tilt": 0.7,
                "revolute_scoop": 0.7,
                "revolute_gripper": 1.05,   # 1.05 = ~60 deg/s
                #"revolute_claw": 1.3,       # 1.3 = ~75 deg/s
            },
            effort_limit={
                "revolute_cabin": 200.0,
                "revolute_lift": 400.0,
                "revolute_tilt": 300.0,
                "revolute_scoop": 350.0,
                "revolute_gripper": 200.0,
                #"revolute_claw": 200.0,
            },

            friction={
                "revolute_cabin": 0.1,
                "revolute_lift": 0.1,
                "revolute_tilt": 0.1,
                "revolute_scoop": 0.1,
                "revolute_gripper": 0.1,
                #"revolute_claw": 0.1,
            },

            armature={
                "revolute_cabin": 0.0,
                "revolute_lift": 0.0,
                "revolute_tilt": 0.0,
                "revolute_scoop": 0.0,
                "revolute_gripper": 0.0,
                #"revolute_claw": 1.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["revolute_claw"],

            stiffness={
            stiffness={
                "revolute_claw": 600.0,
            },
            damping={
                "revolute_claw": 30.0,
            },
            velocity_limit={
                "revolute_claw": 1.05,
            },
            effort_limit={
                "revolute_claw": 250.0,
            },
            friction={
                "revolute_claw": 0.1,
            },

            # armature = "shaft inertia".
            armature={
                "revolute_claw": 0.0,
            },
        ),
    },
)

