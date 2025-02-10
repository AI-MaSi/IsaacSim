"""Test Configuration for the MASI excavator robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sensors import ImuCfg

##
# Configuration
##

MASIV0_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(

        # Hardcoded path to the .usd file!
        usd_path="C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Test_Bench_Model/for_scripts/model0.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Required properties from other examples
            disable_gravity=False,          # Enable/disable gravity
            retain_accelerations=False,     # Whether to keep acceleration data
            linear_damping=0.0,             # Linear motion damping coefficient
            angular_damping=0.0,            # Angular motion damping coefficient
            max_linear_velocity=1000.0,     # Maximum linear velocity limit
            max_angular_velocity=1000.0,    # Maximum angular velocity limit
            max_depenetration_velocity=1.0, # Maximum velocity for collision resolution

            # Optional properties from other examples
            enable_gyroscopic_forces=True,  # Enable gyroscopic forces
            # max_contact_impulse=1e32,               # Maximum contact force allowed
            # enable_momentum_forces=False,           # Enable momentum preservation
            # disable_preprocessing=False,            # Disable constraint preprocessing
            # disable_ccd=False,                      # Disable continuous collision detection (not tested)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(

            enabled_self_collisions=False,      # Whether to enable self-collisions
            solver_position_iteration_count=4,  # Number of position iterations for physics solver
            solver_velocity_iteration_count=1,  # Number of velocity iterations for physics solver
            sleep_threshold=0.005,              # Threshold for putting articulation to sleep
            stabilization_threshold=0.001,      # Threshold for stabilization

            # enable_self_collisions=False,           # Global self-collision setting
            # self_collision_filter_pairs=[],         # Specify collision filtering pairs
            # enable_driving_constraints=True,        # Enable joint constraints
            # enable_projection=True,                 # Enable joint projection
            # projection_iterations=4,                # Number of projection iterations
            # min_position_iteration_count=2,         # Minimum position solver iterations
            # min_velocity_iteration_count=1,         # Minimum velocity solver iterations
        ),
        copy_from_source=True,                        # Copy configs from source USD file (if available, setting values here will override)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),                   # Z1.0 up from ground for debugging. EDIT: dont change here, change in the scene!
        # rot=(1.0, 0.0, 0.0, 0.0),            # Optional: Initial rotation (w,x,y,z)
        # lin_vel=(0.0, 0.0, 0.0),             # Optional: Initial linear velocity
        # ang_vel=(0.0, 0.0, 0.0),             # Optional: Initial angular velocity

        #joint_pos={
        #    "revolute_lift": 0.0,               # radians!
        #    "revolute_tilt": 0.0,
        #    "revolute_scoop": 0.0,
        #},
        # joint_vel={},                         # Optional: Initial joint velocities
        # joint_effort={},                      # Optional: Initial joint efforts
    ),
    actuators={
        "excavator": ImplicitActuatorCfg(
            joint_names_expr=["revolute_lift", "revolute_tilt", "revolute_scoop"],


            # custom actuator properties
             stiffness={
                 "revolute_lift": 600, # Nm/rad
                 "revolute_tilt": 600,
                 "revolute_scoop": 600,
             },
             damping={
                 "revolute_lift": 1.5, # Nm/(rad/s)
                 "revolute_tilt": 1.5,
                 "revolute_scoop": 1.5,
             },
             effort_limit={
                 "revolute_lift": 5000.0, # Nm
                 "revolute_tilt": 5000.0,
                 "revolute_scoop": 5000.0,
             },
             velocity_limit={
                 "revolute_lift": 2000.0, # rad/s
                 "revolute_tilt": 2000.0,
                 "revolute_scoop": 2000.0,
             },
        ),
    },
)


# sensors, now removed for V0 (using direct transforms) but kept for reference
"""
V0_IMU_CFG = {
    "imu_lift": ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_liftboom",
        update_period=0.0,
        debug_vis=True, # Visualize the sensor (red arrow)
        # noise_mean=(0.0, 0.0, 0.0),  # Optional: Mean of Gaussian noise for measurements
        # noise_std=(0.01, 0.01, 0.01),  # Optional: Standard deviation of noise
        # gravity_compensation=True,  # Optional: Compensate for gravity in accelerometer
        # gravity_bias=(0.0, 0.0, 0.0),  # Optional: Bias in gravity measurement
        # offset=ImuCfg.OffsetCfg(  # Optional: Sensor mounting offset
        #     pos=(0.0, 0.0, 0.0),
        #     rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w,x,y,z)
        #     convention="ros"  # Coordinate convention (ros or isaac)
    ),
    "imu_tilt": ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_tiltboom",
        update_period=0.0,
        debug_vis=True,
        # ...
    ),
    "imu_scoop": ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_scoop",
        update_period=0.0,
        debug_vis=True,
    ),
}
"""