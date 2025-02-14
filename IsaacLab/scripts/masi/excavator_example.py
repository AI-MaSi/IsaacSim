# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import MASIV0_CFG # isort:skip


print("I am alive.")


@configclass
class ExampleSceneCfg(InteractiveSceneCfg):

    """
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    """

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = MASIV0_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create IK controller configuration
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False, # If True, then the controller treats the input command as a delta change in the position/pose. Otherwise, the controller treats the input command as the absolute position/pose.

        #ik_method="dls",
        #ik_params={"lambda_val": 0.000},

        # test different ones to find the not worst

        ik_method="pinv",
        ik_params={"k_val": 6.0}
    )

    """
    ik_method: Literal['pinv', 'svd', 'trans', 'dls']
    Method for computing inverse of Jacobian.

    Moore-Penrose pseudo-inverse (“pinv”):
    “k_val”: Scaling of computed delta-joint positions (default: 1.0).

    Adaptive Singular Value Decomposition (“svd”):
    “k_val”: Scaling of computed delta-joint positions (default: 1.0).

    “min_singular_value”: Single values less than this are suppressed to zero (default: 1e-5).

    Jacobian transpose (“trans”):
    “k_val”: Scaling of computed delta-joint positions (default: 1.0).

    Damped Moore-Penrose pseudo-inverse (“dls”):
    “lambda_val”: Damping coefficient (default: 0.01).
    """


    # controller
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy() # basic frame marker config
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05) # size of the marker

    # add markers
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    #scoop_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/scoop"))

    # example movement
    # create goal poses with helper function (euler too hard for me)
    ee_goals = [create_pose(0.42, 0.21, 100.0),  # start position
                create_pose(0.42, 0.21, 100.0),  # start position again since we are just flipping actions each set step times for demo
                create_pose(0.48, 0.00, 70.0),    # getting ready for crazy digging action
                create_pose(0.25, -0.01, 140.0),     # sick digging action
                create_pose(0.23, 0.12, 170.0)   # holding the material
                ]


    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    #robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["scoop", "end_point"])
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["end_point"])


    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset time
            count = 0

            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()

            # reset robot after steps (disabled for now)
            #robot.write_joint_state_to_sim(joint_pos, joint_vel)
            #robot.reset()

            # reset actions (goal targets)
            ik_commands[:] = ee_goals[current_goal_idx]

            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()


            # reset controller
            diff_ik_controller.reset()

            # reset ik commands
            diff_ik_controller.set_command(ik_commands)

            # update goal index
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:

            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] # second body (back to first for testing!), 7 values (posX, posY, posZ, quatW, quatX, quatY, quatZ)



        # scoop_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]  # first body, 7 values (posX, posY, posZ, quatW, quatX, quatY, quatZ)
        #scoop_marker.visualize(scoop_pose_w[:, 0:3], scoop_pose_w[:, 3:7])


        # end effector
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])  # first 3 values are position, last 4 are quaternion
        # goal
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7]) # first 3 values are position, last 4 are quaternion


def create_pose(x, z, y_axis_rotation):
    """Smoothbrain XYZWXYZ helper for people like me:
    - Z axis points UP
    - Y axis points BACK
    - Rotation angle is measured from y axis

    """
    # Convert angle to radians
    angle = torch.tensor(y_axis_rotation * torch.pi / 180.0)

    # Calculate quaternion for Y-axis rotation
    qw = torch.cos(angle / 2)
    qy = torch.sin(angle / 2)

    # Return pose with position and quaternion
    return [x, 0.0, z, float(qw), 0.0, float(qy), 0.0]

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = ExampleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
