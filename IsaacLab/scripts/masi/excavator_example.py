# TODO: limit joints speeds

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--trajectory", type=str, default="circle", choices=["circle", "square", "figure_eight"],
                    help="Type of trajectory to follow.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

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
from isaaclab_assets import MASIV0_CFG  # isort:skip

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


# Trajectory generation functions
def create_square_trajectory(center_x, center_z, side_length, y_axis_rotation, num_steps_per_side=50, device="cuda:0"):
    """
    Creates a square trajectory for the excavator end effector.

    Args:
        center_x (float): X-coordinate of the square center
        center_z (float): Z-coordinate of the square center
        side_length (float): Length of the square side
        y_axis_rotation (float): Rotation around the Y-axis in degrees
        num_steps_per_side (int): Number of steps to divide each side into (for smooth movement)
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_steps_total, 7] containing poses along the square path
    """
    # Calculate the half-side length
    half_side = side_length / 2.0

    # Define the four corners of the square relative to center
    corners = [
        (half_side, half_side),  # top right
        (-half_side, half_side),  # top left
        (-half_side, -half_side),  # bottom left
        (half_side, -half_side),  # bottom right
    ]

    # Create empty list to store all poses
    poses = []

    # Generate poses for each side of the square
    for i in range(4):
        # Get current corner and next corner
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]

        # Interpolate between corners
        for step in range(num_steps_per_side):
            # Linear interpolation between corners
            t = step / num_steps_per_side
            x_offset = start_corner[0] * (1 - t) + end_corner[0] * t
            z_offset = start_corner[1] * (1 - t) + end_corner[1] * t

            # Calculate absolute position
            x = center_x + x_offset
            z = center_z + z_offset

            # Create pose using the existing helper function
            pose = create_pose(x, z, y_axis_rotation)
            poses.append(pose)

    # Convert list of poses to tensor
    return torch.tensor(poses, device=device)


def create_circle_trajectory(center_x, center_z, diameter, y_axis_rotation, num_steps=200, device="cuda:0"):
    """
    Creates a circular trajectory for the excavator end effector.

    Args:
        center_x (float): X-coordinate of the circle center
        center_z (float): Z-coordinate of the circle center
        diameter (float): Diameter of the circle
        y_axis_rotation (float): Rotation around the Y-axis in degrees
        num_steps (int): Number of steps to divide the circle into (for smooth movement)
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_steps, 7] containing poses along the circular path
    """
    # Calculate the radius
    radius = diameter / 2.0

    # Create empty list to store all poses
    poses = []

    # Generate poses along the circle
    for step in range(num_steps):
        # Calculate angle for this step
        angle = 2 * np.pi * step / num_steps

        # Calculate position on circle
        x = center_x + radius * np.cos(angle)
        z = center_z + radius * np.sin(angle)

        # Create pose using the existing helper function
        pose = create_pose(x, z, y_axis_rotation)
        poses.append(pose)

    # Convert list of poses to tensor
    return torch.tensor(poses, device=device)


def create_figure_eight_trajectory(center_x, center_z, width, height, y_axis_rotation, num_steps=200, device="cuda:0"):
    """
    Creates a figure-eight trajectory for the excavator end effector.

    Args:
        center_x (float): X-coordinate of the figure-eight center
        center_z (float): Z-coordinate of the figure-eight center
        width (float): Width of the figure-eight
        height (float): Height of the figure-eight
        y_axis_rotation (float): Rotation around the Y-axis in degrees
        num_steps (int): Number of steps to divide the path into
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_steps, 7] containing poses along the figure-eight path
    """
    # Create empty list to store all poses
    poses = []

    # Generate poses along the figure-eight
    for step in range(num_steps):
        # Calculate parameter t (0 to 2π)
        t = 2 * np.pi * step / num_steps

        # Parametric equation for figure-eight
        x = center_x + (width / 2) * np.sin(t)
        z = center_z + (height / 2) * np.sin(t) * np.cos(t)

        # Create pose using the existing helper function
        pose = create_pose(x, z, y_axis_rotation)
        poses.append(pose)

    # Convert list of poses to tensor
    return torch.tensor(poses, device=device)


def create_pose(x, z, y_axis_rotation):
    """Smoothbrain XYZWXYZ position creator for people like me:
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create IK controller configuration
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        # If True, then the controller treats the input command as a delta change in the position/pose. Otherwise, the controller treats the input command as the absolute position/pose.

        # ik_method="dls",
        # ik_params={"lambda_val": 0.000},

        # test different ones to find the not worst

        ik_method="pinv",
        ik_params={"k_val": 6.0}
    )

    """
    ik_method: Literal['pinv', 'svd', 'trans', 'dls']
    Method for computing inverse of Jacobian.

    Moore-Penrose pseudo-inverse ("pinv"):
    "k_val": Scaling of computed delta-joint positions (default: 1.0).

    Adaptive Singular Value Decomposition ("svd"):
    "k_val": Scaling of computed delta-joint positions (default: 1.0).

    "min_singular_value": Single values less than this are suppressed to zero (default: 1e-5).

    Jacobian transpose ("trans"):
    "k_val": Scaling of computed delta-joint positions (default: 1.0).

    Damped Moore-Penrose pseudo-inverse ("dls"):
    "lambda_val": Damping coefficient (default: 0.01).
    """

    # controller
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()  # basic frame marker config
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)  # size of the marker

    # add markers
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Generate trajectory based on command line argument
    if args_cli.trajectory == "circle":
        trajectory = create_circle_trajectory(
            center_x=0.45,  # X-coordinate of circle center
            center_z=0.05,  # Z-coordinate of circle center
            diameter=0.15,  # Diameter of the circle
            y_axis_rotation=90.0,  # Rotation around Y-axis in degrees
            num_steps=300,  # Number of steps for smooth movement
            device=sim.device  # Device to create tensor on
        )
        print(f"Created circular trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "square":
        trajectory = create_square_trajectory(
            center_x=0.45,  # X-coordinate of square center
            center_z=0.00,  # Z-coordinate of square center
            side_length=0.2,  # Length of square side
            y_axis_rotation=130.0,  # Rotation around Y-axis in degrees
            num_steps_per_side=60,  # Steps per side for smooth movement
            device=sim.device  # Device to create tensor on
        )
        print(f"Created square trajectory with {len(trajectory)} points")
    else:  # figure_eight
        trajectory = create_figure_eight_trajectory(
            center_x=0.60,  # X-coordinate of figure-eight center
            center_z=-0.0,  # Z-coordinate of figure-eight center
            width=0.25,  # Width of figure-eight
            height=0.20,  # Height of figure-eight
            y_axis_rotation=50.0,  # Rotation around Y-axis in degrees
            num_steps=350,  # Number of steps for smooth movement
            device=sim.device  # Device to create tensor on
        )
        print(f"Created figure-eight trajectory with {len(trajectory)} points")

    # Create buffer to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)

    # Initialize trajectory index
    trajectory_idx = 0
    total_trajectory_points = len(trajectory)

    # Specify robot-specific parameters
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

    # Initialize the robot
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # Reset controller
    diff_ik_controller.reset()

    # Set initial command
    ik_commands[:] = trajectory[trajectory_idx]
    diff_ik_controller.set_command(ik_commands)

    # Simulation loop
    while simulation_app.is_running():
        # Update trajectory point every few steps for smooth movement
        if count % 5 == 0:  # Adjust this value to change the speed of movement
            # Move to next point in trajectory
            trajectory_idx = (trajectory_idx + 1) % total_trajectory_points

            # Update commands
            ik_commands[:] = trajectory[trajectory_idx]
            diff_ik_controller.set_command(ik_commands)

        # Obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update sim-time
        count += 1

        # Update buffers
        scene.update(sim_dt)

        # Obtain quantities from simulation for visualization
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # Visualize end effector and goal positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.5, 2.0, 0.5], [0.2, 0.0, 0.0])
    # Design scene
    scene_cfg = ExampleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(f"[INFO]: Running with {args_cli.trajectory} trajectory...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()