"""Launch Isaac Sim Simulator first."""


# TODO: fix trajectories, they are direct copy form model0
# TODO: make sure joint weighting is working


# trajectory generators copied from model0_example!

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller in relative mode.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

parser.add_argument(
    "--trajectory",
    type=str,
    default="circle",
    choices=["circle", "square", "figure_eight", "rotate", "rotate2", "step"],
    help="Type of trajectory to follow."
)

parser.add_argument(
    "--wait_steps",
    type=int,
    default=250,
    help="Number of simulation steps to wait at each target pose for step_test trajectory."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from typing import Literal
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
#from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.masi import DifferentialIKController, DifferentialIKControllerCfg # weighting added
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, matrix_from_quat, quat_inv, combine_frame_transforms

##
# Pre-defined configs
##
#from isaaclab_assets import MASIV2_CFG as CFG  # isort:skip
from isaaclab_assets import MASIV2_MIMIC_CFG as CFG  # isort:skip


@configclass
class ExampleSceneCfg(InteractiveSceneCfg):
    # ground plane
    #ground = AssetBaseCfg(
    #    prim_path="/World/defaultGroundPlane",
    #    spawn=sim_utils.GroundPlaneCfg(),
    #    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Trajectory generation functions (for absolute mode)
def create_square_trajectory(center_x, center_z, side_length, y_axis_rotation, num_steps_per_side, device="cuda:0"):
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


def create_circle_trajectory(center_x, center_z, diameter, y_axis_rotation, num_steps, device="cuda:0"):
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


def create_figure_eight_trajectory(center_x, center_z, width, height, y_axis_rotation, num_steps, device="cuda:0"):
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


def create_rotation_trajectory(center_x, center_z, width, height, frequency, y_axis_rotation,
                               rotation, num_steps, device="cuda:0"):
    """
    Creates a trajectory that follows a sine wave with multiple oscillations along Z with varying rotation.

    Args:
        center_x (float): X-coordinate of the trajectory center
        center_z (float): Z-coordinate of the trajectory center
        width (float): Amplitude of the sine wave in X direction (default: 0.1)
        height (float): Total height of the trajectory in Z direction (default: 0.3)
        frequency (float): Number of complete sine waves in the trajectory (default: 3.0)
        y_axis_rotation (float): Base rotation around the Y-axis in degrees
        rotation (float): Total rotation range in degrees (y_axis_rotation ± rotation/2)
        num_steps (int): Total number of steps in the trajectory
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_steps, 7] containing poses along the path
    """
    # Create empty list to store all poses
    poses = []

    # Calculate rotation range
    half_rotation = rotation / 2.0
    min_rotation = y_axis_rotation - half_rotation
    max_rotation = y_axis_rotation + half_rotation

    # Generate trajectory points
    for step in range(num_steps):
        # Parameter t from 0 to 2π (for a full cycle)
        t = 2 * np.pi * step / num_steps

        # First half goes up, second half goes down (mirroring the motion)
        z_progress = step / num_steps  # Normalized progress (0 to 1)

        # Calculate X position with sine wave
        x = center_x + width * np.sin(frequency * 2 * np.pi * z_progress)

        # Calculate Z position (linear from bottom to top and back)
        if step < num_steps / 2:
            # First half: go up
            z = center_z + height * (2 * step / num_steps)
        else:
            # Second half: go down
            z = center_z + height * (2 - 2 * step / num_steps)

        # Calculate Y-axis rotation (oscillates between min and max)
        current_rotation = min_rotation + (max_rotation - min_rotation) * (0.5 + 0.5 * np.sin(t))

        # Create pose using the existing helper function
        pose = create_pose(x, z, current_rotation)
        poses.append(pose)

    # Convert list of poses to tensor
    return torch.tensor(poses, device=device)


def create_rotation(center_x, center_z, y_axis_rotation, rotation, num_steps, device="cuda:0"):
    """
    Creates a trajectory that keeps the end effector at a fixed position
    while rotating back and forth around the Y-axis.

    Args:
        center_x (float): Fixed X-coordinate of the end effector
        center_z (float): Fixed Z-coordinate of the end effector
        y_axis_rotation (float): Base rotation around the Y-axis in degrees
        rotation (float): Total rotation range in degrees (y_axis_rotation ± rotation/2)
        num_steps (int): Total number of steps in the trajectory
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_steps, 7] containing poses along the path
    """
    # Create empty list to store all poses
    poses = []

    # Calculate rotation range
    half_rotation = rotation / 2.0
    min_rotation = y_axis_rotation - half_rotation
    max_rotation = y_axis_rotation + half_rotation

    # Generate trajectory points
    for step in range(num_steps):
        # Use sine function to create smooth back-and-forth rotation
        # This gives values between -1 and 1, which we scale to our rotation range
        phase = 2 * np.pi * step / num_steps
        rotation_factor = np.sin(phase)

        # Calculate current rotation angle
        current_rotation = y_axis_rotation + rotation_factor * half_rotation

        # Create pose at fixed position with changing rotation
        pose = create_pose(center_x, center_z, current_rotation)
        poses.append(pose)

    # Convert list of poses to tensor
    return torch.tensor(poses, device=device)


def create_pose(x, z, y_axis_rotation):
    """Smoothbrain XYZWXYZ position creator for people like me:
    - Z axis points UP
    - Rotation angle is measured around Y axis (yaw)
    """
    # Convert angle to radians
    angle = torch.tensor(y_axis_rotation * torch.pi / 180.0)

    qw = torch.cos(angle / 2.0)
    qx = 0.0
    qy = torch.sin(angle / 2.0)
    qz = 0.0

    # Return pose with position [x, y, z] and quaternion [qw, qx, qy, qz]
    y_fixed_for_now = 0.0
    return [x, y_fixed_for_now, z, float(qw), float(qx), float(qy), float(qz)]


def create_step_test_trajectory(poses_list, device="cuda:0"):
    """
    Creates a trajectory from a list of direct end-point poses.

    Args:
        poses_list (list[list[float]]): A list where each element is a target point [x, z, y_rot].
        device (str): Device to create the tensor on

    Returns:
        torch.Tensor: Tensor of shape [num_poses, 7] containing the target poses.
    """
    # Convert the simple [x, z, y_rot] list to the full [x, y, z, qw, qx, qy, qz] format
    full_poses = [create_pose(p[0], p[1], p[2]) for p in poses_list]
    return torch.tensor(full_poses, device=device)

def limit_step(vec, max_norm):
    """Limit relative step size to a maximum norm."""
    norm = torch.norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return vec * scale

def compute_relative_ik_command(
        current_ee_pos: torch.Tensor,
        current_ee_quat: torch.Tensor,
        target_ee_pos: torch.Tensor,
        target_ee_quat: torch.Tensor,
        command_type: Literal["position", "pose"],
        device: str = "cuda:0",
) -> torch.Tensor:
    """
    Computes the relative command (delta) for DifferentialIKController based on current and target EE poses.

    Args:
        current_ee_pos: Current end-effector position tensor (Num_envs, 3).
        current_ee_quat: Current end-effector orientation tensor (Num_envs, 4), (w, x, y, z).
        target_ee_pos: Target end-effector position tensor (Num_envs, 3).
        target_ee_quat: Target end-effector orientation tensor (Num_envs, 4), (w, x, y, z).
                         Only used if command_type is "pose".
        command_type: The type of control desired ("position" or "pose").
                      Determines the output dimension (3 for position, 6 for pose).
        device: The torch device for calculations.

    Returns:
        The computed delta command tensor (Num_envs, 3) for "position" or (Num_envs, 6) for "pose".
        For pose, the format is (dx, dy, dz, droll, dpitch, dyaw) using axis-angle error.

    Raises:
        ValueError: If an unknown command_type is provided.
    """
    # Ensure inputs are on the correct device
    current_ee_pos = current_ee_pos.to(device)
    current_ee_quat = current_ee_quat.to(device)
    target_ee_pos = target_ee_pos.to(device)
    target_ee_quat = target_ee_quat.to(device)

    if command_type == "position":
        # Calculate position error (delta position)
        delta_command = target_ee_pos - current_ee_pos
        # Expected output shape: (N, 3)
        return delta_command

    elif command_type == "pose":
        # Calculate pose error using Isaac Lab utility
        # compute_pose_error returns (position_error, axis_angle_error)
        position_error, axis_angle_error = compute_pose_error(
            current_ee_pos,
            current_ee_quat,
            target_ee_pos,
            target_ee_quat,
            rot_error_type="axis_angle",  # Use axis-angle for 6D command
        )
        # Concatenate position and orientation error for the 6D command
        delta_command = torch.cat((position_error, axis_angle_error), dim=1)
        # Expected output shape: (N, 6)
        return delta_command

    else:
        raise ValueError(f"Unknown command_type: '{command_type}'. Must be 'position' or 'pose'.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]


    # Create IK controller configuration
    # We'll use pose relative mode for full control
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="svd",  # damped least squares method
        ik_params={"k_val": 0.6,
                   "lambda_val": 0.002, #for DLS
                   "min_singular_value": 1e-5,  # for SVD
                   "position_weight": 1.0,
                   "rotation_weight": 0.8,
                   # For SVD and DLS methods, we can specify joint weights
                   "joint_weights": [1.0, 1.0, 1.0, 1.0, 1.0]  # rotate, lift, tilt, scoop, gripper_rotate
                }
    )

    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Create markers to visualize origins
    origin_marker_cfg = FRAME_MARKER_CFG.copy()
    origin_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # World origin marker
    world_origin_marker = VisualizationMarkers(origin_marker_cfg.replace(prim_path="/Visuals/world_origin"))
    # Robot base origin marker
    robot_origin_marker = VisualizationMarkers(origin_marker_cfg.replace(prim_path="/Visuals/robot_origin"))

    # Create a marker to visualize the relative command
    relative_marker_cfg = CUBOID_MARKER_CFG.copy()
    relative_marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
    relative_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
    relative_marker = VisualizationMarkers(relative_marker_cfg.replace(prim_path="/Visuals/relative_command"))

    trajectory_mode = "smooth"  # Default mode
    wait_steps_counter = 0
    wait_steps_duration = 0

    # Generate trajectory based on command line argument
    if args_cli.trajectory == "circle":
        trajectory = create_circle_trajectory(
            center_x=0.55, center_z=0.05, diameter=0.15, y_axis_rotation=0.0, num_steps=300, device=sim.device
        )
        print(f"Created circular trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "square":
        trajectory = create_square_trajectory(
            center_x=0.45, center_z=0.00, side_length=0.2, y_axis_rotation=0.0, num_steps_per_side=120,
            device=sim.device
        )
        print(f"Created square trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "figure_eight":
        trajectory = create_figure_eight_trajectory(
            center_x=0.60, center_z=-0.0, width=0.25, height=0.20, y_axis_rotation=0.0, num_steps=350,
            device=sim.device
        )
        print(f"Created figure-eight trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "rotate":
        trajectory = create_rotation_trajectory(
            center_x=0.50, center_z=-0.07, width=0.02, height=0.20, frequency=5.0, y_axis_rotation=0.0, rotation=60.0,
            num_steps=300, device=sim.device
        )
        print(f"Created rotating trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "rotate2":
        trajectory = create_rotation(
            center_x=0.45, center_z=0.0, y_axis_rotation=100.0, rotation=90.0, num_steps=200, device=sim.device
        )
        print(f"Created rotating trajectory with {len(trajectory)} points")
    elif args_cli.trajectory == "step":
        trajectory_mode = "step"
        # Define the sequence of target points [x, z, y_rot]
        step_poses = [
            [0.40, -0.1, 90.0],  # Pose 1
            [0.65, 0.0, 90.0],  # Pose 2
            # Add more poses as needed
        ]
        trajectory = create_step_test_trajectory(step_poses, device=sim.device)
        wait_steps_duration = args_cli.wait_steps  # Get wait duration from args
        print(
            f"Created step test trajectory with {len(trajectory)} points, waiting {wait_steps_duration} steps at each.")
    else:
        raise ValueError(f"Unknown trajectory type: {args_cli.trajectory}")


    # Buffer to store absolute target pose for visualization
    target_ee_pose_b = torch.zeros(scene.num_envs, 7, device=robot.device)
    target_ee_pose_w = torch.zeros(scene.num_envs, 7, device=robot.device)  # Target pose in world frame

    # Initialize trajectory index
    trajectory_idx = 0
    total_trajectory_points = len(trajectory)

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot",
                                      joint_names=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop",
                                                   "revolute_gripper"], body_names=["ee"]) # "ee"
    robot_entity_cfg.resolve(scene)

    print(f"Robot is fixed base: {robot.is_fixed_base}")
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
    robot.write_joint_state_to_sim(joint_pos, joint_vel)  # Set initial state
    scene.write_data_to_sim()  # Write initial state to sim
    sim.step()  # Step once to update physics state
    scene.update(sim_dt)  # Update buffers

    # Reset controller
    diff_ik_controller.reset()


    ee_pose_w_initial = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    root_pose_w_initial = robot.data.root_state_w[:, 0:7]

    # Compute frame in root frame
    ee_pos_b_initial, ee_quat_b_initial = subtract_frame_transforms(
        root_pose_w_initial[:, 0:3], root_pose_w_initial[:, 3:7],
        ee_pose_w_initial[:, 0:3], ee_pose_w_initial[:, 3:7]
    )

    # Set initial target from trajectory
    target_ee_pose_b[:] = trajectory[trajectory_idx]

    # Compute initial relative command
    ik_commands = compute_relative_ik_command(
        ee_pos_b_initial,
        ee_quat_b_initial,
        target_ee_pose_b[:, 0:3],
        target_ee_pose_b[:, 3:7],
        command_type="pose",
        device=sim.device
    )

    # Set initial command
    diff_ik_controller.set_command(ik_commands, ee_pos=ee_pos_b_initial, ee_quat=ee_quat_b_initial)

    # Transform target pose from base frame to world frame for visualization
    target_ee_pos_w, target_ee_quat_w = combine_frame_transforms(
        root_pose_w_initial[:, 0:3], root_pose_w_initial[:, 3:7],
        target_ee_pose_b[:, 0:3], target_ee_pose_b[:, 3:7]
    )
    target_ee_pose_w[:] = torch.cat([target_ee_pos_w, target_ee_quat_w], dim=-1)

    # Visualize goal in world frame
    goal_marker.visualize(target_ee_pose_w[:, 0:3], target_ee_pose_w[:, 3:7])

    # Visualize origins
    world_origin_pos = torch.zeros((scene.num_envs, 3), device=sim.device)
    world_origin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device).expand(scene.num_envs, 4)
    world_origin_marker.visualize(world_origin_pos + scene.env_origins, world_origin_quat)

    # Visualize robot base origin
    robot_origin_marker.visualize(
        root_pose_w_initial[:, 0:3] + scene.env_origins,
        root_pose_w_initial[:, 3:7]
    )

    # Print debug info
    print(
        f"IK Controller initialized with command type: {diff_ik_cfg.command_type}, relative mode: {diff_ik_cfg.use_relative_mode}")
    print(f"Robot root position (world frame): {root_pose_w_initial[:, 0:3]}")
    print(f"Environment origin: {scene.env_origins}")

    # Simulation loop
    while simulation_app.is_running():

        if trajectory_mode == "smooth":
            # Update trajectory point every few steps for smooth movement
            if count > 0 and count % 1 == 0:  # 1 == 0 was 5 == 0 Adjust speed, avoid update on first step (count=0)
                trajectory_idx = (trajectory_idx + 1) % total_trajectory_points
                target_ee_pose_b[:] = trajectory[trajectory_idx]

                # Compute relative command based on current and target poses
                ik_commands = compute_relative_ik_command(
                    ee_pos_b,
                    ee_quat_b,
                    target_ee_pose_b[:, 0:3],
                    target_ee_pose_b[:, 3:7],
                    command_type="pose",
                    device=sim.device
                )

                # Set the relative command
                diff_ik_controller.set_command(ik_commands, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

                # Update target pose in world frame for visualization
                target_ee_pos_w, target_ee_quat_w = combine_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    target_ee_pose_b[:, 0:3], target_ee_pose_b[:, 3:7]
                )
                target_ee_pose_w[:] = torch.cat([target_ee_pos_w, target_ee_quat_w], dim=-1)

                print(f"Step {count}: Set target to trajectory point {trajectory_idx}")


        # Obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Transform Jacobian to base frame (following the pattern from run_ik_demo.txt)
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        # Compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        #print(f"Step {count}: Commanding joint positions: {joint_pos_des}")

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

        # Visualize end effector (world frame)
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # Visualize goal marker in world frame
        goal_marker.visualize(target_ee_pose_w[:, 0:3], target_ee_pose_w[:, 3:7])

        # Visualize relative command as a small marker offset from current position
        # Transform the relative command to world frame for proper visualization
        relative_pos_w = ee_pose_w[:, 0:3] + torch.bmm(
            matrix_from_quat(root_pose_w[:, 3:7]),
            ik_commands[:, 0:3].unsqueeze(-1)
        ).squeeze(-1) * 1.0  # scaling factor for visibility
        relative_marker.visualize(relative_pos_w)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)  # Use device from args
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