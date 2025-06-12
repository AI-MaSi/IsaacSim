"""WIP, mainly contains helper functions for future stuff"""

# TODO: collision detection

import argparse
import numpy as np
import torch
from typing import Tuple, List, Optional

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Path planning foundation with obstacle avoidance.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_walls", type=int, default=3, help="Number of wall obstacles to spawn (0-5).")
parser.add_argument("--shape_noise", type=float, default=0.2, help="Shape noise for walls (0-1).")
parser.add_argument("--pos_noise", type=float, default=0.3, help="Position noise for walls (0-1).")
parser.add_argument("--rot_noise", type=float, default=0.1, help="Rotation noise for walls (0-1).")
parser.add_argument("--trajectory_smoothing", type=float, default=0.5, help="Trajectory smoothing factor (0-1).")
parser.add_argument("--max_spheres", type=int, default=100, help="Maximum number of trajectory spheres to display (for performance).")
parser.add_argument("--step_size", type=float, default=0.005, help="Step size between waypoints in meters (smaller=more waypoints=slower movement).")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from typing import Literal
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.masi import DifferentialIKController, DifferentialIKControllerCfg # custom ik
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, matrix_from_quat, quat_inv, quat_from_euler_xyz, euler_xyz_from_quat, combine_frame_transforms
from collections import deque

##
# Pre-defined configs
##
from isaaclab_assets import MASIV2_MIMIC_CFG as CFG  # isort:skip


@configclass
class PathPlanningSceneCfg(InteractiveSceneCfg):
    """Configuration for the path planning scene."""

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Robot
    robot = CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


class PathPlanningEnvironment:
    """Main class for path planning with obstacle avoidance."""

    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene, max_spheres: int = 100, step_size: float = 0.01):
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.max_spheres = max_spheres
        self.step_size = max(0.001, step_size)  # Minimum step size to prevent infinite loops

        # Initialize markers
        self._init_markers()

        # Initialize IK controller
        self._init_ik_controller()

        # Initialize robot entity config
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["revolute_cabin", "revolute_lift", "revolute_tilt",
                        "revolute_scoop", "revolute_gripper"],
            body_names=["ee"]
        )
        self.robot_entity_cfg.resolve(scene)

        # Compute EE jacobian index
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Storage for obstacles
        self.obstacles: List[RigidObject] = []
        self.obstacle_count = 0

        # Target pose storage (final target)
        self.final_target_pos = torch.zeros(scene.num_envs, 3, device=self.sim.device)
        self.final_target_quat = torch.zeros(scene.num_envs, 4, device=self.sim.device)
        self.final_target_quat[:, 0] = 1.0  # Initialize to identity quaternion

        # Current waypoint storage (what we're currently tracking)
        self.current_waypoint_pos = torch.zeros(scene.num_envs, 3, device=self.sim.device)
        self.current_waypoint_quat = torch.zeros(scene.num_envs, 4, device=self.sim.device)
        self.current_waypoint_quat[:, 0] = 1.0  # Initialize to identity quaternion

        # Trajectory storage and control
        self.trajectory_points = deque(maxlen=self.max_spheres)
        self.interpolated_trajectory: Optional[torch.Tensor] = None
        self.trajectory_step = 0

        print(f"[INFO]: Path planner initialized - Max spheres: {self.max_spheres}, Step size: {self.step_size}m")

    def _print_robot_info(self):
        """Print robot configuration for debugging."""
        print("\n[INFO]: Robot Configuration:")
        print("-" * 30)
        print(f"  Bodies: {len(self.robot.data.body_names)} total")
        print(f"  Joints: {len(self.robot.data.joint_names)} total")
        print(f"  EE Jacobian Index: {self.ee_jacobi_idx}")
        print("-" * 30 + "\n")

    def _init_markers(self):
        """Initialize visualization markers."""
        # Frame markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

        # Goal marker with different color/size for distinction
        goal_frame_marker_cfg = FRAME_MARKER_CFG.copy()
        goal_frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)  # Slightly larger
        self.goal_marker = VisualizationMarkers(goal_frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # Trajectory markers configuration
        # Blue spheres for actual path
        actual_path_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/actual_path",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            }
        )
        self.actual_path_marker = VisualizationMarkers(actual_path_cfg)

        # Red spheres for calculated path
        calculated_path_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/calculated_path",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            }
        )
        self.calculated_path_marker = VisualizationMarkers(calculated_path_cfg)

    def _init_ik_controller(self):
        """Initialize the differential IK controller."""
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="svd",
            ik_params={
                "k_val": 0.8,               # Gain
                "lambda_val": 0.002,        # for DLS
                "min_singular_value": 1e-5, # for SVD
                "position_weight": 1.0,
                "rotation_weight": 0.8,
                # for SVD and DLS methods: joint specific weights
                "joint_weights": [1.0, 1.0, 1.0, 1.0, 1.0]  # rotate, lift, tilt, scoop, gripper_rotate
            }
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=self.scene.num_envs,
            device=self.sim.device
        )

    def spawn_walls(self, num_walls: int = 3, shape_noise: float = 0.2,
                   pos_noise: float = 0.3, rot_noise: float = 0.1):
        """Spawn wall obstacles with random variations.

        Args:
            num_walls: Number of walls to spawn (0-5)
            shape_noise: Noise factor for wall dimensions (0-1)
            pos_noise: Noise factor for wall positions (0-1)
            rot_noise: Noise factor for wall rotations (0-1)
        """
        num_walls = np.clip(num_walls, 0, 5)

        if num_walls == 0:
            print("No wall obstacles spawned")
            return

        # Base wall configuration
        base_size = [0.05, 0.4, 0.3]  # [thickness, width, height]
        base_positions = [
            [0.6, 0.0, 0.15],
            [0.4, 0.2, 0.15],
            [0.4, -0.2, 0.15],
            [0.5, 0.1, 0.15],
            [0.5, -0.1, 0.15]
        ]

        for i in range(num_walls):
            # Apply noise to shape
            size = [
                base_size[0] * (1 + shape_noise * (np.random.rand() - 0.5)),
                base_size[1] * (1 + shape_noise * (np.random.rand() - 0.5)),
                base_size[2] * (1 + shape_noise * (np.random.rand() - 0.5))
            ]

            # Apply noise to position
            pos = np.array(base_positions[i % len(base_positions)])
            pos += pos_noise * (np.random.rand(3) - 0.5) * 0.2

            # Add environment origin offset for proper world positioning
            if hasattr(self.scene, 'env_origins') and len(self.scene.env_origins) > 0:
                # For the first environment (or single environment case)
                env_origin = self.scene.env_origins[0].cpu().numpy()
                pos += env_origin

            # Apply noise to rotation (only around Z axis for walls)
            rot_angle = rot_noise * (np.random.rand() - 0.5) * np.pi/2
            quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=self.sim.device),
                torch.tensor([0.0], device=self.sim.device),
                torch.tensor([rot_angle], device=self.sim.device)
            ).squeeze()

            # Create wall configuration
            wall_cfg = RigidObjectCfg(
                prim_path=f"/World/Obstacle_{self.obstacle_count}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        kinematic_enabled=True, # fixed to world
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2),
                        metallic=0.2,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(pos),
                    rot=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
                ),
            )

            # Spawn the wall
            wall = RigidObject(wall_cfg)
            self.obstacles.append(wall)
            self.obstacle_count += 1

        print(f"Spawned {num_walls} wall obstacles")

    def clear_trajectory_visualization(self):
        """Clear trajectory visualization markers and reset trajectory storage."""
        # Clear stored trajectory points (deque will automatically handle size limits)
        self.trajectory_points.clear()

        # Clear actual path visualization by providing empty positions
        empty_positions = torch.empty(0, 3, device=self.sim.device)
        empty_indices = torch.empty(0, dtype=torch.long, device=self.sim.device)
        self.actual_path_marker.visualize(empty_positions, marker_indices=empty_indices)

        # Clear calculated path visualization
        self.calculated_path_marker.visualize(empty_positions, marker_indices=empty_indices)

    def set_target(self, target_pos: Tuple[float, float, float],
                   target_rot: Tuple[float, float, float]):
        """Set target position and rotation for the end effector.

        Args:
            target_pos: Target position (x, y, z) in robot base frame
            target_rot: Target rotation in Euler angles (roll, pitch, yaw) in degrees
        """
        # Clear previous trajectory visualization for better visibility and performance
        self.clear_trajectory_visualization()

        # Convert position to tensor (in robot base frame)
        self.final_target_pos[:] = torch.tensor(target_pos, device=self.sim.device)

        # Convert Euler angles to quaternion
        roll_rad = torch.tensor([target_rot[0] * np.pi / 180.0], device=self.sim.device)
        pitch_rad = torch.tensor([target_rot[1] * np.pi / 180.0], device=self.sim.device)
        yaw_rad = torch.tensor([target_rot[2] * np.pi / 180.0], device=self.sim.device)

        self.final_target_quat[:] = quat_from_euler_xyz(roll_rad, pitch_rad, yaw_rad).repeat(self.scene.num_envs, 1)

        # Initialize current waypoint to current position
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]

        current_pos_b, current_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        self.current_waypoint_pos[:] = current_pos_b
        self.current_waypoint_quat[:] = current_quat_b

        print(f"Target set to position: {target_pos}, rotation: {target_rot}")

    def generate_trajectory(self, smoothing: float = 0.5):
        """Create smooth trajectory from current pose to target using Bezier curves.

        Doesnt really do much atm, but could be helpful with algo

        Args:
            smoothing: Curve strength (0=linear, 1=maximum curve)
        """
        smoothing = np.clip(smoothing, 0.0, 1.0)

        # Get current EE pose
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]

        # Transform to base frame
        current_pos_b, current_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Calculate path distance and steps
        distance = torch.norm(self.final_target_pos - current_pos_b, dim=-1).item()
        num_steps = max(2, int(np.ceil(distance / self.step_size)))

        # Create Bezier control points
        start_pos = current_pos_b[0].cpu().numpy()
        end_pos = self.final_target_pos[0].cpu().numpy()

        if smoothing > 0:
            # Create control points for cubic Bezier
            direction = end_pos - start_pos
            control_offset = distance * smoothing * 0.3  # Control point distance

            # Control points perpendicular to direct path for natural curve
            perp = np.array([-direction[1], direction[0], 0])
            if np.linalg.norm(perp) > 1e-6:
                perp = perp / np.linalg.norm(perp) * control_offset
            else:
                perp = np.array([0, 0, control_offset])

            p0 = start_pos
            p1 = start_pos + direction * 0.25 + perp
            p2 = end_pos - direction * 0.25 + perp
            p3 = end_pos
        else:
            # Linear interpolation (no curve)
            p0 = p1 = start_pos
            p2 = p3 = end_pos

        # Generate trajectory using Bezier curve
        trajectory = []
        for i in range(num_steps):
            t = i / (num_steps - 1)

            # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            pos = ((1-t)**3 * p0 +
                   3*(1-t)**2*t * p1 +
                   3*(1-t)*t**2 * p2 +
                   t**3 * p3)

            # Interpolate quaternion linearly (SLERP would be better but more complex)
            quat = current_quat_b * (1 - t) + self.final_target_quat * t
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)

            pos_tensor = torch.tensor(pos, device=self.sim.device, dtype=torch.float32).unsqueeze(0)
            trajectory.append(torch.cat([pos_tensor, quat], dim=-1))

        self.interpolated_trajectory = torch.stack(trajectory)
        self.trajectory_step = 0

        curve_type = "curved" if smoothing > 0 else "linear"
        print(f"Created {curve_type} trajectory: {distance:.3f}m distance, {num_steps} waypoints, {self.step_size:.3f}m step size")

    def update_current_waypoint(self):
        """Update the current waypoint based on trajectory progress."""
        if self.interpolated_trajectory is not None and self.trajectory_step < len(self.interpolated_trajectory):
            # Get current waypoint from trajectory
            current_step_pose = self.interpolated_trajectory[self.trajectory_step]
            self.current_waypoint_pos[:] = current_step_pose[:, :3]
            self.current_waypoint_quat[:] = current_step_pose[:, 3:7]

            self.trajectory_step += 1
        else:
            # If we've reached the end of trajectory, use final target
            self.current_waypoint_pos[:] = self.final_target_pos
            self.current_waypoint_quat[:] = self.final_target_quat

        # Convert to world frame and visualize
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        waypoint_pos_w, waypoint_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            self.current_waypoint_pos, self.current_waypoint_quat
        )
        self.goal_marker.visualize(waypoint_pos_w, waypoint_quat_w)

    def visualize_trajectory(self):
        """Visualize the traversed and calculated trajectories with sphere count limit."""
        # Visualize actual traversed path (limited by deque maxlen)
        if len(self.trajectory_points) > 0:
            # Get current root pose for conversion to world frame
            root_pose_w = self.robot.data.root_state_w[:, 0:7]

            # Convert actual traversed path to world frame for visualization
            actual_positions_b = torch.stack([p[:, :3] for p in self.trajectory_points])
            actual_quaternions_b = torch.stack([p[:, 3:7] for p in self.trajectory_points])

            # Convert each point from base frame to world frame
            num_points = actual_positions_b.shape[0]
            actual_positions_w = []

            for i in range(num_points):
                # Use the first environment's root pose for visualization
                pos_w, _ = combine_frame_transforms(
                    root_pose_w[0:1, 0:3], root_pose_w[0:1, 3:7],
                    actual_positions_b[i:i+1, 0], actual_quaternions_b[i:i+1, 0]
                )
                actual_positions_w.append(pos_w)

            if actual_positions_w:
                actual_positions_w = torch.cat(actual_positions_w, dim=0)
                marker_indices = torch.zeros(len(actual_positions_w), dtype=torch.long, device=self.sim.device)
                self.actual_path_marker.visualize(actual_positions_w, marker_indices=marker_indices)

        # Visualize calculated path (limit to max_spheres if needed)
        if self.interpolated_trajectory is not None:
            # Convert calculated path to world frame for visualization
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            calculated_positions_b = self.interpolated_trajectory[:, 0, :3]
            calculated_quaternions_b = self.interpolated_trajectory[:, 0, 3:7]

            # Limit the number of points if trajectory is too long
            num_trajectory_points = calculated_positions_b.shape[0]
            if num_trajectory_points > self.max_spheres:
                # Sample evenly distributed points
                indices = torch.linspace(0, num_trajectory_points - 1, self.max_spheres, dtype=torch.long)
                calculated_positions_b = calculated_positions_b[indices]
                calculated_quaternions_b = calculated_quaternions_b[indices]

            # Convert from base frame to world frame
            calculated_positions_w = []
            for i in range(calculated_positions_b.shape[0]):
                pos_w, _ = combine_frame_transforms(
                    root_pose_w[0:1, 0:3], root_pose_w[0:1, 3:7],
                    calculated_positions_b[i:i+1], calculated_quaternions_b[i:i+1]
                )
                calculated_positions_w.append(pos_w)

            if calculated_positions_w:
                calculated_positions_w = torch.cat(calculated_positions_w, dim=0)
                marker_indices = torch.zeros(len(calculated_positions_w), dtype=torch.long, device=self.sim.device)
                self.calculated_path_marker.visualize(calculated_positions_w, marker_indices=marker_indices)

    def step(self):
        """Execute one simulation step with IK control."""
        # Update current waypoint (this moves the goal marker along the trajectory)
        self.update_current_waypoint()

        # Get current state
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # Transform Jacobian to base frame
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        # Compute EE pose in base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Store trajectory point
        self.trajectory_points.append(torch.cat([ee_pos_b, ee_quat_b], dim=-1).clone())

        # Compute IK command to current waypoint
        pos_error = self.current_waypoint_pos - ee_pos_b
        rot_error_angle, rot_error_axis = compute_pose_error(
            ee_pos_b, ee_quat_b, self.current_waypoint_pos, self.current_waypoint_quat, rot_error_type="axis_angle"
        )

        # Limit step size for stability
        max_linear_step = 0.05
        max_angular_step = 0.5
        pos_error_norm = torch.norm(pos_error, dim=-1, keepdim=True)
        pos_error = pos_error * torch.clamp(max_linear_step / (pos_error_norm + 1e-6), max=1.0)
        rot_error_norm = torch.norm(rot_error_axis, dim=-1, keepdim=True)
        rot_error_axis = rot_error_axis * torch.clamp(max_angular_step / (rot_error_norm + 1e-6), max=1.0)

        # Send command to IK controller
        command = torch.cat([pos_error, rot_error_axis], dim=-1)
        self.diff_ik_controller.set_command(command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        # Compute and apply joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()

        # Visualize current EE position
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main simulation loop."""
    # Create environment with configurable parameters
    env = PathPlanningEnvironment(sim, scene, max_spheres=args_cli.max_spheres, step_size=args_cli.step_size)

    # Initialize robot
    joint_pos = env.robot.data.default_joint_pos.clone()
    joint_vel = env.robot.data.default_joint_vel.clone()
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    # Reset controller
    env.diff_ik_controller.reset()

    # Print robot info for debugging
    env._print_robot_info()

    # Spawn obstacles
    env.spawn_walls(
        num_walls=args_cli.num_walls,
        shape_noise=args_cli.shape_noise,
        pos_noise=args_cli.pos_noise,
        rot_noise=args_cli.rot_noise
    )

    # Set initial target and create trajectory
    env.set_target(
        target_pos=(0.5, 0.0, 0.1),
        target_rot=(0.0, 0.0, 90.0)
    )
    env.generate_trajectory(smoothing=args_cli.trajectory_smoothing)

    # Simulation loop
    count = 0
    trajectory_viz_freq = 10  # Update trajectory visualization every N steps

    while simulation_app.is_running():
        env.step()
        sim.step()
        scene.update(sim.get_physics_dt())
        count += 1

        # Update trajectory visualization periodically
        if count % trajectory_viz_freq == 0:
            env.visualize_trajectory()

        # Set new random target every 350 steps
        if count % 350 == 0:
            new_target_pos = (
                0.3 + 0.3 * np.random.rand(),
                -0.2 + 0.4 * np.random.rand(),
                0.05 + 0.15 * np.random.rand()
            )
            new_target_rot = (0.0, 0.0, -90.0 + 180.0 * np.random.rand())

            env.set_target(new_target_pos, new_target_rot)
            env.generate_trajectory(smoothing=args_cli.trajectory_smoothing)
            print(f"[Step {count}] New target: pos={new_target_pos}, rot={new_target_rot}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([0.5, 2.0, 0.5], [0.2, 0.0, 0.0])

    # Design scene
    scene_cfg = PathPlanningSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: Setup complete...")
    print(f"[INFO]: Spawning {args_cli.num_walls} walls with noise factors:")
    print(f"        Shape noise: {args_cli.shape_noise}")
    print(f"        Position noise: {args_cli.pos_noise}")
    print(f"        Rotation noise: {args_cli.rot_noise}")
    print(f"        Max trajectory spheres: {args_cli.max_spheres}")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()