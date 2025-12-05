"""WIP, mainly contains helper functions for future stuff"""


import argparse
import logging
import numpy as np
import torch
import sys
import os
from typing import Tuple, List, Optional
import datetime
import pandas as pd
import time

# Configure logging - will be set based on CLI args after parsing
# Create logger for this module
logger = logging.getLogger(__name__)

from isaaclab.app import AppLauncher

# Simplified argparse - only keep essential launcher args
parser = argparse.ArgumentParser(description="Path planning pathing demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--algorithm", type=str, default="a_star", choices=["a_star", "rrt", "rrt_star", "prm", "a_star_plane",], help="Path planning algorithm to use.")

# Debug options
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
parser.add_argument("--debug-trajectory", action="store_true", help="Log detailed trajectory waypoint info")
parser.add_argument("--debug-ik", action="store_true", help="Log IK controller commands and joint states")
parser.add_argument("--debug-planning", action="store_true", help="Enable verbose path planning algorithm output")
parser.add_argument("--pause-on-goal", action="store_true", help="Pause simulation when goal is reached for inspection")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Configure logging based on debug flags
if args_cli.debug:
    # Full debug mode - everything verbose
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s'
    )
    logger.setLevel(logging.DEBUG)
    logging.getLogger('pathing.path_planning_algorithms').setLevel(logging.DEBUG)
elif args_cli.debug_planning:
    # Only path planning debug
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    logger.setLevel(logging.INFO)
    logging.getLogger('pathing.path_planning_algorithms').setLevel(logging.DEBUG)
elif args_cli.debug_trajectory or args_cli.debug_ik:
    # Trajectory/IK debug
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    logger.setLevel(logging.INFO)
else:
    # Normal mode - show INFO level for main logger (status updates), WARNING+ for everything else
    logging.basicConfig(
        level=logging.WARNING,
        format='[%(levelname)s] %(message)s'
    )
    logger.setLevel(logging.INFO)  # Our logger shows INFO in normal mode

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from typing import Literal
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, matrix_from_quat, quat_inv, quat_from_euler_xyz, euler_xyz_from_quat, combine_frame_transforms
from collections import deque
from pathing.path_planning_algorithms import (
    ObstacleChecker,
    AStarParams,
    RRTParams,
    RRTStarParams,
    PRMParams,
    PathPlanningError,
    NoPathFoundError,
    CollisionError,
    TimeoutError as PathTimeoutError,
)
from pathing.normalized_planners import (
    NormalizerParams,
    create_astar_plane_trajectory_normalized,
    create_rrt_star_trajectory_normalized,
    create_rrt_trajectory_normalized,
    create_prm_trajectory_normalized,
    plan_to_target,
)
from configuration_files.pathing_config import (
    EnvironmentConfig, PathExecutionConfig,
    DEFAULT_ENV_CONFIG, DEFAULT_CONFIG
)
from pathing.path_utils import (
    calculate_path_length,
    precompute_cumulative_distances,
    interpolate_at_s,
)


# Use local modified IK controller implementation
from sim_controllers.differential_ik_cfg import DifferentialIKControllerCfg
from sim_controllers.differential_ik import DifferentialIKController
from isaaclab_assets import MASI_PATHING_CFG as CFG  # isort:skip

# Visualization toggles (sim-only)
SHOW_RELATIVE_PULL_MARKER: bool = True
RELATIVE_PULL_MARKER_SCALE_M: float = 0.01  # 10 mm sphere/frame size
RELATIVE_PULL_MARKER_POSITION_SCALE: float = 3.0  # Exaggerate the pull target position for visibility

@configclass
class PathPlanningSceneCfg(InteractiveSceneCfg):
    """Configuration for the path planning scene."""

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Ground plane
    #ground = AssetBaseCfg(
    #    prim_path="/World/defaultGroundPlane",
    #    spawn=sim_utils.GroundPlaneCfg(),
    #    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #)

    # Robot
    robot = CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


# Helper used only for A* debug logging.
# NOTE: The actual A* implementation snaps to the grid internally using
# its own GridConfig (with bounds_min as origin). This helper ignores
# bounds_min and therefore is only an approximate visualization of what
# the planner does, not the exact internal mapping.
def snap_to_grid(pos: Tuple[float, float, float], resolution: float) -> Tuple[float, float, float]:
    return tuple(round(p / resolution) * resolution for p in pos)


class PathPlanningEnvironment:
    """Main class for path planning with obstacle avoidance."""

    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.obstacle_data = []

        # Initialize path configuration BEFORE IK controller (which depends on it)
        self.path_config = DEFAULT_CONFIG  # Use shared config
        self.env_config = DEFAULT_ENV_CONFIG  # Store environment config for metrics

        # Initialize markers
        self._init_markers()

        # Initialize IK controller
        self._init_ik_controller()

        # Initialize robot entity config
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["revolute_cabin", "revolute_lift", "revolute_tilt",
                        "revolute_scoop"],
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

        # Debug options (controlled by CLI args)
        self.debug_pose_logging: bool = args_cli.debug or args_cli.debug_trajectory
        self.debug_trajectory: bool = args_cli.debug_trajectory
        self.debug_ik: bool = args_cli.debug_ik

        # Goal evaluation tolerances come from shared config (pathing_config.PathExecutionConfig)

        # Trajectory storage and control (time-based like real hardware)
        self.interpolated_trajectory: Optional[torch.Tensor] = None
        self.trajectory_start_time = None
        self.trajectory_total_time = None

        # Current waypoint tracking
        self.current_waypoint_index = 0
        self.current_progress = 0.0

        self.trajectory_log = []  # Store (x_g, y_g, z_g, x_e, y_e, z_e) tuples
        self.execution_start_time = None
        self.calculation_time = 0.0
        self.execution_time = 0.0
        self.total_distance_planned = 0.0
        self.total_distance_executed = 0.0
        self.max_tracking_error = 0.0
        self.avg_tracking_error = 0.0
        self.at_threshold_time = None  # Time when robot first reached within target threshold
        self.trajectory_counter = 0
        self.trajectory_waypoints = 0
        self.simulation_steps_logged = 0

        logger.debug("Path planner initialized")

    def init_logging(self, algorithm_name: str, base_log_dir: str = "logs"):
        """Initialize logging directories and files."""
        # Create main logs directory
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Find next available folder number for this algorithm
        algo_folders = [f for f in os.listdir(base_log_dir) if f.startswith(f"{algorithm_name}_sim_")]
        if algo_folders:
            existing_nums = [int(f.split("_")[2]) for f in algo_folders if len(f.split("_")) > 2 and f.split("_")[2].isdigit()]
            folder_num = max(existing_nums) + 1 if existing_nums else 1
        else:
            folder_num = 1
        
        # Create algorithm-specific folder
        self.log_dir = os.path.join(base_log_dir, f"{algorithm_name}_sim_{folder_num}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"[INFO] Logging to: {self.log_dir}")
        
    def start_trajectory_tracking(self):
        """Start tracking a new trajectory execution."""
        self.trajectory_log.clear()
        self.execution_start_time = time.time()
        self.max_tracking_error = 0.0
        self.total_distance_executed = 0.0
        self.at_threshold_time = None
        self.prev_ee_pos = None
        self.trajectory_time_complete = False
        if hasattr(self, '_execution_complete_logged'):
            delattr(self, '_execution_complete_logged')
        
    
    def log_trajectory_step(self, planned_pos: torch.Tensor, actual_pos: torch.Tensor, 
                           planned_quat: torch.Tensor, actual_quat: torch.Tensor, joint_angles: torch.Tensor):
        """Log a single step of trajectory execution with waypoint tracking, orientations, and joint angles."""
        # Convert to numpy for easier handling
        planned_pos_np = planned_pos.cpu().numpy()
        actual_pos_np = actual_pos.cpu().numpy()
        planned_quat_np = planned_quat.cpu().numpy()
        actual_quat_np = actual_quat.cpu().numpy()
        joint_angles_np = joint_angles.cpu().numpy()
        
        # Store trajectory data
        # TODO: zero if below 1e-5!
        self.trajectory_log.append([
            # Position data
            float(planned_pos_np[0]),  # x_g
            float(planned_pos_np[1]),  # y_g  
            float(planned_pos_np[2]),  # z_g
            float(actual_pos_np[0]),   # x_e
            float(actual_pos_np[1]),   # y_e
            float(actual_pos_np[2]),   # z_e
            # Planned orientation (quaternion: w, x, y, z)
            float(planned_quat_np[0]), # quat_g_w
            float(planned_quat_np[1]), # quat_g_x
            float(planned_quat_np[2]), # quat_g_y
            float(planned_quat_np[3]), # quat_g_z
            # Actual orientation (quaternion: w, x, y, z)
            float(actual_quat_np[0]),  # quat_e_w
            float(actual_quat_np[1]),  # quat_e_x
            float(actual_quat_np[2]),  # quat_e_y
            float(actual_quat_np[3]),  # quat_e_z
            # Joint angles (5 joints)
            float(joint_angles_np[0]), # joint_1 (revolute_cabin)
            float(joint_angles_np[1]), # joint_2 (revolute_lift)
            float(joint_angles_np[2]), # joint_3 (revolute_tilt)
            float(joint_angles_np[3]), # joint_4 (revolute_scoop)
            #float(joint_angles_np[4]), # joint_5 (revolute_gripper)
            # Tracking data
            int(self.current_waypoint_index),  # Current interpolated waypoint index
            float(self.current_progress)       # Overall path progress (0.0-1.0)
        ])
        
        # Calculate tracking error
        error = torch.norm(planned_pos - actual_pos).item()
        self.max_tracking_error = max(self.max_tracking_error, error)
        
        # Calculate executed distance
        if self.prev_ee_pos is not None:
            self.total_distance_executed += torch.norm(actual_pos - self.prev_ee_pos).item()
        self.prev_ee_pos = actual_pos.clone()
    
    def save_trajectory_log(self, algorithm_name: str):
        """Save trajectory log to CSV file."""
        if not self.trajectory_log:
            return
    
        self.trajectory_counter += 1
        self.execution_time = time.time() - self.execution_start_time
        self.simulation_steps_logged = len(self.trajectory_log)
        
        # Calculate metrics
        errors = [np.linalg.norm(np.array(log[3:6]) - np.array(log[0:3])) for log in self.trajectory_log]
        self.avg_tracking_error = np.mean(errors) if errors else 0.0
        
        # Save trajectory data with waypoint tracking, orientations, and joint angles
        columns = ['x_g', 'y_g', 'z_g', 'x_e', 'y_e', 'z_e',
                   'quat_g_w', 'quat_g_x', 'quat_g_y', 'quat_g_z',
                   'quat_e_w', 'quat_e_x', 'quat_e_y', 'quat_e_z',
                   'joint_1', 'joint_2', 'joint_3', 'joint_4',
                   'waypoint_idx', 'progress']
        df = pd.DataFrame(self.trajectory_log, columns=columns)
        csv_path = os.path.join(self.log_dir, f"{algorithm_name}_{self.trajectory_counter}_sim.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate time to reach threshold (excludes settling time after reaching target)
        at_threshold_time_s = None
        if self.at_threshold_time is not None:
            at_threshold_time_s = self.at_threshold_time - self.execution_start_time

        # Save metrics with more detailed waypoint counts
        metrics = {
            'trajectory_id': self.trajectory_counter,
            'data_source': 'simulation',  # Mark as simulation data
            'algorithm': algorithm_name,
            'calculation_time_s': self.calculation_time,
            'execution_time_s': self.execution_time - self.calculation_time,
            'at_threshold_time_s': at_threshold_time_s,
            'trajectory_waypoints': self.trajectory_waypoints,  # Execution waypoints
            'simulation_steps': self.simulation_steps_logged,   # Logged simulation steps
            'planned_distance_m': self.total_distance_planned,
            'executed_distance_m': self.total_distance_executed,
            'max_tracking_error_m': self.max_tracking_error,
            'avg_tracking_error_m': self.avg_tracking_error,
            'efficiency_ratio': self.total_distance_executed / self.total_distance_planned if self.total_distance_planned > 0 else 0,
            # Relative-mode gains (set below if mode is enabled)
            'relative_pos_gain': None,
            'relative_rot_gain': None,
            # Velocity-mode settings (set below if available)
            'ik_velocity_mode': None,
            'ik_velocity_error_gain': None,
            'ik_use_rotational_velocity': None,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        # Add path configuration parameters (matching hardware logger)
        metrics['speed_mps'] = self.path_config.speed_mps
        metrics['dt'] = self.path_config.dt
        metrics['update_frequency'] = self.path_config.update_frequency
        metrics['grid_resolution'] = self.path_config.grid_resolution
        metrics['safety_margin'] = self.path_config.safety_margin
        metrics['use_3d'] = self.path_config.use_3d
        metrics['final_target_tolerance'] = self.path_config.final_target_tolerance
        metrics['orientation_tolerance'] = self.path_config.orientation_tolerance
        metrics['ik_velocity_mode'] = getattr(self.path_config, 'ik_velocity_mode', None)
        metrics['ik_velocity_error_gain'] = getattr(self.path_config, 'ik_velocity_error_gain', None)
        metrics['ik_use_rotational_velocity'] = getattr(self.path_config, 'ik_use_rotational_velocity', None)
        metrics['ik_method'] = self.path_config.ik_method
        metrics['ik_command_type'] = self.path_config.ik_command_type
        metrics['ik_use_relative_mode'] = self.path_config.ik_use_relative_mode
        if getattr(self.path_config, 'ik_use_relative_mode', False):
            metrics['relative_pos_gain'] = getattr(self.path_config, 'relative_pos_gain', None)
            metrics['relative_rot_gain'] = getattr(self.path_config, 'relative_rot_gain', None)

        # Add environment/obstacle configuration parameters (matching hardware logger)
        metrics['wall_size_x'] = self.env_config.wall_size[0]
        metrics['wall_size_y'] = self.env_config.wall_size[1]
        metrics['wall_size_z'] = self.env_config.wall_size[2]
        metrics['wall_pos_x'] = self.env_config.wall_pos[0]
        metrics['wall_pos_y'] = self.env_config.wall_pos[1]
        metrics['wall_pos_z'] = self.env_config.wall_pos[2]
        metrics['wall_rot_w'] = self.env_config.wall_rot[0]
        metrics['wall_rot_x'] = self.env_config.wall_rot[1]
        metrics['wall_rot_y'] = self.env_config.wall_rot[2]
        metrics['wall_rot_z'] = self.env_config.wall_rot[3]
        
        metrics_path = os.path.join(self.log_dir, "metrics.csv")
        metrics_df = pd.DataFrame([metrics])
        
        # Append to existing metrics file or create new one
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)
        
        real_execution_time = self.execution_time - self.calculation_time
        logger.info(f"Saved trajectory {self.trajectory_counter} to {csv_path}")
        logger.info(f"Calculation time: {self.calculation_time:.2f}s, Execution time: {real_execution_time:.2f}s, "
            f"Avg Error: {self.avg_tracking_error:.4f}m")
    
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

        # Final target marker (static target pose)
        final_frame_marker_cfg = FRAME_MARKER_CFG.copy()
        final_frame_marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
        self.final_target_marker = VisualizationMarkers(final_frame_marker_cfg.replace(prim_path="/Visuals/ee_final_target"))

        # Relative pull marker (where relative IK pulls toward this step)
        pull_marker_cfg = FRAME_MARKER_CFG.copy()
        s = float(RELATIVE_PULL_MARKER_SCALE_M)
        pull_marker_cfg.markers["frame"].scale = (s, s, s)
        self.rel_pull_marker = VisualizationMarkers(pull_marker_cfg.replace(prim_path="/Visuals/ee_rel_pull"))

    def _init_ik_controller(self):
        """Initialize the differential IK controller (SIM) to mirror IRL behaviour."""
        # Optional joint limits: convert from config (deg or rad) to radians
        joint_limits = None
        if getattr(self.path_config, "joint_limits_relative", None):
            jl = []
            for lo, hi in self.path_config.joint_limits_relative:
                lo_f = float(lo)
                hi_f = float(hi)
                # If values look like degrees (>|pi|), convert
                if abs(lo_f) > np.pi or abs(hi_f) > np.pi:
                    lo_f = np.deg2rad(lo_f)
                    hi_f = np.deg2rad(hi_f)
                jl.append((lo_f, hi_f))
            joint_limits = jl

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type=self.path_config.ik_command_type,
            use_relative_mode=self.path_config.ik_use_relative_mode,
            ik_method=self.path_config.ik_method,
            ik_params=self.path_config.ik_params,
            ignore_axes=list(self.path_config.ignore_axes),
            use_reduced_jacobian=self.path_config.use_reduced_jacobian,
            joint_limits=joint_limits,
            velocity_mode=bool(self.path_config.ik_velocity_mode),
            velocity_error_gain=float(self.path_config.ik_velocity_error_gain),
            use_rotational_velocity=bool(self.path_config.ik_use_rotational_velocity),
        )

        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=self.scene.num_envs,
            device=self.sim.device,
        )

    def spawn_wall(self, env_config: EnvironmentConfig = None):
        """Spawn single wall obstacle using environment configuration (matches real hardware)."""
        if env_config is None:
            env_config = DEFAULT_ENV_CONFIG
        
        # Use wall configuration from environment config (matches hardware exactly)
        size = list(env_config.wall_size)
        pos = np.array(env_config.wall_pos)
        quat_array = np.array(env_config.wall_rot)  # [w, x, y, z]

        # No environment origin offset (hardware has single world frame, num_envs=1)

        # Convert quaternion array to torch tensor
        quat = torch.tensor(quat_array, device=self.sim.device)
        
        # Create wall configuration
        wall_cfg = RigidObjectCfg(
            prim_path=f"/World/Obstacle_{self.obstacle_count}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    kinematic_enabled=True,  # Fixed to world
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
        self.obstacle_data.append({
            "size": np.array(size),
            "pos": np.array(pos),
            "rot": np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]),
        })
        self.obstacle_count += 1

        logger.debug(f"Spawned wall: size={size}, pos={pos}, rot={quat_array}")
        logger.debug("Wall matches real hardware configuration")


    def set_target(self, target_pos: Tuple[float, float, float],
                   target_rot: Tuple[float, float, float]):
        """Set target position and rotation for the end effector.

        Args:
            target_pos: Target position (x, y, z) in robot base frame
            target_rot: Target rotation in Euler angles (roll, pitch, yaw) in degrees
        """
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

        logger.info(f"Target set to position: {target_pos}, rotation: {target_rot}")

    #def spawn_simple_target(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Alternately spawns a fixed target point on either side (left/right) of the wall.
        Wall is assumed to be centered at (0.5, 0.0, 0.1), rotated 90° about Z.

        Returns:
            Tuple of (target_pos, target_rot)
        """
        # Fixed height above ground
        #z_pos = -0.05  # slightly above center of the wall

        # X stays constant (aligned with wall center)
        #x_pos = 0.5

        # Y is alternated each time
        #y_offset = 0.3  #

        #if not hasattr(self, "_last_target_side") or self._last_target_side == "right":
        #    self._last_target_side = "left"
        #    target_pos = (x_pos, y_offset, z_pos) # (0.5, 0.3, -0.05)
        #else:
        #    self._last_target_side = "right"
        #    target_pos = (x_pos + 0.1, -y_offset, z_pos) # (0.6, -0.3, -0.05)

        # Random flat yaw
        #target_rot = (0.0, 0.0, np.random.uniform(-90.0, 90.0))

        #print(f"[spawn_target] Target: pos={target_pos}, rot={target_rot}")
        #return target_pos, target_rot

    
    def generate_trajectory(self, target_pos: Tuple[float, float, float],
                     target_rot: Tuple[float, float, float],
                     algorithm = args_cli.algorithm):
        """Generate trajectory using standardized pathfinding algorithms."""

        calc_start_time = time.time()

        # Get current robot poses
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]

        # Get current EE position in world frame
        current_pos_w = ee_pose_w[0, :3].cpu().numpy()

        # Convert target from base frame to world frame
        target_pos_tensor = torch.tensor(target_pos, device=self.sim.device).unsqueeze(0)
        target_pos_w, _ = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_tensor, torch.tensor([1, 0, 0, 0], device=self.sim.device).unsqueeze(0)
        )
        target_pos_w = target_pos_w[0].cpu().numpy()

        logger.debug(f"Current EE world: {current_pos_w}")
        logger.debug(f"Target base frame: {target_pos}")
        logger.debug(f"Target world frame: {target_pos_w}")

        try:
            # Create normalizer config from path_config
            normalizer = NormalizerParams(
                speed_mps=self.path_config.speed_mps,
                dt=self.path_config.dt,
                return_poses=self.path_config.normalizer_return_poses,
                force_goal=self.path_config.normalizer_force_goal,
            )

            # Use normalized algorithms for consistent comparison
            if algorithm == "a_star":
                algorithm_name = "A*"
                astar_params = AStarParams()

                # DEBUG: Show how grid snapping would look, but feed raw
                # world positions into the planner (A* snaps internally).
                if self.debug_trajectory or args_cli.debug:
                    snapped_start = snap_to_grid(current_pos_w, self.path_config.grid_resolution)
                    snapped_goal = snap_to_grid(target_pos_w, self.path_config.grid_resolution)
                    print(f"\n[DEBUG A*] Grid Snapping (resolution={self.path_config.grid_resolution}m):")
                    print(f"  Start before snap: {current_pos_w}")
                    print(f"  Start after snap:  {snapped_start}")
                    print(f"  Start snap delta:  {(np.array(snapped_start) - current_pos_w)*1000} mm")
                    print(f"  Goal before snap:  {target_pos_w}")
                    print(f"  Goal after snap:   {snapped_goal}")
                    print(f"  Goal snap delta:   {(np.array(snapped_goal) - target_pos_w)*1000} mm")

                # Plan from current EE pose to target; A* will handle its own
                # grid snapping internally inside the planner.
                result = plan_to_target(
                    start_pos_world=tuple(current_pos_w),
                    target_pos_world=tuple(target_pos_w),
                    obstacle_data=self.obstacle_data,
                    algorithm="a_star",
                    grid_resolution=self.path_config.grid_resolution,
                    safety_margin=self.path_config.safety_margin,
                    normalizer_params=normalizer,
                    astar_params=astar_params,
                    use_3d=True,
                    verbose=True,  # Print A* iteration progress (every 5k)
                )
            elif algorithm == "rrt":
                algorithm_name = "RRT"
                rrt_params = RRTParams(max_acceptable_cost=0.768)
                result = plan_to_target(
                    start_pos_world=tuple(current_pos_w),
                    target_pos_world=tuple(target_pos_w),
                    obstacle_data=self.obstacle_data,
                    algorithm="rrt",
                    grid_resolution=self.path_config.grid_resolution,
                    safety_margin=self.path_config.safety_margin,
                    normalizer_params=normalizer,
                    rrt_params=rrt_params,
                    use_3d=False,
                )
            elif algorithm == "rrt_star":
                algorithm_name = "RRT*"
                rrt_star_params = RRTStarParams(max_acceptable_cost=0.768)
                result = plan_to_target(
                    start_pos_world=tuple(current_pos_w),
                    target_pos_world=tuple(target_pos_w),
                    obstacle_data=self.obstacle_data,
                    algorithm="rrt_star",
                    grid_resolution=self.path_config.grid_resolution,
                    safety_margin=self.path_config.safety_margin,
                    normalizer_params=normalizer,
                    rrt_star_params=rrt_star_params,
                    use_3d=False,
                )
            elif algorithm == "prm":
                algorithm_name = "PRM"
                prm_params = PRMParams()
                result = plan_to_target(
                    start_pos_world=tuple(current_pos_w),
                    target_pos_world=tuple(target_pos_w),
                    obstacle_data=self.obstacle_data,
                    algorithm="prm",
                    grid_resolution=self.path_config.grid_resolution,
                    safety_margin=self.path_config.safety_margin,
                    normalizer_params=normalizer,
                    prm_params=prm_params,
                    use_3d=False,
                )
            elif algorithm == "a_star_plane":
                algorithm_name = "A* Plane"
                astar_plane_params = AStarParams()
                result = create_astar_plane_trajectory_normalized(
                    start_pos=tuple(current_pos_w),
                    goal_pos=tuple(target_pos_w),
                    obstacle_data=self.obstacle_data,
                    grid_resolution=self.path_config.grid_resolution,
                    safety_margin=self.path_config.safety_margin,
                    astar_params=astar_plane_params,
                    normalizer_params=normalizer,
                    verbose=True,  # Print planar A* iteration progress (every 5k)
                )

            # Extract trajectory data from standardized result
            exec_poses_world = result["exec_poses"]  # [K, 7] NumPy array: xyz + quat(w,x,y,z)

            # Check if we need to append final target (only if path doesn't end close enough)
            last_waypoint_to_target_dist = np.linalg.norm(exec_poses_world[-1][:3] - target_pos_w)
            min_append_threshold = self.path_config.speed_mps * self.path_config.dt * 0.5  # Half a timestep

            if last_waypoint_to_target_dist > min_append_threshold:
                # Append exact final target to ensure smooth arrival
                # (path planners may not end exactly at target due to grid snapping)
                # Since we use pitch=0, we can just use identity quaternion [1,0,0,0]
                final_target_pose_world = np.concatenate([target_pos_w, [1, 0, 0, 0]]).astype(np.float32)
                exec_poses_world = np.vstack([exec_poses_world, final_target_pose_world])
                logger.info(f"[{algorithm_name}] Appended final target ({last_waypoint_to_target_dist*1000:.2f}mm gap)")
            else:
                logger.info(f"[{algorithm_name}] Path already ends at target ({last_waypoint_to_target_dist*1000:.2f}mm)")

            self.trajectory_waypoints = len(exec_poses_world)  # Execution waypoints

            # Calculate timing for user-configurable speed
            self.total_distance_planned = float(result["total_length_m"][0])
            if last_waypoint_to_target_dist > min_append_threshold:
                self.total_distance_planned += last_waypoint_to_target_dist
            self.trajectory_total_time = self.total_distance_planned / self.path_config.speed_mps

            logger.info(f"[{algorithm_name}] Path length: {self.total_distance_planned:.3f}m")
            logger.info(f"[{algorithm_name}] Speed: {self.path_config.speed_mps:.3f}m/s")
            logger.info(f"[{algorithm_name}] Estimated time: {self.trajectory_total_time:.1f}s")
            logger.info(f"[{algorithm_name}] Execution waypoints: {self.trajectory_waypoints}")

            # Convert poses from world frame to base frame
            # Use identity quaternions from path planner directly (no interpolation needed)
            # IK controller handles orientation via ignore_axes, we only care about final target Y rotation
            trajectory = []
            for i, pose_w in enumerate(exec_poses_world):
                # Extract position and quaternion (world frame)
                pos_w = torch.tensor(pose_w[:3], device=self.sim.device).unsqueeze(0)
                quat_w = torch.tensor(pose_w[3:7], device=self.sim.device).unsqueeze(0)

                # Convert to base frame
                pos_b, quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    pos_w, quat_w
                )

                # Combine position and quaternion
                pose_7d = torch.cat([pos_b, quat_b], dim=-1)  # [1, 7]
                trajectory.append(pose_7d)

            # Stack trajectory (base-frame poses with identity quaternions from path planner)
            self.interpolated_trajectory = torch.stack(trajectory, dim=0)  # [K, 1, 7]

            # Start time-based execution (like real hardware, but using standardized constant-speed samples)
            self.trajectory_start_time = time.time()

            self.calculation_time = time.time() - calc_start_time
            logger.info(f"{algorithm_name} calculation time: {self.calculation_time:.3f}s")
            logger.debug(f"Average point spacing: {self.total_distance_planned/(len(exec_poses_world)-1):.4f}m")

            # DEBUG: Analyze trajectory start/end points
            if self.debug_trajectory or args_cli.debug:
                # TODO: this
                pass

        except CollisionError as e:
            print(f"[ERROR] {algorithm_name} planning failed: Start or goal position is in collision")
            print(f"        Details: {e}")
            raise
        except NoPathFoundError as e:
            print(f"[ERROR] {algorithm_name} planning failed: No valid path exists")
            print(f"        Details: {e}")
            raise
        except PathTimeoutError as e:
            print(f"[ERROR] {algorithm_name} planning failed: Planning exceeded time limit")
            print(f"        Details: {e}")
            raise
        except PathPlanningError as e:
            print(f"[ERROR] {algorithm_name} planning failed: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] {algorithm_name} planning failed with unexpected error: {e}")
            raise PathPlanningError(f"{algorithm_name} path planning failed: {e}") from e

    
    def is_goal_reached(self, position_tolerance: float = 0.018, orientation_tolerance: float = 0.2) -> bool:
        """Check if the end effector has reached the current target.

        This variant enforces Y-only orientation checking (cartesian + Yrot),
        matching the IK controller and IRL behavior.

        Args:
            position_tolerance: Position error tolerance in meters
            orientation_tolerance: Orientation error tolerance around Y-axis in radians

        Returns:
            True if goal is reached within tolerances
        """
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]

        # Get current EE pose in base frame
        current_pos_b, current_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Require full progress completion to avoid early skipping
        if hasattr(self, "current_progress") and self.current_progress < 1.0:
            return False

        # Calculate position error (vector and magnitude)
        pos_error_vec = self.final_target_pos - current_pos_b
        pos_error = torch.norm(pos_error_vec, dim=-1).item()

        # Check orientation error (Y-axis only)
        _, rot_error_axis = compute_pose_error(
            current_pos_b, current_quat_b,
            self.final_target_pos, self.final_target_quat,
            rot_error_type="axis_angle"
        )
        rot_error_y = torch.abs(rot_error_axis[:, 1]).item()

        # Use tolerances from shared path configuration (ignore function defaults)
        pos_tol = float(self.path_config.final_target_tolerance)
        rot_tol = float(self.path_config.orientation_tolerance)

        # Print clear error status (matching IRL format)
        pos_error_vec_mm = pos_error_vec[0].cpu().numpy() * 1000.0
        pos_error_mm = pos_error * 1000.0
        rot_error_deg = rot_error_y * 180.0 / np.pi
        pos_tol_mm = pos_tol * 1000.0
        rot_tol_deg = rot_tol * 180.0 / np.pi

        goal_reached = pos_error < pos_tol and rot_error_y < rot_tol

        if goal_reached:
            # Mark the first time we reach the threshold (for metrics logging)
            if self.at_threshold_time is None:
                self.at_threshold_time = time.time()

            logger.info(
                f"Target reached: "
                f"pos_err=[{pos_error_vec_mm[0]:.2f}, {pos_error_vec_mm[1]:.2f}, {pos_error_vec_mm[2]:.2f}]mm "
                f"({pos_error_mm:.2f}mm), rot_err={rot_error_deg:.2f}deg"
            )
        else:
            logger.info(
                f"Waiting for target: "
                f"pos_err=[{pos_error_vec_mm[0]:.2f}, {pos_error_vec_mm[1]:.2f}, {pos_error_vec_mm[2]:.2f}]mm "
                f"({pos_error_mm:.2f}mm < {pos_tol_mm:.2f}mm tol), "
                f"rot_err={rot_error_deg:.2f}deg (< {rot_tol_deg:.2f}deg tol)"
            )

        return goal_reached


    def update_current_waypoint(self):
        """Update waypoint along standardized constant-speed trajectory."""
        if (
            self.interpolated_trajectory is None
            or self.trajectory_start_time is None
            or self.trajectory_total_time is None
        ):
            return

        # Time tracking
        elapsed = time.time() - self.trajectory_start_time

        # Calculate normalized progress (0-1) using planned duration
        progress = min(elapsed / self.trajectory_total_time, 1.0) if self.trajectory_total_time > 0 else 1.0
        self.current_progress = progress

        # Map progress directly to trajectory index (samples are already constant-speed)
        num_steps = self.interpolated_trajectory.shape[0]
        if num_steps <= 1:
            target_index = 0
        else:
            target_index = min(int(progress * (num_steps - 1)), num_steps - 1)

        self.current_waypoint_index = target_index

        if progress >= 1.0:
            self.trajectory_time_complete = True

            # Trajectory naturally ends at final target (no snap needed)
            # Just hold the last waypoint
            current_step_pose = self.interpolated_trajectory[-1]

            # Fix tensor dimensions - interpolated_trajectory is [N, 1, 7]
            if current_step_pose.dim() == 2:  # [1, 7]
                current_step_pose = current_step_pose[0]  # [7]
            elif current_step_pose.dim() == 3:  # [1, 1, 7]
                current_step_pose = current_step_pose[0, 0]  # [7]

            self.current_waypoint_pos[:] = current_step_pose[:3].unsqueeze(0)  # [1, 3]
            self.current_waypoint_quat[:] = current_step_pose[3:7].unsqueeze(0)  # [1, 4]

            if not hasattr(self, '_execution_complete_logged'):
                logger.info("Path execution completed")
                self._execution_complete_logged = True
        elif target_index < len(self.interpolated_trajectory):
            # Get current waypoint from trajectory
            current_step_pose = self.interpolated_trajectory[target_index]

            # Fix tensor dimensions - interpolated_trajectory is [N, 1, 7], we need [1, 7] then [7]
            if current_step_pose.dim() == 2:  # [1, 7]
                current_step_pose = current_step_pose[0]  # [7]
            elif current_step_pose.dim() == 3:  # [1, 1, 7] - shouldn't happen but handle it
                current_step_pose = current_step_pose[0, 0]  # [7]

            # DEBUG: Log waypoint changes (helps identify sudden jumps)
            if self.debug_trajectory and not hasattr(self, '_last_logged_waypoint_idx'):
                self._last_logged_waypoint_idx = -1

            if self.debug_trajectory and target_index != self._last_logged_waypoint_idx:
                old_pos = self.current_waypoint_pos[0].cpu().numpy()
                new_pos = current_step_pose[:3].cpu().numpy()
                waypoint_jump = np.linalg.norm(new_pos - old_pos)
                if waypoint_jump > 0.01:  # 10mm threshold
                    print(f"\n[DEBUG TRAJECTORY] Waypoint {self._last_logged_waypoint_idx} → {target_index}")
                    print(f"  Old waypoint: {old_pos}")
                    print(f"  New waypoint: {new_pos}")
                    print(f"  Waypoint jump: {waypoint_jump*1000:.2f}mm")
                self._last_logged_waypoint_idx = target_index

            # Set current waypoint - robot tries its best to follow
            self.current_waypoint_pos[:] = current_step_pose[:3].unsqueeze(0)  # [1, 3]
            self.current_waypoint_quat[:] = current_step_pose[3:7].unsqueeze(0)  # [1, 4]
            
            # Progress feedback (matches real hardware style)
            if int(elapsed) % int(self.path_config.progress_update_interval) == 0 and elapsed > 0:
                remaining_time = max(0, self.trajectory_total_time - elapsed)

                # Compute current tracking error (EE vs current waypoint) in base frame
                ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                root_pose_w = self.robot.data.root_state_w[:, 0:7]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                pos_err_vec = self.current_waypoint_pos - ee_pos_b
                pos_err = torch.norm(pos_err_vec, dim=-1).item()
                _, rot_err_vec = compute_pose_error(
                    ee_pos_b, ee_quat_b,
                    self.current_waypoint_pos, self.current_waypoint_quat,
                    rot_error_type="axis_angle"
                )
                rot_err_y = torch.abs(rot_err_vec[:, 1]).item()

                if not hasattr(self, '_last_progress_print') or time.time() - self._last_progress_print > self.path_config.progress_update_interval:
                    # Present errors in human-friendly units
                    pos_err_mm = pos_err * 1000.0
                    pos_tol_mm = float(self.path_config.final_target_tolerance) * 1000.0
                    rot_err_deg = float(rot_err_y) * 180.0 / np.pi
                    rot_tol_deg = float(self.path_config.orientation_tolerance) * 180.0 / np.pi

                    # Use print instead of logger for this status message - it should always show
                    print(
                        f"[Progress] {progress*100:.1f}% "
                        f"({elapsed:.1f}s/{self.trajectory_total_time:.1f}s, {remaining_time:.1f}s remaining) "
                        f"| err_pos={pos_err_mm:.1f}mm, err_rot={rot_err_deg:.1f}deg"
                    )

                    # Extra debug: show where target, marker, and EE are
                    if self.debug_pose_logging:
                        current_pos_w = ee_pose_w[0, 0:3].detach().cpu().numpy()
                        print(f"[DEBUG] EE_w={current_pos_w}")
                        # Waypoint in base frame
                        wp_b = self.current_waypoint_pos[0].detach().cpu().numpy()
                        ee_b = ee_pos_b[0].detach().cpu().numpy()
                        dpos_b = float(np.linalg.norm(wp_b - ee_b))

                        # Final target in base frame and its error (Y-only rot)
                        tgt_b = self.final_target_pos[0].detach().cpu().numpy()
                        pos_err_tgt = float(np.linalg.norm(tgt_b - ee_b))
                        _, rot_err_axis_tgt = compute_pose_error(
                            ee_pos_b, ee_quat_b, self.final_target_pos, self.final_target_quat, rot_error_type="axis_angle"
                        )
                        rot_err_y_tgt = float(torch.abs(rot_err_axis_tgt[:, 1]).item())

                        # Waypoint marker in world frame vs EE world frame
                        wp_w_pos, wp_w_quat = combine_frame_transforms(
                            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                            self.current_waypoint_pos, self.current_waypoint_quat
                        )
                        wp_w = wp_w_pos[0].detach().cpu().numpy()
                        ee_w = ee_pose_w[0, 0:3].detach().cpu().numpy()
                        dpos_w = float(np.linalg.norm(wp_w - ee_w))

                        # Log which body we use for EE
                        try:
                            ee_body_id = int(self.robot_entity_cfg.body_ids[0])
                            ee_body_name = str(self.robot.data.body_names[ee_body_id]) if hasattr(self.robot.data, 'body_names') else str(ee_body_id)
                        except Exception:
                            ee_body_id = int(self.robot_entity_cfg.body_ids[0])
                            ee_body_name = str(ee_body_id)

                        print(
                            f"[DEBUG Base] EE={ee_b}, WP={wp_b}, dpos={dpos_b:.4f}m | Target={tgt_b}, dpos_tgt={pos_err_tgt:.4f}m, errY_tgt={rot_err_y_tgt:.3f}rad"
                        )
                        print(
                            f"[DEBUG World] EE_w={ee_w}, Marker_w={wp_w}, dpos_w={dpos_w:.4f}m | EE body id={ee_body_id} name={ee_body_name}"
                        )
                    self._last_progress_print = time.time()

        # Visualize targets without rotating them by the base heading (match IRL view)
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        root_pos_w = root_pose_w[:, 0:3]

        # Place markers using root translation only (no yaw-based rotation) and show orientations in world frame
        waypoint_pos_w = root_pos_w + self.current_waypoint_pos
        final_pos_w = root_pos_w + self.final_target_pos

        # Orientations: use identity unless an explicit world-frame orientation is provided
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device).unsqueeze(0)
        wp_quat_w = self.current_waypoint_quat if self.current_waypoint_quat is not None else ident
        final_quat_w = self.final_target_quat if self.final_target_quat is not None else ident

        self.goal_marker.visualize(waypoint_pos_w, wp_quat_w)
        self.final_target_marker.visualize(final_pos_w, final_quat_w)

        
    def step(self):
        """Execute one simulation step with IK control and collision awareness."""
        # Update current waypoint
        self.update_current_waypoint()

        # Get current state
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]

        # Transform Jacobian to base frame
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        # Compute EE pose in base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        if hasattr(self, 'current_waypoint_pos') and hasattr(self, 'trajectory_log'):
            self.log_trajectory_step(
                planned_pos=self.current_waypoint_pos[0],  # Remove batch dimension
                actual_pos=ee_pos_b[0],                    # Remove batch dimension
                planned_quat=self.current_waypoint_quat[0],  # Remove batch dimension
                actual_quat=ee_quat_b[0],                    # Remove batch dimension
                joint_angles=joint_pos[0]                    # Remove batch dimension
            )
        
        # Store trajectory point in trajectory log
        # Note: trajectory_log expects (x_g, y_g, z_g, x_e, y_e, z_e) tuples

        # Prepare IK command based on mode (relative vs absolute)
        use_relative = getattr(self.diff_ik_controller.cfg, "use_relative_mode", False)

        # DEBUG: Log IK inputs
        if self.debug_ik:
            print(f"\n[DEBUG IK] Step input:")
            print(f"  Current EE pos:     {ee_pos_b[0].cpu().numpy()}")
            print(f"  Target waypoint:    {self.current_waypoint_pos[0].cpu().numpy()}")
            print(f"  Position error:     {(self.current_waypoint_pos - ee_pos_b)[0].cpu().numpy()}")
            print(f"  Mode: {'RELATIVE' if use_relative else 'ABSOLUTE'}")

        if use_relative:
            # RELATIVE MODE: Send position and rotation deltas
            pos_error = self.current_waypoint_pos - ee_pos_b

            # FRAME FIX: Transform waypoint orientation to robot's local frame
            # This ensures pitch commands are relative to robot's heading, not global frame
            # Extract robot's current yaw from slew joint (joint 0 = revolute_cabin)
            slew_angle_rad = joint_pos[:, 0]  # Slew joint angle

            # Create quaternion for yaw-only rotation around Z-axis: [w, x, y, z]
            slew_quat = torch.zeros(self.scene.num_envs, 4, device=self.sim.device)
            slew_quat[:, 0] = torch.cos(slew_angle_rad / 2)  # w
            slew_quat[:, 3] = torch.sin(slew_angle_rad / 2)  # z (yaw axis)

            # Transform waypoint to robot's local frame: quat_local = slew_inv * quat_global
            from isaaclab.utils.math import quat_mul, quat_conjugate
            slew_quat_inv = quat_conjugate(slew_quat)
            waypoint_quat_local = quat_mul(slew_quat_inv, self.current_waypoint_quat)

            # Compute error with local-frame waypoint (pitch now relative to robot heading)
            rot_error_angle, rot_error_axis = compute_pose_error(
                ee_pos_b, ee_quat_b, self.current_waypoint_pos, waypoint_quat_local, rot_error_type="axis_angle"
            )

            # Apply relative-mode gains (IK controller handles all limits internally)
            pos_error = float(self.path_config.relative_pos_gain) * pos_error
            rot_error_axis = float(self.path_config.relative_rot_gain) * rot_error_axis

            # Command is [dx, dy, dz, droll, dpitch, dyaw] (6 elements)
            command = torch.cat([pos_error, rot_error_axis], dim=-1)

            # DEBUG: Log relative command
            if self.debug_ik:
                print(f"  Relative command:   {command[0].cpu().numpy()}")
                print(f"    Pos delta (mm):   {(pos_error[0]*1000).cpu().numpy()}")
                print(f"    Rot delta (deg):  {(rot_error_axis[0]*180/np.pi).cpu().numpy()}")
        else:
            # ABSOLUTE MODE: Send target pose directly
            # Command is [x, y, z, qw, qx, qy, qz] (7 elements)
            # Waypoint orientation is already in base frame; keep it (no extra world rotation)
            command = torch.cat([self.current_waypoint_pos, self.current_waypoint_quat], dim=-1)

            # DEBUG: Log absolute command
            if self.debug_ik:
                print(f"  Absolute command:   {command[0].cpu().numpy()}")

        # Send command to IK controller
        self.diff_ik_controller.set_command(command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        # Optional desired EE velocity for velocity_mode (use simulated measurements)
        desired_ee_velocity = None
        if getattr(self.path_config, "ik_velocity_mode", False):
            # EE twist from PhysX body state (world frame) -> base frame
            lin_vel_w = self.robot.data.body_lin_vel_w[:, self.robot_entity_cfg.body_ids[0], :]
            ang_vel_w = self.robot.data.body_ang_vel_w[:, self.robot_entity_cfg.body_ids[0], :]
            lin_vel_b = torch.bmm(base_rot_matrix, lin_vel_w.unsqueeze(-1)).squeeze(-1)
            ang_vel_b = torch.bmm(base_rot_matrix, ang_vel_w.unsqueeze(-1)).squeeze(-1)

            if not getattr(self.path_config, "ik_use_rotational_velocity", True):
                ang_vel_b = torch.zeros_like(ang_vel_b)

            desired_ee_velocity = torch.cat([lin_vel_b, ang_vel_b], dim=1)

        # Compute and apply joint commands
        joint_pos_des = self.diff_ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
            desired_ee_velocity=desired_ee_velocity,
            dt=float(self.path_config.dt),
        )

        # DEBUG: Log joint commands
        if self.debug_ik:
            joint_delta = (joint_pos_des - joint_pos)[0].cpu().numpy()
            print(f"  Joint positions (current): {joint_pos[0].cpu().numpy()}")
            print(f"  Joint positions (desired): {joint_pos_des[0].cpu().numpy()}")
            print(f"  Joint delta (rad):         {joint_delta}")
            print(f"  Joint delta (deg):         {joint_delta * 180/np.pi}")

        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()

        # Visualize current EE position
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # Visualize relative pull target (only in relative mode)
        if use_relative and SHOW_RELATIVE_PULL_MARKER:
            # pull target in base frame is ee_pos_b + pos_error (after gains and clamp)
            # Apply position scaling to exaggerate the pull target for visibility
            pull_pos_b = ee_pos_b + (RELATIVE_PULL_MARKER_POSITION_SCALE * pos_error)
            pull_quat_b = ee_quat_b  # orientation visualization not critical here
            pull_pos_w, pull_quat_w = combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                pull_pos_b, pull_quat_b
            )
            self.rel_pull_marker.visualize(pull_pos_w, pull_quat_w)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main simulation loop with clean A* path planning."""
    # Create clean environment
    env = PathPlanningEnvironment(sim, scene)

    # Initialize logging
    algorithm_logging_name = args_cli.algorithm.replace("_", "")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env.init_logging(algorithm_logging_name, os.path.join(script_dir, "logs_sim"))
    
    if args_cli.algorithm == "a_star":
        algorithm_name = "A*"
    elif args_cli.algorithm == "rrt":
        algorithm_name = "RRT"
    elif args_cli.algorithm == "rrt_star":
        algorithm_name = "RRT*"
    elif args_cli.algorithm == "prm":
        algorithm_name = "PRM"
    elif args_cli.algorithm == "a_star_plane":
        algorithm_name = "A* Plane"
    else:
        algorithm_name = "Unknown"
        
    # Initialize robot
    joint_pos = env.robot.data.default_joint_pos.clone()
    joint_vel = env.robot.data.default_joint_vel.clone()
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    # Reset controller
    env.diff_ik_controller.reset()
    env._print_robot_info()

    # Spawn single wall obstacle using environment configuration
    env.spawn_wall(DEFAULT_ENV_CONFIG)
    scene.write_data_to_sim()  # Write obstacle to simulation
    sim.step()                  # Step simulation to render
    scene.update(sim.get_physics_dt())  # Update scene

    # State management for target reaching
    current_target = None
    goal_reached_counter = 0
    wait_at_goal_steps = 50
    trajectory_viz_freq = 10
    count = 0

    # Use environment configuration for targets (matches real hardware)
    env_config = DEFAULT_ENV_CONFIG
    # Start at point A first (blind move to staging pose), then alternate to B
    target_positions = [
        (env_config.point_a_pos, (0.0, 0.0, env_config.point_a_rotation_deg)),
        (env_config.point_b_pos, (0.0, 0.0, env_config.point_b_rotation_deg)),
    ]

    # Initialize target index
    env._target_index = 0
    is_first_trajectory = True  # Skip logging the blind move to Point A (matches hardware)

    # Set initial target and plan (blind move to Point A - not logged)
    try:
        target_pos, target_rot = target_positions[env._target_index]
        env.set_target(target_pos, target_rot)
        env.start_trajectory_tracking()
        logger.info("Blind move to Point A (not logged, matches hardware)")
        env.generate_trajectory(target_pos, target_rot)
        current_target = (target_pos, target_rot)
    except CollisionError as e:
        print(f"\n" + "="*60)
        print(f"[FATAL] Failed to set initial target: Start or goal is in collision")
        print(f"        Consider adjusting target positions or obstacle placement")
        print(f"        Details: {e}")
        print("="*60)
        print("[PAUSED] Path planning failed. Press Enter to exit...")
        input()
        return
    except NoPathFoundError as e:
        print(f"\n" + "="*60)
        print(f"[FATAL] Failed to set initial target: No path exists to target")
        print(f"        The target may be unreachable with current obstacles")
        print(f"        Details: {e}")
        print("="*60)
        print("[PAUSED] Path planning failed. Press Enter to exit...")
        input()
        return
    except PathPlanningError as e:
        print(f"\n" + "="*60)
        print(f"[FATAL] Failed to set initial target: {e}")
        print("="*60)
        print("[PAUSED] Path planning failed. Press Enter to exit...")
        input()
        return

    while simulation_app.is_running():
        env.step()
        sim.step()
        scene.update(sim.get_physics_dt())
        count += 1

        # Update trajectory visualization periodically
        # if count % trajectory_viz_freq == 0:
        #     env.visualize_trajectory()

        # Check if goal is reached
        if current_target is not None and env.is_goal_reached():
            goal_reached_counter += 1

            if goal_reached_counter == 1:  # First time reaching goal
                logger.info(f"Goal reached! Waiting for {wait_at_goal_steps} steps...")

                # Pause for inspection if requested
                if args_cli.pause_on_goal:
                    print("\n" + "="*60)
                    print("[PAUSED] Goal reached - simulation paused for inspection")
                    print("         Press Enter to continue...")
                    print("="*60)
                    input()

            # Wait at goal for specified steps
            if goal_reached_counter >= wait_at_goal_steps:
                logger.info("Settling complete. Saving trajectory log...")

                # Skip logging the first trajectory (blind move to Point A, matches hardware)
                if is_first_trajectory:
                    logger.info("First trajectory (blind move) - not logged (matches hardware)")
                    is_first_trajectory = False
                else:
                    # Save log AFTER settling period (matches hardware behavior)
                    env.save_trajectory_log(algorithm_logging_name)

                logger.info("Alternating target...")

                try:
                    # Alternate to next target
                    env._target_index = (env._target_index + 1) % len(target_positions)
                    target_pos, target_rot = target_positions[env._target_index]
                    env.set_target(target_pos, target_rot)
                    env.start_trajectory_tracking()
                    logger.info(f"Using {algorithm_name} path planning")
                    env.generate_trajectory(target_pos, target_rot)
                    current_target = (target_pos, target_rot)
                    goal_reached_counter = 0

                except CollisionError as e:
                    print(f"\n" + "="*60)
                    print(f"[ERROR] Failed to plan to next target: Start or goal is in collision")
                    print(f"        Details: {e}")
                    print("="*60)
                    print("[PAUSED] Path planning failed. Press Enter to continue or Ctrl+C to exit...")
                    input()
                    goal_reached_counter = 0
                except NoPathFoundError as e:
                    print(f"\n" + "="*60)
                    print(f"[ERROR] Failed to plan to next target: No path exists")
                    print(f"        Details: {e}")
                    print("="*60)
                    print("[PAUSED] Path planning failed. Press Enter to continue or Ctrl+C to exit...")
                    input()
                    goal_reached_counter = 0
                except PathPlanningError as e:
                    print(f"\n" + "="*60)
                    print(f"[ERROR] Failed to find new safe target: {e}")
                    print("="*60)
                    print("[PAUSED] Path planning failed. Press Enter to continue or Ctrl+C to exit...")
                    input()
                    goal_reached_counter = 0
        else:
            # Reset counter if we're not at goal
            goal_reached_counter = 0


def main():
    """Main function."""
    # Load kit helper
    # Use update frequency from config for simulation dt
    sim_cfg = sim_utils.SimulationCfg(dt=(1/DEFAULT_CONFIG.update_frequency), render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([0.5, 2.0, 0.5], [0.2, 0.0, 0.0])

    # Design scene
    scene_cfg = PathPlanningSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: setup complete...")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
