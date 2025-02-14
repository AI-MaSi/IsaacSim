# todo: register the environment in the gym registry
# todo: basic RL usage

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Excavator RL environment with IK control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg

from isaaclab_assets import MASIV0_CFG

@configclass
class ExcavatorEnvCfg(DirectRLEnvCfg):
    """Configuration for the excavator environment."""

    # Basic environment parameters
    decimation = 2  # Run environment stepping every N simulation steps
    episode_length_s = 20.0  # Maximum episode length in seconds

    # Observation and action spaces
    action_space = 3  # Target position (x, z, rotation)
    observation_space = 9  # Current end effector (x, z, rot) + target (x, z, rot) + previous action (x, z, rot)
    state_space = 0  # No separate state space for now

    # Environment configuration
    sim = sim_utils.SimulationCfg(
        dt=0.01,
        render_interval=decimation,
    )

    # Scene configuration
    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.0,
        replicate_physics=True
    )

    # Robot configuration
    robot_cfg: ArticulationCfg = MASIV0_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

class ExcavatorEnv(DirectRLEnv):
    """RL environment for excavator control with simplified XZRot interface."""

    cfg: ExcavatorEnvCfg

    def __init__(self, cfg: ExcavatorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
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

        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Initialize visualization markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # Initialize buffers for XZRot control
        self.current_xzrot = torch.zeros(self.num_envs, 3, device=self.device)  # x, z, rotation
        self.target_xzrot = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 3, device=self.device)

        # Buffer for full pose needed by IK controller
        self.ik_commands = torch.zeros(self.num_envs, 7, device=self.device)

        # Setup robot configuration
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["end_point"])
        self.robot_entity_cfg.resolve(self.scene)

        # Get end-effector Jacobian index
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Set default XZRot state
        self.default_xzrot = torch.tensor([0.42, 0.5, 100.0], device=self.device)  # x, z, rotation
        self.target_xzrot[:] = self.default_xzrot

    def _setup_scene(self):
        """Set up the environment scene."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = actions.clone()

        # Directly construct IK command pose from XZRot
        angle = actions[:, 2] * torch.pi / 180.0
        self.ik_commands[:, 0] = actions[:, 0]  # x
        self.ik_commands[:, 1] = 0.0  # y always 0
        self.ik_commands[:, 2] = actions[:, 1]  # z
        self.ik_commands[:, 3] = torch.cos(angle / 2)  # qw
        self.ik_commands[:, 4] = 0.0  # qx
        self.ik_commands[:, 5] = torch.sin(angle / 2)  # qy
        self.ik_commands[:, 6] = 0.0  # qz

        self.previous_actions = actions.clone()

    def _apply_action(self):
        """Apply actions using IK controller."""
        # Get current robot state
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # Transform to root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Update controller command
        self.diff_ik_controller.set_command(self.ik_commands)

        # Compute and apply joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

    def _get_observations(self) -> dict:
        """Get observations from environment."""
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]

        # Extract XZRot from current pose
        angle = 2 * torch.atan2(ee_pose_w[:, 5], ee_pose_w[:, 3]) * 180.0 / torch.pi
        current_xzrot = torch.stack([
            ee_pose_w[:, 0],  # x
            ee_pose_w[:, 2],  # z
            angle  # rotation
        ], dim=-1)

        obs = torch.cat([
            current_xzrot,
            self.actions,  # Current target XZRot
            self.previous_actions
        ], dim=-1)

        # Update visualization markers
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(
            self.ik_commands[:, 0:3] + self.scene.env_origins,
            self.ik_commands[:, 3:7]
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]

        # Extract current XZRot
        angle = 2 * torch.atan2(ee_pose_w[:, 5], ee_pose_w[:, 3]) * 180.0 / torch.pi
        current_xzrot = torch.stack([
            ee_pose_w[:, 0],  # x
            ee_pose_w[:, 2],  # z
            angle  # rotation
        ], dim=-1)

        # Compute distance in XZRot space
        pos_error = torch.norm(current_xzrot[:, :2] - self.actions[:, :2], dim=-1)

        # Rotation error (considering circular nature of angles)
        rot_error = torch.abs(current_xzrot[:, 2] - self.actions[:, 2])
        rot_error = torch.min(rot_error, 360.0 - rot_error)

        reward = -(pos_error + 0.1 * rot_error)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset controller and commands
        self.diff_ik_controller.reset()

        # Reset to default XZRot
        self.actions[env_ids] = self.default_xzrot
        angle = self.default_xzrot[2] * torch.pi / 180.0
        self.ik_commands[env_ids, 0] = self.default_xzrot[0]  # x
        self.ik_commands[env_ids, 1] = 0.0  # y
        self.ik_commands[env_ids, 2] = self.default_xzrot[1]  # z
        self.ik_commands[env_ids, 3] = torch.cos(angle / 2)  # qw
        self.ik_commands[env_ids, 4] = 0.0  # qx
        self.ik_commands[env_ids, 5] = torch.sin(angle / 2)  # qy
        self.ik_commands[env_ids, 6] = 0.0  # qz

        self.previous_actions[env_ids] = self.default_xzrot

        # Make sure to set the command after reset
        self.diff_ik_controller.set_command(self.ik_commands)


@torch.jit.script
def compute_rewards():
    total_reward = 0.0
    return total_reward


# direct run for now:

#isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/masi/excv_env.py

def main():
    # Main function
    # Create environment
    env_cfg = ExcavatorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ExcavatorEnv(cfg=env_cfg)

    # Create a simple oscillating motion for testing
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            # Placeholder for actions
            # position (0.45), size (0.1), speed (0.01)
            x = 0.45 + 0.1 * torch.sin(torch.tensor(step * 0.01))
            z = 0.05 + 0.1 * torch.cos(torch.tensor(step * 0.01))
            rot = 100.0 + 30.0 * torch.sin(torch.tensor(step * 0.02))  # oscillate rotation

            actions = torch.tensor([[x, z, rot]], device=env.device).expand(env.num_envs, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            step += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
