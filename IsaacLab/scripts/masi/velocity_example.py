#!/usr/bin/env python3

"""
Velocity-integrated control example using a simple actuator (no linkage scaling).

This script mirrors test_linkage_corrected.py in scene setup and controls but
routes keyboard velocity commands through a clean VelocityIntegratedActuator
that integrates joint velocities into position targets. It does NOT perform
any linkage-rate based scaling or cylinder geometry measurements.

Usage:
    ./isaaclab.sh -p scripts/masi/velocity_example.py

Controls:
    Q/A: Rotate Cabin
    W/S: Move Lift Boom
    E/D: Move Tilt Boom
    Z/X: Move Scoop
    C/V: Rotate Gripper
    B/N: Open/Close Claws
    R: Reset joint positions
    ESC: Exit
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="MASI Velocity-Integrated Actuator Example")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to clone.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import math
import traceback

import torch

import carb
import carb.input
import omni
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sensors.frame_transformer import FrameTransformerCfg, OffsetCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_assets.robots.masiV0 import MASI_CFG

from actuators import VelocityIntegratedActuator


def y_rot_to_quat(deg: float):
    rad = math.radians(deg)
    return (math.cos(rad / 2), 0.0, math.sin(rad / 2), 0.0)


@configclass
class MasiSceneCfg(InteractiveSceneCfg):
    """Configuration for MASI robot scene with sensors (kept same as test)."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = MASI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # IMUs (kept for parity; not required by this actuator)
    imu_lift = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lift_boom",
        debug_vis=False,
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.125, 0.0, 0.300),
            rot=y_rot_to_quat(-50.09 - 13.85),
        ),
    )

    imu_tilt = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tilt_boom",
        debug_vis=False,
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.110, 0.0, -0.010),
            rot=y_rot_to_quat(0.61),
        ),
    )

    imu_tool = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool_body",
        debug_vis=False,
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0250, 0.0, -0.005),
            rot=y_rot_to_quat(0.0),
        ),
    )

    # Frame transformers (kept for parity/visualization; not used by the actuator)
    frame_transformer_lift = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/upper_carriage",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer/Lift").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.05, 0.05, 0.05))}
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="lift_upper",
                prim_path="{ENV_REGEX_NS}/Robot/lift_boom",
                offset=OffsetCfg(pos=(0.0082, 0.0, 0.2029)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="lift_lower",
                prim_path="{ENV_REGEX_NS}/Robot/upper_carriage",
                offset=OffsetCfg(pos=(0.0615, 0.0, 0.0105)),
            ),
        ],
    )

    frame_transformer_tilt = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lift_boom",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer/Tilt").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.05, 0.05, 0.05))}
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="tilt_upper",
                prim_path="{ENV_REGEX_NS}/Robot/tilt_boom",
                offset=OffsetCfg(pos=(-0.065, 0.0, 0.02858)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="tilt_lower",
                prim_path="{ENV_REGEX_NS}/Robot/lift_boom",
                offset=OffsetCfg(pos=(0.01121, 0.0, 0.2697)),
            ),
        ],
    )

    frame_transformer_scoop = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tilt_boom",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer/Scoop").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.05, 0.05, 0.05))}
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="scoop_upper",
                prim_path="{ENV_REGEX_NS}/Robot/scoop_link_3",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="scoop_lower",
                prim_path="{ENV_REGEX_NS}/Robot/tilt_boom",
                offset=OffsetCfg(pos=(0.0225, 0.0, 0.06033)),
            ),
        ],
    )


class KeyboardController:
    """Simple keyboard controller for velocity commands."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.velocity_commands = torch.zeros((num_envs, 7), device=device)
        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = appwindow.get_keyboard()
        self.vel_scale = 1.0

    def update(self) -> torch.Tensor:
        self.velocity_commands.zero_()

        # Q/A: revolute_lift
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q):
            self.velocity_commands[:, 0] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A):
            self.velocity_commands[:, 0] = -self.vel_scale

        # W/S: revolute_tilt
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W):
            self.velocity_commands[:, 1] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S):
            self.velocity_commands[:, 1] = -self.vel_scale

        # E/D: revolute_scoop
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E):
            self.velocity_commands[:, 2] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D):
            self.velocity_commands[:, 2] = -self.vel_scale

        # Z/X: revolute_cabin
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Z):
            self.velocity_commands[:, 3] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.X):
            self.velocity_commands[:, 3] = -self.vel_scale

        # C/V: revolute_gripper
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.C):
            self.velocity_commands[:, 4] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.V):
            self.velocity_commands[:, 4] = -self.vel_scale

        # B/N: revolute_claw_1 and revolute_claw_2
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.B):
            self.velocity_commands[:, 5] = self.vel_scale
            self.velocity_commands[:, 6] = self.vel_scale
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.N):
            self.velocity_commands[:, 5] = -self.vel_scale
            self.velocity_commands[:, 6] = -self.vel_scale

        return self.velocity_commands

    def is_reset_pressed(self) -> bool:
        return self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.R)


def main():
    # Create simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0"))
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # Scene
    scene_cfg = MasiSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Visualization markers (optional)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    imu_lift_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/IMU/lift_boom"))
    imu_tilt_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/IMU/tilt_boom"))
    imu_tool_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/IMU/tool_body"))

    # Inputs
    keyboard = KeyboardController(num_envs=args_cli.num_envs, device=sim.device)

    # Collect joint names from MASI config actuator groups (same approach as test)
    all_joint_names = []
    for actuator_name, actuator_cfg in MASI_CFG.actuators.items():
        joint_names = actuator_cfg.joint_names_expr
        all_joint_names.extend(joint_names)
        print(f"[INFO] Found actuator group '{actuator_name}' with joints: {joint_names}")

    print(f"[INFO] Total controlled joints: {all_joint_names}")

    # Create the simple velocity-integrated actuator (no linkage scaling)
    actuator = VelocityIntegratedActuator(
        scene=scene,
        joint_names=all_joint_names,
        sim_dt=sim.get_physics_dt(),
        clamp_to_limits=True,
    )

    print("=" * 80)
    print("VELOCITY-INTEGRATED ACTUATOR EXAMPLE (no linkage correction)")
    print("=" * 80)
    print("Controls:")
    print("  Q/A: Move revolute_lift joint (up/down)")
    print("  W/S: Move revolute_tilt joint (forward/back)")
    print("  E/D: Move revolute_scoop joint (rotate)")
    print("  Z/X: Move revolute_cabin joint (rotate)")
    print("  C/V: Move revolute_gripper joint (rotate)")
    print("  B/N: Open/close claws (symmetric)")
    print("  R:   Reset joint positions to current")
    print("  ESC: Exit")
    print("=" * 80)

    step_count = 0
    log_interval = 100

    while simulation_app.is_running():
        if sim.is_stopped():
            break

        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue

        velocity_commands = keyboard.update()

        if torch.any(torch.abs(velocity_commands) > 1e-6):
            print(f"[DEBUG] Keyboard velocity commands: {velocity_commands[0].tolist()}")

        if keyboard.is_reset_pressed():
            actuator.reset()
            print("[INFO] Actuator reset - target positions set to current")

        # Apply velocity commands (straight joint-space integration)
        actuator.apply_velocity_command(velocity_commands)

        # Write data and step
        scene.write_data_to_sim()
        sim.step()

        # Update scene and optional markers
        scene.update(sim.get_physics_dt())
        imu_lift_marker.visualize(scene["imu_lift"].data.pos_w, scene["imu_lift"].data.quat_w)
        imu_tilt_marker.visualize(scene["imu_tilt"].data.pos_w, scene["imu_tilt"].data.quat_w)
        imu_tool_marker.visualize(scene["imu_tool"].data.pos_w, scene["imu_tool"].data.quat_w)

        if step_count % log_interval == 0:
            current_pos = scene["robot"].data.joint_pos[0, actuator.joint_ids]
            target_pos = actuator._target_position[0]
            print(f"\n[Step {step_count}]")
            print(f"  Target Positions:  {[f'{x:.4f}' for x in target_pos.tolist()]}")
            print(f"  Current Positions: {[f'{x:.4f}' for x in current_pos.tolist()]}")

        step_count += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()

