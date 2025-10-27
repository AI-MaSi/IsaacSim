from __future__ import annotations

from typing import Iterable

import torch

from isaaclab.scene import InteractiveScene


class SimpleGripperController:
    """Minimal PD-based gripper controller for MASI claws.

    Sets joint position targets directly on the robot PD for the two claw joints,
    giving both claws the same target value. Intended for scripts/tests where a
    high-level open()/close()/set() interface is convenient.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        claw_joint_names: Iterable[str] = ("revolute_claw_1", "revolute_claw_2"),
    ) -> None:
        self.scene = scene
        self.robot = scene["robot"]
        self.claw_joint_names = list(claw_joint_names)
        self.claw_joint_ids, _ = self.robot.find_joints(self.claw_joint_names)

    def set(self, value: float) -> None:
        """Set both claws to the same target position value (radians)."""
        target = torch.tensor([[value] * len(self.claw_joint_ids)], device=self.robot.device).repeat(
            self.scene.num_envs, 1
        )
        self.robot.set_joint_position_target(target, joint_ids=self.claw_joint_ids)

    def open(self) -> None:
        """Open claws to a preset angle."""
        self.set(0.2)

    def close(self) -> None:
        """Close claws to a preset angle."""
        self.set(0.0)


__all__ = ["SimpleGripperController"]

