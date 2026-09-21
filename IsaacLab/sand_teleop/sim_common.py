"""Shared control-pipeline settings for live simulation and recorded replay."""

from __future__ import annotations


SIM_HZ = 100.0
ARM_JOINT_NAMES = ["revolute_lift", "revolute_tilt", "revolute_tool"]

ARM_STIFFNESS = 600.0
ARM_DAMPING = 40.0
CARRIAGE_STIFFNESS = 600.0
CARRIAGE_DAMPING = 40.0
SOLVER_VELOCITY_ITERATIONS = 1
DISABLE_ROBOT_GRAVITY = True

NN_ARM_VEL_LIMIT_DEFAULT = 10.0
NN_ARM_VEL_WARN_DEFAULT = 2.0


def configure_arm_drive(robot, joint_ids, direct: bool) -> None:
    """Release the PhysX arm drive for direct state integration, otherwise enable PD."""
    stiffness = 0.0 if direct else ARM_STIFFNESS
    damping = 0.0 if direct else ARM_DAMPING
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)
    robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids)


def sanitize_velocity_prediction(prediction, limit: float):
    """Zero non-finite output and apply the shared last-resort velocity clamp."""
    import numpy as np

    velocity = np.asarray(prediction, dtype=np.float32)
    if not np.all(np.isfinite(velocity)):
        return np.zeros_like(velocity), 0.0, False, False
    max_abs = float(np.max(np.abs(velocity)))
    clipped = max_abs > limit
    return np.clip(velocity, -limit, limit), max_abs, True, clipped


__all__ = [
    "ARM_DAMPING",
    "ARM_JOINT_NAMES",
    "ARM_STIFFNESS",
    "CARRIAGE_DAMPING",
    "CARRIAGE_STIFFNESS",
    "DISABLE_ROBOT_GRAVITY",
    "NN_ARM_VEL_LIMIT_DEFAULT",
    "NN_ARM_VEL_WARN_DEFAULT",
    "SIM_HZ",
    "SOLVER_VELOCITY_ITERATIONS",
    "configure_arm_drive",
    "sanitize_velocity_prediction",
]
