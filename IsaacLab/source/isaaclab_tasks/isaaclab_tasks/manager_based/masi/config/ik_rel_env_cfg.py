from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.masiV0 import MASI_CFG

from ..masi_env_cfg import MasiBaseEnvCfg


@configclass
class MasiIkRelEnvCfg(MasiBaseEnvCfg):
    """MASI env with relative end-effector IK actions for lifting.

    - The arm action is a 6-DoF end-effector delta pose command (position+rotation)
      in the MASI base/root frame, tracked via a Differential IK controller.
    - Joint targets produced by the IK controller are sent to the robot's PDs.
    - The MASI custom velocity-integrated actuator can still be used in scripts
      for joint-velocity control (see `masi/velocity_example.py`).
    """

    def __post_init__(self):
        super().__post_init__()

        # Plug in the MASI robot
        self.scene.robot = MASI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions: Relative end-effector IK on MASI
        # Exclude claws from IK by specifying only arm joints
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "revolute_cabin",
                "revolute_lift",
                "revolute_tilt",
                "revolute_scoop",
                "revolute_gripper",
            ],
            body_name="gripper_frame",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            # EE offset: Z = -0.115 m from gripper_frame
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.115]),
        )
