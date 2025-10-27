from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.masiV0 import MASI_CFG

from .. import mdp


@configclass
class MasiLogSceneCfg(InteractiveSceneCfg):
    """Scene with MASI and a simple log object to reach."""

    robot: ArticulationCfg = MISSING
    log: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Log",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.20, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.100),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        ),
        # place on ground: center at z ~ half thickness (0.03)
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: Policy = Policy()


@configclass
class ActionsCfg:
    # IK with EE at 115 mm below gripper_frame
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        # Exclude claws from IK: only arm joints
        joint_names=[
            "revolute_cabin",
            "revolute_lift",
            "revolute_tilt",
            "revolute_scoop",
            "revolute_gripper",
        ],
        body_name="gripper_frame",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.115]),
    )
    # Binary gripper action controlling both claws with the same value
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["revolute_claw_.*"],
        open_command_expr={"revolute_claw_.*": 0.2},
        close_command_expr={"revolute_claw_.*": 0.0},
    )


@configclass
class RewardsCfg:
    # Encourage reaching the log (negative distance)
    reaching_log = RewTerm(func=mdp.ee_to_log_distance, weight=1.0)
    # Mild action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Simple spawn: random box in front of robot, fixed Z at ground height, random yaw
    reset_log = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.75), "y": (-0.25, 0.25), "z": (0.03, 0.03), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14159, 3.14159)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("log"),
        },
    )


@configclass
class MasiIkReachLogEnvCfg(ManagerBasedRLEnvCfg):
    scene: MasiLogSceneCfg = MasiLogSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Robot with higher PD stiffness for IK tracking
        self.scene.robot = MASI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Rollout settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


@configclass
class MasiIkReachLogEnvCfg_PLAY(MasiIkReachLogEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for quick runs
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        # Disable observation corruption for play
        self.observations.policy.enable_corruption = False
