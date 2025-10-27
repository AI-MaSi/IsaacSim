from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class MasiLiftSceneCfg(InteractiveSceneCfg):
    """Minimal scene for MASI: robot, a liftable object, ground, and light."""

    # Robot articulation (provided by agent env cfg)
    robot: ArticulationCfg = MISSING

    # A simple rigid object to lift
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        # Start in front of the robot base; adjust as needed
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.10)),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ObservationsCfg:
    """Observation specs (kept minimal and generic)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Basic proprioception and last action history
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms (placeholder; to be extended when task is defined)."""

    # Encourage small actions as a gentle regularizer
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms (time-out only for now)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Reset scene to default state on episode reset."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class MasiBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base Manager-based env config for the MASI robot.

    This config sets up the scene and generic MDP scaffolding. Specific action
    heads (e.g., IK) are provided by derived configs under `config/`.

    Custom Actuator Compatibility
    - The MASI robot is controlled via joint position targets (PD) by default.
    - The custom velocity-integrated actuator in `masi/actuators` integrates
      joint-velocity commands into position targets and writes them to the same
      PD interface, so it coexists with this setup.
    """

    # Scene and MDP settings
    scene: MasiLiftSceneCfg = MasiLiftSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # General rollouts
        self.decimation = 2
        self.episode_length_s = 5.0
        # Simulation settings (100 Hz)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


