from omni.isaac.examples.base_sample import BaseSample

import asyncio
import weakref

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import numpy as np

import carb
import omni
import omni.kit.commands
import omni.physx as _physx
import omni.ui as ui
from omni.isaac.sensor import _sensor
from omni.isaac.ui.ui_utils import LABEL_WIDTH, get_style, setup_ui_headers
from pxr import Gf, UsdGeom

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class TestBench(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.body_path = "/kaivuri"
        self.sensor_paths = [
            "/carriage_upper/imu_carriage",
            "/liftBoom/imu_liftBoom",
            "/tiltBoom/imu_tiltBoom"
        ]


    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()

        #self._assets_root_path = "C:/Users/sh23937/Documents/IsaacLab/my_local_server"
        self._assets_root_path = "C:/Users/sh23937/Documents/GitHub/IsaacSim/IsaacSim Models/Test_Bench_Model/test_bench"
        print(f"Assets root path: {self._assets_root_path}")
        print("-"*50)
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())

        self._events = omni.usd.get_context().get_stage_event_stream()

        #self._stage_event_subscription = self._events.create_subscription_to_pop(
            #self._on_stage_event, name="IMU Sensor Sample stage Watch")

        # Load the Kaivuri model
        #kaivuri_path = self._assets_root_path + "/kaivuri.usd"
        kaivuri_path = self._assets_root_path + "/test_bench.usd"
        prim_path = "/World/kaivuri"

        add_reference_to_stage(usd_path=kaivuri_path, prim_path=prim_path)
        kaivuri = world.scene.add(Robot(prim_path=prim_path, name="kaivuri"))

        print("Num of degrees of freedom before first reset: " + str(kaivuri.num_dof)) # prints None

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._kaivuri = self._world.scene.get_object("kaivuri")
        # Print info about the gaevur,, after the first reset is called
        print("Num of degrees of freedom after first reset: " + str(self._kaivuri.num_dof)) # prints 3
        print("Joint Positions after first reset: " + str(self._kaivuri.get_joint_positions()))


        self._kaivuri_controller = self._kaivuri.get_articulation_controller()
        # Adding a physics callback to send the actions to apply actions with every
        # physics step executed.
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return
    
    def send_robot_actions(self, step_size):
        # Send random joint velocities to the robot
        self._kaivuri_controller.apply_action(ArticulationAction(joint_positions=None,
                                                                            joint_efforts=None,
                                                                            joint_velocities=50 * np.random.rand(3,)))
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return
    

    def world_cleanup(self):
        return
