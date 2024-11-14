from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.controllers import BaseController

# omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import IMUSensor
import numpy as np
import carb
import omni
import omni.kit.commands
import os
from pxr import UsdGeom

import random


class TestBench(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        
        # Assets root path
        self._assets_root_path = self.get_assets_root_path()
        
        if not self._assets_root_path:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            
        # Path to the base test bench model
        folder = "Test_Bench_Model/test_bench/"
        model = "test_bench.usd"
        self.kaivuri_path = os.path.join(self._assets_root_path, folder, model)
        
        # Primary body path inside the simulation environment
        self.body_path = "/World/kaivuri"


        self.i = 0
        self.commands = np.array([0, 0, 0])


    def get_assets_root_path(self):
        user_profile = os.getenv('USERPROFILE') or os.getenv('HOME')
        if user_profile:
            return os.path.join(user_profile, "Documents/GitHub/IsaacSim/IsaacSim Models/")
        else:
            carb.log_error("Could not find the USERPROFILE or HOME environment variable.")
            return None

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())
        self._events = omni.usd.get_context().get_stage_event_stream()
        
        # Add the Kaivuri model
        add_reference_to_stage(usd_path=self.kaivuri_path, prim_path=self.body_path)
        
        self.kaivuri = Articulation(
            prim_path=self.body_path,
            name="kaivuri",
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1],
        )
        world.scene.add(self.kaivuri)
        
        # Add IMU sensors
        self.add_imu_sensors()

        return

    def add_imu_sensors(self):

        # Sensor paths for the IMU sensors
        self.sensor_paths = [
            "/World/kaivuri/liftboom/imu_liftBoom",
            "/World/kaivuri/tiltboom/imu_tiltBoom",
            "/World/kaivuri/scoop/imu_scoop",
        ]

        sensor_names = ["imu_liftBoom", "imu_tiltBoom", "imu_scoop"]

        # Roughly estimated positions and orientations of the sensors
        sensor_positions = [
            np.array([0.1275, 0.0000, 0.2909]),
            np.array([0.1059, 0.0000, -0.0180]),
            np.array([0.0247, 0.0000, -0.0135]),
        ]
        # WIP: Sensor orientations are not accurate
        sensor_orientations = [
            np.array([1, 0, 0.0, 0]),
            np.array([1, 0, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]
        
        for i, sensor_path in enumerate(self.sensor_paths):
            sensor = IMUSensor(
                prim_path=sensor_path,
                name=sensor_names[i],
                frequency=60,
                translation=sensor_positions[i],
                orientation=sensor_orientations[i],
                linear_acceleration_filter_size=10,
                angular_velocity_filter_size=10,
                orientation_filter_size=10,
            )
            setattr(self, sensor_names[i], sensor)
        return


    async def setup_post_load(self):
        self._world = self.get_world()
        
        # Initialize the hydraulics controller
        #self.hydraulics_controller = HydraulicsController()
        #self.kaivuri_controller = self.kaivuri.get_articulation_controller()
        
        # Add physics callback
        self._world.add_physics_callback("sending_actions", callback_fn=self.apply_robot_control)
        return


    def apply_robot_control(self, step_size):
        
        # Generate command array every two seconds
        if self.i % 120 == 0:
            self.commands = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
    

        # Read the joint positions
        current_position = self.kaivuri.get_joint_positions()
        print(current_position)


        # add command array to the current position array (to make the movement smooth)
        self.commands += current_position


        # Apply the commands to the robot
        self.kaivuri.set_joint_positions(self.commands)



        self.i += 1
        return



    # Example IMU reading function
    def read_imu_data(self):
        lift_boom_value = self.imu_liftBoom.get_current_frame()
        tilt_boom_value = self.imu_tiltBoom.get_current_frame()
        scoop_value = self.imu_scoop.get_current_frame()

        # Example of reading IMU data:
        print(f"LiftBoom IMU orientation: {lift_boom_value['orientation']}")

        return

    async def setup_pre_reset(self):
        await self._world.play_async()
        self.kaivuri.initialize()

        print("pre reset")
        return

    async def setup_post_reset(self):
        # Placeholder for further setup after reset
        print("post reset")
        return

    def world_cleanup(self):
        if self._world.physics_callback_exists("sending_actions"):
            self._world.remove_physics_callback("sending_actions")
        return