from omni.isaac.examples.base_sample import BaseSample

import os

from omni.isaac.sensor import IMUSensor
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import numpy as np

import carb
import omni
import omni.kit.commands

# wip import for UI etc.
import asyncio
import weakref
import omni.physx as _physx
import omni.ui as ui
from omni.isaac.ui.ui_utils import LABEL_WIDTH, get_style, setup_ui_headers
from pxr import Gf, UsdGeom

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class TestBench(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # Assets root path
        self._assets_root_path = self.get_assets_root_path()
        print(f"Assets root path: {self._assets_root_path}")
        print("-" * 50)

        # Error check if the path is None
        if not self._assets_root_path:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Path to the Kaivuri model. Modify this for different model
        folder = "Test_Bench_Model/test_bench velocity control/"    # inside root path
        model = "test_bench velocity control.usd"
        self.kaivuri_path = os.path.join(self._assets_root_path, folder, model)

        # Primary body path inside the simulation environment
        self.body_path = "/World/kaivuri"

        # Sensor paths for the IMU sensors inside the model
        self.sensor_paths = [
            "/World/kaivuri/liftboom/imu_liftBoom",
            "/World/kaivuri/tiltboom/imu_tiltBoom",
            "/World/kaivuri/scoop/imu_scoop",
        ]

        # For debugging and testing purposes!
        self.i = 0
        self.simulation_time = 0.0


    def get_assets_root_path(self):
        """
        Load models directly from GitHub repository
        """

        # Try to get USERPROFILE environment variable (Windows) or HOME (Linux/Mac)
        user_profile = os.getenv('USERPROFILE') or os.getenv('HOME')
    
        if user_profile:
            # Join the user profile path with the rest of the asset path
            # return os.path.join(user_profile, "Documents/GitHub/IsaacSim/IsaacSim Models/Test_Bench_Model/test_bench")
            return os.path.join(user_profile, "Documents/GitHub/IsaacSim/IsaacSim Models/")
        else:
            # Log error if USERPROFILE or HOME is not found
            carb.log_error("Could not find the USERPROFILE or HOME environment variable.")
            return None
        
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())
        self._events = omni.usd.get_context().get_stage_event_stream()
        

        # Add the Kaivuri model to the scene
        add_reference_to_stage(usd_path=self.kaivuri_path, prim_path=self.body_path)

        self.kaivuri = Robot(prim_path=self.body_path,
                             name="kaivuri",
                             position=[0, 0, 0],
                             orientation=[0, 0, 0, 1],
                             )
        world.scene.add(self.kaivuri)



        # Add IMU sensors to the Kaivuri model
        self.add_imu_sensors()

        return

    def add_imu_sensors(self):
    
        sensor_names = ["imu_liftBoom", "imu_tiltBoom", "imu_scoop"]
        
        # Sensor positions from the SolidWorks model
        sensor_positions = [
            np.array([0.1275, 0.0000, 0.2909]),
            np.array([0.1059, 0.0000, -0.0180]),
            np.array([0.0247, 0.0000, -0.0135]),
        ]

        # Sensor orientations. WIP!
        sensor_orientations = [
            np.array([1, 0, 0.0, 0]),
            np.array([1, 0, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]

        # Add sensors to the Kaivuri model
        for i, sensor_path in enumerate(self.sensor_paths):
            sensor = IMUSensor(
                prim_path=sensor_path,
                name=sensor_names[i],  # Use the sensor name from the list
                frequency=60,
                translation=sensor_positions[i],
                orientation=sensor_orientations[i],
                linear_acceleration_filter_size=10,
                angular_velocity_filter_size=10,
                orientation_filter_size=10,
            )
            # Dynamically assign each sensor to self as an attribute
            setattr(self, sensor_names[i], sensor)
            
        return

    async def setup_post_load(self):
        self._world = self.get_world()

        print("Num of degrees of freedom after first reset: " + str(self.kaivuri.num_dof)) # prints 3
        print("Joint Positions after first reset: " + str(self.kaivuri.get_joint_positions()))
        print(f"Joint names: {self.kaivuri.dof_names}")


        self.kaivuri_controller = self.kaivuri.get_articulation_controller()

        # Callback for general physics updates
        self._world.add_physics_callback("sending_actions", callback_fn=self.control_and_read_imu)

        return
    
    def control_and_read_imu(self, step_size):
        # Control the joints
        self.apply_robot_control()

        # Read the IMU sensors
        self.read_imu_data()

        # Debugging. Increment simulation time by the step size (1/60 seconds per frame)
        self.simulation_time += step_size

        return

    def apply_robot_control(self):
        # Apply control to the robot


        # Control the joints using a sine wave. This is just for demonstration purposes.

        amplitude = 10  # Control value should be between -1 and 1, as in the real excavator.
        frequency = 0.2  # Set frequency for sine wave control
        phase = 0      # Set the phase shift
        joint_vel = np.array([
            self.generate_sine_wave(amplitude, frequency, phase, self.simulation_time),
            self.generate_sine_wave(amplitude, frequency, phase, self.simulation_time),
            self.generate_sine_wave(amplitude, frequency, phase, self.simulation_time)
        ])

        #print(f"Joint velocities: {joint_vel}")

        # Apply the control to the robot
        self.kaivuri_controller.apply_action(ArticulationAction(
            joint_positions=None,               # Direct position control
            joint_efforts=None,                 # Force-based control
            joint_velocities=joint_vel          # Velocity control using sine wave
        ))

        return

    def generate_sine_wave(self, amplitude, frequency, phase, time):
        # Generate a sine wave. Used to demonstrate the control of the joints
        return amplitude * np.sin(2 * np.pi * frequency * time + phase)

    def read_imu_data(self):
        # Reading data from imu_liftBoom sensor
        lift_boom_value = self.imu_liftBoom.get_current_frame()
        #print(f"LiftBoom IMU acceleration: {lift_boom_value['lin_acc']}")
        #print(f"LiftBoom IMU angular velocity: {lift_boom_value['ang_vel']}")
        #print(f"LiftBoom IMU orientation: {lift_boom_value['orientation']}")
        
        # Reading data from imu_tiltBoom sensor
        tilt_boom_value = self.imu_tiltBoom.get_current_frame()
        #print(f"TiltBoom IMU acceleration: {tilt_boom_value['lin_acc']}")
        #print(f"TiltBoom IMU angular velocity: {tilt_boom_value['ang_vel']}")
        #print(f"TiltBoom IMU orientation: {tilt_boom_value['orientation']}")
        
        # Reading data from imu_scoop sensor
        scoop_value = self.imu_scoop.get_current_frame()
        #print(f"Scoop IMU acceleration: {scoop_value['lin_acc']}")
        #print(f"Scoop IMU angular velocity: {scoop_value['ang_vel']}")
        #print(f"Scoop IMU orientation: {scoop_value['orientation']}")


        # Debug. Print every 30th frame (~0.5 second)
        if self.i % 30 == 0:
            print(f"LiftBoom IMU orientation: {lift_boom_value['orientation']}")
        

        self.i += 1
        return  # here you could e.g. return a list containing all the sensor values

    async def setup_pre_reset(self):
        await self._world.play_async()
        self.kaivuri.initialize()
        return

    async def setup_post_reset(self):
        return
    
    def world_cleanup(self):
        if self._world.physics_callback_exists("sending_actions"):
            self._world.remove_physics_callback("sending_actions")
        return