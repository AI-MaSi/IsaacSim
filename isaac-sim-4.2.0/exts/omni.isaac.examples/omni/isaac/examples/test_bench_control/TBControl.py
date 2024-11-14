# super simple example of how to load a model into Isaac Sim and control it using python.

from omni.isaac.examples.base_sample import BaseSample

import os
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot

import carb



class TBController(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.controller = None

        # Assets root path
        self._assets_root_path = self.get_assets_root_path()

        # Error check if the path is None
        if not self._assets_root_path:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Path to the Kaivuri model. Modify this for different model
        folder = "Test_Bench_Model/test_bench/"    # inside root path
        model = "test_bench_control.usd"
        self.kaivuri_path = os.path.join(self._assets_root_path, folder, model)

        # Primary body path inside the simulation environment
        self.body_path = "/World/kaivuri"



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
        # add ground plane
        world.scene.add_default_ground_plane()

        # Add the model to the scene
        add_reference_to_stage(usd_path=self.kaivuri_path, prim_path=self.body_path)
        world.scene.add(Robot(prim_path=self.body_path, name="kaivuri"))
        return
    
    def world_cleanup(self):
        # placeholder
        return
