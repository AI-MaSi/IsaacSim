import os

from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.test_bench import TestBench


class TestBenchExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Other",
            submenu_name="",
            name="Test Bench",
            title="Masi",
            doc_link="None",
            overview="No idea what is happening",
            file_path=os.path.abspath(__file__),
            sample=TestBench(),
        )
        return
