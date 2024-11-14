import os

from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.test_bench_control import TBController


class TBControlExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Other",
            submenu_name="",
            name="Test Bench control",
            title="Xbox controller input example",
            doc_link="None",
            overview="simple articulation input example with ActionGraphs",
            file_path=os.path.abspath(__file__),
            sample=TBController(),
        )
        return
