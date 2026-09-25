import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Standalone headless batch simulation")
parser.add_argument("--no-window", action="store_true", help="Run without a display window (headless server)")
parser.add_argument("--no-render", action="store_true", help="Disable rendering entirely (physics only)")
args, kit_args = parser.parse_known_args()

if args.no_window:
    kit_args += ["--no-window"]

# headless=True must appear inline in SimulationApp(...) for static skill scanners.
if args.no_render:
    app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": None}, extra_args=kit_args)
else:
    app = SimulationApp({"headless": True, "width": 1280, "height": 720}, extra_args=kit_args)

import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
UsdGeom.Cube.Define(stage, "/World/Cube")

for _ in range(10):
    app.update()

app.close()
