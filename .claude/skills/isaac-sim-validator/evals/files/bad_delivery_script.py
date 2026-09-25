from isaacsim import SimulationApp

app = SimulationApp({})

import omni.replicator.core as rep
from omni.isaac.core import World
from pxr import UsdGeom

world = World()
world.scene.add_default_ground_plane()
UsdGeom.Cube.Define(world.stage, "/World/Cube")
world.reset()
rep.settings.set_render_rtx_pathtracing(spp=64)

for _ in range(50):
    world.step(render=True)

app.close()
