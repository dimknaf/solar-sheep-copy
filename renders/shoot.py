import os, sys
import mujoco, numpy as np
try:
    import imageio.v2 as imageio
except ImportError:
    os.system(f'"{sys.executable}" -m pip install -q imageio')
    import imageio.v2 as imageio

m = mujoco.MjModel.from_xml_path("robot/rover.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for _ in range(200):      # let it settle on its wheels
    mujoco.mj_step(m, d)

cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print("cameras:", cams)

r = mujoco.Renderer(m, height=720, width=1280)
for cam in cams:
    r.update_scene(d, camera=cam)
    px = r.render()
    out = f"renders/rover_{cam}.png"
    imageio.imwrite(out, px)
    print(f"  {out}  mean_pixel={px.mean():.1f}")
r.close()
print("DONE")
