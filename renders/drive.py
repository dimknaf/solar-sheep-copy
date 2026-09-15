import numpy as np, mujoco, imageio.v2 as imageio, os
m = mujoco.MjModel.from_xml_path("robot/rover.xml")
d = mujoco.MjData(m)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
print("actuators:", names)
L=[i for i,n in enumerate(names) if n and n.endswith(("fl","rl"))]
R=[i for i,n in enumerate(names) if n and n.endswith(("fr","rr"))]
hi=float(m.actuator_ctrlrange[0,1]); print("ctrl max:", round(hi,3))

r = mujoco.Renderer(m, height=720, width=1280)
cam = mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.distance=3.2; cam.elevation=-18; cam.lookat[:]=[0,0,0.2]

FPS=30; sub=max(1,int(round((1.0/FPS)/m.opt.timestep)))
frames=[]
# script: drive straight, spin in place, drive straight again
plan=[(4.0, hi, hi), (3.0, -hi*0.7, hi*0.7), (4.0, hi, hi)]
t=0.0
for dur, l, rr in plan:
    n=int(dur/m.opt.timestep)
    for k in range(n):
        d.ctrl[L]=l; d.ctrl[R]=rr
        mujoco.mj_step(m,d)
        t+=m.opt.timestep
        if k % sub == 0:
            cam.lookat[:2]=d.qpos[:2]           # follow the rover
            cam.azimuth = 130 + 8*np.sin(t*0.25)
            r.update_scene(d, camera=cam)
            frames.append(r.render())
r.close()
out="renders/rover_drive.mp4"
try:
    imageio.mimwrite(out, frames, fps=FPS, codec="libx264")
except Exception as e:
    print("pyav/ffmpeg mp4 failed:", e, "-> writing GIF instead")
    out = out.replace(".mp4", ".gif")
    imageio.mimwrite(out, frames[::2], fps=15)
print(f"{out}  frames={len(frames)}  {os.path.getsize(out)/1e6:.2f} MB")
print("final xy:", np.round(d.qpos[:2],3), " upright_z:", round(float(d.xmat[1,8]),3))
