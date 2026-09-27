"""envs/isaac_rover/measure_rover.py -- gate G2: does the rover in Omniverse drive like robot/SPEC.md?

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    U=/data/runs/usd/rover_train/rover_train/rover_train.usda
    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd $U                    # PhysX 240 Hz
    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd $U --hz 480           # timestep check
    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd $U --physics mjwarp   # MuJoCo-Warp check
    add --realtime --loops 5 to watch it at real speed in the live view (http://localhost:8080 via watch.sh)

Open loop, no policy: this measures the machine, not a controller. Every lane is a separate
copy of the rover in one scene, on the ground or on its own ramp:

    straight   all four wheels +pi rad/s for 8 s          -> top speed   SPEC 0.279 m/s (+-5 %)
    spin       left -pi / right +pi for 8 s               -> yaw rate    SPEC 0.571 rad/s (+-20 %)
    climb_N    full forward up an N-degree ramp           -> climbs 14 deg, stalls by 16 deg (SPEC)
    hold_N     parked (wheel targets 0) across an N-degree
               side slope                                 -> holds to 33 deg, slides by 36 (SPEC)

The wheel actuators are set here, in SI units, from robot/rover.xml -- NOT from the drive
gains the importer wrote into the USD (docs/decisions.md D6):
    velocity servo kv 18 N*m*s/rad -> damping 18, stiffness 0;  forcerange +-2.4 N*m -> effort 2.4
    armature 0.01;  frictionloss 0.60 N*m -> friction 0.60;  joint damping 0.02 -> viscous 0.02
Ground and ramps: mu 1.0 with the "multiply" combine mode, so the tyre-ground pair takes the
tyre's 0.65 -- the PhysX equivalent of MJCF priority=1 (Isaac Lab's velocity-env idiom).

Viewing: a live Viser view bound to the box's 127.0.0.1 only (reached through the SSH tunnel)
and an RTX clip of the two flat lanes from Isaac Lab's headless Kit visualizer, written next to
the JSON results. The script refuses to run if any viewer would listen beyond localhost.
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the imported rover against robot/SPEC.md.")
parser.add_argument("--usd", required=True, help="rover USD from robot/usd/convert_rover.py")
parser.add_argument("--physics", choices=["physx", "mjwarp"], default="physx")
parser.add_argument("--hz", type=int, default=240, help="physics steps per second")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--out", default="/data/runs/measure", help="results + clip directory")
parser.add_argument("--realtime", action="store_true", help="pace to wall-clock time (for watching live)")
parser.add_argument("--loops", type=int, default=1, help="repeat the drive (results from the first)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Launch Kit with rendering (for the RTX clip) plus the Viser web view. Their settings come from
# SimulationCfg.visualizer_cfgs below; with both listed on the CLI, Isaac Lab keeps our configs.
args_cli.visualizer = ["kit", "viser"]     # counts as an explicit --viz request (app_launcher.py:973)
args_cli.video = True                      # needs MoviePy - scripts/gpu/Dockerfile adds it
args_cli.enable_cameras = True             # rendering-capable Kit experience, as `train --video` uses
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402
from isaaclab_visualizers.viser import ViserVisualizerCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

CHASSIS_Z = 0.175          # chassis origin above the surface with wheels touching (rover.xml)
PI = math.pi
SPEC = {"speed": 0.279, "yaw": 0.571, "climbs_deg": 14, "stalls_deg": 16, "holds_deg": 33, "slides_deg": 36}

# (name, kind, angle_deg, lane y). Flat lanes close together so one camera frames both.
LANES = [("straight", "flat", 0, 0.0), ("spin", "flat", 0, 3.0)]
LANES += [(f"climb_{a}", "climb", a, 9.0 + 6.0 * i) for i, a in enumerate((10, 12, 14, 16))]
LANES += [(f"hold_{a}", "hold", a, 33.0 + 6.0 * i) for i, a in enumerate((30, 33, 36))]


def physics_cfg():
    if args_cli.physics == "physx":
        from isaaclab_physx.physics import PhysxCfg
        return PhysxCfg()                                   # TGS, external forces every iteration
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    # rover.xml's own contact model: elliptic cone, impratio 1, implicitfast
    return NewtonCfg(solver_cfg=MJWarpSolverCfg(cone="elliptic", impratio=1.0, integrator="implicitfast",
                                                njmax=2000, nconmax=600))


def ground_material():
    return sim_utils.PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                               friction_combine_mode="multiply",
                                               restitution_combine_mode="multiply")


def quat_xyzw(axis: str, angle: float) -> tuple:
    """Rotation about a world axis, Isaac Lab 3.0 order (x, y, z, w)."""
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return {"x": (s, 0.0, 0.0, c), "y": (0.0, s, 0.0, c)}[axis]


def lane_pose(kind: str, angle_deg: float, y0: float):
    """Start pose of the rover in a lane and, for ramps, the ramp box to spawn under it."""
    a = math.radians(angle_deg)
    if kind == "flat":
        return (0.0, y0, CHASSIS_Z + 0.005), (0.0, 0.0, 0.0, 1.0), None
    if kind == "climb":        # surface rises along +x: rotate -a about y
        q = quat_xyzw("y", -a)
        n = (-math.sin(a), 0.0, math.cos(a))          # surface normal
        d = (math.cos(a), 0.0, math.sin(a))           # uphill direction
    else:                      # hold: surface tilts sideways, downhill is -y: rotate +a about x
        q = quat_xyzw("x", a)
        n = (0.0, -math.sin(a), math.cos(a))
        d = (0.0, math.cos(a), math.sin(a))
    thick, centre_up = 0.2, 1.5                        # ramp box thickness; ramp centre height
    centre = (0.0, y0, centre_up)
    top = tuple(centre[i] + thick / 2 * n[i] for i in range(3))
    start = tuple(top[i] - 1.5 * d[i] * (kind == "climb") + (CHASSIS_Z + 0.005) * n[i] for i in range(3))
    return start, q, dict(size=(8.0, 4.0, thick), pos=centre, quat=q)


def main():
    viser = ViserVisualizerCfg(bind_address="127.0.0.1", port=8080, open_browser=False, share=False)
    kit = KitVisualizerCfg(headless=True, eye=(4.5, -4.0, 2.6), lookat=(1.0, 1.5, 0.2),
                           window_width=1280, window_height=720)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args_cli.hz, device=args_cli.device,
                                                    physics=physics_cfg(), visualizer_cfgs=[viser, kit]))
    ground = sim_utils.GroundPlaneCfg(physics_material=ground_material())
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/light", light)

    starts = []
    for k, (name, kind, angle, y0) in enumerate(LANES):
        pos, quat, box = lane_pose(kind, angle, y0)
        starts.append((pos, quat))
        sim_utils.create_prim(f"/World/Lane_{k}", "Xform", translation=(0.0, y0, 0.0))  # rovers never overlap
        if box:
            cfg = sim_utils.CuboidCfg(size=box["size"], collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=ground_material())
            cfg.func(f"/World/Ramp_{k}", cfg, translation=box["pos"], orientation=box["quat"])

    # The "mujoco" physics variant replaces the drives with MuJoCo servos that nothing commands
    # (they act as brakes under Newton); Newton runs use the backend-portable "physics" variant.
    variants = {"Physics": "physx" if args_cli.physics == "physx" else "physics"}
    rover_cfg = ArticulationCfg(
        prim_path="/World/Lane_.*/Rover",
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.usd, variants=variants),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, CHASSIS_Z + 0.005), joint_pos={".*": 0.0}),
        actuators={"wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"], stiffness=0.0, damping=18.0, joint_effort_limit=2.4,
            armature=0.01, friction=0.60, dynamic_friction=0.60, viscous_friction=0.02)},
    )
    rover = Articulation(rover_cfg)
    sim.reset()

    # Safety: every viewer must be box-local; the owner reaches them only through SSH.
    for v in getattr(sim, "visualizers", []):
        addr = getattr(v.cfg, "bind_address", "127.0.0.1")
        if addr not in ("127.0.0.1", "localhost") or getattr(v.cfg, "share", False):
            raise SystemExit(f"refusing to run: visualizer {v.cfg.visualizer_type} would listen on {addr}")
    kit_viz = next((v for v in getattr(sim, "visualizers", []) if v.cfg.visualizer_type == "kit"), None)

    n, dev = len(LANES), sim.device
    ids, names = rover.find_joints(["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"], preserve_order=True)
    left, right = [0, 2], [1, 3]
    cmd = torch.zeros(n, 4, device=dev)
    for k, (name, kind, _, _) in enumerate(LANES):
        if name == "spin":
            cmd[k, left], cmd[k, right] = -PI, PI
        elif kind in ("flat", "climb"):
            cmd[k, :] = PI
    mass = getattr(rover.data, "body_mass", None)
    mass_kg = float(mass.torch[0].sum()) if mass is not None and hasattr(mass, "torch") else None
    print(f"[measure] physics={args_cli.physics} hz={args_cli.hz} joints={names} lanes={n} mass={mass_kg}")

    def reset_lanes():
        pose = torch.zeros(n, 7, device=dev)
        for k, (pos, quat) in enumerate(starts):
            pose[k, :3] = torch.tensor(pos, device=dev)
            pose[k, 3:] = torch.tensor(quat, device=dev)
        rover.write_root_pose_to_sim_index(root_pose=pose)
        rover.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
        zeros = torch.zeros(n, rover.num_joints, device=dev)
        rover.write_joint_position_to_sim_index(position=zeros)
        rover.write_joint_velocity_to_sim_index(velocity=zeros)
        rover.reset()

    dt = sim.get_physics_dt()
    steps, settle = int(round(args_cli.seconds / dt)), int(round(0.5 / dt))
    every = max(1, int(round(0.1 / dt)))
    frame_every = max(1, int(round(1.0 / 30.0 / dt)))          # 30 fps clip
    zero = torch.zeros_like(cmd)
    frames, track = [], []
    for loop in range(args_cli.loops):
        reset_lanes()
        t0 = time.perf_counter()
        for i in range(settle + steps):
            rover.actuators.target_command.set_velocity_index(value=zero if i < settle else cmd, joint_ids=ids)
            rover.write_data_to_sim()
            sim.step()
            rover.update(dt)
            if loop == 0:
                if i >= settle and (i - settle) % every == 0:
                    track.append((rover.data.root_pos_w.torch.clone(), rover.data.root_ang_vel_w.torch.clone(),
                                  rover.data.projected_gravity_b.torch.clone()))
                if kit_viz is not None and i % frame_every == 0:
                    f = kit_viz.render_rgb_array()
                    if f is not None and f.size and f.any():
                        frames.append(np.asarray(f)[..., :3])
            if args_cli.realtime:
                lag = (i + 1) * dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)

    p0, ph, p1 = track[0][0], track[len(track) // 2][0], track[-1][0]
    half_t = (len(track) - 1 - len(track) // 2) * every * dt
    yaw_rate = torch.stack([t[1][:, 2] for t in track[len(track) // 2:]]).mean(0)
    tipped = torch.stack([t[2][:, 2] for t in track]).max(0).values > -0.5   # gravity no longer "down"

    res = {"physics": args_cli.physics, "hz": args_cli.hz, "mass_kg": mass_kg, "lanes": {}}
    for k, (name, kind, angle, _) in enumerate(LANES):
        a = math.radians(angle)
        if kind == "climb":
            d = torch.tensor([math.cos(a), 0.0, math.sin(a)], device=dev)
        elif kind == "hold":
            d = torch.tensor([0.0, -math.cos(a), -math.sin(a)], device=dev)   # downhill
        else:
            d = torch.tensor([1.0, 0.0, 0.0], device=dev)
        total = float(((p1[k] - p0[k]) * d).sum())
        speed = float(((p1[k] - ph[k]) * d).sum()) / half_t
        res["lanes"][name] = {"distance_m": round(total, 3), "speed_mps": round(speed, 4),
                              "yaw_rate": round(float(yaw_rate[k]), 4), "tipped": bool(tipped[k])}
        print(f"  {name:10s} dist={total:+.3f} m  speed={speed:+.4f} m/s  yaw={float(yaw_rate[k]):+.4f} rad/s"
              f"  tipped={bool(tipped[k])}")

    L = res["lanes"]
    checks = {
        "speed": abs(L["straight"]["speed_mps"] - SPEC["speed"]) <= 0.05 * SPEC["speed"],
        "spin": abs(abs(L["spin"]["yaw_rate"]) - SPEC["yaw"]) <= 0.20 * SPEC["yaw"],
        "climbs_14": L["climb_14"]["speed_mps"] > 0.05,
        "stalls_by_16": L["climb_16"]["speed_mps"] < 0.05,
        "holds_33": L["hold_33"]["distance_m"] < 0.05,
        "nothing_tipped": not any(v["tipped"] for v in L.values()),
    }
    res["spec"], res["checks"] = SPEC, checks
    for c, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {c}")
    print("G2 PARITY PASSED" if all(checks.values()) else "G2 PARITY: see FAIL lines")

    os.makedirs(args_cli.out, exist_ok=True)
    stem = os.path.join(args_cli.out, f"measure_{args_cli.physics}_{args_cli.hz}hz")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=2)
    if frames:
        import imageio
        imageio.mimwrite(stem + ".mp4", frames, fps=30, codec="libx264", quality=7)
        print(f"[measure] clip: {stem}.mp4 ({len(frames)} frames)")
    else:
        print("[measure] no clip frames captured")
    print(f"[measure] results: {stem}.json")


if __name__ == "__main__":
    main()
    simulation_app.close()
