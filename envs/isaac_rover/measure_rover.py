"""envs/isaac_rover/measure_rover.py -- gate G2: does the rover in Omniverse drive like robot/SPEC.md?

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd /data/runs/usd/rover_train/rover_train.usda
    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd ... --physics mjwarp   # MuJoCo-Warp check
    bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd ... --hz 480           # timestep check

Open loop, no policy: this measures the machine, not a controller. Every lane is a
separate copy of the rover in one scene, far apart, on the ground or on its own ramp:

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
"""

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the imported rover against robot/SPEC.md.")
parser.add_argument("--usd", required=True, help="rover USD from robot/usd/convert_rover.py")
parser.add_argument("--physics", choices=["physx", "mjwarp"], default="physx")
parser.add_argument("--hz", type=int, default=240, help="physics steps per second")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--json", default=None, help="write the results here")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

WHEEL_R = 0.09
CHASSIS_Z = 0.175          # chassis origin above the surface with wheels touching (rover.xml)
PI = math.pi
SPEC = {"speed": 0.279, "yaw": 0.571, "climbs_deg": 14, "stalls_deg": 16, "holds_deg": 33, "slides_deg": 36}

# (name, kind, angle_deg); lanes are laid out along +y, 8 m apart
LANES = [("straight", "flat", 0), ("spin", "flat", 0)]
LANES += [(f"climb_{a}", "climb", a) for a in (10, 12, 14, 16)]
LANES += [(f"hold_{a}", "hold", a) for a in (30, 33, 36)]
LANE_DY = 8.0


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


def lane_pose(k: int, kind: str, angle_deg: float):
    """Start pose of the rover in lane k and, for ramps, the ramp box to spawn under it."""
    y0 = k * LANE_DY
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
    box = dict(size=(8.0, 4.0, thick), pos=centre, quat=q)
    return start, q, box


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args_cli.hz, device=args_cli.device,
                                                    physics=physics_cfg()))
    ground = sim_utils.GroundPlaneCfg(physics_material=ground_material())
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/light", light)

    starts = []
    for k, (name, kind, angle) in enumerate(LANES):
        pos, quat, box = lane_pose(k, kind, angle)
        starts.append((pos, quat))
        sim_utils.create_prim(f"/World/Lane_{k}", "Xform")
        if box:
            cfg = sim_utils.CuboidCfg(size=box["size"], collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=ground_material())
            cfg.func(f"/World/Ramp_{k}", cfg, translation=box["pos"], orientation=box["quat"])

    variants = {"Physics": "mujoco"} if args_cli.physics == "mjwarp" else {"Physics": "physx"}
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

    n = len(LANES)
    dev = sim.device
    pose = torch.zeros(n, 7, device=dev)
    for k, (pos, quat) in enumerate(starts):
        pose[k, :3] = torch.tensor(pos, device=dev)
        pose[k, 3:] = torch.tensor(quat, device=dev)
    rover.write_root_pose_to_sim_index(root_pose=pose)
    rover.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
    rover.reset()

    ids, names = rover.find_joints(["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"], preserve_order=True)
    left, right = [0, 2], [1, 3]
    cmd = torch.zeros(n, 4, device=dev)
    for k, (name, kind, _) in enumerate(LANES):
        if name == "spin":
            cmd[k, left], cmd[k, right] = -PI, PI
        elif kind in ("flat", "climb"):
            cmd[k, :] = PI
    print(f"[measure] physics={args_cli.physics} hz={args_cli.hz} joints={names} lanes={n}")

    # settle 0.5 s with zero command, then drive
    dt = sim.get_physics_dt()
    steps = int(round(args_cli.seconds / dt))
    settle = int(round(0.5 / dt))
    zero = torch.zeros_like(cmd)
    track = []
    for i in range(settle + steps):
        rover.actuators.target_command.set_velocity_index(value=zero if i < settle else cmd, joint_ids=ids)
        rover.write_data_to_sim()
        sim.step()
        rover.update(dt)
        if i >= settle and (i - settle) % max(1, int(round(0.1 / dt))) == 0:
            track.append((rover.data.root_pos_w.torch.clone(), rover.data.root_ang_vel_w.torch.clone(),
                          rover.data.projected_gravity_b.torch.clone()))
    p0 = track[0][0]
    ph = track[len(track) // 2][0]
    p1 = track[-1][0]
    half_t = (len(track) - 1 - len(track) // 2) * 0.1
    yaw_rate = torch.stack([t[1][:, 2] for t in track[len(track) // 2:]]).mean(0)
    tipped = torch.stack([t[2][:, 2] for t in track]).max(0).values > -0.5   # gravity no longer "down"

    res = {"physics": args_cli.physics, "hz": args_cli.hz, "lanes": {}}
    for k, (name, kind, angle) in enumerate(LANES):
        a = math.radians(angle)
        if kind == "climb":
            d = torch.tensor([math.cos(a), 0.0, math.sin(a)], device=dev)
        elif kind == "hold":
            d = torch.tensor([0.0, -math.cos(a), -math.sin(a)], device=dev)   # downhill
        else:
            d = torch.tensor([1.0, 0.0, 0.0], device=dev)
        total = float(((p1[k] - p0[k]) * d).sum())
        speed = float(((p1[k] - ph[k]) * d).sum()) / half_t
        r = {"distance_m": round(total, 3), "speed_mps": round(speed, 4),
             "yaw_rate": round(float(yaw_rate[k]), 4), "tipped": bool(tipped[k])}
        res["lanes"][name] = r
        print(f"  {name:10s} dist={total:+.3f} m  speed={speed:+.4f} m/s  yaw={float(yaw_rate[k]):+.4f} rad/s"
              f"  tipped={bool(tipped[k])}")

    L = res["lanes"]
    checks = {
        "speed": abs(L["straight"]["speed_mps"] - SPEC["speed"]) <= 0.05 * SPEC["speed"],
        "spin": abs(abs(L["spin"]["yaw_rate"]) - SPEC["yaw"]) <= 0.20 * SPEC["yaw"],
        "climbs_12": L["climb_12"]["speed_mps"] > 0.05,
        "stalls_by_16": L["climb_16"]["speed_mps"] < 0.05,
        "holds_33": L["hold_33"]["distance_m"] < 0.05,
        "nothing_tipped": not any(v["tipped"] for v in L.values()),
    }
    res["spec"], res["checks"] = SPEC, checks
    for c, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {c}")
    print("G2 PARITY PASSED" if all(checks.values()) else "G2 PARITY: see FAIL lines")
    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
    simulation_app.close()
