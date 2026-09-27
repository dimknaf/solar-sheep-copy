"""envs/swap/latch_spike.py -- can Omniverse (PhysX) release and re-latch the battery pack at runtime?

The MuJoCo swap flips `eq_active` on the `pack_latch` weld. The Isaac Sim importer turns that weld
into a USD PhysicsFixedJoint excluded from the articulation (`/<rover>/Physics/pack_latch`). Isaac Lab
3.0 EA has no runtime joint-enable API, but the stage stays attached to PhysX while playing, so an edit
of `physics:jointEnabled` should reach the solver -- on CPU at least (surface grippers, the one
official runtime attach, are CPU-only for the same reason). This spike tests exactly that, on one
rover, before the dock adapter (envs/swap/isaac_hardware.py) is built on it:

    A  latched, 1 s        the pack hangs in its bay (offset (0, 0, -0.0875) from the chassis)
    B  jointEnabled=False  the pack drops onto the ground (~6 cm) within 1 s
    C  drive 2 s           the rover leaves; the released pack stays where it fell
    D  re-latch            pack written to the bay pose (chassis velocity), jointEnabled=True
    E  drive 2 s           the pack travels with the chassis, still in its bay

On the GPU box:
    bash scripts/gpu/isaac.sh envs/swap/latch_spike.py --usd <rover_swap.usda>                 # CPU
    bash scripts/gpu/isaac.sh envs/swap/latch_spike.py --usd <rover_swap.usda> --device cuda:0 # GPU?
Prints LATCH SPIKE PASS / FAIL and writes /data/runs/swap/latch_spike_<device>.json.
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Runtime pack release / re-latch through USD jointEnabled.")
parser.add_argument("--usd", required=True, help="rover_swap USD from robot/usd/convert_rover.py")
parser.add_argument("--out", default="/data/runs/swap")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

CHASSIS_Z = 0.175                 # chassis origin above flat ground, wheels touching (rover.xml)
BAY = (0.0, 0.0, -0.0875)         # pack_latch localPos0: pack origin in the chassis frame
ROVER = "/World/Rover"


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args_cli.device, physics=PhysxCfg()))
    ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="multiply",
        restitution_combine_mode="multiply"))
    ground.func("/World/ground", ground)
    rover = Articulation(ArticulationCfg(
        prim_path=ROVER,
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.usd, variants={"Physics": "physx"}),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, CHASSIS_Z + 0.005), joint_pos={".*": 0.0}),
        actuators={"wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"], stiffness=0.0, damping=18.0, joint_effort_limit=2.4,
            armature=0.01, friction=0.60, dynamic_friction=0.60, viscous_friction=0.02)}))
    pack = RigidObject(RigidObjectCfg(prim_path=f"{ROVER}/Geometry/pack", spawn=None))
    stage = sim_utils.get_current_stage()
    latch = stage.GetPrimAtPath(f"{ROVER}/Physics/pack_latch")
    if not latch.IsValid():
        raise SystemExit(f"no pack_latch under {ROVER}/Physics - is this the rover_swap USD?")
    sim.reset()
    dev, dt = sim.device, sim.get_physics_dt()
    ids, _ = rover.find_joints(["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"], preserve_order=True)
    bay = torch.tensor([BAY], device=dev)

    def set_latch(on: bool):
        from pxr import Sdf
        attr = latch.GetAttribute("physics:jointEnabled")
        if not attr.IsValid():
            attr = latch.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(bool(on))

    def run(seconds: float, wheel: float):
        cmd = torch.full((1, 4), wheel, device=dev)
        for _ in range(int(round(seconds / dt))):
            rover.actuators.target_command.set_velocity_index(value=cmd, joint_ids=ids)
            rover.write_data_to_sim()
            pack.write_data_to_sim()
            sim.step()
            rover.update(dt)
            pack.update(dt)

    def offset():
        """Pack origin in the chassis frame minus the bay offset (0 when latched in place) [m]."""
        c, q = rover.data.root_pos_w.torch, rover.data.root_quat_w.torch
        want = c + quat_apply(q, bay)
        return (pack.data.root_pos_w.torch - want)[0]

    res = {"device": dev, "usd": args_cli.usd}
    run(1.0, 0.0)
    res["A_latched_offset"] = offset().tolist()
    res["A_pack_z"] = float(pack.data.root_pos_w.torch[0, 2])

    set_latch(False)
    run(1.0, 0.0)
    res["B_pack_z"] = float(pack.data.root_pos_w.torch[0, 2])
    res["B_drop"] = res["A_pack_z"] - res["B_pack_z"]

    pack_xy0, rover_xy0 = pack.data.root_pos_w.torch[0, :2].clone(), rover.data.root_pos_w.torch[0, :2].clone()
    run(2.0, math.pi)
    res["C_rover_moved"] = float((rover.data.root_pos_w.torch[0, :2] - rover_xy0).norm())
    res["C_pack_moved"] = float((pack.data.root_pos_w.torch[0, :2] - pack_xy0).norm())

    c, q = rover.data.root_pos_w.torch, rover.data.root_quat_w.torch
    pose = torch.cat([c + quat_apply(q, bay), q], dim=1)
    pack.write_root_pose_to_sim_index(root_pose=pose)
    pack.write_root_velocity_to_sim_index(root_velocity=rover.data.root_com_vel_w.torch.clone())
    set_latch(True)
    run(0.5, 0.0)
    res["D_relatched_offset"] = offset().tolist()

    rover_xy1 = rover.data.root_pos_w.torch[0, :2].clone()
    run(2.0, math.pi)
    res["E_rover_moved"] = float((rover.data.root_pos_w.torch[0, :2] - rover_xy1).norm())
    res["E_offset"] = offset().tolist()

    checks = {
        "A_pack_in_bay": float(torch.tensor(res["A_latched_offset"]).norm()) < 0.005,
        "B_pack_dropped": res["B_drop"] > 0.03,
        "C_rover_left_pack": res["C_rover_moved"] > 0.3 and res["C_pack_moved"] < 0.05,
        "D_relatched_in_bay": float(torch.tensor(res["D_relatched_offset"]).norm()) < 0.005,
        "E_pack_travels": res["E_rover_moved"] > 0.3 and float(torch.tensor(res["E_offset"]).norm()) < 0.01,
    }
    res["checks"] = checks
    ok = all(checks.values())
    os.makedirs(args_cli.out, exist_ok=True)
    tag = "cpu" if str(dev).startswith("cpu") else "gpu"
    with open(os.path.join(args_cli.out, f"latch_spike_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("LATCH SPIKE PASS" if ok else "LATCH SPIKE FAIL: " + ", ".join(k for k, v in checks.items() if not v))
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        app.close()
    raise SystemExit(code)
