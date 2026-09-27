"""envs/swap/isaac_swap.py -- the battery swap, physically, in Omniverse (Isaac Sim 6.1 PhysX via Isaac Lab 3.0 EA).

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    bash scripts/gpu/isaac.sh envs/swap/isaac_swap.py                      # tilt20 USDs, GPU PhysX, -> /data/runs/swap
    bash scripts/gpu/isaac.sh envs/swap/isaac_swap.py --realtime           # watch it in the live view
    bash scripts/gpu/isaac.sh envs/swap/isaac_swap.py --rover-usd U --dock-usd D --out /data/runs/swap_x

The Omniverse twin of renders/swap_demo.py (MuJoCo), one continuous take:

    approach  ->  berth on the pad  ->  lift takes the pack  ->  swap  ->  drive away

* The rover is the official-importer USD of robot/import/rover_swap.xml: the pack is its own rigid
  body, held by the imported `pack_latch` FixedJoint (excludeFromArticulation).
* The dock is the importer USD of robot/import/dock.xml (envs/swap/dock_mjcf.py --write-import): a
  fixed-base articulation whose prismatic `dock_lift` is driven by an Isaac Lab position actuator.
* The spare pack is a plain 4 kg rigid box on the `empty_ready` station, with its own latch.
* The phases are B's phase machine: `swap_battery()` from envs/swap/scripts/swap_battery.py, ticked
  unchanged every step through `IsaacDockHardware` (envs/swap/isaac_hardware.py). The controller
  starts with no pose and gets `aligned=True` only after the rover has stopped inside B's published
  berth tolerances (XY_TOL_M 0.035 m, YAW_TOL_RAD, HEIGHT_GAP_TOL_M) measured off its bay.
* The pack really comes off: releasing and latching toggle `physics:jointEnabled` on the USD
  FixedJoints while PhysX runs (proven on cuda:0 by the lead's envs/swap/latch_spike.py, 27 Sep; the
  CPU pipeline died at sim start there, so the default device is Isaac Lab's). The lift carriage and
  the deck shuttles are the scripted conveyor (pose writes), as in scripts/mujoco_hardware.py.

Driving onto the berth is renders/swap_demo.py's P controller, unchanged (open loop in the sense that no
learned policy is involved: the trained traverse policy only reaches +-0.35 m, 4x looser than the berth).

Outputs in --out: isaac_swap.json (berth report, phase log, hardware events, assertion results) and
isaac_swap.mp4 (1280x720 RTX clip from Isaac Lab's headless Kit visualizer). A Viser live view is bound
to the box's 127.0.0.1 only (reached through the SSH tunnel); the script refuses any other binding.
The last line is `ISAAC SWAP: ALL ASSERTIONS PASSED` or the list of failures.
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

USD_ROOT = "/data/runs/usd/tilt20"
parser = argparse.ArgumentParser(description="Physical battery swap in Isaac Sim / PhysX.")
parser.add_argument("--rover-usd", default=f"{USD_ROOT}/rover_swap/rover_swap/rover_swap.usda")
parser.add_argument("--dock-usd", default=f"{USD_ROOT}/dock/dock/dock.usda")
parser.add_argument("--out", default="/data/runs/swap", help="results + clip directory")
parser.add_argument("--hz", type=int, default=240, help="physics steps per second")
parser.add_argument("--realtime", action="store_true", help="pace to wall-clock time (for watching live)")
parser.add_argument("--author-disabled", action="store_true",
                    help="author the spare latch disabled before play (Contract 2 as written) instead of the "
                         "default: authored enabled, opened after sim.reset() before the first step")
AppLauncher.add_app_launcher_args(parser)     # --device: Isaac Lab's default (cuda:0); cpu is optional
args_cli = parser.parse_args()
args_cli.visualizer = ["kit", "viser"]     # as envs/isaac_rover/measure_rover.py: RTX clip + Viser live view
args_cli.video = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402
from isaaclab_visualizers.viser import ViserVisualizerCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from envs.swap.dock_mjcf import DOCK  # noqa: E402
from envs.swap.isaac_hardware import (  # noqa: E402
    IsaacDockHardware, LatchBank, author_spare_pack, dock_stations, find_named_prim, quat_yaw)
from envs.swap.scripts.status import Status  # noqa: E402
from envs.swap.scripts.swap_battery import (  # noqa: E402
    HEIGHT_GAP_TOL_M, XY_TOL_M, YAW_TOL_RAD, Phase, SwapBatteryController, SwapRequest, swap_battery)

# ---- shot (renders/swap_demo.py) ----------------------------------------------------------------
FPS = 30
START_X = -1.90            # behind the dock, clear of the magazine end stop
CHASSIS_Z = 0.175          # chassis origin above the surface with wheels touching (rover.xml)
SETTLE0_S = 0.5            # let the spawned rover land before driving
APPROACH_MAX_S = 12.0
SETTLE_S = 1.2
DEPART_S = 5.0
SWAP_MAX_S = 20.0

# ---- drive (renders/swap_demo.py, unchanged) ------------------------------------------------------
WHEEL_R = 0.090
V_CRUISE = 0.279           # measured top speed, robot/SPEC.md
KP_X, KP_YAW, KP_Y = 1.9, 2.2, 2.6
RAMP_S = 1.0
WHEEL_HI = 3.1416          # rover.xml ctrlrange, rad/s
WHEELS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]


def ground_material():
    """measure_rover.py: mu 1.0 with "multiply", so the tyre-ground pair takes the tyre's 0.65."""
    return sim_utils.PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                               friction_combine_mode="multiply",
                                               restitution_combine_mode="multiply")


def v3(t) -> tuple:
    return tuple(float(x) for x in t)


class SwapScene:
    def __init__(self) -> None:
        self.t = 0.0
        viser = ViserVisualizerCfg(bind_address="127.0.0.1", port=8080, open_browser=False, share=False)
        kit = KitVisualizerCfg(headless=True, eye=(1.2, -3.2, 1.3), lookat=(-0.6, 0.0, 0.12),
                               window_width=1280, window_height=720)
        self.sim = SimulationContext(sim_utils.SimulationCfg(
            dt=1.0 / args_cli.hz, device=args_cli.device, physics=PhysxCfg(), visualizer_cfgs=[viser, kit]))
        ground = sim_utils.GroundPlaneCfg(physics_material=ground_material())
        ground.func("/World/ground", ground)
        light = sim_utils.DomeLightCfg(intensity=2500.0)
        light.func("/World/light", light)

        # Dock at the origin: fixed base, the lift is the only joint. disable_gravity stands in for
        # MJCF gravcomp=1 (a screw jack does not droop); stiffness is N/m on a prismatic joint.
        self.dock = Articulation(ArticulationCfg(
            prim_path="/World/Dock",
            # No fix_root_link: the dock USD (imported with fix_base=True) already carries its world
            # FixedJoint; a second one breaks the Viser scene build ("Cannot merge joint ... FixedJoint").
            spawn=sim_utils.UsdFileCfg(usd_path=args_cli.dock_usd, variants={"Physics": "physx"},
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"dock_lift": 0.0}),
            actuators={"lift": ImplicitActuatorCfg(joint_names_expr=["dock_lift"], stiffness=DOCK.lift_kp,
                                                   damping=DOCK.lift_kv, joint_effort_limit=500.0)},
        ))
        # Rover: rover_cfg.py's actuators (G2 parity), the swap USD. The spawn translation moves the
        # chassis AND its separate pack; both are placed exactly again after reset (_place_rover).
        self.rover = Articulation(ArticulationCfg(
            prim_path="/World/Rover",
            spawn=sim_utils.UsdFileCfg(usd_path=args_cli.rover_usd, variants={"Physics": "physx"}),
            init_state=ArticulationCfg.InitialStateCfg(pos=(START_X, 0.0, CHASSIS_Z + 0.005),
                                                       joint_pos={".*": 0.0}),
            actuators={"wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*"], stiffness=0.0, damping=18.0, joint_effort_limit=2.4,
                armature=0.01, friction=0.60, dynamic_friction=0.60, viscous_friction=0.02)},
        ))

        stage = sim_utils.get_current_stage()
        self._carriage_colliders_off(stage)
        chassis = find_named_prim(stage, "/World/Rover", "chassis", "body")
        pack = find_named_prim(stage, "/World/Rover", "pack", "body")
        latch = find_named_prim(stage, "/World/Rover", "pack_latch", "joint")
        self.paths = {"chassis": str(chassis.GetPath()), "pack": str(pack.GetPath()),
                      "pack_latch": str(latch.GetPath()), "spare": "/World/SparePack"}
        print(f"[swap] rover prims: {self.paths}")
        self.pack = RigidObject(RigidObjectCfg(prim_path=self.paths["pack"], spawn=None))
        st = dock_stations(layout=DOCK)
        rp, rq = st["empty_ready"]
        self.spare = author_spare_pack(self.paths["spare"], (rp[0], rp[1], rp[2] + 0.0005), rq)
        self.bank = LatchBank(stage, [self.paths["chassis"]], [self.paths["pack"], self.paths["spare"]],
                              self.paths["pack_latch"], initial={0: 0})
        self.bank.author(prime=not args_cli.author_disabled)

        self.sim.reset()
        self.bank.settle_primed()              # before the first sim.step(): no step ever sees them closed
        for v in getattr(self.sim, "visualizers", []):          # owner's rule: box-local viewers only
            addr = getattr(v.cfg, "bind_address", "127.0.0.1")
            if addr not in ("127.0.0.1", "localhost") or getattr(v.cfg, "share", False):
                raise SystemExit(f"refusing to run: visualizer {v.cfg.visualizer_type} would listen on {addr}")
        self.kit_viz = next((v for v in getattr(self.sim, "visualizers", [])
                             if v.cfg.visualizer_type == "kit"), None)

        self.dt = self.sim.get_physics_dt()
        self.dev = self.sim.device
        self.wheel_ids, _ = self.rover.find_joints(WHEELS, preserve_order=True)
        self.wheel_body_ids, self.wheel_body_names = self.rover.find_bodies("wheel_.*")
        self.cmd = [0.0, 0.0, 0.0, 0.0]
        self._place_rover(START_X)
        # sim.reset()'s warm-up step runs with the primed spare latch still closed, which yanks the spare
        # from empty_ready toward the rover's bay (box run 27 Sep: it landed in the rover's path). Put it
        # back on its station at rest, as _place_rover does for the rover and its pack.
        self.spare.write_root_pose_to_sim_index(
            root_pose=torch.tensor([[rp[0], rp[1], rp[2] + 0.0005, *rq]], device=self.dev))
        self.spare.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
        zj = torch.zeros(1, self.dock.num_joints, device=self.dev)           # carriage down, at rest
        self.dock.write_joint_position_to_sim_index(position=zj)
        self.dock.write_joint_velocity_to_sim_index(velocity=zj)

        self.hw = IsaacDockHardware(dock=self.dock, rovers=self.rover, packs=[self.pack, self.spare],
                                    bank=self.bank, stations=st, on_event=self._hw_event)
        self.ctrl = SwapBatteryController(self.hw)
        self.stage_name, self.stage_t, self.t = "land", 0.0, 0.0
        self.phase_log, self._last_phase = [], self.ctrl.phase
        self._aligned_flag, self.result = None, None
        self.swap_status = Status.RUNNING
        self.berth_report = None
        self.m = dict(max_lift_top=-1e9, max_lift_cmd=-1e9, min_wheel_bottom=1e9, min_chassis_z=1e9,
                      wheel_margin=1e9, wheel_xy={}, welded_pack_lo=[])
        self.frames, self.frame_sum = [], 0.0

    # ---- scene helpers ------------------------------------------------------------------------------
    @staticmethod
    def _carriage_colliders_off(stage) -> None:
        """The carriage is a kinematic jack and must not collide (dock_mjcf.py: a colliding carriage
        levers the 29 kg rover off its wheels). dock.xml authors it contype 0; if the importer made
        it a collider anyway, switch that off here and say so."""
        from pxr import Usd, UsdPhysics

        carriage = find_named_prim(stage, "/World/Dock", "dock_carriage", "body")
        for p in Usd.PrimRange(carriage):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
                print(f"[swap] WARNING importer made {p.GetPath()} a collider; disabled it")

    def _place_rover(self, x: float) -> None:
        """Chassis AND pack together (swap_demo.py _place_rover): the latch would snap them otherwise."""
        cpos, cquat = (x, 0.0, CHASSIS_Z + 0.005), (0.0, 0.0, 0.0, 1.0)
        pose = torch.tensor([[*cpos, *cquat]], device=self.dev)
        self.rover.write_root_pose_to_sim_index(root_pose=pose)
        self.rover.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
        zeros = torch.zeros(1, self.rover.num_joints, device=self.dev)
        self.rover.write_joint_position_to_sim_index(position=zeros)
        self.rover.write_joint_velocity_to_sim_index(velocity=zeros)
        self.rover.reset()
        bp, bq = self.bank.bay_pose(cpos, cquat)
        self.pack.write_root_pose_to_sim_index(root_pose=torch.tensor([[*bp, *bq]], device=self.dev))
        self.pack.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))

    def _hw_event(self, text: str) -> None:
        print(f"[{self.t:7.3f} s]   hw: {text}")

    def _pose2d(self):
        p = v3(self.rover.data.root_link_pos_w.torch[0])
        q = v3(self.rover.data.root_link_quat_w.torch[0])
        return p, q, quat_yaw(q)

    # ---- drive (swap_demo.py _drive) ------------------------------------------------------------------
    def _drive(self, x_goal) -> None:
        if x_goal is None:
            l_cmd = r_cmd = 0.0
        else:
            (x, y, _), _, yaw = self._pose2d()
            v = float(np.clip(KP_X * (x_goal - x), -V_CRUISE, V_CRUISE))
            if abs(x_goal - x) < 0.015:
                v = 0.0
            base = v / WHEEL_R
            diff = float(np.clip(-(KP_YAW * yaw + KP_Y * y), -0.9, 0.9))
            l_cmd, r_cmd = base - diff, base + diff
        slew = WHEEL_HI / RAMP_S * self.dt
        for i, cmd in enumerate((l_cmd, r_cmd, l_cmd, r_cmd)):          # fl, fr, rl, rr
            cur = self.cmd[i] + float(np.clip(cmd - self.cmd[i], -slew, slew))
            self.cmd[i] = float(np.clip(cur, -WHEEL_HI, WHEEL_HI))
        self.rover.actuators.target_command.set_velocity_index(
            value=torch.tensor([self.cmd], device=self.dev), joint_ids=self.wheel_ids)

    def _stopped(self) -> bool:
        vx = float(self.rover.data.root_link_lin_vel_w.torch[0, 0])
        return abs(vx) < 0.015 and abs(self._pose2d()[0][0]) < 0.05

    def _check_alignment(self) -> bool:
        (x, y, _), _, yaw = self._pose2d()
        z_mount = self.hw.mount_underside_z(0)
        z_ref = DOCK.pad_top_z + DOCK.pack_underside_z
        xy_err = math.hypot(x - DOCK.berth_x, y - DOCK.berth_y)
        yaw_err = abs((yaw + math.pi) % (2 * math.pi) - math.pi)
        z_err = abs(z_mount - z_ref)
        ok = xy_err <= XY_TOL_M and yaw_err <= YAW_TOL_RAD and z_err <= HEIGHT_GAP_TOL_M
        self.berth_report = dict(x=x, y=y, yaw=yaw, z_mount=z_mount, z_ref=z_ref,
                                 xy_err=xy_err, yaw_err=yaw_err, z_err=z_err, ok=ok)
        print(f"[{self.t:7.3f} s] BERTH  xy_err={xy_err*1000:6.1f} mm (tol {XY_TOL_M*1000:.0f})  "
              f"yaw_err={math.degrees(yaw_err):5.2f} deg (tol {math.degrees(YAW_TOL_RAD):.1f})  "
              f"z_mount={z_mount:.4f} m vs {z_ref:.4f} (tol {HEIGHT_GAP_TOL_M*1000:.0f} mm)  -> "
              f"{'ALIGNED' if ok else 'MISALIGNED'}")
        return ok

    # ---- stages (swap_demo.py _advance_stage) ---------------------------------------------------------
    def _advance_stage(self):
        self.stage_t += self.dt
        if self.stage_name == "land":
            if self.stage_t >= SETTLE0_S:
                self.stage_name, self.stage_t = "approach", 0.0
                # B's controller starts with no pose: it genuinely waits in APPROACH.
                self.result = swap_battery(self.ctrl, "rover0", "dock0", dt=0.0,
                                           request=SwapRequest(vehicle_id="rover0", dock_id="dock0",
                                                               empty_pack_count=self.hw.empty_pack_count()))
                print(f"[{self.t:7.3f} s] PHASE idle -> {self.ctrl.phase.value} (controller waiting for berth)")
                self._last_phase = self.ctrl.phase
            return None
        if self.stage_name == "approach":
            if self._stopped() or self.stage_t >= APPROACH_MAX_S:
                self.stage_name, self.stage_t = "settle", 0.0
                print(f"[{self.t:7.3f} s] STAGE approach -> settle (x={self._pose2d()[0][0]:+.4f} m)")
            return DOCK.berth_x
        if self.stage_name == "settle":
            if self.stage_t >= SETTLE_S:
                self._aligned_flag = self._check_alignment()
                self.stage_name, self.stage_t = "swap", 0.0
            return None
        if self.stage_name == "swap":
            res = self.result
            if (res is not None and res.status is not Status.RUNNING) or self.stage_t >= SWAP_MAX_S:
                print(f"[{self.t:7.3f} s] SWAP  {res.status.value}" + (f" ({res.reason})" if res.reason else ""))
                self.swap_status = res.status
                self.stage_name, self.stage_t = "depart", 0.0
            return None
        if self.stage_name == "depart":
            if self.stage_t >= DEPART_S:
                self.stage_name = "done"
            return 3.0
        return None

    def _log_phase(self) -> None:
        p = self.ctrl.phase
        if p is not self._last_phase:
            print(f"[{self.t:7.3f} s] PHASE {self._last_phase.value} -> {p.value}")
            self.phase_log.append((round(self.t, 4), self._last_phase.value, p.value))
            self._last_phase = p

    def _measure(self) -> None:
        m = self.m
        m["max_lift_top"] = max(m["max_lift_top"], self.hw.carriage_top)
        m["max_lift_cmd"] = max(m["max_lift_cmd"], self.hw.lift_command)
        if self.bank.enabled(0, 0):
            m["welded_pack_lo"].append(self.hw.pack_pos(0)[2] - DOCK.pack_half[2])
        if self.stage_name in ("settle", "swap"):
            m["min_chassis_z"] = min(m["min_chassis_z"], self._pose2d()[0][2])
            pos = self.rover.data.body_link_pos_w.torch[0]
            for bid, name in zip(self.wheel_body_ids, self.wheel_body_names):
                wx, wy, wz = v3(pos[bid])
                m["min_wheel_bottom"] = min(m["min_wheel_bottom"], wz - WHEEL_R)
                m["wheel_margin"] = min(m["wheel_margin"], DOCK.pad_half_x - abs(wx), DOCK.pad_half_y - abs(wy))
                m["wheel_xy"][name] = (wx, wy)

    # Camera: swap_demo.py's framing, as (eye offset from look, look). The lift phases are shot from a
    # LOW FRONT QUARTER: side-on, both tyres stand between the camera and the bay; down the machine's
    # axis the sightline threads between the front wheels and under the chassis front edge.
    CAM_BY_PHASE = {
        Phase.LIFT_FULL: ((1.89, -0.65, 0.12), (0.02, 0.0, 0.08)),
        Phase.STOW_FULL: ((1.87, -2.08, 0.75), (-0.40, 0.0, 0.09)),
        Phase.OFFER_EMPTY: ((2.23, -1.68, 0.75), (0.40, 0.0, 0.09)),
        Phase.LATCH_EMPTY: ((1.89, -0.65, 0.12), (0.02, 0.0, 0.08)),
        Phase.RELEASE: ((2.20, -0.90, 0.34), (0.0, 0.0, 0.11)),
    }
    CAM_TAU = 0.55

    def _camera_target(self):
        x = self._pose2d()[0][0]
        if self.stage_name in ("land", "approach"):
            return (1.47, -2.18, 0.70), (0.55 * x, 0.0, 0.16)
        if self.stage_name == "depart":
            return (1.75, -1.72, 0.76), (0.62 * x, 0.0, 0.16)
        return self.CAM_BY_PHASE.get(self.ctrl.phase, ((1.83, -1.24, 0.56), (0.0, 0.0, 0.13)))

    def _frame(self, k: int, every: int) -> None:
        if self.kit_viz is None:
            return
        off, look = self._camera_target()
        tgt = np.array([*off, *look])
        if not hasattr(self, "_cam"):
            self._cam = tgt
        self._cam = self._cam + (1.0 - math.exp(-self.dt / self.CAM_TAU)) * (tgt - self._cam)
        if k % every:
            return
        look = tuple(float(v) for v in self._cam[3:])
        eye = tuple(float(self._cam[i] + self._cam[3 + i]) for i in range(3))
        self.kit_viz.set_camera_view(eye, look)
        f = self.kit_viz.render_rgb_array()
        if f is not None and f.size and f.any():
            f = np.asarray(f)[..., :3]
            self.frames.append(f)
            self.frame_sum += float(f.mean())

    # ---- main loop (swap_demo.py order: stage, drive, hw.sync, controller, step, measure) -------------
    def run(self) -> None:
        every = max(1, int(round(1.0 / FPS / self.dt)))
        k, t0 = 0, time.perf_counter()
        while True:
            x_goal = self._advance_stage()
            if self.stage_name == "done":
                break
            self._drive(x_goal)
            self.hw.sync(self.dt)
            if self.stage_name in ("approach", "settle", "swap"):
                self.result = swap_battery(self.ctrl, "rover0", "dock0", dt=self.dt, aligned=self._aligned_flag,
                                           empty_pack_count=self.hw.empty_pack_count())
                self._aligned_flag = None
            self._log_phase()
            self.rover.write_data_to_sim()
            self.dock.write_data_to_sim()
            self.sim.step()
            for a in (self.rover, self.dock, self.pack, self.spare):
                a.update(self.dt)
            self.t += self.dt
            self._measure()
            self._frame(k, every)
            k += 1
            if args_cli.realtime:
                lag = k * self.dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)

    # ---- assertions 2-8 of renders/swap_demo.py, jointEnabled in place of eq_active -------------------
    def assertions(self) -> dict:
        m, L, out = self.m, DOCK, {}

        def check(name, ok, detail):
            out[name] = {"pass": bool(ok), "detail": detail}
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

        print("\n" + "=" * 72 + "\nASSERTIONS\n" + "=" * 72)
        seq = [b for _, _, b in self.phase_log]
        expect = ["lift_full", "stow_full", "offer_empty", "latch_empty", "release", "done"]
        check("2_phase_machine", self.swap_status is Status.SUCCESS and self.ctrl.phase is Phase.DONE
              and seq == expect, f"status={self.swap_status.value} phase={self.ctrl.phase.value} seq={seq}")

        b = self.berth_report or {}
        check("3_berth_tolerance", bool(b.get("ok")),
              f"xy {b.get('xy_err', float('nan'))*1000:.1f}/{XY_TOL_M*1000:.0f} mm, yaw "
              f"{math.degrees(b.get('yaw_err', float('nan'))):.2f}/{math.degrees(YAW_TOL_RAD):.1f} deg, mount z "
              f"{b.get('z_err', float('nan'))*1000:.1f}/{HEIGHT_GAP_TOL_M*1000:.0f} mm")

        ys = sorted(abs(y) for _, y in m["wheel_xy"].values())
        over_slot = [n for n, (wx, _) in m["wheel_xy"].items() if abs(wx) <= L.slot_half_x]
        check("4_wheels_on_pad", m["wheel_margin"] > 0.05 and len(ys) == 4
              and all(abs(v - 0.40) < 0.02 for v in ys) and not over_slot,
              f"|y|={[round(v, 3) for v in ys]} margin={m['wheel_margin']*1000:.0f} mm over_slot={over_slot}")

        check("5_lift_stroke", L.lift_stroke - 1e-3 <= m["max_lift_top"] <= L.lift_stroke + 5e-4
              and m["max_lift_cmd"] <= L.lift_stroke + 1e-9,
              f"top max {m['max_lift_top']:.5f} m (LOCKED {L.lift_stroke}), max command {m['max_lift_cmd']:.5f} m")

        nominal = CHASSIS_Z + L.pad_top_z
        check("6_no_sinking", m["min_wheel_bottom"] >= L.pad_top_z - 0.005
              and abs(m["min_chassis_z"] - nominal) < 0.010,
              f"wheel bottom min {m['min_wheel_bottom']*1000:+.2f} mm vs pad {L.pad_top_z*1000:.1f} mm; "
              f"chassis min {m['min_chassis_z']:.4f} (nominal {nominal:.4f})")

        stow = self.hw.pack_pos(0)
        cpos, cquat = self.hw.chassis_pose(0)
        bay, _ = self.bank.bay_pose(cpos, cquat)
        carried = self.hw.pack_pos(1)
        bay_err = math.dist(carried, bay)
        ok7 = (not self.bank.enabled(0, 0) and self.bank.enabled(0, 1) and self.hw.latched_pack(0) == 1
               and abs(stow[0] + L.station_x) < 0.03 and abs(stow[1]) < 0.03
               and abs(stow[2] - L.deck_pack_z) < 0.005 and bay_err < 0.006
               and self.hw.lift_height <= 0.004 and cpos[0] > 0.8)
        check("7_latches_swapped", ok7,
              f"pack_latch jointEnabled={self.bank.enabled(0, 0)}, spare latch jointEnabled={self.bank.enabled(0, 1)}; "
              f"old pack at {tuple(round(v, 3) for v in stow)} (full_stow x={-L.station_x}, z={L.deck_pack_z}); "
              f"fresh pack {bay_err*1000:.2f} mm off the bay; lift {self.hw.lift_height:.4f} m; rover x={cpos[0]:.2f}")

        lo = np.array(m["welded_pack_lo"]) if m["welded_pack_lo"] else np.array([float("nan")])
        check("8_welded_pack_height", abs(float(lo.mean()) - 0.060) < 0.004,
              f"underside {lo.min():.4f}..{lo.max():.4f} m (mean {lo.mean():.4f}) vs LOCKED 0.060")
        return out

    def save(self, checks: dict) -> bool:
        os.makedirs(args_cli.out, exist_ok=True)
        stem = os.path.join(args_cli.out, "isaac_swap")
        clip = None
        if self.frames:
            import imageio
            imageio.mimwrite(stem + ".mp4", self.frames, fps=FPS, codec="libx264", quality=7)
            clip = stem + ".mp4"
            print(f"[swap] clip: {clip} ({len(self.frames)} frames, mean pixel "
                  f"{self.frame_sum / len(self.frames):.1f})")
        else:
            print("[swap] no clip frames captured")
        ok = all(c["pass"] for c in checks.values())
        res = {"rover_usd": args_cli.rover_usd, "dock_usd": args_cli.dock_usd, "device": args_cli.device,
               "latch_authoring": "disabled" if args_cli.author_disabled else "primed",
               "hz": args_cli.hz, "prims": self.paths, "latches": {f"{r}-{p}": j for (r, p), j in
                                                                 self.bank.joint_paths.items()},
               "berth": self.berth_report, "phase_log": self.phase_log, "hw_events": self.hw.events,
               "metrics": {k: v for k, v in self.m.items() if k != "welded_pack_lo"},
               "checks": checks, "all_passed": ok, "clip": clip}
        with open(stem + ".json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"[swap] results: {stem}.json")
        return ok


def main() -> None:
    scene = SwapScene()
    scene.run()
    checks = scene.assertions()
    ok = scene.save(checks)
    print("=" * 72)
    if ok:
        print("ISAAC SWAP: ALL ASSERTIONS PASSED")
    else:
        print("ISAAC SWAP: FAILED: " + ", ".join(k for k, c in checks.items() if not c["pass"]))


if __name__ == "__main__":
    main()
    simulation_app.close()
