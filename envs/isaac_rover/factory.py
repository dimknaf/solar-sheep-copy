"""envs/isaac_rover/factory.py -- the whole Solar Sheep factory in one continuous Omniverse run (Contract 3).

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    bash scripts/gpu/isaac.sh envs/isaac_rover/factory.py --n 5 [--llm] [--seconds 180]
        [--policy P] [--usd ROVER_SWAP_USD] [--dock-usd DOCK_USD] [--out /data/runs/factory]
        [--inject-fault blocked_cell|no_empty_pack|none] [--realtime] [--dock-mode physical|logical]
    add --realtime to watch it at real speed in the live view (http://localhost:18080 via watch.sh)

What runs, every policy step (48 Hz; PhysX 240 Hz underneath):
    econ clock (time-lapse K x physics time) -> sun direction -> the DistantLight is re-aimed
    fleet brain (orchestrator.FleetBrain: Nemotron / the model in SOLAR_TF_MODEL with --llm, or the
        labelled rule fallback) gets a FleetSnapshot every BRAIN_TICK_S and hands back one Command
        per rover; it never blocks this loop
    per-rover executor turns the Command into motion:
        harvest -> the TRAINED traverse policy (TorchScript policy.pt) drives to the cell through
                   carrot waypoints <= 2 m (heading along travel, the real heading on the last leg),
                   routed around the dock; parked (action 0) within 0.35 m / 0.5 rad; then a scripted
                   in-place spin faces the tilted panel to the sun, re-aimed when the sun moves > 8 deg
        dock    -> policy to the rover's slot in the dock line (queue_xy, >= 1.2 m apart)
        head of the line, dock free, an empty pack on the rack -> admitted: policy to the approach
                   point, spin onto the dock axis, then the berth P controller of renders/swap_demo.py
                   (pure pursuit on the dock centre line) for the last 1.5 m -- the policy is only
                   good to 0.35 m, the swap needs 0.035 m -- settle, and the UNCHANGED swap_battery()
                   phase machine drives IsaacDockHardware (Contract 2): the pack is released and a
                   different rigid body is latched by toggling physical FixedJoints. Then the rover
                   drives off the pad forward to the exit and goes back to the field.
        rest / hold -> park row / stop
    plant  -> while the dock is idle, the conveyor (IsaacDockHardware.convey, scripted pose writes) moves
              the stowed full pack off the rover lane into a plant slot, drains it (PLANT_DRAIN_S) and
              stages ONE empty pack (magazine spare or drained pack) on empty_ready, so a rover is only
              admitted with full_stow clear and a pack on the deck, and every offer shuttle runs along
              the deck centre line between the tyres. Rovers always leave the pad forward (exit_xy).
    SoC lives on the PACK (as in the browser sim): it rises with the real exposure max(0, n_w . s)
    of the 20-deg panel (normal (sin20, 0, cos20) in the body, rotated by the root quaternion), only
    while harvesting, and falls with wheel power (full drive ~ 30 W). Staged SoCs make a queue form
    early; one injected fault shows the brain's recovery on camera.

Stage markers to watch: "[factory] scene built", "[factory] latches authored", "[factory] env ready",
"[factory] rovers placed", "[factory] step N ...", "[factory] SWAP ...", "[factory] FAULT ...",
"[factory] clip: ...", then "FACTORY: ALL ASSERTIONS PASSED" (or the failures).

Outputs (--out, default /data/runs/factory): factory.mp4 (1920x1080 Kit RTX clip with overlays drawn
by factory_overlay.py; no overlays if Pillow is missing), factory.json (counters + assertions),
orchestrator.jsonl (the brain's log, key redacted by the orchestrator).

Why the dock, spare packs and the fault rock are NOT scene entities: InteractiveScene resets every
asset with per-env indices (interactive_scene.py:479-494), and these are single global objects
(count 1). They are built in the "prestartup" event hook -- the documented point between scene
creation and sim.reset() (manager_based_env.py:184-196), which is also where the LatchBank authors
its FixedJoints -- and stepped next to the scene by wrapping scene.write_data_to_sim / scene.update,
which env.step() calls once per PHYSICS step (manager_based_rl_env.py:229-244); that is also where
hw.sync(dt) runs, once per physics step, before the data is written to the sim (Contract 2).

Safety (owner rule): same refusals as train.py / play.py; Viser bound to 127.0.0.1 with share=False;
the script refuses to run if any visualizer would listen beyond localhost. No uploads; the Token
Factory key is read by the orchestrator from the environment only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)            # orchestrator/, envs.swap
sys.path.insert(0, HERE)            # train.py (guards), solar_sheep_tasks, factory_overlay
from train import refuse_unsafe, take_usd  # noqa: E402  (this folder's train.py: first on sys.path)

from isaaclab.app import AppLauncher  # noqa: E402

# ---- defaults: today's converted assets and policy on the box (contracts_today.md) -------------------
DEFAULT_ROVER_USD = "/data/runs/usd/tilt20/rover_swap/rover_swap/rover_swap.usda"
DEFAULT_DOCK_USD = "/data/runs/usd/tilt20/dock/dock/dock.usda"
DEFAULT_POLICY = "/data/runs/rsl_rl/solar_sheep_traverse/2026-09-27_11-19-43/exported/policy.pt"  # rough, 1498 it
DEFAULT_SOC = "0.90,0.75,0.55,0.35,0.18"     # 0.18, not 0.15: rules.py docks a rover at soc <= 0.15 (brownout)

# ---- scene -----------------------------------------------------------------------------------------
ENV_SPACING = 4.0                  # clone grid only (4 m: no rover inside the dock during warm-up); moved right after reset
PASTURE_HALF = 8.0                 # the pasture is +-8 m; leaving it fails the run
SUN_INTENSITY = 3000.0             # DistantLight (as Isaac Lab's ant env)
SUN_COLOR = (1.0, 0.95, 0.85)
SUN_ANGLE_DEG = 0.53
SKY_INTENSITY = 600.0              # dome light dimmed from 2500 so the sun casts visible shadows
SUN_UPDATE_EVERY = 12              # policy steps between light re-aims (0.25 s)
DOCK_PRIM = "/World/Dock"
DOCK_ID = "dock0"
LIFT_STIFFNESS, LIFT_DAMPING = 20000.0, 400.0     # dock_mjcf lift_kp / lift_kv
SPARE_ROOT = "/World/SparePacks"
PACK_SIZE = (0.20, 0.45, 0.055)    # LOCKED (envs/swap/SPEC.md)
PACK_MASS = 4.0
PACK_COLOR = (0.10, 0.45, 0.85)
BAY = (0.0, 0.0, -0.0875)          # pack origin in the chassis frame (pack_latch localPos0)
RACK_Y, RACK_X0, RACK_PITCH = 0.90, -0.60, 0.30   # ground rack beside the dock (dock -Y side), slots along X:
PLANT_SLOTS = 3                    # the extra spares first, then the plant's drained-pack slots
PLANT_DRAIN_S = 15.0               # physics s a full pack sits in the plant before it returns empty
STATION_CLEAR_M = 0.15             # a station counts as occupied if a pack is this close
ROCK_PRIM = "/World/FaultRock"
ROCK_SIZE, ROCK_MASS = (0.70, 0.70, 0.35), 300.0
ROCK_PARK = (0.0, 30.0)            # off camera until the fault is injected

# ---- executor ---------------------------------------------------------------------------------------
BRAIN_TICK_S = 0.5                 # physics seconds between FleetBrain.tick calls
CARROT_M = 2.0                     # training targets were 0.8-2.5 m away; obs clips the offset at 4 m
PARK_R, PARK_YAW = 0.35, 0.5       # the policy's accuracy (reach radius / heading of the task)
UNPARK_R = 0.70                    # a parked rover re-drives only if pushed this far off
NEAR_R, ARRIVE_TIMEOUT_S = 0.60, 6.0    # close enough for this long -> accept the arrival
VIA_R = 0.60                       # route waypoints (dock corners, recovery via) count as passed here
STALL_S, STALL_PROGRESS_M = 6.0, 0.10   # no progress for this long -> traverse FAILURE "stuck"
FAIL_PAUSE_S = 1.5                 # a failed rover stands still this long (the brain sees FAILURE)
FAULT_HOLD_S = 4.0                 # state "fault" after a tip-over
SPIN_GAIN = 5.5                    # wheel rad/s per rad/s of yaw rate (pi rad/s -> 0.571 rad/s measured)
WHEEL_R = 0.090
KP_SPIN, YAW_RATE_MAX = 1.5, 0.5   # in-place spin: rad/s per rad, cap
AIM_TOL = math.radians(3.0)        # panel aimed at the sun within this
REAIM = math.radians(8.0)          # re-aim when the sun has moved this far
SLEW_PER_S = 2.0                   # scripted wheel commands ramp (SPEC: no gearmotor snaps)
AVOID_R, AVOID_COS = 1.10, math.cos(math.radians(55.0))   # another rover this close, ahead -> yield
YIELD_S, SIDESTEP_S, SIDESTEP_M = 3.0, 2.5, 1.0            # then sidestep to the right
# dock obstacle for routing, dock frame, already inflated by the rover's half width
DOCK_OBS_X = 1.45                  # deck ends at 1.18; the approach (-1.5) and exit (+2.0) stay outside
DOCK_OBS_Y_POS = 1.55              # cabinet to 1.04 + 0.5
DOCK_OBS_Y_NEG = 1.65              # the rack at -0.9 - 0.225, + 0.5 (the park row at -2.4 stays outside)
CORNER_MARGIN = 0.30
# berth (renders/swap_demo.py gains, pure pursuit on the dock centre line)
APPROACH_R, APPROACH_YAW = 0.35, 0.5
ALIGN_TOL = 0.05                   # rad, spin onto the dock axis before rolling in
LOOKAHEAD = 0.45                   # m, pure-pursuit lookahead on the centre line
K_HEAD = 2.0                       # yaw rate per rad of pursuit heading error
KP_X, V_BERTH, V_CREEP = 1.9, 0.15, 0.03
BERTH_STOP_X = 0.008               # stop this short of the berth (coast)
BERTH_MAX_S, SETTLE_S = 20.0, 1.2
MAX_BERTH_TRIES = 2
REVERSE_TO_X = -1.40
DEPART_X = 2.0                     # = layout.exit_fwd_m
SWAP_MAX_S = 30.0
Z_MOUNT_OFFSET = BAY[2] - PACK_SIZE[2] / 2    # pack underside below the chassis origin
MOTOR_IDLE_W, MOTOR_FULL_W = 3.0, 30.0        # electronics + drive at full wheel power (robot/SPEC.md 30.2 W)
TILT = math.radians(20.0)
PANEL_N_B = (math.sin(TILT), 0.0, math.cos(TILT))
# ---- video ------------------------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080
CAM_EYE = (7.6, -14.0, 9.8)        # wide 3/4 view from the south-east: pasture, queue, dock, park row
CAM_LOOKAT = (-1.6, 1.2, 0.0)
CAM_HFOV_DEG = 60.0                # Kit's default perspective camera (18.15 mm / 20.955 mm aperture)
TAG_Z = 0.85                       # name tags drawn this high above the rover origin
FAULT_BANNER_S = 12.0
LOG_EVERY_S = 5.0

parser = argparse.ArgumentParser(description="Solar Sheep: the whole factory in one Omniverse run.")
parser.add_argument("--n", type=int, default=5, help="rovers (one Isaac env each, all in one world)")
parser.add_argument("--policy", default=DEFAULT_POLICY, help="TorchScript traverse policy (exported/policy.pt)")
parser.add_argument("--usd", default=DEFAULT_ROVER_USD, help="rover_swap USD (pack separate + pack_latch)")
parser.add_argument("--dock-usd", default=DEFAULT_DOCK_USD)
parser.add_argument("--dock-mode", choices=["physical", "logical"], default="physical",
                    help="logical = no IsaacDockHardware: the phase machine runs on NullHardware and the SoC "
                         "is reset on SUCCESS (labelled on screen; the physical-swap assertion then fails)")
parser.add_argument("--llm", action="store_true", help="let the LLM dispatch (key from the environment)")
parser.add_argument("--model", default=None, help="Token Factory model (default: SOLAR_TF_MODEL)")
parser.add_argument("--max-llm-calls", type=int, default=300)
parser.add_argument("--seconds", type=float, default=180.0, help="physics seconds to run")
parser.add_argument("--out", default="/data/runs/factory")
parser.add_argument("--realtime", action="store_true", help="pace to wall-clock time (for watching live)")
parser.add_argument("--inject-fault", choices=["blocked_cell", "no_empty_pack", "none"], default="blocked_cell")
parser.add_argument("--fault-at", type=float, default=20.0, help="physics s after which the fault is injected")
parser.add_argument("--spares", type=int, default=2, help="empty packs at the dock at the start")
parser.add_argument("--latch-disabled", action="store_true",
                    help="author the open latches disabled (Contract 2 text) instead of primed-closed and opened "
                         "before the first step (isaac_swap.py's default; the path the latch spike proved)")
parser.add_argument("--soc", default=DEFAULT_SOC, help="staged pack SoC per rover, comma separated")
parser.add_argument("--timelapse", type=float, default=120.0)
parser.add_argument("--start-hour", type=float, default=8.0)
parser.add_argument("--weather", default="clear", help="clear | cloudy | overcast | random")
parser.add_argument("--weather-dynamic", action="store_true", help="let the weather change (seeded)")
parser.add_argument("--start", choices=["field", "park"], default="field")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--no-overlay", action="store_true")
parser.add_argument("--no-tags", action="store_true", help="no name tags projected onto the rovers")
parser.add_argument("--no-goal-markers", action="store_true", help="hide the policy's target arrows")
parser.add_argument("--cam-eye", type=float, nargs=3, default=CAM_EYE)
parser.add_argument("--cam-lookat", type=float, nargs=3, default=CAM_LOOKAT)
AppLauncher.add_app_launcher_args(parser)      # --device: Isaac Lab's default (cuda:0); the latch works there
_argv = sys.argv[1:]
refuse_unsafe(_argv)
_usd_given = any(a == "--usd" or a.startswith("--usd=") for a in _argv)
_argv = take_usd(_argv)                        # --usd -> ROVER_USD, before rover_cfg.py is imported
args_cli = parser.parse_args(_argv)
if not _usd_given:
    os.environ["ROVER_USD"] = args_cli.usd
ROVER_USD = os.environ["ROVER_USD"]
for _label, _path in (("rover USD (--usd)", ROVER_USD), ("policy (--policy)", args_cli.policy)):
    if not os.path.isfile(_path):
        raise SystemExit(f"[factory] {_label} not found: {_path}")
args_cli.visualizer = ["kit", "viser"]         # counts as an explicit --viz request (app_launcher.py:973)
args_cli.video = True                          # rendering-capable Kit experience, as measure_rover.py
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import warp as wp  # noqa: E402

wp.config.enable_backward = False              # before any kernel module is built (as train.py)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402
from isaaclab_visualizers.viser import ViserVisualizerCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import (Articulation, ArticulationCfg, AssetBaseCfg, RigidObject,  # noqa: E402
                             RigidObjectCfg)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import solar_sheep_tasks  # noqa: E402,F401  registers the tasks (and imports their configs)
from solar_sheep_tasks.traverse.rover_cfg import CHASSIS_Z, WHEEL_JOINTS  # noqa: E402
from solar_sheep_tasks.traverse.traverse_env_cfg import TraverseFlatEnvCfg, TraverseSceneCfg  # noqa: E402

try:
    from orchestrator import (CAPACITY_WH, RESERVE, DockTelemetry, Economics, FleetBrain,  # noqa: E402
                              FleetSnapshot, Layout, RoverTelemetry)
except Exception as _e:  # noqa: BLE001
    raise SystemExit(f"[factory] cannot import orchestrator/ (Contract 1): {type(_e).__name__}: {_e}")
try:
    from envs.swap.scripts.status import Status  # noqa: E402
    from envs.swap.scripts.swap_battery import (HEIGHT_GAP_TOL_M, XY_TOL_M, YAW_TOL_RAD,  # noqa: E402
                                                SwapBatteryController, SwapRequest, swap_battery)
except Exception as _e:  # noqa: BLE001
    raise SystemExit(f"[factory] cannot import envs/swap/scripts/swap_battery.py: {type(_e).__name__}: {_e}")
try:
    from envs.swap.dock_mjcf import DOCK as _DOCK   # geometry only; mujoco is optional there
    STATION_X, DECK_PACK_Z, PAD_TOP_Z = float(_DOCK.station_x), float(_DOCK.deck_pack_z), float(_DOCK.pad_top_z)
except Exception:  # noqa: BLE001  (values of dock_mjcf.DockLayout, 27 Sep)
    STATION_X, DECK_PACK_Z, PAD_TOP_Z = 1.00, 0.012 + 0.0005 + PACK_SIZE[2] / 2, 0.001
PHYSICAL = args_cli.dock_mode == "physical"
if PHYSICAL:
    try:
        from envs.swap.isaac_hardware import IsaacDockHardware, LatchBank, dock_stations  # noqa: E402
    except Exception as _e:  # noqa: BLE001
        raise SystemExit(f"[factory] cannot import envs/swap/isaac_hardware.py (Contract 2): "
                         f"{type(_e).__name__}: {_e}\n  (run with --dock-mode logical to test the rest)")

N = int(args_cli.n)
K = max(1, int(args_cli.spares)) if PHYSICAL else 0
RIDS = [f"rover{i}" for i in range(N)]
_soc = [float(v) for v in args_cli.soc.split(",") if v.strip()]
STAGED_SOC = [_soc[i % len(_soc)] for i in range(N)]
LAYOUT = Layout.default(n_rovers=N)
DX, DY, DYAW = float(LAYOUT.dock_xy[0]), float(LAYOUT.dock_xy[1]), float(LAYOUT.dock_yaw)
if max(0, K - 1) + PLANT_SLOTS > 7:
    raise SystemExit(f"[factory] --spares {K}: the rack beside the dock holds {7 - PLANT_SLOTS + 1} spares at most")
FACTORY: dict = {}                              # what the prestartup hook builds, read back after env creation


# ---- small math -------------------------------------------------------------------------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_quat(yaw: float) -> tuple:
    """(x, y, z, w) of a rotation about +Z (Isaac Lab 3.0 order)."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def sun_quat(sun: tuple) -> tuple:
    """DistantLight orientation (x, y, z, w): its local -Z (the light direction) points along -sun.

    Rz(az) Ry(pi/2 - el) takes local +Z to s = (cos el cos az, cos el sin az, sin el); the same
    product as isaaclab.utils.math.quat_from_euler_xyz(0, pi/2 - el, az) (utils/math.py:275-301).
    """
    el = math.asin(max(-1.0, min(1.0, sun[2])))
    az = math.atan2(sun[1], sun[0])
    p = math.pi / 2 - el
    cz, sz, cp, sp = math.cos(az / 2), math.sin(az / 2), math.cos(p / 2), math.sin(p / 2)
    return (-sz * sp, cz * sp, sz * cp, cz * cp)


def qrot(q, v) -> tuple:
    """Rotate v by the (x, y, z, w) quaternion q."""
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def to_dock(x: float, y: float, yaw: float = 0.0) -> tuple:
    c, s = math.cos(DYAW), math.sin(DYAW)
    dx, dy = x - DX, y - DY
    return (c * dx + s * dy, -s * dx + c * dy, wrap(yaw - DYAW))


def from_dock(xd: float, yd: float) -> tuple:
    c, s = math.cos(DYAW), math.sin(DYAW)
    return (DX + c * xd - s * yd, DY + s * xd + c * yd)


def wheels(v: float, yaw_rate: float) -> list:
    """(fl, fr, rl, rr) actions in [-1, 1] for a body speed [m/s] and yaw rate [rad/s]."""
    base, diff = v / WHEEL_R, SPIN_GAIN * yaw_rate
    l, r = (base - diff) / math.pi, (base + diff) / math.pi
    l, r = max(-1.0, min(1.0, l)), max(-1.0, min(1.0, r))
    return [l, r, l, r]


# ---- routing around the dock (visibility graph over the inflated dock rectangle's corners) ---------------
def _seg_hits_rect(a, b, x0, x1, y0, y1) -> bool:
    """Liang-Barsky: does segment a-b pass through the open rectangle?"""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return False
    return t1 - t0 > 1e-6


def route(start: tuple, goal: tuple) -> list:
    """World waypoints from start to goal (goal last), around the dock footprint."""
    rect = (-DOCK_OBS_X, DOCK_OBS_X, -DOCK_OBS_Y_NEG, DOCK_OBS_Y_POS)
    s, g = to_dock(*start)[:2], to_dock(*goal)[:2]
    inside = lambda p: rect[0] < p[0] < rect[1] and rect[2] < p[1] < rect[3]  # noqa: E731
    if inside(s) or inside(g) or not _seg_hits_rect(s, g, *rect):
        return [goal]
    m = CORNER_MARGIN
    nodes = [s, g] + [(x, y) for x in (rect[0] - m, rect[1] + m) for y in (rect[2] - m, rect[3] + m)]
    dist, prev, heap = {0: 0.0}, {}, [(0.0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == 1 or d > dist.get(u, 1e18):
            continue
        for v in range(len(nodes)):
            if v == u or _seg_hits_rect(nodes[u], nodes[v], *rect):
                continue
            nd = d + math.dist(nodes[u], nodes[v])
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if 1 not in prev:
        return [goal]
    path, v = [], 1
    while v != 0:
        path.append(v)
        v = prev[v]
    return [from_dock(*nodes[k]) if k != 1 else goal for k in reversed(path)]


# ---- scene ------------------------------------------------------------------------------------------
@configclass
class FactorySceneCfg(TraverseSceneCfg):
    """The traverse scene (one rover per env, shared ground) + the sun + a view over the rovers' packs."""

    sun = AssetBaseCfg(prim_path="/World/sun",
                       spawn=sim_utils.DistantLightCfg(intensity=SUN_INTENSITY, color=SUN_COLOR, angle=SUN_ANGLE_DEG))
    # the rover_swap USD's pack: its own rigid body, held by the imported FixedJoint pack_latch
    rover_packs = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/pack", spawn=None)


@configclass
class FactoryEnvCfg(TraverseFlatEnvCfg):
    scene: FactorySceneCfg = FactorySceneCfg(num_envs=5, env_spacing=ENV_SPACING)


def _pack_cfg(color=PACK_COLOR) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=PACK_SIZE, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=PACK_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))


def station_pose(xd: float, yd: float, z: float) -> tuple:
    x, y = from_dock(xd, yd)
    return (x, y, z) + yaw_quat(DYAW)


def row_pose(slot: int) -> tuple:
    """Ground slot of the rack beside the dock (dock -Y side): the spare magazine, then the plant slots."""
    return station_pose(RACK_X0 + RACK_PITCH * slot, -RACK_Y, PACK_SIZE[2] / 2 + 0.002)


def factory_prestartup(env, env_ids) -> None:
    """Event hook, mode "prestartup": after the scene exists, before sim.reset() plays it.

    Builds the global dock / spare packs / fault rock (single objects, outside the per-env scene) and
    authors the LatchBank FixedJoints while the stage is still editable before physics parses it.
    """
    stage = env.sim.stage
    env_paths = list(env.scene.env_prim_paths)
    chassis = [f"{p}/Robot/Geometry/chassis" for p in env_paths]
    packs = [f"{p}/Robot/Geometry/pack" for p in env_paths]
    latch = f"{env_paths[0]}/Robot/Physics/pack_latch"
    for path in (chassis[0], packs[0], latch):
        if not stage.GetPrimAtPath(path).IsValid():
            raise SystemExit(f"[factory] {path} not in the stage -- is --usd the rover_swap USD? ({ROVER_USD})")
    print(f"[factory] scene built: {len(env_paths)} rovers, pack + pack_latch found under {env_paths[0]}/Robot")

    # dock: a fixed-base articulation with one prismatic lift, placed at the layout's dock pose
    dock_usd = args_cli.dock_usd
    if os.path.isfile(dock_usd):
        FACTORY["dock"] = Articulation(ArticulationCfg(
            prim_path=DOCK_PRIM,
            # no fix_root_link: the imported dock already has its world FixedJoint (a second breaks Viser)
            spawn=sim_utils.UsdFileCfg(usd_path=dock_usd, variants={"Physics": "physx"},
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(DX, DY, 0.0), rot=yaw_quat(DYAW), joint_pos={".*": 0.0}),
            actuators={"lift": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=LIFT_STIFFNESS,
                                                   damping=LIFT_DAMPING)}))
        from pxr import Usd, UsdPhysics             # the carriage is a jack: it must never collide
        for prim in Usd.PrimRange(stage.GetPrimAtPath(DOCK_PRIM)):
            if prim.GetName() == "dock_carriage":
                for q in Usd.PrimRange(prim):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr().Set(False)
                        print(f"[factory] WARNING importer made {q.GetPath()} a collider; disabled it")
    elif PHYSICAL:
        raise SystemExit(f"[factory] dock USD not found: {dock_usd} (--dock-usd)")
    else:
        pad = sim_utils.CuboidCfg(size=(1.10, 1.00, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.65, 0.1)))
        pad.func(DOCK_PRIM, pad, translation=(DX, DY, -0.009), orientation=yaw_quat(DYAW))
        print(f"[factory] WARNING: no dock USD at {dock_usd}; a flat pad stands in (logical mode)")

    # spare (empty) packs: #0 on the empty_ready station of the deck, the rest in a magazine beside the dock
    spares, spare_paths = [], []
    if K:
        sim_utils.create_prim(SPARE_ROOT, "Xform")
    spare_poses = []
    for k in range(K):
        pose = station_pose(STATION_X, 0.0, DECK_PACK_Z) if k == 0 else row_pose(k - 1)
        path = f"{SPARE_ROOT}/Pack_{k}"
        spares.append(RigidObject(RigidObjectCfg(
            prim_path=path, spawn=_pack_cfg(),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pose[:3], rot=pose[3:]))))
        spare_paths.append(path)
        spare_poses.append(pose)
    FACTORY["spares"], FACTORY["spare_poses"] = spares, spare_poses

    if args_cli.inject_fault == "blocked_cell":
        FACTORY["rock"] = RigidObject(RigidObjectCfg(
            prim_path=ROCK_PRIM,
            spawn=sim_utils.CuboidCfg(
                size=ROCK_SIZE, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=ROCK_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.40, 0.37))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(ROCK_PARK[0], ROCK_PARK[1], ROCK_SIZE[2] / 2 + 0.002))))

    if PHYSICAL:
        # Contract 2: every chassis x every pack gets a disabled FixedJoint (+ filtered pairs); the
        # rovers' own imported pack_latch joints start enabled. Pack index = rover packs 0..N-1, spares N..
        bank = LatchBank(stage, chassis, packs + spare_paths, latch, {i: i for i in range(N)})
        # primed: every latch parsed closed, opened by settle_primed() before our first step. NOTE the
        # PhysX warm-up inside sim.reset() runs one physics step (physx_manager.py:957-960) with them all
        # closed; Factory.place() therefore rewrites every rover, pack and spare pose afterwards.
        bank.author(prime=not args_cli.latch_disabled)
        FACTORY["bank"] = bank
        print(f"[factory] latches authored: {N} chassis x {N + K} packs (template {latch}, "
              f"{'disabled' if args_cli.latch_disabled else 'primed'})")
    FACTORY["pack_paths"] = packs + spare_paths


def build_cfg() -> FactoryEnvCfg:
    cfg = FactoryEnvCfg()
    cfg.play_mode()                                        # no pushes, no corruption
    if args_cli.device:
        cfg.sim.device = args_cli.device
    cfg.scene.num_envs = N
    cfg.scene.env_spacing = ENV_SPACING
    cfg.scene.filter_collisions = False                    # rovers really collide with each other
    cfg.scene.replicate_physics = False                    # per-env parsing: required for prestartup events
    cfg.scene.sky_light.spawn.intensity = SKY_INTENSITY    # (event_manager.py:378) and the cross-env latches
    cfg.episode_length_s = 1.0e5                           # no 25 s timeout teleports
    cfg.commands.pose_command.reach_radius = -1.0          # never "reached", so never resampled
    cfg.commands.pose_command.debug_vis = not args_cli.no_goal_markers
    cfg.events.reset_base.params["pose_range"] = {"z": (0.005, 0.005)}   # a tip-over reset lands at the park spot
    cfg.events.factory_setup = EventTerm(func=factory_prestartup, mode="prestartup")
    cfg.sim.render_interval = cfg.decimation * capture_every()             # render only on captured steps
    cfg.sim.visualizer_cfgs = [
        ViserVisualizerCfg(bind_address="127.0.0.1", port=8080, open_browser=False, share=False, max_visible_envs=64),
        KitVisualizerCfg(headless=True, origin_type="world", eye=tuple(args_cli.cam_eye),
                         lookat=tuple(args_cli.cam_lookat), window_width=WIDTH, window_height=HEIGHT),
    ]
    return cfg


def capture_every() -> int:
    """Policy steps between captured frames (48 Hz policy -> about --fps)."""
    return max(1, int(round(48.0 / max(1, args_cli.fps))))


# ---- the executor -----------------------------------------------------------------------------------
class Rover:
    """One rover's side of the fleet: its task, its drive mode and the telemetry the brain reads."""

    def __init__(self, i: int):
        self.i, self.id = i, RIDS[i]
        self.state, self.status, self.reason = "resting", "", ""
        self.task, self.cmd_key, self.cmd = None, None, None
        self.goal, self.goal_yaw, self.via, self.face = None, None, None, "keep"
        self.cell = None
        self.parked, self.near_t = False, 0.0
        self.aim_yaw, self.aimed = None, False
        self.stage, self.stage_t, self.berth_tries = None, 0.0, 0
        self.pause_until, self.fault_until = -1.0, -1.0
        self.node, self.best, self.best_t = None, 1e18, 0.0
        self.yield_t, self.sidestep_until, self.sidestep_xy = 0.0, -1.0, None
        self.act = [0.0] * 4
        self.pv_w, self.exposure = 0.0, 0.0
        self.harvested_wh, self.cycle = 0.0, {"harvested": False, "swapped": False}
        self.pack_before, self.swap_res, self.swap_phase = None, None, ""
        self.outside = False


class Factory:
    def __init__(self, env: ManagerBasedRLEnv, policy):
        self.env, self.policy = env, policy
        self.t, self.events = 0.0, []
        self.dev = env.device
        self.robot = env.scene["robot"]
        self.packs_view = env.scene["rover_packs"]
        self.cmd_term = env.command_manager.get_term("pose_command")
        self.wheel_ids, _ = self.robot.find_joints(WHEEL_JOINTS, preserve_order=True)
        self.step_dt, self.physics_dt = float(env.step_dt), float(env.physics_dt)
        self.stage = env.sim.stage
        self.dock = FACTORY.get("dock")
        self.spares = FACTORY.get("spares", [])
        self.rock = FACTORY.get("rock")
        self.bank = FACTORY.get("bank")
        self.pack_paths = FACTORY.get("pack_paths", [])
        # pack handles: (view, index in view) for every pack the bank knows, same order as its pack_paths
        self.pack_handles = [(self.packs_view, i) for i in range(N)] + [(s, 0) for s in self.spares]
        self.pack_soc = list(STAGED_SOC) + [RESERVE] * K
        self.econ = Economics(timelapse=args_cli.timelapse, start_hour=args_cli.start_hour, weather=args_cli.weather)
        self.brain = FleetBrain(RIDS, LAYOUT, use_llm=args_cli.llm, model=args_cli.model,
                                log_path=os.path.join(args_cli.out, "orchestrator.jsonl"),
                                max_calls=args_cli.max_llm_calls)
        self.hw, self.stations = None, {}
        self.plant: dict = {}                              # pack idx -> [stage "stowed"|"draining", t, slot]
        if PHYSICAL:
            # full_stow / empty_ready (isaac_hardware.dock_stations) + the plant rack for drained packs
            self.stations = dict(dock_stations(dock_pos=(DX, DY, 0.0), dock_yaw=DYAW))
            for s in range(PLANT_SLOTS):
                p = row_pose(max(0, K - 1) + s)
                self.stations[f"plant_{s}"] = (tuple(p[:3]), tuple(p[3:]))
            self.hw = IsaacDockHardware(dock=self.dock, rovers=self.robot, packs=self.pack_handles,
                                        bank=self.bank, stations=self.stations,
                                        on_event=lambda text: self.log(f"hw: {text}"))
        self.ctrl = SwapBatteryController(self.hw)            # None -> NullHardware (logical mode)
        self.rovers = [Rover(i) for i in range(N)]
        self.queue: list[int] = []
        self.current = None
        self.k = 0
        self.sun = (0.0, 0.0, 1.0)
        self.sun_yaw = 0.0
        self.no_pack_until = -1.0
        self.fault = {"injected": False, "kind": args_cli.inject_fault, "t": None, "detail": ""}
        self.banner, self.banner_until = None, -1.0
        self.last_cmd = {"json": "", "reason": ""}
        self.plan_id = None
        self.counters = {"swaps_ok": 0, "swaps_physical": 0, "swap_failures": 0, "logical_swaps": 0,
                         "full_cycles": 0, "delivered_wh": 0.0, "llm_calls": 0, "plans": {"llm": 0, "rules": 0},
                         "tasks": {"llm": 0, "rules": 0}, "faults": 0, "recoveries": 0, "tipovers": 0,
                         "nonfinite_resets": 0, "left_pasture": 0, "max_abs_xy": 0.0, "stuck": 0}
        self.swaps: list = []
        self.pending_recovery: set = set()
        self._install_scene_hook()

    # -- the global dock objects are stepped next to the scene, once per physics step ---------------------
    def _install_scene_hook(self) -> None:
        scene, extras = self.env.scene, [a for a in [self.dock, *self.spares, self.rock] if a is not None]
        write0, update0, hw, pdt = scene.write_data_to_sim, scene.update, self.hw, self.physics_dt

        def write_data_to_sim():
            if hw is not None:
                hw.sync(pdt)                   # Contract 2: once per physics step, before the write
            for a in extras:
                a.write_data_to_sim()
            write0()

        def update(dt):
            update0(dt)
            for a in extras:
                a.update(dt)

        scene.write_data_to_sim, scene.update = write_data_to_sim, update

    # -- helpers ---------------------------------------------------------------------------------------
    def log(self, text: str, **kv) -> None:
        print(f"[factory] t={self.t:6.1f}s {text}")
        if len(self.events) < 2000:
            self.events.append({"t": round(self.t, 2), "event": text, **kv})

    def pack_of(self, i: int):
        return self.bank.pack_of(i) if self.bank is not None else i

    def soc(self, i: int) -> float:
        p = self.pack_of(i)
        return self.pack_soc[p] if p is not None else 0.0

    def empty_count(self) -> int:
        if self.t < self.no_pack_until:
            return 0
        if self.hw is not None:
            return int(self.hw.empty_pack_count())
        return 99 if not PHYSICAL else 0                  # logical mode: the rack never runs dry

    def pack_pos(self, p: int):
        view, j = self.pack_handles[p]
        return view.data.root_pos_w.torch[j].tolist()

    def seat_pack(self, i: int, c=None, q=None) -> None:
        """Write rover i's latched pack into its bay (after a teleport / reset: the joint must not snap).

        ``c`` / ``q``: the chassis position / (x, y, z, w) just written; default = read back from the data
        (write_root_pose_to_sim_index updates the data buffers, articulation.py:546-563).
        """
        p = self.pack_of(i)
        if p is None or p >= len(self.pack_handles):
            return
        c = list(c) if c is not None else self.robot.data.root_pos_w.torch[i].tolist()
        q = list(q) if q is not None else self.robot.data.root_quat_w.torch[i].tolist()
        b = qrot(q, BAY)
        pose = torch.tensor([[c[0] + b[0], c[1] + b[1], c[2] + b[2], *q]], device=self.dev)
        view, j = self.pack_handles[p]
        view.write_root_pose_to_sim_index(root_pose=pose, env_ids=[j])
        view.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev), env_ids=[j])

    # -- start -----------------------------------------------------------------------------------------
    def place(self) -> None:
        env = self.env
        origins = torch.tensor([[*LAYOUT.park_xy(i), 0.0] for i in range(N)], device=self.dev)
        env.scene.terrain.env_origins[:] = origins          # tip-over resets land on the park row
        env.reset()
        cells = LAYOUT.spread_cells()
        t_day = self.econ.day_seconds(0.0)
        sun_yaw = self.econ.sun_yaw(t_day)
        pose = torch.zeros(N, 7, device=self.dev)
        for i, r in enumerate(self.rovers):
            if args_cli.start == "field":
                r.cell = cells[i % len(cells)]
                x, y = LAYOUT.slot_xy(*r.cell)
                yaw, r.state, r.parked = sun_yaw, "harvesting", True
                r.task, r.goal, r.goal_yaw, r.face = "harvest", (x, y), None, "sun"
            else:
                (x, y), yaw = LAYOUT.park_xy(i), float(getattr(LAYOUT, "park_yaw", 0.0))
                r.state, r.parked, r.task, r.goal = "resting", True, "rest", (x, y)
            pose[i, :3] = torch.tensor([x, y, CHASSIS_Z + 0.005], device=self.dev)
            pose[i, 3:] = torch.tensor(yaw_quat(yaw), device=self.dev)
        self.robot.write_root_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=self.dev))
        start = pose.tolist()
        for i in range(N):                                 # chassis and pack move together
            self.seat_pack(i, start[i][:3], start[i][3:])
        zero = torch.zeros(1, 6, device=self.dev)          # undo whatever the warm-up step did to the loose bodies
        for obj, p in zip(self.spares, FACTORY.get("spare_poses", [])):
            obj.write_root_pose_to_sim_index(root_pose=torch.tensor([list(p)], device=self.dev))
            obj.write_root_velocity_to_sim_index(root_velocity=zero)
        if self.rock is not None:
            self.rock.write_root_pose_to_sim_index(root_pose=torch.tensor(
                [[ROCK_PARK[0], ROCK_PARK[1], ROCK_SIZE[2] / 2 + 0.002, 0.0, 0.0, 0.0, 1.0]], device=self.dev))
            self.rock.write_root_velocity_to_sim_index(root_velocity=zero)
        if self.dock is not None:                          # carriage down, at rest
            zj = torch.zeros(1, self.dock.num_joints, device=self.dev)
            self.dock.write_joint_position_to_sim_index(position=zj)
            self.dock.write_joint_velocity_to_sim_index(velocity=zj)
        env.sim.forward()
        self.log(f"rovers placed ({args_cli.start}); staged soc {STAGED_SOC}; "
                 f"dock at ({DX:.2f}, {DY:.2f}) yaw {math.degrees(DYAW):.0f} deg; {K} empty packs")

    # -- sun -------------------------------------------------------------------------------------------
    def update_sun(self, force: bool = False) -> None:
        t_day = self.econ.day_seconds(self.t)
        self.sun = tuple(self.econ.sun_dir(t_day))
        self.sun_yaw = self.econ.sun_yaw(t_day)
        if not (force or self.k % SUN_UPDATE_EVERY == 0):
            return
        prim = self.stage.GetPrimAtPath("/World/sun")
        if not prim.IsValid():
            return
        try:
            from pxr import Gf
            x, y, z, w = sun_quat(self.sun)
            attr = prim.GetAttribute("xformOp:orient")
            if attr.IsValid():
                attr.Set(Gf.Quatf(w, x, y, z) if "quatf" in str(attr.GetTypeName()) else Gf.Quatd(w, x, y, z))
            inten = prim.GetAttribute("inputs:intensity")
            if inten.IsValid():
                inten.Set(SUN_INTENSITY if self.sun[2] > 0.0 else 0.0)
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_sun_warned", False):
                self._sun_warned = True
                print(f"[factory] WARNING: cannot re-aim the sun light ({e}); the clip keeps a fixed sun")

    # -- brain -----------------------------------------------------------------------------------------
    def snapshot(self, pos, yaw) -> FleetSnapshot:
        t_day = self.econ.day_seconds(self.t)
        rovers = [RoverTelemetry(r.id, pos[i][0], pos[i][1], yaw[i], self.soc(i), r.state, r.pv_w,
                                 r.status, r.reason) for i, r in enumerate(self.rovers)]
        dock = DockTelemetry(busy=self.current is not None,
                             current=RIDS[self.current] if self.current is not None else None,
                             queue=[RIDS[i] for i in self.queue], empty_packs=self.empty_count())
        return FleetSnapshot(t_day, self.sun_yaw, self.econ.sun_el(t_day), self.econ.weather, rovers, dock)

    def follow(self, r: Rover, cmd, pos) -> None:
        """Start a new task when the rover's Command changed (Command.key())."""
        if cmd is None or r.stage is not None or self.t < r.pause_until:
            return
        if self.t < r.fault_until:
            return
        if r.state == "fault":                             # fault hold over: back to a task-shaped state
            r.state = {"harvest": "to_field", "dock": "to_dock"}.get(r.task, "resting")
        key = cmd.key() if hasattr(cmd, "key") else (cmd.action, cmd.row, cmd.slot, cmd.face, cmd.dock_order)
        if key == r.cmd_key:
            return
        r.cmd_key, r.cmd = key, cmd
        src = "llm" if str(getattr(cmd, "source", "rules")).startswith("llm") else "rules"
        self.counters["tasks"][src] += 1
        self.last_cmd = {"json": json.dumps({"rover": r.id, "action": cmd.action, "row": cmd.row, "slot": cmd.slot,
                                             "face": cmd.face, "dock_order": cmd.dock_order, "src": src}),
                         "reason": str(cmd.reason)}
        r.via = tuple(cmd.via_xy) if getattr(cmd, "via_xy", None) else None
        r.face = cmd.face
        prior_reason = r.reason
        r.status, r.reason = "RUNNING", ""
        r.best, r.best_t, r.node = 1e18, self.t, None
        if cmd.action == "harvest":
            if not LAYOUT.in_grid(cmd.row, cmd.slot):
                r.status, r.reason = "FAILURE", "bad_cell"
                return
            self.leave_queue(r.i)
            r.task, r.cell = "harvest", (cmd.row, cmd.slot)
            new_goal = LAYOUT.slot_xy(cmd.row, cmd.slot)
            if r.goal != new_goal or r.via is not None:
                r.parked, r.aimed = False, False
            r.goal, r.goal_yaw = new_goal, None
            r.state = "harvesting" if r.parked else "to_field"
            if r.parked:
                r.status = "SUCCESS"                       # already on that cell
        elif cmd.action in ("dock", "swap"):
            if self.current == r.i:
                return
            r.task = "dock"
            if r.i not in self.queue:
                ax, ay = LAYOUT.approach_xy
                front = prior_reason == "misaligned" or math.hypot(pos[r.i][0] - ax, pos[r.i][1] - ay) < 1.0
                if front:
                    self.queue.insert(0, r.i)
                else:
                    self.queue.append(r.i)
                r.parked = False
                r.state = "to_dock"
        elif cmd.action == "rest":
            self.leave_queue(r.i)
            new_goal = LAYOUT.park_xy(r.i)
            if r.goal != new_goal:
                r.parked = False
            r.task, r.goal, r.goal_yaw = "rest", new_goal, float(getattr(LAYOUT, "park_yaw", 0.0))
            r.state = "resting"
        else:                                              # hold
            r.task, r.goal, r.parked = "hold", None, True

    def leave_queue(self, i: int) -> None:
        if i in self.queue:
            self.queue.remove(i)

    # -- dock ------------------------------------------------------------------------------------------
    def bay_err(self, i: int) -> float | None:
        """Distance of rover i's latched pack from its bay pose [m] (None without a pack)."""
        p = self.pack_of(i)
        if p is None:
            return None
        c = self.robot.data.root_pos_w.torch[i].tolist()
        q = self.robot.data.root_quat_w.torch[i].tolist()
        b = qrot(q, BAY)
        return math.dist(self.pack_pos(p), [c[k] + b[k] for k in range(3)])

    def all_pack_xy(self) -> list:
        """xy of every pack, in pack-index order (one read per view)."""
        xy = [v[:2] for v in self.packs_view.data.root_pos_w.torch[:, :3].tolist()]
        return xy + [s.data.root_pos_w.torch[0, :2].tolist() for s in self.spares]

    def station_clear(self, name: str) -> bool:
        if name not in self.stations:
            return True
        sx, sy = self.stations[name][0][:2]
        return all(math.hypot(x - sx, y - sy) > STATION_CLEAR_M for p, (x, y) in enumerate(self.all_pack_xy())
                   if self.bank is None or self.bank.rover_of(p) is None)

    def plant_step(self) -> None:
        """The plant's conveyor, only while the dock is idle and nobody is near the berth:
        clear full_stow into a plant slot (drain), and keep ONE empty pack staged on empty_ready --
        a magazine spare or a drained pack -- so the swap's offer shuttle always runs along the deck
        centre line, between the tyres, never through them."""
        if self.hw is None:
            return
        bx, by = LAYOUT.berth_xy
        if self.current is not None or any(math.hypot(q[0] - bx, q[1] - by) < 1.3 for q in self.last_pos):
            return
        if any(self.hw.pack_mode(p)[0] == "pinned" for p in range(len(self.pack_handles))):
            return                                         # one shuttle at a time
        for p, (stage, _, _) in list(self.plant.items()):
            if stage == "stowed" and self.hw.pack_mode(p)[1] == "full_stow":
                used = {v[2] for v in self.plant.values() if v[2] is not None}
                free = next((f"plant_{s}" for s in range(PLANT_SLOTS) if f"plant_{s}" not in used), None)
                if free is not None:
                    self.hw.convey(p, free)
                    self.plant[p] = ["draining", self.t, free]
                    self.log(f"plant: pack {p} full_stow -> {free} (draining to the grid)")
                    return
        if not self.station_clear("empty_ready"):
            return
        ex, ey = self.stations["empty_ready"][0][:2]
        xy = self.all_pack_xy()
        for p in range(len(self.pack_handles)):           # a magazine spare waiting in the rack
            mode, carrier = self.hw.pack_mode(p)
            if (self.bank.rover_of(p) is None and mode == "free" and carrier == "empty_ready" and p not in self.plant
                    and math.hypot(xy[p][0] - ex, xy[p][1] - ey) > STATION_CLEAR_M):
                self.hw.convey(p, "empty_ready")
                self.log(f"plant: spare pack {p} magazine -> empty_ready")
                return
        for p, (stage, t0, _) in list(self.plant.items()):   # or a drained one
            if stage == "draining" and self.t - t0 >= PLANT_DRAIN_S:
                self.hw.convey(p, "empty_ready")
                del self.plant[p]
                self.log(f"plant: pack {p} drained -> empty_ready (soc {self.pack_soc[p]:.2f})")
                return

    def admit(self) -> None:
        if self.current is not None or not self.queue:
            return
        head = self.rovers[self.queue[0]]
        if head.state != "queuing" or head.cmd is None or head.cmd.action not in ("dock", "swap"):
            return
        if self.empty_count() <= 0:
            return
        if self.hw is not None and (not self.station_clear("full_stow") or self.station_clear("empty_ready")
                                    or any(self.hw.pack_mode(p)[0] == "pinned" for p in range(len(self.pack_handles)))):
            return                     # a stowed pack in the lane / no pack staged on the deck / a shuttle moving
        self.queue.pop(0)
        self.current = head.i
        head.stage, head.stage_t, head.berth_tries = "approach", 0.0, 0
        head.state, head.status, head.reason = "swapping", "RUNNING", ""
        head.parked, head.best, head.best_t, head.node = False, 1e18, self.t, None
        head.pack_before = self.pack_of(head.i)
        if self.hw is not None:
            self.hw.set_active(head.i)
        self.log(f"SWAP admit {head.id} (soc {self.soc(head.i):.2f}, pack {head.pack_before}); "
                 f"queue {[RIDS[i] for i in self.queue]}")

    def release(self, r: Rover) -> None:
        if self.current == r.i:
            self.current = None
        r.stage = None
        r.cmd_key = None                                   # take the brain's next command as new

    def berth_step(self, r: Rover, i: int, pos, yaw):
        """Scripted dock stages -> (mode, payload). Mode 'wheels' = direct wheel actions."""
        xd, yd, yawd = to_dock(pos[i][0], pos[i][1], yaw[i])
        r.stage_t += self.step_dt
        st = r.stage
        if st == "approach":                               # policy to the approach point, dock heading
            ax, ay = LAYOUT.approach_xy
            if math.hypot(pos[i][0] - ax, pos[i][1] - ay) < APPROACH_R and abs(wrap(DYAW - yaw[i])) < APPROACH_YAW:
                r.stage, r.stage_t = "align", 0.0
            elif r.stage_t > 40.0:
                return self.swap_failed(r, "timeout", reverse=False)
            return ("policy", (ax, ay), DYAW)
        if st == "align":                                  # spin onto the dock axis
            if abs(yawd) < ALIGN_TOL or r.stage_t > 8.0:
                r.stage, r.stage_t = "berth", 0.0
                return ("wheels", [0.0] * 4)
            return ("wheels", wheels(0.0, max(-YAW_RATE_MAX, min(YAW_RATE_MAX, -KP_SPIN * yawd))))
        if st == "berth":                                  # pure pursuit on the centre line, creep to x = 0
            if xd >= -BERTH_STOP_X or r.stage_t > BERTH_MAX_S:
                r.stage, r.stage_t = "settle", 0.0
                return ("wheels", [0.0] * 4)
            yaw_des = math.atan2(-yd, LOOKAHEAD)
            rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, K_HEAD * wrap(yaw_des - yawd)))
            v = max(V_CREEP, min(V_BERTH, KP_X * (-xd)))
            return ("wheels", wheels(v, rate))
        if st == "settle":
            if r.stage_t < SETTLE_S:
                return ("wheels", [0.0] * 4)
            z_mount = pos[i][2] + Z_MOUNT_OFFSET
            xy_err, z_err = math.hypot(xd, yd), abs(z_mount - (PAD_TOP_Z + 0.060))
            ok = xy_err <= XY_TOL_M and abs(yawd) <= YAW_TOL_RAD and z_err <= HEIGHT_GAP_TOL_M
            self.log(f"BERTH {r.id} xy_err {xy_err * 1000:.1f} mm (tol {XY_TOL_M * 1000:.0f}) "
                     f"yaw_err {math.degrees(abs(yawd)):.2f} deg z_err {z_err * 1000:.1f} mm -> "
                     f"{'ALIGNED' if ok else 'MISALIGNED'} (try {r.berth_tries + 1})",
                     xy_err=xy_err, yaw_err=abs(yawd), z_err=z_err)
            r.berth_tries += 1
            if not ok and r.berth_tries < MAX_BERTH_TRIES:
                r.stage, r.stage_t = "retry", 0.0
                return ("wheels", [0.0] * 4)
            inject = self.fault["kind"] == "no_empty_pack" and self.t < self.no_pack_until
            n_empty = 0 if inject else self.empty_count()
            req = SwapRequest(vehicle_id=r.id, dock_id=DOCK_ID, empty_pack_count=n_empty)
            res = swap_battery(self.ctrl, r.id, DOCK_ID, dt=self.step_dt, request=req, aligned=ok,
                               empty_pack_count=n_empty)
            r.stage, r.stage_t = "swap", 0.0
            return self.swap_result(r, res)
        if st == "swap":
            inject = self.fault["kind"] == "no_empty_pack" and self.t < self.no_pack_until
            res = swap_battery(self.ctrl, r.id, DOCK_ID, dt=self.step_dt,
                               empty_pack_count=0 if inject else self.empty_count())
            if res.status == Status.RUNNING and r.stage_t > SWAP_MAX_S:
                self.ctrl.reset()
                self.abort_swap()
                return self.swap_failed(r, "timeout", reverse=False)
            return self.swap_result(r, res)
        if st in ("retry", "reverse"):                     # straight back to the approach, yaw held
            if xd <= REVERSE_TO_X or r.stage_t > 15.0:
                if st == "retry":
                    r.stage, r.stage_t = "align", 0.0
                else:
                    self.release(r)
                return ("wheels", [0.0] * 4)
            return ("wheels", wheels(-V_BERTH, max(-YAW_RATE_MAX, min(YAW_RATE_MAX, -K_HEAD * yawd))))
        if st == "depart":                                 # forward off the pad to the exit
            if xd >= DEPART_X - 0.1 or r.stage_t > 25.0:
                rec = self.swaps[-1] if self.swaps and self.swaps[-1]["rover"] == r.id else None
                if rec is not None and rec["physical"] and "bay_err_after_depart_m" not in rec:
                    err = self.bay_err(r.i)                # the new pack really travels with the rover
                    rec["bay_err_after_depart_m"] = round(err, 4) if err is not None else None
                self.log(f"SWAP {r.id} clear of the dock (exit)"
                         + (f"; new pack {rec.get('bay_err_after_depart_m')} m from its bay" if rec else ""))
                self.release(r)
                return ("wheels", [0.0] * 4)
            yaw_des = math.atan2(-yd, LOOKAHEAD)
            rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, K_HEAD * wrap(yaw_des - yawd)))
            return ("wheels", wheels(V_BERTH * 1.5, rate))
        return ("wheels", [0.0] * 4)

    def swap_result(self, r: Rover, res):
        r.swap_phase = getattr(res.phase, "value", str(res.phase))
        if res.status == Status.RUNNING:
            return ("wheels", [0.0] * 4)
        if res.status == Status.SUCCESS:
            before, after = r.pack_before, self.pack_of(r.i)
            physical = self.bank is not None and after is not None and after != before
            rec = {"t": round(self.t, 2), "rover": r.id, "pack_before": before, "pack_after": after,
                   "soc_delivered": round(self.pack_soc[before], 3) if before is not None else None,
                   "physical": physical}
            if physical:
                rec["new_pack_bay_err_m"] = round(self.bay_err(r.i), 4)
                stow = self.stations["full_stow"][0]
                rec["old_pack_to_stow_m"] = round(math.dist(self.pack_pos(before), stow), 4)
                self.counters["swaps_physical"] += 1
                self.plant[before] = ["stowed", self.t, None]
            if before is not None:
                self.counters["delivered_wh"] += (self.pack_soc[before] - RESERVE) * CAPACITY_WH
                self.pack_soc[before] = RESERVE           # the plant drains it to the grid (bookkeeping)
            if not PHYSICAL:
                self.counters["logical_swaps"] += 1
            self.counters["swaps_ok"] += 1
            self.swaps.append(rec)
            r.cycle["swapped"] = r.cycle["harvested"] and physical      # the assertion wants a PHYSICAL swap
            r.status, r.reason = "SUCCESS", ""
            self.log(f"SWAP SUCCESS {r.id}: pack {before} -> {after} "
                     f"({'physical latch' if physical else 'LOGICAL'}); rec {rec}")
            r.stage, r.stage_t = "depart", 0.0
            return ("wheels", [0.0] * 4)
        # FAILURE: nothing moved if it failed while still approaching (misaligned) -> back out; else forward
        why = res.reason or "failure"
        return self.swap_failed(r, why, reverse=(why == "misaligned"))

    def abort_swap(self) -> None:
        """Abandoned swap (failure, timeout, reset on the berth): packs still pinned on the carriage go
        to full_stow and join the plant, so admission (full_stow clear, nothing pinned) can resume."""
        hw = getattr(self, "hw", None)
        if hw is None:
            return
        for p in hw.abort():
            self.plant.setdefault(p, ["stowed", self.t, None])

    def swap_failed(self, r: Rover, why: str, reverse: bool):
        self.abort_swap()
        self.counters["swap_failures"] += 1
        self.counters["faults"] += 1
        self.pending_recovery.add(r.i)
        r.status, r.reason = "FAILURE", why
        self.set_banner(f"FAULT: {r.id} swap FAILURE ({why}) -> brain recovers")
        self.log(f"SWAP FAILURE {r.id}: {why}")
        r.stage, r.stage_t = ("reverse" if reverse else "depart"), 0.0
        return ("wheels", [0.0] * 4)

    # -- fault injection -------------------------------------------------------------------------------
    def set_banner(self, text: str) -> None:
        self.banner, self.banner_until = text, self.t + FAULT_BANNER_S

    def maybe_inject(self, pos) -> None:
        f = self.fault
        if f["injected"] or f["kind"] == "none" or self.t < args_cli.fault_at:
            return
        if f["kind"] == "no_empty_pack":
            self.no_pack_until = self.t + 25.0
            f.update(injected=True, t=self.t, detail="rack reports 0 empty packs for 25 s")
            self.set_banner("FAULT INJECTED: empty-pack rack jammed (0 empty packs for 25 s)")
            self.log("FAULT injected: no_empty_pack window 25 s")
            return
        for r in self.rovers:                              # blocked_cell: drop the rock on a cell a rover drives to
            if r.state == "to_field" and r.goal is not None and r.via is None and r.stage is None:
                if math.dist(pos[r.i][:2], r.goal) < 2.0:
                    continue
                gx, gy = r.goal
                pose = torch.tensor([[gx, gy, ROCK_SIZE[2] / 2 + 0.002, 0.0, 0.0, 0.0, 1.0]], device=self.dev)
                self.rock.write_root_pose_to_sim_index(root_pose=pose)
                self.rock.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
                f.update(injected=True, t=self.t, detail=f"rock on cell {r.cell} in {r.id}'s path")
                self.set_banner(f"FAULT INJECTED: rock dropped on cell {r.cell} ({r.id} is heading there)")
                self.log(f"FAULT injected: rock on cell {r.cell} at ({gx:.1f}, {gy:.1f}) for {r.id}")
                return

    # -- driving ---------------------------------------------------------------------------------------
    def drive(self, r: Rover, i: int, pos, yaw):
        """(mode, payload): ('policy', (x, y), heading) or ('wheels', [4 actions])."""
        if r.stage is not None:
            return self.berth_step(r, i, pos, yaw)
        if self.t < r.pause_until or self.t < r.fault_until or r.goal is None:
            return ("wheels", [0.0] * 4)
        # the rover's leg goal: its queue slot while in the dock line
        goal, goal_yaw = r.goal, r.goal_yaw
        if r.task == "dock" and r.i in self.queue:
            k = self.queue.index(r.i)
            goal, goal_yaw = LAYOUT.queue_xy(k), DYAW
            if r.goal != goal:                             # the line moved up: drive to the new slot
                r.goal, r.parked, r.state = goal, False, "to_dock"
            elif r.state not in ("queuing", "to_dock"):
                r.state = "to_dock"
        p = (pos[i][0], pos[i][1])
        d_goal = math.dist(p, goal)
        if r.parked:
            if d_goal > UNPARK_R:
                r.parked = False
            else:
                return self.at_goal(r, i, yaw)
        if goal_yaw is None:                               # harvest cells: arrive facing the sun
            goal_yaw = self.sun_yaw if r.face == "sun" else yaw[i]
        # arrival
        if d_goal < NEAR_R:
            r.near_t += self.step_dt
        else:
            r.near_t = 0.0
        if (d_goal < PARK_R and abs(wrap(goal_yaw - yaw[i])) < PARK_YAW) or r.near_t > ARRIVE_TIMEOUT_S:
            r.parked, r.near_t, r.via = True, 0.0, None
            self.arrived(r)
            return self.at_goal(r, i, yaw)
        # recovery waypoint first, then around the dock
        if r.via is not None:
            if math.dist(p, r.via) < VIA_R:
                r.via = None
            else:
                goal, goal_yaw = r.via, None
        path = route(p, goal)
        while len(path) > 1 and math.dist(p, path[0]) < VIA_R:
            path.pop(0)
        node = path[0]
        final = len(path) == 1 and node == goal and (r.via is None or goal != r.via)
        if r.node is None or math.dist(node, r.node) > 0.3:
            r.node, r.best, r.best_t = node, math.dist(p, node), self.t
        # rover-rover avoidance: yield, then sidestep right
        blocked = self.blocked_by_rover(i, pos, node)
        if self.t < r.sidestep_until and r.sidestep_xy is not None:
            node, final = r.sidestep_xy, False
            r.best_t = self.t
        elif blocked:
            r.yield_t += self.step_dt
            r.best_t = self.t
            if r.yield_t > YIELD_S:
                ux, uy = node[0] - p[0], node[1] - p[1]
                n = math.hypot(ux, uy) or 1.0
                a = math.radians(-70.0)
                sx, sy = (ux * math.cos(a) - uy * math.sin(a)) / n, (ux * math.sin(a) + uy * math.cos(a)) / n
                r.sidestep_xy, r.sidestep_until, r.yield_t = (p[0] + SIDESTEP_M * sx, p[1] + SIDESTEP_M * sy), \
                    self.t + SIDESTEP_S, 0.0
            return ("wheels", [0.0] * 4)
        else:
            r.yield_t = 0.0
        # stall -> traverse FAILURE (the brain retries via an offset waypoint, then another cell)
        dn = math.dist(p, node)
        if dn < r.best - STALL_PROGRESS_M:
            r.best, r.best_t = dn, self.t
        elif self.t - r.best_t > STALL_S:
            return self.traverse_failed(r, "stuck")
        # carrot <= 2 m, heading along travel, the real heading only on the last leg
        dx, dy = node[0] - p[0], node[1] - p[1]
        if dn > CARROT_M:
            tgt, hdg = (p[0] + dx / dn * CARROT_M, p[1] + dy / dn * CARROT_M), math.atan2(dy, dx)
        elif final:
            tgt, hdg = node, goal_yaw
        else:
            nxt = path[1] if len(path) > 1 else goal
            tgt, hdg = node, math.atan2(nxt[1] - node[1], nxt[0] - node[0]) if math.dist(nxt, node) > 0.1 \
                else math.atan2(dy, dx)
        return ("policy", tgt, hdg)

    def blocked_by_rover(self, i: int, pos, node) -> bool:
        p = pos[i]
        ux, uy = node[0] - p[0], node[1] - p[1]
        n = math.hypot(ux, uy)
        if n < 1e-6:
            return False
        for j in range(N):
            if j == i:
                continue
            dx, dy = pos[j][0] - p[0], pos[j][1] - p[1]
            d = math.hypot(dx, dy)
            if d < AVOID_R and (dx * ux + dy * uy) / (d * n + 1e-9) > AVOID_COS:
                # the one closer to its own goal keeps going when both are moving (fewer deadlocks)
                o = self.rovers[j]
                if o.stage is None and not o.parked and o.goal is not None and j > i:
                    continue
                return True
        return False

    def arrived(self, r: Rover) -> None:
        if r.task == "harvest":
            r.state, r.status, r.reason = "harvesting", "SUCCESS", ""
            if r.cycle.get("swapped"):
                self.counters["full_cycles"] += 1
                self.log(f"CYCLE {r.id}: harvest -> dock -> physical swap -> back on cell {r.cell}")
                r.cycle = {"harvested": False, "swapped": False}
            if r.i in self.pending_recovery:
                self.pending_recovery.discard(r.i)
                self.counters["recoveries"] += 1
                self.log(f"RECOVERED {r.id}: harvesting on cell {r.cell}")
        elif r.task == "dock":
            r.state = "queuing"
        elif r.task == "rest":
            r.state, r.status = "resting", "SUCCESS"
            if r.i in self.pending_recovery:
                self.pending_recovery.discard(r.i)
                self.counters["recoveries"] += 1

    def at_goal(self, r: Rover, i: int, yaw):
        if r.task == "harvest" and r.face in ("sun", "dock"):
            want = self.sun_yaw if r.face == "sun" else math.atan2(DY - self.last_pos[i][1], DX - self.last_pos[i][0])
            if r.aimed and r.aim_yaw is not None and abs(wrap(want - r.aim_yaw)) > REAIM:
                r.aimed = False
            if not r.aimed:
                err = wrap(want - yaw[i])
                if abs(err) < AIM_TOL:
                    r.aimed, r.aim_yaw = True, want
                    return ("wheels", [0.0] * 4)
                return ("wheels", wheels(0.0, max(-YAW_RATE_MAX, min(YAW_RATE_MAX, KP_SPIN * err))))
        return ("wheels", [0.0] * 4)

    def traverse_failed(self, r: Rover, why: str):
        self.counters["faults"] += 1
        self.counters["stuck"] += 1
        self.pending_recovery.add(r.i)
        r.status, r.reason = "FAILURE", why
        r.pause_until, r.cmd_key, r.node = self.t + FAIL_PAUSE_S, None, None
        r.best, r.best_t = 1e18, self.t
        self.set_banner(f"FAULT: {r.id} traverse FAILURE ({why}) -> brain replans")
        self.log(f"FAULT {r.id}: traverse FAILURE ({why}) at ({self.last_pos[r.i][0]:.2f}, "
                 f"{self.last_pos[r.i][1]:.2f}) heading to {r.goal}")
        return ("wheels", [0.0] * 4)

    # -- one policy step -------------------------------------------------------------------------------
    def step(self) -> None:
        env, rb = self.env, self.robot
        pos = rb.data.root_pos_w.torch[:, :3].tolist()
        yaw = rb.data.heading_w.torch.tolist()
        quat = rb.data.root_quat_w.torch.tolist()
        self.last_pos = pos
        self.update_sun()
        if args_cli.weather_dynamic and self.econ.step_weather(self.step_dt):
            self.log(f"weather -> {self.econ.weather}")
        self.maybe_inject(pos)
        if self.k % max(1, int(round(BRAIN_TICK_S / self.step_dt))) == 0:
            self.brain.tick(self.snapshot(pos, yaw), time.monotonic())
        plan = self.brain.plan
        if id(plan) != self.plan_id:
            self.plan_id = id(plan)
            self.counters["plans"]["llm" if str(plan.source).startswith("llm") else "rules"] += 1
        for r in self.rovers:
            self.follow(r, plan.commands.get(r.id), pos)
        self.admit()
        self.plant_step()

        # targets for the policy, scripted wheel commands for everything else
        modes = [self.drive(r, i, pos, yaw) for i, r in enumerate(self.rovers)]
        tx = torch.tensor([m[1][0] if m[0] == "policy" else pos[i][0] for i, m in enumerate(modes)], device=self.dev)
        ty = torch.tensor([m[1][1] if m[0] == "policy" else pos[i][1] for i, m in enumerate(modes)], device=self.dev)
        th = torch.tensor([m[2] if m[0] == "policy" else yaw[i] for i, m in enumerate(modes)], device=self.dev)
        self.cmd_term.pos_command_w[:, 0] = tx
        self.cmd_term.pos_command_w[:, 1] = ty
        self.cmd_term.heading_command_w[:] = th
        self.cmd_term._update_command()                   # the new target is in this step's observation
        obs = env.observation_manager.compute()
        act = self.policy(obs["policy"]).clamp(-1.0, 1.0).tolist()
        slew = SLEW_PER_S * self.step_dt
        for i, (m, r) in enumerate(zip(modes, self.rovers)):
            if m[0] == "wheels":
                act[i] = [r.act[k] + max(-slew, min(slew, m[1][k] - r.act[k])) for k in range(4)]
            r.act = list(act[i])
        # SoC: exposure of the tilted panel from the real pose; charge only while harvesting
        wheel_vel = rb.data.joint_vel.torch[:, self.wheel_ids].abs().tolist()
        for i, r in enumerate(self.rovers):
            n_w = qrot(quat[i], PANEL_N_B)
            r.exposure = max(0.0, sum(a * b for a, b in zip(n_w, self.sun))) if self.sun[2] > 0 else 0.0
            r.pv_w = self.econ.panel_power_w(n_w, self.sun)
            p = self.pack_of(i)
            if p is None:
                continue
            power = sum(abs(a) * w / math.pi for a, w in zip(r.act, wheel_vel[i])) / 4.0
            motor_w = MOTOR_IDLE_W + MOTOR_FULL_W * min(1.0, power)
            harvesting = r.state == "harvesting" and r.stage is None
            before = self.pack_soc[p]
            self.pack_soc[p] = self.econ.step_soc(before, r.pv_w, motor_w, self.step_dt, harvesting)
            if harvesting and self.pack_soc[p] > before:
                r.harvested_wh += (self.pack_soc[p] - before) * CAPACITY_WH
                r.cycle["harvested"] = True
        _, _, terminated, truncated, _ = env.step(torch.tensor(act, device=self.dev))
        self.k += 1
        self.t = self.k * self.step_dt
        self.after_step(terminated | truncated)

    def after_step(self, done) -> None:
        ids = done.nonzero(as_tuple=False).flatten().tolist()
        if ids:
            try:
                tipped = self.env.termination_manager.get_term("tipped").tolist()
            except Exception:  # noqa: BLE001
                tipped = [True] * N
            for i in ids:
                r = self.rovers[i]
                self.seat_pack(i)                          # the reset moved the chassis; bring the pack along
                kind = "tipped" if tipped[i] else "reset"
                self.counters["tipovers" if tipped[i] else "nonfinite_resets"] += 1
                self.counters["faults"] += 1
                self.pending_recovery.add(i)
                self.leave_queue(i)
                if self.current == i:
                    self.ctrl.reset()
                    self.abort_swap()
                    self.release(r)
                r.stage, r.parked, r.goal, r.cmd_key = None, False, None, None
                r.state, r.status, r.reason, r.fault_until = "fault", "FAILURE", kind, self.t + FAULT_HOLD_S
                self.set_banner(f"FAULT: {r.id} {kind} -> reset to its park spot")
                self.log(f"FAULT {r.id}: {kind} (episode reset to the park row)")
        pos = self.robot.data.root_pos_w.torch[:, :2].tolist()
        for i, r in enumerate(self.rovers):
            m = max(abs(pos[i][0]), abs(pos[i][1]))
            self.counters["max_abs_xy"] = max(self.counters["max_abs_xy"], round(m, 3))
            out = m > PASTURE_HALF
            if out and not r.outside:
                self.counters["left_pasture"] += 1
                self.log(f"{r.id} LEFT THE PASTURE at ({pos[i][0]:.2f}, {pos[i][1]:.2f})")
            r.outside = out
        if self.t >= self.banner_until:
            self.banner = None

    # -- overlay ---------------------------------------------------------------------------------------
    def overlay_info(self) -> dict:
        from orchestrator.telemetry import hhmm
        t_day = self.econ.day_seconds(self.t)
        el = math.degrees(math.asin(max(-1.0, min(1.0, self.sun[2]))))
        az = (math.degrees(math.atan2(self.sun[0], self.sun[1])) + 360.0) % 360.0   # compass
        pos = self.robot.data.root_pos_w.torch[:, :3].tolist()
        rovers = []
        for i, r in enumerate(self.rovers):
            stage = r.stage or ""
            if r.stage == "swap" and r.swap_phase:
                stage = r.swap_phase
            rovers.append({"id": r.id, "state": r.state, "stage": stage, "soc": self.soc(i),
                           "exposure": r.exposure, "pv_w": r.pv_w, "has_pack": self.pack_of(i) is not None,
                           "status": r.status, "reason": r.reason,
                           "tag_px": None if args_cli.no_tags else project((pos[i][0], pos[i][1], pos[i][2] + TAG_Z))})
        cur = self.rovers[self.current] if self.current is not None else None
        st = self.brain.status
        plan = self.brain.plan
        return {
            "title": "Solar Sheep · Omniverse factory",
            "clock": hhmm(t_day), "timelapse": self.econ.timelapse, "t_phys": self.t,
            "sun_az_deg": az, "sun_el_deg": el, "weather": self.econ.weather, "weather_f": self.econ.weather_f,
            "rovers": rovers,
            "dock": {"status": ("swapping" if cur is not None else "idle"), "current": cur.id if cur else None,
                     "phase": (cur.swap_phase if cur and cur.stage == "swap" else (cur.stage if cur else "")),
                     "queue": [RIDS[i] for i in self.queue], "empty_packs": self.empty_count() if PHYSICAL else "-",
                     "swaps": self.counters["swaps_ok"],
                     "mode": ("physical latch swap: USD FixedJoints toggled at runtime" if PHYSICAL
                              else "LOGICAL swap (dock hardware disabled)")},
            "brain": {"label": brain_label(plan.source), "calls": st.get("calls", 0),
                      "cost": st.get("cost_usd_est", 0.0), "in_flight": st.get("in_flight", False),
                      "cmd_json": self.last_cmd["json"], "cmd_reason": self.last_cmd["reason"],
                      "summary": plan.summary},
            "banner": self.banner,
        }


def brain_label(source: str) -> str:
    source = str(source)
    if source.startswith("llm:"):
        model = source[4:]
        name = "Nemotron" if "nemotron" in model.lower() else model.split("/")[-1]
        return f"{name} ({model})"
    if source.startswith("rules ("):
        return "rule fallback " + source[len("rules "):]
    return "rule fallback"


def project(p) -> tuple | None:
    """World point -> pixel in the Kit clip (pinhole, horizontal FOV CAM_HFOV_DEG)."""
    e, la = args_cli.cam_eye, args_cli.cam_lookat
    f = [la[k] - e[k] for k in range(3)]
    fn = math.sqrt(sum(v * v for v in f)) or 1.0
    f = [v / fn for v in f]
    rgt = [f[1], -f[0], 0.0]                             # f x z_up
    rn = math.sqrt(sum(v * v for v in rgt)) or 1.0
    rgt = [v / rn for v in rgt]
    up = [rgt[1] * f[2] - rgt[2] * f[1], rgt[2] * f[0] - rgt[0] * f[2], rgt[0] * f[1] - rgt[1] * f[0]]
    d = [p[k] - e[k] for k in range(3)]
    zc = sum(d[k] * f[k] for k in range(3))
    if zc <= 0.1:
        return None
    fx = (WIDTH / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)
    return (int(WIDTH / 2 + fx * sum(d[k] * rgt[k] for k in range(3)) / zc),
            int(HEIGHT / 2 - fx * sum(d[k] * up[k] for k in range(3)) / zc))


class ClipWriter:
    """Streams frames to an H.264 mp4 (imageio-ffmpeg); falls back to numbered JPEG/PNG frames."""

    def __init__(self, path: str, fps: int):
        self.path, self.n, self.writer, self.frames_dir = path, 0, None, None
        try:
            import imageio
            self.writer = imageio.get_writer(path, format="FFMPEG", mode="I", fps=fps, codec="libx264",
                                             quality=7, macro_block_size=8)
        except Exception as e:  # noqa: BLE001
            self.frames_dir = os.path.splitext(path)[0] + "_frames"
            os.makedirs(self.frames_dir, exist_ok=True)
            print(f"[factory] WARNING: no ffmpeg writer ({e}); writing frames to {self.frames_dir}")

    def add(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.append_data(frame)
        else:
            import imageio
            imageio.imwrite(os.path.join(self.frames_dir, f"{self.n:06d}.png"), frame)
        self.n += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def main() -> int:
    os.makedirs(args_cli.out, exist_ok=True)
    cfg = build_cfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    for v in getattr(env.sim, "visualizers", []):        # safety: every viewer box-local (as measure_rover.py)
        addr = getattr(v.cfg, "bind_address", "127.0.0.1")
        if addr not in ("127.0.0.1", "localhost") or getattr(v.cfg, "share", False):
            env.close()
            raise SystemExit(f"refusing to run: visualizer {v.cfg.visualizer_type} would listen on {addr}")
    kit_viz = next((v for v in getattr(env.sim, "visualizers", []) if v.cfg.visualizer_type == "kit"), None)
    if FACTORY.get("bank") is not None:
        FACTORY["bank"].settle_primed()                    # open the primed latches before our first step
    print(f"[factory] env ready: {N} rovers on {env.device}, step_dt {env.step_dt:.4f} s, "
          f"dock {'physical' if PHYSICAL else 'logical'}, brain llm={args_cli.llm}, kit clip={kit_viz is not None}")

    policy = torch.jit.load(args_cli.policy, map_location=env.device).eval()
    overlay = None
    if not args_cli.no_overlay:
        try:
            import factory_overlay as overlay
            if not overlay.available():
                print("[factory] WARNING: Pillow missing -- the clip is written without overlays")
                overlay = None
        except Exception as e:  # noqa: BLE001
            print(f"[factory] WARNING: overlay module failed ({e}) -- the clip is written without overlays")
            overlay = None

    with torch.inference_mode():
        fac = Factory(env, policy)
        fac.place()
        fac.update_sun(force=True)
        for _ in range(int(round(0.5 / fac.step_dt))):     # settle on the ground, packs in their bays
            env.step(torch.zeros(N, 4, device=env.device))
        clip = None if (args_cli.no_video or kit_viz is None) else ClipWriter(
            os.path.join(args_cli.out, "factory.mp4"), args_cli.fps)
        steps, every = int(round(args_cli.seconds / fac.step_dt)), capture_every()
        log_every = max(1, int(round(LOG_EVERY_S / fac.step_dt)))
        t0, blank = time.perf_counter(), 0
        for n in range(steps):
            fac.step()
            if clip is not None and fac.k % every == 0:
                frame = kit_viz.render_rgb_array()
                if frame is not None and frame.size and frame.any():
                    frame = np.ascontiguousarray(np.asarray(frame)[..., :3])
                    if overlay is not None:
                        try:
                            frame = overlay.draw(frame, fac.overlay_info())
                        except Exception as e:  # noqa: BLE001
                            print(f"[factory] WARNING: overlay failed ({e}); continuing without")
                            overlay = None
                    clip.add(frame)
                else:
                    blank += 1
            if n % log_every == 0:
                st = fac.brain.status
                print(f"[factory] step {n} t={fac.t:.1f}s clock {fac.overlay_info()['clock']} "
                      f"states {[r.state[:5] + (':' + r.stage if r.stage else '') for r in fac.rovers]} "
                      f"soc {[round(fac.soc(i), 2) for i in range(N)]} queue {[RIDS[i] for i in fac.queue]} "
                      f"dock {RIDS[fac.current] if fac.current is not None else '-'} "
                      f"brain {fac.brain.plan.source} calls {st.get('calls', 0)} frames {clip.n if clip else 0}")
            if args_cli.realtime:
                lag = (n + 1) * fac.step_dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)
        wall = time.perf_counter() - t0
        if clip is not None:
            clip.close()
            print(f"[factory] clip: {clip.path} ({clip.n} frames at {args_cli.fps} fps, {blank} blank skipped)")
        st = fac.brain.status
        fac.brain.close()

    c = fac.counters
    c["llm_calls"] = st.get("calls", 0)
    c["harvested_wh"] = {r.id: round(r.harvested_wh, 1) for r in fac.rovers}
    c["delivered_wh"] = round(c["delivered_wh"], 1)
    checks = {
        "full_cycle": c["full_cycles"] >= 1,               # harvest -> dock -> PHYSICAL swap -> back on a cell
        "physical_swap": c["swaps_physical"] >= 1 and all(
            (s.get("new_pack_bay_err_m") or 0.0) < 0.02 and (s.get("bay_err_after_depart_m") or 0.0) < 0.02
            for s in fac.swaps if s["physical"]),
        "no_rover_left_pasture": c["left_pasture"] == 0,
        "no_tipover": c["tipovers"] == 0,
    }
    if not PHYSICAL:
        c["note"] = "logical dock mode: swaps are not physical, full_cycle/physical_swap cannot pass"
    res = {"args": {k: v for k, v in vars(args_cli).items() if isinstance(v, (int, float, str, bool, list, type(None)))},
           "rover_usd": ROVER_USD, "physics_seconds": round(fac.t, 2), "wall_seconds": round(wall, 1),
           "counters": c, "swaps": fac.swaps, "fault": fac.fault, "brain_status": st,
           "final": {r.id: {"state": r.state, "soc": round(fac.soc(r.i), 3), "pack": fac.pack_of(r.i)}
                     for r in fac.rovers},
           "checks": checks, "events": fac.events}
    path = os.path.join(args_cli.out, "factory.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"[factory] results: {path}  (swaps {c['swaps_ok']} physical {c['swaps_physical']}, cycles "
          f"{c['full_cycles']}, faults {c['faults']} recovered {c['recoveries']}, llm calls {c['llm_calls']})")
    ok = all(checks.values())
    print("FACTORY: ALL ASSERTIONS PASSED" if ok else
          "FACTORY: FAILED " + ", ".join(k for k, v in checks.items() if not v))
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    raise SystemExit(code)
