"""envs/swap/isaac_hardware.py -- B's `DockHardware` Protocol in Omniverse (Isaac Lab 3.0 EA, PhysX).

The Isaac twin of `scripts/mujoco_hardware.py`, used by `envs/swap/isaac_swap.py` (one rover) and by
the multi-rover factory (`envs/isaac_rover/factory.py`, N rovers sharing one dock). `swap_battery.py`
is untouched: it drives the six Protocol methods, and `sync(dt)` is where each request meets physics.

    from envs.swap.isaac_hardware import LatchBank, IsaacDockHardware, author_spare_pack

    # BEFORE sim.reset(): author the latches (disabled FixedJoints chassis<->pack for every pair)
    bank = LatchBank(stage, chassis_paths, pack_paths, latch_template_path, initial={0: 0})
    bank.author()
    sim.reset()
    hw = IsaacDockHardware(dock=dock, rovers=rover, packs=[pack, spare], bank=bank,
                           stations=dock_stations())
    hw.set_active(0); ...; every physics step: hw.sync(dt) before *.write_data_to_sim()

WHAT MAKES THIS A SWAP AND NOT AN ANIMATION
Each pack is a real PhysX rigid body, held to a chassis by a USD `UsdPhysics.FixedJoint`
(excludeFromArticulation) that this class switches with `physics:jointEnabled` -- the engine-level
equivalent of MuJoCo's `eq_active`. The rover's own imported `pack_latch` is one of those joints; the
others are authored by `LatchBank` before play. Releasing is one attribute going False and the pack
is no longer part of the vehicle; latching is one going True and the rover drives away carrying a
different rigid body. The LATCH is physical and is never a teleport. PhysX picks the attribute up
through its USD stage listener while it runs; the lead's envs/swap/latch_spike.py proved both
directions on the imported pack_latch on cuda:0 (27 Sep: 1e-6 m bay error, pack falls exactly 0.060 m
when released). The CPU pipeline died at sim start in that spike, so nothing here forces a device.

Between release and latch the dock owns the pack and moves it KINEMATICALLY (pose written, velocity
zeroed every step): the lift carriage and the two deck shuttles are the plant's conveyor, scripted
exactly as `mujoco_hardware.py`'s PINNED mode ("scripted - not learned", envs/swap/SPEC.md). Ownership
is explicit in `_Pack.mode`:

    "weld"   a FixedJoint holds it to a chassis; physics owns it; nothing here writes to it
    "pinned" the dock holds it; pose/velocity written every step
    "free"   nobody holds it; it rests on the transfer deck under gravity

Ported unchanged from mujoco_hardware.py: Protocol calls ARM a transfer and `sync` resolves it when
the carriage has physically arrived (4 mm); the lift is rate-limited (0.055 m/s); the legacy 0.12
"raise fully" is normalised to the LOCKED 0.060 stroke; before a latch closes the pack is written to
the exact bay pose WITH the chassis velocity, so the joint has nothing to snap.

Engine touch-points (MuJoCo -> here):
    eq_active[...] = 0/1              -> FixedJoint `physics:jointEnabled` (LatchBank.attach/detach)
    d.ctrl[lift]                      -> dock.actuators.target_command.set_position_index
    site_xpos carriage top / mount    -> dock.data.joint_pos + dock root z / chassis pose (x) latch frames
    xpos/xquat of the chassis         -> rovers.data.root_link_pos_w / root_link_quat_w (x, y, z, w)
    qpos/qvel writes                  -> RigidObject.write_root_pose/velocity_to_sim_index
    d.time                            -> the accumulated dt of sync()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # normal package import
    from .dock_mjcf import DOCK, DockLayout
except ImportError:  # pragma: no cover - direct execution
    from envs.swap.dock_mjcf import DOCK, DockLayout  # type: ignore

LEGACY_FULL_STROKE = 0.12   # swap_battery.py's hard-coded "raise fully"

_CAPTURE_EPS = 0.004        # carriage counts as "arrived" within 4 mm
_STOWED_EPS = 0.004         # carriage counts as "down" within 4 mm
_PIN_FLOAT = 0.0005         # hold a pinned pack 0.5 mm off the deck (no chatter against the pin)

VEHICLE = "vehicle"
CARRIAGE = "carriage"
RACK_FULL = "full_stow"
RACK_EMPTY = "empty_ready"
WELD, PINNED, FREE = "weld", "pinned", "free"

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]     # (x, y, z, w), Isaac Lab 3.0 order
IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


# ---- quaternion helpers, (x, y, z, w) --------------------------------------------------------
def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_conj(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    x, y, z, _ = quat_mul(quat_mul(q, (v[0], v[1], v[2], 0.0)), quat_conj(q))
    return (x, y, z)


def quat_yaw(q: Quat) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw: float) -> Quat:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def pose_mul(p1: Vec3, q1: Quat, p2: Vec3, q2: Quat) -> Tuple[Vec3, Quat]:
    r = quat_rotate(q1, p2)
    return (p1[0] + r[0], p1[1] + r[1], p1[2] + r[2]), quat_mul(q1, q2)


def pose_inv(p: Vec3, q: Quat) -> Tuple[Vec3, Quat]:
    qi = quat_conj(q)
    r = quat_rotate(qi, p)
    return (-r[0], -r[1], -r[2]), qi


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


# ---- stage lookup (the importer's prim layout is not hard-coded) ------------------------------
def find_named_prim(stage, root_path: str, name: str, kind: str = "body"):
    """First prim called `name` under `root_path` that is a rigid body (kind="body") or a joint.

    The MJCF importer's nesting (Geometry/..., Physics/..., payloads) is an implementation detail of
    Isaac Sim 6.1, so bodies and joints are found by their MJCF names rather than by path.
    """
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"no prim at {root_path}")
    for prim in Usd.PrimRange(root):
        if prim.GetName() != name:
            continue
        if kind == "body" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        if kind == "joint" and prim.IsA(UsdPhysics.Joint):
            return prim
    raise RuntimeError(f"no {kind} named {name!r} under {root_path}")


# ================================================================================================
# LatchBank: one FixedJoint per (chassis, pack) pair, exactly one enabled per rover at most.
# ================================================================================================
class LatchBank:
    """Every chassis<->pack latch in the scene, as USD FixedJoints toggled with jointEnabled.

    `latch_template_path` is an imported `pack_latch` (rover_swap USD); its local frames are copied
    to every authored latch, so the bay pose is whatever the importer wrote, not an assumed
    (0, 0, -0.0875). A joint that already links a chassis and a pack (each rover's own imported
    pack_latch) is adopted rather than duplicated. `initial` maps rover index -> pack index for the
    latches that start closed; every other latch starts open.
    """

    def __init__(self, stage, chassis_paths: Sequence[str], pack_paths: Sequence[str],
                 latch_template_path: str, initial: Dict[int, int],
                 scope: str = "/World/SwapLatches") -> None:
        self.stage = stage
        self.chassis_paths = [str(p) for p in chassis_paths]
        self.pack_paths = [str(p) for p in pack_paths]
        self.template_path = str(latch_template_path)
        self.initial = dict(initial)
        self.scope = scope
        self.joint_paths: Dict[Tuple[int, int], str] = {}
        self._primed: List[Tuple[int, int]] = []
        self._attached: Dict[int, Optional[int]] = {r: None for r in range(len(self.chassis_paths))}
        # chassis-side and pack-side joint frames, positions in metres, quats (x, y, z, w)
        self.frame_chassis: Tuple[Vec3, Quat] = ((0.0, 0.0, DOCK.pack_bay_local_z), IDENTITY)
        self.frame_pack: Tuple[Vec3, Quat] = ((0.0, 0.0, 0.0), IDENTITY)
        if len(set(self.initial.values())) != len(self.initial):
            raise ValueError(f"two rovers cannot start on the same pack: {self.initial}")

    # ---- authoring (before sim.reset) ---------------------------------------------------------
    def author(self, prime: bool = False) -> None:
        """Author/adopt every latch. Call before sim.reset().

        prime=False (Contract 2): open latches are authored with jointEnabled=False.
        prime=True: every latch is authored ENABLED, so PhysX certainly creates it at parse time, and
        `settle_primed()` opens the ones that should start open after sim.reset() and before the first
        sim.step() (reset does not simulate: isaaclab_physx physx_manager.py reset/_warmup_and_create_views).
        That is exactly the path the latch spike proved (parsed enabled -> disabled -> enabled at runtime);
        whether PhysX creates a joint that is disabled at parse time is not proven yet.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        stage = self.stage
        tpl = stage.GetPrimAtPath(self.template_path)
        if not tpl or not tpl.IsA(UsdPhysics.Joint):
            raise RuntimeError(f"latch template {self.template_path} is not a UsdPhysics joint")
        tj = UsdPhysics.Joint(tpl)
        b0 = [str(t) for t in tj.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in tj.GetBody1Rel().GetTargets()]

        def frame(pos_attr, rot_attr):
            p = pos_attr.Get() or Gf.Vec3f(0, 0, 0)
            q = rot_attr.Get() or Gf.Quatf(1, 0, 0, 0)
            im = q.GetImaginary()
            return ((float(p[0]), float(p[1]), float(p[2])),
                    (float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())))

        f0 = frame(tj.GetLocalPos0Attr(), tj.GetLocalRot0Attr())
        f1 = frame(tj.GetLocalPos1Attr(), tj.GetLocalRot1Attr())
        chassis_is_body0 = bool(b0) and any(b0[0] == c for c in self.chassis_paths)
        if not chassis_is_body0 and not (b1 and any(b1[0] == c for c in self.chassis_paths)):
            # Template belongs to a rover not in this bank: trust the importer's MJCF order
            # (weld body1=chassis -> body0) and say so.
            print(f"[LatchBank] template bodies {b0} / {b1} match no chassis; assuming body0 = chassis")
            chassis_is_body0 = True
        self.frame_chassis, self.frame_pack = (f0, f1) if chassis_is_body0 else (f1, f0)

        # Existing joints linking a chassis and a pack (the imported pack_latch of each rover).
        existing: Dict[Tuple[int, int], str] = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(prim)
            t0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            t1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not t0 or not t1:
                continue
            for a, b in ((t0[0], t1[0]), (t1[0], t0[0])):
                if a in self.chassis_paths and b in self.pack_paths:
                    existing[(self.chassis_paths.index(a), self.pack_paths.index(b))] = str(prim.GetPath())

        UsdGeom.Scope.Define(stage, Sdf.Path(self.scope))
        c_pos, c_rot = self.frame_chassis
        p_pos, p_rot = self.frame_pack
        for r, cpath in enumerate(self.chassis_paths):
            chassis = stage.GetPrimAtPath(cpath)
            if not chassis or not chassis.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"chassis {cpath} is not a rigid body")
            pairs = UsdPhysics.FilteredPairsAPI.Apply(chassis).CreateFilteredPairsRel()
            for p, ppath in enumerate(self.pack_paths):
                pack = stage.GetPrimAtPath(ppath)
                if not pack or not pack.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(f"pack {ppath} is not a rigid body")
                pairs.AddTarget(Sdf.Path(ppath))          # the MJCF <exclude> chassis<->pack
                if (r, p) in existing:
                    jp = existing[(r, p)]
                    joint = UsdPhysics.Joint(stage.GetPrimAtPath(jp))
                else:
                    jp = f"{self.scope}/latch_r{r}_p{p}"
                    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(jp))
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(cpath)])
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(ppath)])
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*c_pos))
                    joint.CreateLocalRot0Attr().Set(Gf.Quatf(c_rot[3], c_rot[0], c_rot[1], c_rot[2]))
                    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*p_pos))
                    joint.CreateLocalRot1Attr().Set(Gf.Quatf(p_rot[3], p_rot[0], p_rot[1], p_rot[2]))
                joint.CreateExcludeFromArticulationAttr().Set(True)
                on = self.initial.get(r) == p
                joint.CreateJointEnabledAttr().Set(on or prime)
                if prime and not on:
                    self._primed.append((r, p))
                self.joint_paths[(r, p)] = jp
                if on:
                    self._attached[r] = p
        print(f"[LatchBank] {len(self.joint_paths)} latches ({len(existing)} adopted from the import), "
              f"closed: {self.initial}; bay frame chassis-side {self.frame_chassis}, pack-side {self.frame_pack}")

    def settle_primed(self) -> None:
        """Open the latches author(prime=True) left closed. After sim.reset(), before any sim.step()."""
        for r, p in self._primed:
            self._set(r, p, False)
        if self._primed:
            print(f"[LatchBank] opened {len(self._primed)} primed latches before the first step")
        self._primed = []

    # ---- runtime -------------------------------------------------------------------------------
    def _set(self, r: int, p: int, on: bool) -> None:
        from pxr import UsdPhysics

        prim = self.stage.GetPrimAtPath(self.joint_paths[(r, p)])
        UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(bool(on))

    def enabled(self, r: int, p: int) -> bool:
        """What the stage says (the assertion source), not the bookkeeping."""
        prim = self.stage.GetPrimAtPath(self.joint_paths[(r, p)])
        return bool(prim.GetAttribute("physics:jointEnabled").Get())

    def attach(self, rover_idx: int, pack_idx: int) -> None:
        if self._attached.get(rover_idx) is not None:
            raise RuntimeError(f"rover {rover_idx} already carries pack {self._attached[rover_idx]}")
        holder = self.rover_of(pack_idx)
        if holder is not None:
            raise RuntimeError(f"pack {pack_idx} is latched to rover {holder}")
        self._set(rover_idx, pack_idx, True)
        self._attached[rover_idx] = pack_idx

    def detach(self, rover_idx: int) -> Optional[int]:
        p = self._attached.get(rover_idx)
        if p is not None:
            self._set(rover_idx, p, False)
            self._attached[rover_idx] = None
        return p

    def pack_of(self, rover_idx: int) -> Optional[int]:
        return self._attached.get(rover_idx)

    def rover_of(self, pack_idx: int) -> Optional[int]:
        for r, p in self._attached.items():
            if p == pack_idx:
                return r
        return None

    def bay_pose(self, chassis_pos: Vec3, chassis_quat: Quat) -> Tuple[Vec3, Quat]:
        """World pose a pack must have for its latch to be exactly satisfied."""
        jp, jq = pose_mul(chassis_pos, chassis_quat, *self.frame_chassis)
        return pose_mul(jp, jq, *pose_inv(*self.frame_pack))


# ================================================================================================
# The spare pack (the dock's fresh one): a plain 4 kg rigid box, like dock_mjcf's pack_spare.
# ================================================================================================
C_SPARE_PACK = (0.20, 0.80, 0.42)


def spare_pack_cfg(prim_path: str, pos: Vec3, rot: Quat = IDENTITY, layout: DockLayout = DOCK,
                   mass: float = 4.0, friction: float = 0.4):
    """RigidObjectCfg of one spare pack: 0.20 x 0.45 x 0.055 m, 4 kg, mu 0.4, green."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=tuple(layout.pack_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=C_SPARE_PACK),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot)),
    )


def author_spare_pack(prim_path: str, pos: Vec3, rot: Quat = IDENTITY, layout: DockLayout = DOCK, **kw):
    """Spawn one spare pack on the stage (before sim.reset) and return its RigidObject."""
    from isaaclab.assets import RigidObject

    return RigidObject(spare_pack_cfg(prim_path, pos, rot, layout, **kw))


def dock_stations(dock_pos: Vec3 = (0.0, 0.0, 0.0), dock_yaw: float = 0.0,
                  layout: DockLayout = DOCK) -> Dict[str, Tuple[Vec3, Quat]]:
    """World poses of a pack resting on the full_stow / empty_ready stations of a placed dock."""
    q = yaw_quat(dock_yaw)
    out = {}
    for name, sgn in ((RACK_FULL, -1.0), (RACK_EMPTY, 1.0)):
        out[name] = pose_mul(dock_pos, q, (sgn * layout.station_x, 0.0, layout.deck_pack_z), IDENTITY)
    return out


# ================================================================================================
# IsaacDockHardware
# ================================================================================================
@dataclass
class _Pack:
    """One pack body, which RigidObject instance it is, and whatever is holding it."""

    idx: int
    obj: object
    env: int
    mode: str
    carrier: str
    shuttle_from: Optional[Vec3] = None
    shuttle_to: str = ""
    shuttle_t: float = 0.0
    shuttle_T: float = 1.0
    release_on_arrival: bool = False

    @property
    def shuttling(self) -> bool:
        return self.shuttle_from is not None


class IsaacDockHardware:
    """B's `DockHardware`, realised against PhysX through Isaac Lab assets.

    dock      Articulation of the dock USD (fixed base, prismatic `dock_lift`), instance `dock_env`
    rovers    Articulation holding every rover (one instance per rover; root link = chassis)
    packs     one entry per pack, in LatchBank.pack_paths order: a single-instance RigidObject, or
              (RigidObject, instance index) when one RigidObject holds several packs
    stations  name -> world pose (pos, quat) of a pack resting there; "full_stow" and "empty_ready"
              are required (see dock_stations()); others are conveyor targets for convey()
    Unlatched packs start FREE at empty_ready unless `pack_carriers` says otherwise.
    """

    def __init__(self, dock, rovers, packs: Sequence, bank: LatchBank,
                 stations: Dict[str, Tuple[Vec3, Quat]], layout: DockLayout = DOCK, *,
                 dock_env: int = 0, lift_joint: str = "dock_lift", lift_rate: float = 0.055,
                 shuttle_seconds: float = 0.90, pack_carriers: Optional[Dict[int, str]] = None,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        import torch  # noqa: F401  (Isaac Lab's tensors; imported here so the module loads without it)

        self.dock, self.rovers, self.bank, self.L = dock, rovers, bank, layout
        self.dock_env = int(dock_env)
        self.stations = {k: (tuple(v[0]), tuple(v[1])) for k, v in stations.items()}
        for need in (RACK_FULL, RACK_EMPTY):
            if need not in self.stations:
                raise ValueError(f"stations must include {need!r}")
        self.lift_rate = float(lift_rate)
        self.shuttle_seconds = float(shuttle_seconds)
        self._on_event = on_event
        self.events: list = []
        self.time = 0.0

        ids, _ = dock.find_joints([lift_joint])
        if len(ids) != 1:
            raise RuntimeError(f"dock has no joint {lift_joint!r}: {dock.joint_names}")
        self._lift_id = int(ids[0])

        carriers = dict(pack_carriers or {})
        self._packs: List[_Pack] = []
        for i, entry in enumerate(packs):
            obj, env = (entry if isinstance(entry, tuple) else (entry, 0))
            r = bank.rover_of(i)
            if r is not None:
                pk = _Pack(i, obj, int(env), WELD, VEHICLE)
            else:
                pk = _Pack(i, obj, int(env), FREE, carriers.get(i, RACK_EMPTY))
            self._packs.append(pk)
        if len(self._packs) != len(bank.pack_paths):
            raise ValueError(f"{len(self._packs)} pack handles for {len(bank.pack_paths)} latch packs")

        bank.settle_primed()                    # no-op unless author(prime=True) and not yet settled
        self.active = 0
        self._full: Optional[_Pack] = None      # the pack taken off the active rover
        self._spare: Optional[_Pack] = None     # the pack being offered to it
        self._lift_target = 0.0
        self._lift_cmd = 0.0
        self._reset_arms()
        self.sync(0.0)

    # ---- factory hooks ---------------------------------------------------------------------------
    def _reset_arms(self) -> None:
        self._detach_armed = self._stow_armed = self._stage_armed = self._latch_armed = False
        self._carriage_cleared = False

    def set_active(self, rover_idx: int) -> None:
        """Which rover is on the berth. Only switch between swaps (the controller is idle)."""
        if int(rover_idx) != self.active:
            self.active = int(rover_idx)
            self._full = self._spare = None
            self._reset_arms()
            self._event(f"active rover -> {self.active}")

    def empty_pack_count(self) -> int:
        """Packs waiting at empty_ready (swap_battery's rack count)."""
        return sum(1 for p in self._packs if p.carrier == RACK_EMPTY and p.mode != WELD)

    def convey(self, pack_idx: int, station: str, *, release: bool = True) -> None:
        """Plant conveyor: shuttle a pack that no rover holds to a named station (factory logistics,
        e.g. clear full_stow before the next swap, or return a charged pack to empty_ready)."""
        pk = self._packs[pack_idx]
        if pk.mode == WELD:
            raise RuntimeError(f"pack {pack_idx} is latched to a rover")
        if station not in self.stations:
            raise ValueError(f"unknown station {station!r}")
        self._start_shuttle(pk, station, release_on_arrival=release)

    # ---- introspection ---------------------------------------------------------------------------
    def _dock_pose(self) -> Tuple[Vec3, Quat]:
        d = self.dock.data
        return (tuple(float(v) for v in d.root_link_pos_w.torch[self.dock_env]),
                tuple(float(v) for v in d.root_link_quat_w.torch[self.dock_env]))

    @property
    def lift_height(self) -> float:
        """Achieved stroke, 0.000 .. 0.060 (the prismatic joint position)."""
        return float(self.dock.data.joint_pos.torch[self.dock_env, self._lift_id])

    @property
    def carriage_top(self) -> float:
        """World z of the carriage top face (B's `pack_attach` frame): dock z + stroke."""
        return self._dock_pose()[0][2] + self.L.carriage_top_stowed + self.lift_height

    @property
    def lift_command(self) -> float:
        return self._lift_cmd

    def chassis_pose(self, rover_idx: Optional[int] = None) -> Tuple[Vec3, Quat]:
        r = self.active if rover_idx is None else rover_idx
        d = self.rovers.data
        return (tuple(float(v) for v in d.root_link_pos_w.torch[r]),
                tuple(float(v) for v in d.root_link_quat_w.torch[r]))

    def mount_underside_z(self, rover_idx: Optional[int] = None) -> float:
        """World z of the rover's `battery_mount` (pack underside in the bay)."""
        cp, cq = self.chassis_pose(rover_idx)
        bp, bq = self.bank.bay_pose(cp, cq)
        down = quat_rotate(bq, (0.0, 0.0, -self.L.pack_half[2]))
        return bp[2] + down[2]

    def latched_pack(self, rover_idx: Optional[int] = None) -> Optional[int]:
        r = self.active if rover_idx is None else rover_idx
        p = self.bank.pack_of(r)
        return p if p is not None and self.bank.enabled(r, p) else None

    def pack_pos(self, pack_idx: int) -> Vec3:
        pk = self._packs[pack_idx]
        return tuple(float(v) for v in pk.obj.data.root_link_pos_w.torch[pk.env])

    def pack_mode(self, pack_idx: int) -> Tuple[str, str]:
        pk = self._packs[pack_idx]
        return pk.mode, pk.carrier

    def _event(self, text: str) -> None:
        self.events.append((round(self.time, 4), text))
        if self._on_event is not None:
            self._on_event(text)

    # ==============================================================================================
    # DockHardware Protocol - all six methods, all idempotent.
    # ==============================================================================================
    def set_lift_height(self, height_m: float) -> None:
        if abs(float(height_m) - LEGACY_FULL_STROKE) < 1e-9:
            target = self.L.lift_stroke              # "raise fully" -> 0.060
        else:
            target = min(max(float(height_m), 0.0), self.L.lift_stroke)
        if abs(target - self._lift_target) > 1e-12:
            self._lift_target = target
            self._event(f"lift commanded to {target:.3f} m")

    def detach_full_to_carriage(self, vehicle_id: str) -> None:
        if not self._detach_armed:
            self._detach_armed = True
            self._carriage_cleared = False
            self._event(f"latch arming release from {vehicle_id} (rover {self.active})")

    def stow_full_pack(self) -> None:
        if not self._stow_armed:
            self._stow_armed = True
            self._event("full pack routed to full_stow")

    def stage_empty_pack(self) -> None:
        if not self._stage_armed:
            self._stage_armed = True
            self._event("fresh pack called from empty_ready")

    def latch_empty_to_vehicle(self, vehicle_id: str) -> None:
        if not self._latch_armed:
            self._latch_armed = True
            self._event(f"latch arming onto {vehicle_id} (rover {self.active})")

    def clear_carriage(self) -> None:
        if not self._carriage_cleared:
            self._carriage_cleared = True
            self._detach_armed = False
            self._latch_armed = False
            self._stow_armed = self._stage_armed = False   # re-armed by the per-tick re-issue; done
            self._full = None       # this swap's full pack is the plant's now
            self._event("carriage cleared")

    # ==============================================================================================
    # Simulation side: request meets physics.
    # ==============================================================================================
    def sync(self, dt: float) -> None:
        """Advance the mechanism by `dt`. Call once per physics step, before write_data_to_sim()."""
        self.time += max(dt, 0.0)
        self._advance_lift(dt)
        self._resolve_transfers()
        for pk in self._packs:
            self._advance_shuttle(pk, dt)
            if pk.mode == PINNED:
                self._pin(pk)

    def _advance_lift(self, dt: float) -> None:
        import torch

        step = self.lift_rate * max(dt, 0.0)
        if step > 0.0:
            err = self._lift_target - self._lift_cmd
            self._lift_cmd += min(max(err, -step), step)
        self._lift_cmd = min(max(self._lift_cmd, 0.0), self.L.lift_stroke)
        self.dock.actuators.target_command.set_position_index(
            value=torch.tensor([[self._lift_cmd]], dtype=torch.float32, device=self.dock.device),
            joint_ids=[self._lift_id], env_ids=[self.dock_env])

    def _pick_spare(self) -> Optional[_Pack]:
        """The waiting pack closest to the empty_ready station."""
        ready = [p for p in self._packs if p.carrier == RACK_EMPTY and p.mode != WELD and not p.shuttling]
        if not ready:
            return None
        sx, sy, _ = self.stations[RACK_EMPTY][0]
        return min(ready, key=lambda p: math.hypot(self.pack_pos(p.idx)[0] - sx, self.pack_pos(p.idx)[1] - sy))

    def _resolve_transfers(self) -> None:
        top = self.carriage_top
        stowed = self._dock_pose()[0][2] + self.L.carriage_top_stowed

        # 1. lift_full - the carriage must REACH the pack before the latch opens.
        if self._detach_armed and self._full is None:
            p = self.bank.pack_of(self.active)
            if p is not None and top >= self.mount_underside_z() - _CAPTURE_EPS:
                self.bank.detach(self.active)
                self._full = self._packs[p]
                self._full.mode, self._full.carrier = PINNED, CARRIAGE
                self._event(f"latch r{self.active}/p{p} RELEASED (jointEnabled=False) - full pack on the carriage")

        # 2. stow_full - only once the carriage has come back down.
        f = self._full
        if (self._stow_armed and f is not None and f.carrier == CARRIAGE and not f.shuttling
                and top <= stowed + _STOWED_EPS):
            self._start_shuttle(f, RACK_FULL, release_on_arrival=True)
            self._stow_armed = False

        # 3. offer_empty - shuttle a fresh pack onto the lowered carriage.
        if self._stage_armed and self._spare is None and top <= stowed + _STOWED_EPS:
            sp = self._pick_spare()
            if sp is not None:
                self._spare = sp
                self._start_shuttle(sp, CARRIAGE)
                self._stage_armed = False

        # 4. latch_empty - mirror of (1): seat it at the bay pose, then close the latch.
        s = self._spare
        if (self._latch_armed and s is not None and s.carrier == CARRIAGE and not s.shuttling
                and s.mode == PINNED and top >= self.mount_underside_z() - _CAPTURE_EPS):
            held = self.bank.pack_of(self.active)
            if held is not None:            # the release never happened (rover sat too high?)
                msg = f"cannot latch pack {s.idx}: rover {self.active} still carries pack {held}"
                if not self.events or self.events[-1][1] != msg:
                    self._event(msg)
                return
            pos, quat = self.bank.bay_pose(*self.chassis_pose())
            lin, ang = self._chassis_point_velocity(pos)
            self._write(s, pos, quat, lin, ang)
            self.bank.attach(self.active, s.idx)
            s.mode, s.carrier = WELD, VEHICLE
            self._spare = None      # (_full stays set until clear_carriage: step 1 must not re-fire)
            self._event(f"latch r{self.active}/p{s.idx} CLOSED (jointEnabled=True) - fresh pack on the vehicle")

    def _chassis_point_velocity(self, point: Vec3) -> Tuple[Vec3, Vec3]:
        """Velocity of the chassis at `point`, so a pack joins the rover with zero relative motion."""
        d = self.rovers.data
        r = self.active
        v = [float(x) for x in d.root_link_lin_vel_w.torch[r]]
        w = [float(x) for x in d.root_link_ang_vel_w.torch[r]]
        c, _ = self.chassis_pose()
        rx, ry, rz = point[0] - c[0], point[1] - c[1], point[2] - c[2]
        return ((v[0] + w[1] * rz - w[2] * ry, v[1] + w[2] * rx - w[0] * rz, v[2] + w[0] * ry - w[1] * rx),
                (w[0], w[1], w[2]))

    def _start_shuttle(self, pk: _Pack, to_carrier: str, *, release_on_arrival: bool = False) -> None:
        pk.shuttle_from = self.pack_pos(pk.idx)
        pk.shuttle_to = to_carrier
        pk.shuttle_t = 0.0
        pk.shuttle_T = self.shuttle_seconds
        pk.release_on_arrival = release_on_arrival
        pk.mode = PINNED
        self._event(f"pack {pk.idx} shuttling -> {to_carrier}")

    def _advance_shuttle(self, pk: _Pack, dt: float) -> None:
        if not pk.shuttling:
            return
        pk.shuttle_t += max(dt, 0.0)
        if pk.shuttle_t >= pk.shuttle_T:
            pk.carrier = pk.shuttle_to
            pk.shuttle_from = None
            if pk.release_on_arrival:
                self._pin(pk)                 # seat it exactly, then let go
                pk.mode = FREE
                pk.release_on_arrival = False
                self._event(f"pack {pk.idx} set down at {pk.carrier}")
            else:
                self._event(f"pack {pk.idx} seated at {pk.carrier}")

    # ---- pose sources ----------------------------------------------------------------------------
    def _carrier_pose(self, carrier: str) -> Tuple[Vec3, Quat]:
        hz = self.L.pack_half[2]
        dp, dq = self._dock_pose()
        if carrier == CARRIAGE:
            # Never dragged below the transfer deck: the carriage sets it down there and keeps
            # retracting on its own.
            z = max(self.lift_height + self.L.carriage_top_stowed, self.L.deck_top_z + _PIN_FLOAT)
            return pose_mul(dp, dq, (0.0, 0.0, z + hz), IDENTITY)
        pos, quat = self.stations[carrier]
        return (pos[0], pos[1], pos[2] + _PIN_FLOAT), quat

    def _pin(self, pk: _Pack) -> None:
        pos, quat = self._carrier_pose(pk.shuttle_to if pk.shuttling else pk.carrier)
        if pk.shuttling:
            u = _smoothstep(pk.shuttle_t / max(pk.shuttle_T, 1e-9))
            a = pk.shuttle_from
            pos = (a[0] + u * (pos[0] - a[0]), a[1] + u * (pos[1] - a[1]), a[2] + u * (pos[2] - a[2]))
        self._write(pk, pos, quat)

    def _write(self, pk: _Pack, pos: Vec3, quat: Quat, lin: Vec3 = (0.0, 0.0, 0.0),
               ang: Vec3 = (0.0, 0.0, 0.0)) -> None:
        import torch

        dev = pk.obj.device
        pk.obj.write_root_pose_to_sim_index(
            root_pose=torch.tensor([[*pos, *quat]], dtype=torch.float32, device=dev), env_ids=[pk.env])
        pk.obj.write_root_velocity_to_sim_index(
            root_velocity=torch.tensor([[*lin, *ang]], dtype=torch.float32, device=dev), env_ids=[pk.env])


__all__ = ["LatchBank", "IsaacDockHardware", "author_spare_pack", "spare_pack_cfg", "dock_stations",
           "find_named_prim", "quat_yaw", "yaw_quat", "LEGACY_FULL_STROKE", "VEHICLE", "CARRIAGE",
           "RACK_FULL", "RACK_EMPTY", "WELD", "PINNED", "FREE"]
