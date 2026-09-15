"""MuJoCo build of the drive-over battery-swap dock, fused with A's rover.

`robot/rover.xml` is loaded READ-ONLY and all dock geometry is grafted on in
memory with MjSpec.  Nothing is written back to the rover file: the assertion
`nq 11 / nu 4 / nbody 6` must keep holding for `robot/rover.xml` on disk.

Every number below marked LOCKED comes from `robot/SPEC.md` (15 Sep 2026) and
supersedes the PROPOSED values in `envs/swap/SPEC.md`:

  * forward is **+X** (wheel hinges are axis 0 1 0), so the whole dock is laid
    out along X.  B's frames, authored for a +Y approach, rotate 90 deg about Z:
    (x, y) -> (y, -x).  `approach (0,-1.5,0)` therefore becomes `(-1.5, 0, 0)`.
  * the pad spans the **0.80 m track** (wheels at y = +/-0.40): 1.10 x 1.00 m,
    so each wheel sits 0.10 m inside the lateral edge.  B's 0.70 m pad missed
    both wheels entirely.
  * the pack is **0.20 (X) x 0.45 (Y) x 0.055 (Z)**.
  * the lift stroke is **0 -> 0.060**; the pack underside is at z = 0.060.

STRAP CLEARANCE.  The rover's retaining straps sit at x = +/-0.06 with a
half-width of 0.012, i.e. their inner faces are at x = +/-0.048, and they hang
2.5 mm below the pack to z = 0.0575.  The carriage is made **0.090 m wide in X**
(half 0.045) so it rises *between* the straps with 3 mm of clearance each side
and never touches them.  See the module docstring of `scripts/mujoco_hardware.py`
for the pack-side half of the story.

DATUM.  The pad top is the datum the rover stands on, and it is deliberately
placed 1 mm proud of the grass plane (z = 0.001) rather than exactly at z = 0:
flush would z-fight with the ground plane, and 1 mm is 1/90th of a wheel radius
so there is no step to climb (SPEC constraint 4 - ramp <= 14 deg - is satisfied
vacuously).  World z and pad-relative z are therefore interchangeable to 1 mm,
which is what lets the lift stroke be asserted literally against 0.060.

TWO CONSTRAINTS FROM `envs/traverse/train.py`, honoured here so the same dock
can be dropped straight into the training arena:

  1. COLLISION BITS.  The training arena filters contacts with
     `BIT_ROVER, BIT_WORLD = 1, 2`; every static geom is re-bitted to
     `(contype, conaffinity) = (BIT_WORLD, BIT_ROVER)`.  Every colliding dock
     geom is authored with exactly those bits (`WB_BITS`), so the rover still
     lands on the pad whether or not the filter is applied.  Note train.py only
     re-bits geoms that already have `contype or conaffinity` set, so a dock
     geom left at 0/0 would stay uncollidable there too - deliberate choices
     only, each justified at its call site.

  2. RANGEFINDER FAN at z ~ 0.180 m.  The rover's 9 forward rays are horizontal
     at chassis-local z = 0.005, i.e. world z = 0.180.  **`contype = 0` does NOT
     hide a geom from a rangefinder ray** - a decorative geom above 0.180 would
     read as a phantom obstacle in the policy's observation.  So NOTHING in this
     dock reaches 0.180: the tallest structure is the control cabinet's beacon
     at z = 0.155, and the tallest moving part is the carriage top at 0.060.
     `check_rangefinder_clearance()` asserts this against the compiled model
     rather than trusting the comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROVER_XML = REPO_ROOT / "robot" / "rover.xml"

# Mirrors envs/traverse/train.py: BIT_ROVER, BIT_WORLD = 1, 2 and every static
# geom gets WB_BITS = (BIT_WORLD, BIT_ROVER).  Against an unfiltered rover
# (contype 1 / conaffinity 1, as rover.xml ships) the pair still collides,
# because conaffinity BIT_ROVER meets the rover's contype bit 1.
BIT_ROVER, BIT_WORLD = 1, 2
WB_BITS = (BIT_WORLD, BIT_ROVER)

# World height of the rangefinder fan: chassis at z = 0.175 + RF_POS z = 0.005.
RF_FAN_Z = 0.180
RF_CLEARANCE = 0.020  # keep this much under the fan


@dataclass(frozen=True)
class DockLayout:
    """All dock numbers in one place. `LOCKED` = from robot/SPEC.md."""

    # ---- LOCKED (robot/SPEC.md) -------------------------------------------
    track: float = 0.80                       # wheel centres at y = +/-0.40
    wheelbase: float = 0.22                   # axles at x = +/-0.11
    pack_size: Tuple[float, float, float] = (0.20, 0.45, 0.055)
    pack_underside_z: float = 0.060           # lift must reach exactly this
    lift_stroke: float = 0.060                # 0 -> 0.060, NOT 0.12
    strap_x: float = 0.06                     # straps at x = +/-0.06 ...
    strap_half_x: float = 0.012               # ... half-width 0.012
    strap_bottom_z: float = 0.0575            # 2.5 mm below the pack

    # ---- Pad: must span the track ----------------------------------------
    pad_size_x: float = 1.10
    pad_size_y: float = 1.00
    pad_top_z: float = 0.001
    pad_thick: float = 0.060
    slot_half_x: float = 0.060                # well the carriage retracts into
    slot_half_y: float = 0.230

    # ---- Lift carriage ----------------------------------------------------
    # half_x 0.045 -> 0.090 m wide -> passes between the straps (inner faces
    # at +/-0.048) with 3 mm clearance a side.
    carriage_half: Tuple[float, float, float] = (0.045, 0.200, 0.012)
    lift_kp: float = 8000.0
    lift_kv: float = 300.0

    # ---- Magazine ---------------------------------------------------------
    # In-line fore/aft at y = 0 so the shuttle path never crosses the wheels
    # (pack |y| <= 0.225, tyres at |y| in [0.37, 0.43] -> 145 mm clear).
    # full_stow is BEHIND the berth (-X) and empty_ready AHEAD (+X): the rover
    # drives in over an empty full_stow rack and drives out over an empty_ready
    # rack it has just emptied, so it never straddles an occupied station.
    rack_x: float = 1.00                      # empty_ready +x, full_stow -x
    rack_half_x: float = 0.145
    rack_rail_y: float = 0.245
    rack_rail_half_y: float = 0.012
    rack_rail_half_z: float = 0.015           # top 0.030: 30 mm under the pack

    # ---- Furniture (all of it must stay under RF_FAN_Z) -------------------
    cabinet_pos: Tuple[float, float, float] = (-0.10, 0.92, 0.065)
    cabinet_half: Tuple[float, float, float] = (0.16, 0.12, 0.065)
    beacon_z: float = 0.143
    beacon_half_h: float = 0.012
    kerb_half_z: float = 0.020

    # ---- Approach / berth -------------------------------------------------
    approach_x: float = -1.50                 # B's (0,-1.5,0) rotated 90 deg
    berth_x: float = 0.0
    berth_y: float = 0.0

    # ---- derived ----------------------------------------------------------
    @property
    def pad_half_x(self) -> float:
        return 0.5 * self.pad_size_x

    @property
    def pad_half_y(self) -> float:
        return 0.5 * self.pad_size_y

    @property
    def pack_half(self) -> Tuple[float, float, float]:
        return (self.pack_size[0] / 2, self.pack_size[1] / 2, self.pack_size[2] / 2)

    @property
    def carriage_top_stowed(self) -> float:
        """World z of the carriage top face at stroke 0."""
        return 0.0

    @property
    def carriage_top_raised(self) -> float:
        """World z of the carriage top face at full stroke = 0.060."""
        return self.lift_stroke

    @property
    def rack_pack_z(self) -> float:
        """Pack centre height when resting on a rack."""
        return self.pad_top_z + self.pack_half[2]

    @property
    def strap_inner_x(self) -> float:
        return self.strap_x - self.strap_half_x


DOCK = DockLayout()

# ---- names the rest of the code addresses things by ------------------------
CARRIAGE_BODY = "dock_carriage"
CARRIAGE_GEOM = "dock_carriage_plate"
LIFT_JOINT = "dock_lift"
LIFT_ACTUATOR = "dock_lift_act"
PACK_ATTACH_SITE = "dock_pack_attach"
BERTH_SITE = "dock_berth"
APPROACH_SITE = "dock_approach"
EMPTY_READY_SITE = "dock_empty_ready"
FULL_STOW_SITE = "dock_full_stow"
PACK_FULL_BODY = "pack_full"
PACK_EMPTY_BODY = "pack_empty"
ROVER_BATTERY_GEOM = "battery"
ROVER_MOUNT_SITE = "battery_mount"
CHASSIS_BODY = "chassis"

DOCK_GEOM_PREFIXES = ("dock_", "pack_full", "pack_empty")

# ---- palette (sits alongside robot/rover.xml's hero palette) ---------------
C_PAD = (0.255, 0.265, 0.290, 1.0)
C_PAD_EDGE = (0.175, 0.185, 0.205, 1.0)
C_STRIPE = (0.93, 0.78, 0.16, 1.0)
C_CARRIAGE = (0.90, 0.42, 0.06, 1.0)
C_RACK_FULL = (0.46, 0.33, 0.20, 1.0)
C_RACK_EMPTY = (0.16, 0.42, 0.46, 1.0)
C_CABINET = (0.20, 0.215, 0.245, 1.0)
C_BEACON = (0.98, 0.62, 0.08, 1.0)
C_PACK_FULL = (0.95, 0.78, 0.10, 1.0)      # same hazard yellow as rover battery
C_PACK_EMPTY = (0.20, 0.80, 0.42, 1.0)     # fresh pack reads green


def _box(parent, name, pos, half, rgba, *, collide: bool, group: int = 0):
    """Static box. `collide=True` stamps the traverse arena's WB_BITS."""
    contype, conaffinity = WB_BITS if collide else (0, 0)
    return parent.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos),
        size=list(half),
        rgba=list(rgba),
        contype=contype,
        conaffinity=conaffinity,
        group=group,
        mass=0.0,
    )


def _hide_rover_pack(spec: mujoco.MjSpec) -> None:
    """Stop drawing the rover's own battery geom; a mocap twin replaces it.

    The geom keeps its 4.0 kg (mass is independent of `group`), so the rover's
    29.4 kg / CoM 0.1865 m stay exactly as measured in robot/SPEC.md.  Group 3
    is outside the renderer's default geomgroup mask, and contype/conaffinity
    are cleared so the rising carriage can never fight it.  The geom sits at
    z in [0.060, 0.115], far under the 0.180 m ray fan, so clearing its bits
    costs the perception model nothing.
    """
    for g in spec.geoms:
        if g.name == ROVER_BATTERY_GEOM:
            g.group = 3
            g.contype = 0
            g.conaffinity = 0
            return
    raise RuntimeError(f"rover geom {ROVER_BATTERY_GEOM!r} not found")


def _add_pad(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """Drive-over plate, built as four plates around a central lift well."""
    wb = spec.worldbody
    zc = L.pad_top_z - L.pad_thick / 2
    hz = L.pad_thick / 2

    fore_half_x = (L.pad_half_x - L.slot_half_x) / 2
    fore_cx = (L.pad_half_x + L.slot_half_x) / 2
    # fore / aft plates carry BOTH wheel contacts (|x| = 0.11 >= slot 0.060)
    _box(wb, "dock_pad_fore", (fore_cx, 0, zc), (fore_half_x, L.pad_half_y, hz),
         C_PAD, collide=True)
    _box(wb, "dock_pad_aft", (-fore_cx, 0, zc), (fore_half_x, L.pad_half_y, hz),
         C_PAD, collide=True)
    side_half_y = (L.pad_half_y - L.slot_half_y) / 2
    side_cy = (L.pad_half_y + L.slot_half_y) / 2
    _box(wb, "dock_pad_left", (0, side_cy, zc), (L.slot_half_x, side_half_y, hz),
         C_PAD, collide=True)
    _box(wb, "dock_pad_right", (0, -side_cy, zc), (L.slot_half_x, side_half_y, hz),
         C_PAD, collide=True)

    # Painted wheel lanes at y = +/-0.40 and a stop bar. Paint is not a
    # collider, so these are the one place contype 0 is physically correct.
    sz = L.pad_top_z + 0.0015
    for sgn, tag in ((1, "l"), (-1, "r")):
        _box(wb, f"dock_lane_{tag}", (0, sgn * L.track / 2, sz),
             (L.pad_half_x - 0.02, 0.018, 0.0015), C_STRIPE, collide=False, group=2)
    _box(wb, "dock_stopbar", (0.36, 0, sz),
         (0.022, L.pad_half_y - 0.02, 0.0015), C_STRIPE, collide=False, group=2)

    # Corner kerbs: real kerbs, so they collide. Clear of the panel edge
    # (y = 0.505) and of the y = 0 shuttle lane. 0.040 m tall.
    for sx in (1, -1):
        for sy in (1, -1):
            tag = f"{'f' if sx > 0 else 'a'}{'l' if sy > 0 else 'r'}"
            _box(wb, f"dock_kerb_{tag}",
                 (sx * 0.40, sy * 0.575, L.pad_top_z + L.kerb_half_z),
                 (0.11, 0.025, L.kerb_half_z), C_PAD_EDGE, collide=True)


def _add_lift(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """Prismatic carriage: top face travels z = 0.000 -> 0.060, no further."""
    hx, hy, hz = L.carriage_half
    body = spec.worldbody.add_body(name=CARRIAGE_BODY, pos=[0.0, 0.0, -hz])
    body.add_joint(
        name=LIFT_JOINT,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[0.0, L.lift_stroke],
        limited=True,
        damping=40.0,
    )
    # The carriage is deliberately non-colliding.  It is a kinematic jack: give
    # it contacts and a stiff position servo will lever the 29.4 kg rover off
    # its own wheels, which is exactly the artefact we must not put in front of
    # judges.  Pack capture is resolved geometrically in MuJoCoDockHardware.
    # It is NOT hidden from perception by this: its top never exceeds 0.060 m,
    # which is 0.120 m below the rangefinder fan, so it is out of the rays on
    # height alone - the property that actually matters.
    body.add_geom(
        name=CARRIAGE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, hz],
        rgba=list(C_CARRIAGE),
        contype=0,
        conaffinity=0,
        mass=2.0,
    )
    # pack_attach frame = carriage top face (B's /Dock/frames/pack_attach).
    body.add_site(
        name=PACK_ATTACH_SITE, pos=[0.0, 0.0, hz], size=[0.01, 0.01, 0.01], group=3
    )
    act = spec.add_actuator(
        name=LIFT_ACTUATOR, target=LIFT_JOINT, trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    act.set_to_position(kp=L.lift_kp, kv=L.lift_kv)
    act.ctrlrange = [0.0, L.lift_stroke]
    act.ctrllimited = 1


def _add_racks(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """Two in-line rack stations. Real structure, so they collide.

    The rover drives over both, with 0.030 m of clearance between the rails'
    0.030 m top and the 0.060 m pack underside.
    """
    wb = spec.worldbody
    rz = L.rack_rail_half_z
    for sgn, name, col in ((-1, "full_stow", C_RACK_FULL), (1, "empty_ready", C_RACK_EMPTY)):
        x = sgn * L.rack_x
        _box(wb, f"dock_rack_{name}_floor", (x, 0, L.pad_top_z / 2),
             (L.rack_half_x, L.rack_rail_y, max(L.pad_top_z / 2, 0.0005)),
             col, collide=True)
        for sy in (1, -1):
            _box(wb, f"dock_rack_{name}_rail_{'l' if sy > 0 else 'r'}",
                 (x, sy * L.rack_rail_y, rz),
                 (L.rack_half_x, L.rack_rail_half_y, rz), col, collide=True)
        _box(wb, f"dock_rack_{name}_stop", (x + sgn * (L.rack_half_x + 0.012), 0, rz),
             (0.012, L.rack_rail_y, rz), col, collide=True)


def _add_furniture(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """Control cabinet + beacon.

    Kept SQUAT ON PURPOSE.  A dock mast or signage would be the right thing
    visually and the wrong thing for the traverse policy: anything crossing
    z = 0.180 lands in the rangefinder fan as an obstacle, and the rover is
    supposed to drive onto this dock, not around it.  Beacon tops out at
    0.155 m - 25 mm of margin under the rays.
    """
    wb = spec.worldbody
    _box(wb, "dock_cabinet", L.cabinet_pos, L.cabinet_half, C_CABINET, collide=True)
    _box(wb, "dock_cabinet_face",
         (L.cabinet_pos[0], L.cabinet_pos[1] - L.cabinet_half[1] - 0.005, 0.085),
         (0.10, 0.006, 0.032), C_RACK_EMPTY, collide=False, group=2)
    wb.add_geom(
        name="dock_beacon",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[L.cabinet_pos[0], L.cabinet_pos[1], L.beacon_z],
        size=[0.030, L.beacon_half_h],
        rgba=list(C_BEACON),
        contype=WB_BITS[0],
        conaffinity=WB_BITS[1],
        group=0,
        mass=0.0,
    )


def _add_packs(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """Two mocap packs: the one the rover arrives with, and the fresh one.

    Mocap because the swap is a *scripted* mechanism (B's SPEC: "scripted - not
    learned"); driving them through `d.mocap_pos` keeps the rover's own
    dynamics untouched while the transfer stays frame-accurate.  contype 0
    because a mocap body is infinite-mass - a colliding one would jam the rover
    rather than be pushed.  Highest either pack ever reaches is 0.115 m (latched
    in the bay), so neither enters the rangefinder fan.
    """
    hx, hy, hz = L.pack_half
    full = spec.worldbody.add_body(name=PACK_FULL_BODY, pos=[0, 0, 0.0875], mocap=True)
    full.add_geom(
        name="pack_full_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, hz], rgba=list(C_PACK_FULL), contype=0, conaffinity=0,
    )
    empty = spec.worldbody.add_body(
        name=PACK_EMPTY_BODY, pos=[L.rack_x, 0, L.rack_pack_z], mocap=True
    )
    empty.add_geom(
        name="pack_empty_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, hz], rgba=list(C_PACK_EMPTY), contype=0, conaffinity=0,
    )


def _add_frames(spec: mujoco.MjSpec, L: DockLayout) -> None:
    """B's named frames (frames.py), rotated 90 deg into the +X convention."""
    wb = spec.worldbody
    for name, pos in (
        (APPROACH_SITE, (L.approach_x, 0.0, L.pad_top_z)),
        (BERTH_SITE, (L.berth_x, L.berth_y, L.pad_top_z)),
        (EMPTY_READY_SITE, (L.rack_x, 0.0, L.rack_pack_z)),
        (FULL_STOW_SITE, (-L.rack_x, 0.0, L.rack_pack_z)),
    ):
        wb.add_site(name=name, pos=list(pos), size=[0.01, 0.01, 0.01], group=3)


def _extend_keyframes(spec: mujoco.MjSpec) -> None:
    """The lift adds one qpos/ctrl slot; rover.xml's `home` key must grow."""
    for key in spec.keys:
        key.qpos = list(key.qpos) + [0.0]
        key.ctrl = list(key.ctrl) + [0.0]


def build_spec(rover_xml: Path | str = ROVER_XML, layout: DockLayout = DOCK) -> mujoco.MjSpec:
    """Load rover.xml read-only and graft the dock on in memory."""
    spec = mujoco.MjSpec.from_file(str(rover_xml))
    _hide_rover_pack(spec)
    _add_pad(spec, layout)
    _add_lift(spec, layout)
    _add_racks(spec, layout)
    _add_furniture(spec, layout)
    _add_packs(spec, layout)
    _add_frames(spec, layout)
    _extend_keyframes(spec)
    return spec


def build_model(
    rover_xml: Path | str = ROVER_XML, layout: DockLayout = DOCK
) -> mujoco.MjModel:
    """Compiled model containing rover + dock."""
    return build_spec(rover_xml, layout).compile()


# ---------------------------------------------------------------------------
# Assertions, not comments.
# ---------------------------------------------------------------------------
def dock_geom_ids(model: mujoco.MjModel) -> list[int]:
    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(DOCK_GEOM_PREFIXES):
            out.append(g)
    return out


def check_rangefinder_clearance(model: mujoco.MjModel, layout: DockLayout = DOCK) -> float:
    """Assert no dock geom can enter the 0.180 m rangefinder fan.

    Static geoms are measured from the compiled model; the two moving parts are
    bounded analytically (carriage top <= 0.060, a latched pack top <= 0.115).
    """
    L = layout
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    worst_name, worst_z = "", -1e9
    for g in dock_geom_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name in (CARRIAGE_GEOM, "pack_full_geom", "pack_empty_geom"):
            continue  # moving parts, bounded below
        top = float(data.geom_xpos[g][2] + model.geom_size[g][2])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            top = float(data.geom_xpos[g][2] + model.geom_size[g][1])
        if top > worst_z:
            worst_name, worst_z = name, top

    carriage_top = L.carriage_top_raised                      # 0.060
    pack_top_latched = L.pack_underside_z + L.pack_size[2]    # 0.115
    for label, z in (
        (worst_name, worst_z),
        (CARRIAGE_GEOM + " (full stroke)", carriage_top),
        ("pack (latched in bay)", pack_top_latched),
    ):
        assert z < RF_FAN_Z - RF_CLEARANCE, (
            f"{label} tops out at z={z:.4f}, inside the rangefinder fan "
            f"(rays at {RF_FAN_Z:.3f} m). It would read as a phantom obstacle."
        )
    return max(worst_z, carriage_top, pack_top_latched)


def check_collision_bits(model: mujoco.MjModel) -> int:
    """Every colliding dock geom must carry the traverse arena's WB_BITS."""
    n = 0
    for g in dock_geom_ids(model):
        ct, ca = int(model.geom_contype[g]), int(model.geom_conaffinity[g])
        if ct or ca:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            assert (ct, ca) == WB_BITS, f"{name}: contype/conaffinity {(ct, ca)} != {WB_BITS}"
            n += 1
    return n


def self_check(layout: DockLayout = DOCK) -> mujoco.MjModel:
    """Run with `python -m envs.swap.dock_mjcf`."""
    L = layout
    wheel_y, wheel_x = L.track / 2, L.wheelbase / 2
    assert abs((L.pad_half_y - wheel_y) - 0.10) < 1e-9, "wheels must sit 0.10 m inside"
    assert wheel_x > L.slot_half_x, "wheel contact falls into the lift well"
    assert L.carriage_half[0] < L.strap_inner_x, "carriage would hit the straps"
    assert L.carriage_top_raised == L.pack_underside_z == 0.060
    assert L.pack_size == (0.20, 0.45, 0.055)
    assert L.pack_half[1] < 0.37, "shuttle lane would cross the tyres"

    model = build_model(layout=layout)
    tallest = check_rangefinder_clearance(model, layout)
    n_bitted = check_collision_bits(model)

    print("dock_mjcf self_check OK")
    print(f"  pad            {L.pad_size_x:.2f} x {L.pad_size_y:.2f} m, top z={L.pad_top_z}")
    print(f"  wheel margin   {L.pad_half_y - wheel_y:.3f} m inside each lateral edge")
    print(f"  carriage       {2*L.carriage_half[0]:.3f} m in X vs strap gap "
          f"{2*L.strap_inner_x:.3f} m -> "
          f"{(L.strap_inner_x - L.carriage_half[0])*1000:.1f} mm/side")
    print(f"  lift stroke    {L.lift_stroke:.3f} m, top "
          f"{L.carriage_top_stowed:.3f} -> {L.carriage_top_raised:.3f}")
    print(f"  collision bits {n_bitted} dock geoms at (contype,conaffinity)={WB_BITS}")
    print(f"  tallest dock z {tallest:.3f} m vs rangefinder fan {RF_FAN_Z:.3f} m "
          f"-> {(RF_FAN_Z - tallest)*1000:.0f} mm clear")
    return model


if __name__ == "__main__":
    m = self_check()
    print(f"compiled rover+dock: nq={m.nq} nu={m.nu} nbody={m.nbody} "
          f"ngeom={m.ngeom} nmocap={m.nmocap}")
