"""MuJoCo realisation of B's `DockHardware` Protocol.

`swap_battery.py` is untouched.  Its phase machine is engine-independent - it
drives a Protocol with six methods and a `NullHardware` stand-in - so bringing
the swap into MuJoCo means implementing that Protocol once, here, exactly as
declared:

    set_lift_height(height_m)            -> commanded carriage height
    detach_full_to_carriage(vehicle_id)  -> full pack: rover  -> carriage
    stow_full_pack()                     -> full pack: carriage -> full_stow
    stage_empty_pack()                   -> empty pack: empty_ready -> carriage
    latch_empty_to_vehicle(vehicle_id)   -> empty pack: carriage -> rover
    clear_carriage()                     -> carriage released, lift stowed

`sync(dt)` is the one addition and is not part of the Protocol: the controller
only ever *requests*, and `sync` is where the request meets physics.  Call it
once per sim step, before `mj_step`.

--------------------------------------------------------------------------
WHAT MAKES THIS A SWAP AND NOT AN ANIMATION
--------------------------------------------------------------------------
The pack is a real body (`pack` in robot/rover.xml, and `pack_spare` added by
`dock_mjcf`), each held to the chassis by a **weld equality** that this class
switches with `data.eq_active`.  Detaching is one byte going to 0 and the pack
is genuinely no longer part of the vehicle; latching is one byte going to 1 and
the machine drives away carrying a different rigid body.  Nothing pretends.

Between those two events the dock owns the pack and moves it KINEMATICALLY
(free-joint qpos written, qvel held at zero).  That is not a dodge - B's SPEC
says the mechanism is "scripted - not learned", a real screw jack and shuttle
are position-controlled, and modelling a conveyor as a contact problem buys
nothing but jitter.  Ownership is explicit in `_Pack.mode`:

    "weld"   the weld holds it; physics owns it; this class does not write to it
    "pinned" the dock holds it; qpos/qvel written every step
    "free"   nobody holds it; it rests on the transfer deck under gravity

--------------------------------------------------------------------------
THE TWO THINGS THAT MAKE IT LOOK RIGHT RATHER THAN TELEPORT
--------------------------------------------------------------------------
1. B's controller calls `detach_full_to_carriage()` on the FIRST tick of
   `lift_full` (in `_enter_lift_full`), i.e. while the carriage is still down.
   Taken literally the pack would drop 60 mm to meet it.  So Protocol calls ARM
   a transfer rather than executing one: the weld opens only when the carriage
   has physically arrived (`_CAPTURE_EPS` = 4 mm).  The same rule runs in
   reverse for `latch_empty_to_vehicle`.  Every method stays idempotent, which
   it must be - `_drive_hardware_for_phase()` re-issues them on every one of the
   ~2000 ticks in a phase.
2. `set_lift_height` is rate-limited (`lift_rate`, m/s) before it reaches the
   position servo, so the jack screws up and down instead of snapping.

--------------------------------------------------------------------------
LIFT STROKE: 0.060, NOT 0.12
--------------------------------------------------------------------------
`swap_battery.py` hard-codes `set_lift_height(0.12)` for "raise fully", from the
superseded 0.12 m stroke in `envs/swap/SPEC.md`.  robot/SPEC.md LOCKS the pack
underside at z = 0.060, and a 0.12 m stroke drives the carriage into the
chassis.  `swap_battery.py` is not ours to edit, so the legacy constant is
interpreted rather than obeyed:

    height == LEGACY_FULL_STROKE (0.12)  ->  full stroke, i.e. 0.060
    anything else                        ->  a literal height, clamped to 0.060

Both directions are safe: a caller updated to ask for 0.060 gets 0.060, and
nothing can command the carriage past the pack.

--------------------------------------------------------------------------
STRAP RESOLUTION
--------------------------------------------------------------------------
The retaining straps travel with the pack, sit at x = +/-0.06 with half-width
0.012, and hang to z = 0.0575 - 2.5 mm below the pack.  They, not the pack, are
what a carriage rising flat meets first.  Resolution taken: **the carriage goes
BETWEEN them** - 0.090 m wide in X against the 0.096 m gap between their inner
faces, so it rises past 0.0575 with 3 mm of clearance a side and never touches
them (`dock_mjcf.DockLayout.carriage_half`).  At |x| <= 0.045 it also clears the
visual stub axles at x = +/-0.11.  The 2.5 mm is therefore never spent: no
allowance is made for it anywhere, because nothing ever reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mujoco
import numpy as np

try:  # normal package import
    from ..dock_mjcf import (
        CARRIAGE_BODY, CHASSIS_BODY, DOCK, DockLayout, LIFT_ACTUATOR,
        PACK_ATTACH_SITE, PACK_LATCH_EQ, ROVER_MOUNT_SITE, ROVER_PACK_BODY,
        SPARE_LATCH_EQ, SPARE_PACK_BODY,
    )
except ImportError:  # pragma: no cover - direct execution
    from envs.swap.dock_mjcf import (  # type: ignore
        CARRIAGE_BODY, CHASSIS_BODY, DOCK, DockLayout, LIFT_ACTUATOR,
        PACK_ATTACH_SITE, PACK_LATCH_EQ, ROVER_MOUNT_SITE, ROVER_PACK_BODY,
        SPARE_LATCH_EQ, SPARE_PACK_BODY,
    )

LEGACY_FULL_STROKE = 0.12   # swap_battery.py's hard-coded "raise fully"

_CAPTURE_EPS = 0.004        # carriage counts as "arrived" within 4 mm
_STOWED_EPS = 0.004         # carriage counts as "down" within 4 mm
_PIN_FLOAT = 0.0005         # hold a pinned pack 0.5 mm off the deck, so a
                            # zero-gap contact cannot chatter against the pin

VEHICLE = "vehicle"
CARRIAGE = "carriage"
RACK_FULL = "full_stow"
RACK_EMPTY = "empty_ready"

WELD, PINNED, FREE = "weld", "pinned", "free"


@dataclass
class _Pack:
    """One pack body, its weld, and whatever is currently holding it."""

    name: str
    body_id: int
    qadr: int
    dofadr: int
    eq_id: int
    mode: str
    carrier: str
    shuttle_from: Optional[np.ndarray] = None
    shuttle_to: str = ""
    shuttle_t: float = 0.0
    shuttle_T: float = 1.0
    release_on_arrival: bool = False

    @property
    def shuttling(self) -> bool:
        return self.shuttle_from is not None


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class MuJoCoDockHardware:
    """B's `DockHardware`, realised against a compiled rover+dock MjModel.

    Structurally typed against the Protocol in `swap_battery.py`: the six
    methods are the whole contract.  `sync(dt)` and `on_event` are additions for
    the simulation loop and are invisible to the controller.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        layout: DockLayout = DOCK,
        *,
        lift_rate: float = 0.055,
        shuttle_seconds: float = 0.90,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.m, self.d, self.L = model, data, layout
        self.lift_rate = float(lift_rate)
        self.shuttle_seconds = float(shuttle_seconds)
        self._on_event = on_event
        self.events: list = []

        nid = mujoco.mj_name2id
        self._lift_act = nid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
        self._carriage_body = nid(model, mujoco.mjtObj.mjOBJ_BODY, CARRIAGE_BODY)
        self._attach_site = nid(model, mujoco.mjtObj.mjOBJ_SITE, PACK_ATTACH_SITE)
        self._chassis_body = nid(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS_BODY)
        self._mount_site = nid(model, mujoco.mjtObj.mjOBJ_SITE, ROVER_MOUNT_SITE)
        for label, idx in (
            (LIFT_ACTUATOR, self._lift_act), (CARRIAGE_BODY, self._carriage_body),
            (PACK_ATTACH_SITE, self._attach_site), (CHASSIS_BODY, self._chassis_body),
            (ROVER_MOUNT_SITE, self._mount_site),
        ):
            if idx < 0:
                raise RuntimeError(
                    f"model is missing {label!r}: build it with dock_mjcf.build_model()")

        self.full = self._make_pack(ROVER_PACK_BODY, PACK_LATCH_EQ, WELD, VEHICLE)
        self.spare = self._make_pack(SPARE_PACK_BODY, SPARE_LATCH_EQ, FREE, RACK_EMPTY)
        self._packs = (self.full, self.spare)
        self.d.eq_active[self.full.eq_id] = 1
        self.d.eq_active[self.spare.eq_id] = 0

        self._lift_target = 0.0     # metres, already normalised to the stroke
        self._lift_cmd = 0.0
        self._detach_armed = False
        self._stow_armed = False
        self._stage_armed = False
        self._latch_armed = False
        self._carriage_cleared = False
        self.sync(0.0)

    # ---- wiring ------------------------------------------------------------
    def _make_pack(self, body: str, eq: str, mode: str, carrier: str) -> _Pack:
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        eid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
        if bid < 0 or eid < 0:
            raise RuntimeError(f"model is missing body {body!r} or equality {eq!r}")
        jid = int(self.m.body_jntadr[bid])
        if jid < 0 or self.m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"body {body!r} must be on a free joint to be swappable")
        return _Pack(body, bid, int(self.m.jnt_qposadr[jid]),
                     int(self.m.jnt_dofadr[jid]), eid, mode, carrier)

    # ---- introspection -----------------------------------------------------
    @property
    def carriage_top(self) -> float:
        """World z of the carriage top face (B's `pack_attach` frame)."""
        return float(self.d.site_xpos[self._attach_site][2])

    @property
    def lift_height(self) -> float:
        """Achieved stroke, 0.000 .. 0.060."""
        return self.carriage_top - self.L.carriage_top_stowed

    @property
    def lift_command(self) -> float:
        return float(self.d.ctrl[self._lift_act])

    @property
    def mount_underside_z(self) -> float:
        """World z of the rover's `battery_mount` site - the pack underside."""
        return float(self.d.site_xpos[self._mount_site][2])

    def latched_pack(self) -> Optional[str]:
        for p in self._packs:
            if p.mode == WELD and int(self.d.eq_active[p.eq_id]):
                return p.name
        return None

    def pack_pos(self, pack: _Pack) -> np.ndarray:
        return np.array(self.d.xpos[pack.body_id], dtype=float)

    def _event(self, text: str) -> None:
        self.events.append((float(self.d.time), text))
        if self._on_event is not None:
            self._on_event(text)

    # ======================================================================
    # DockHardware Protocol - all six methods, all idempotent.
    # ======================================================================
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
            self._event(f"pack_latch arming release from {vehicle_id}")

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
            self._event(f"spare_latch arming onto {vehicle_id}")

    def clear_carriage(self) -> None:
        if not self._carriage_cleared:
            self._carriage_cleared = True
            self._detach_armed = False
            self._latch_armed = False
            self._event("carriage cleared")

    # ======================================================================
    # Simulation-side: request meets physics.
    # ======================================================================
    def sync(self, dt: float) -> None:
        """Advance the mechanism by `dt`. Call once per step, before mj_step."""
        self._advance_lift(dt)
        self._resolve_transfers()
        for pack in self._packs:
            self._advance_shuttle(pack, dt)
            if pack.mode == PINNED:
                self._pin(pack)

    def _advance_lift(self, dt: float) -> None:
        step = self.lift_rate * max(dt, 0.0)
        if step > 0.0:
            err = self._lift_target - self._lift_cmd
            self._lift_cmd += min(max(err, -step), step)
        self._lift_cmd = min(max(self._lift_cmd, 0.0), self.L.lift_stroke)
        self.d.ctrl[self._lift_act] = self._lift_cmd

    def _resolve_transfers(self) -> None:
        top = self.carriage_top

        # 1. lift_full - the carriage must REACH the pack before the weld opens.
        if (self._detach_armed and self.full.mode == WELD
                and top >= self.mount_underside_z - _CAPTURE_EPS):
            self.d.eq_active[self.full.eq_id] = 0
            self.full.mode, self.full.carrier = PINNED, CARRIAGE
            self._event("pack_latch RELEASED - full pack is on the carriage")

        # 2. stow_full - only once the carriage has come back down.
        if (self._stow_armed and self.full.carrier == CARRIAGE
                and not self.full.shuttling
                and top <= self.L.carriage_top_stowed + _STOWED_EPS):
            self._start_shuttle(self.full, RACK_FULL, release_on_arrival=True)
            self._stow_armed = False

        # 3. offer_empty - shuttle the fresh pack onto a lowered carriage.
        if (self._stage_armed and self.spare.carrier == RACK_EMPTY
                and not self.spare.shuttling
                and top <= self.L.carriage_top_stowed + _STOWED_EPS):
            self.spare.mode = PINNED
            self._start_shuttle(self.spare, CARRIAGE)
            self._stage_armed = False

        # 4. latch_empty - mirror of (1): seat it, then close the weld.
        if (self._latch_armed and self.spare.carrier == CARRIAGE
                and not self.spare.shuttling and self.spare.mode == PINNED
                and top >= self.mount_underside_z - _CAPTURE_EPS):
            pos, quat = self._carrier_pose(VEHICLE)
            self._write(self.spare, pos, quat)
            self.d.eq_active[self.spare.eq_id] = 1
            self.spare.mode, self.spare.carrier = WELD, VEHICLE
            self._event("spare_latch CLOSED - fresh pack is on the vehicle")

    def _start_shuttle(self, pack: _Pack, to_carrier: str, *,
                       release_on_arrival: bool = False) -> None:
        pack.shuttle_from = np.array(self.d.qpos[pack.qadr:pack.qadr + 3], dtype=float)
        pack.shuttle_to = to_carrier
        pack.shuttle_t = 0.0
        pack.shuttle_T = self.shuttle_seconds
        pack.release_on_arrival = release_on_arrival
        pack.mode = PINNED
        self._event(f"{pack.name} shuttling -> {to_carrier}")

    def _advance_shuttle(self, pack: _Pack, dt: float) -> None:
        if not pack.shuttling:
            return
        pack.shuttle_t += max(dt, 0.0)
        if pack.shuttle_t >= pack.shuttle_T:
            pack.carrier = pack.shuttle_to
            pack.shuttle_from = None
            if pack.release_on_arrival:
                self._pin(pack)                 # seat it exactly, then let go
                pack.mode = FREE
                pack.release_on_arrival = False
                self._event(f"{pack.name} set down at {pack.carrier}")
            else:
                self._event(f"{pack.name} seated at {pack.carrier}")

    # ---- pose sources ------------------------------------------------------
    def _carrier_pose(self, carrier: str):
        hz = self.L.pack_half[2]
        if carrier == VEHICLE:
            R = np.array(self.d.xmat[self._chassis_body], dtype=float).reshape(3, 3)
            bay = np.array([0.0, 0.0, self.L.pack_bay_local_z])
            return (np.array(self.d.xpos[self._chassis_body], dtype=float) + R @ bay,
                    np.array(self.d.xquat[self._chassis_body], dtype=float))
        if carrier == CARRIAGE:
            # The pack is never dragged below the transfer deck: the carriage
            # sets it down there and keeps retracting on its own.
            z = max(self.carriage_top, self.L.deck_top_z + _PIN_FLOAT)
            c = self.d.xpos[self._carriage_body]
            return np.array([float(c[0]), float(c[1]), z + hz]), np.array([1.0, 0, 0, 0])
        sgn = -1.0 if carrier == RACK_FULL else 1.0
        return (np.array([sgn * self.L.station_x, 0.0,
                          self.L.deck_top_z + _PIN_FLOAT + hz]),
                np.array([1.0, 0, 0, 0]))

    def _pin(self, pack: _Pack) -> None:
        pos, quat = self._carrier_pose(
            pack.shuttle_to if pack.shuttling else pack.carrier)
        if pack.shuttling:
            u = _smoothstep(pack.shuttle_t / max(pack.shuttle_T, 1e-9))
            pos = pack.shuttle_from + u * (pos - pack.shuttle_from)
            quat = np.array([1.0, 0, 0, 0])
        self._write(pack, pos, quat)

    def _write(self, pack: _Pack, pos, quat) -> None:
        self.d.qpos[pack.qadr:pack.qadr + 3] = pos
        self.d.qpos[pack.qadr + 3:pack.qadr + 7] = quat
        self.d.qvel[pack.dofadr:pack.dofadr + 6] = 0.0


__all__ = ["MuJoCoDockHardware", "LEGACY_FULL_STROKE", "VEHICLE", "CARRIAGE",
           "RACK_FULL", "RACK_EMPTY", "WELD", "PINNED", "FREE"]
