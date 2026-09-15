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

`sync(dt)` is the one addition, and it is not part of the Protocol: the
controller only ever *requests*, and `sync` is where the request meets physics.
Call it once per sim step, before `mj_step`.

--------------------------------------------------------------------------
THE TWO THINGS THAT MAKE THIS LOOK RIGHT RATHER THAN TELEPORT
--------------------------------------------------------------------------
1. B's controller calls `detach_full_to_carriage()` on the FIRST tick of
   `lift_full` (in `_enter_lift_full`), i.e. while the carriage is still down.
   Taken literally the pack would drop 60 mm to meet it.  So the Protocol calls
   are treated as ARMING a transfer, not executing one: the pack changes hands
   only when the carriage has physically arrived (`_CAPTURE_EPS` = 4 mm).  The
   same rule runs in reverse for `latch_empty_to_vehicle`.  Every method stays
   idempotent, which it has to be - `_drive_hardware_for_phase()` re-issues them
   on every one of the ~2000 ticks in a phase.

2. `set_lift_height` is rate-limited (`lift_rate`, m/s) before it reaches the
   position servo, so the jack screws up and down instead of snapping.

--------------------------------------------------------------------------
LIFT STROKE: 0.060, NOT 0.12
--------------------------------------------------------------------------
`swap_battery.py` hard-codes `set_lift_height(0.12)` for "raise fully", which
came from the superseded 0.12 m stroke in `envs/swap/SPEC.md`.  robot/SPEC.md
LOCKS the pack underside at z = 0.060, and a 0.12 m stroke drives the carriage
into the chassis.  `swap_battery.py` is not ours to edit, so the legacy
constant is interpreted rather than obeyed:

    height == LEGACY_FULL_STROKE (0.12)  ->  full stroke, i.e. 0.060
    anything else                        ->  a literal height, clamped to 0.060

Both directions are safe: a caller that has been updated to ask for 0.060
directly gets 0.060, and nothing can ever command the carriage past the pack.

--------------------------------------------------------------------------
STRAP RESOLUTION
--------------------------------------------------------------------------
The retaining straps sit at x = +/-0.06, half-width 0.012, and hang to
z = 0.0575 - 2.5 mm below the pack.  Two separate clearances:

  * THE CARRIAGE goes BETWEEN them.  It is 0.090 m wide in X against a 0.096 m
    gap between the straps' inner faces, so it rises past z = 0.0575 with 3 mm
    of clearance a side and never touches them.  This is the resolution chosen
    (see `dock_mjcf.DockLayout.carriage_half`).
  * THE PACK cannot: it is 0.20 m wide and the straps wrap it.  That 2.5 mm
    overlap is a property of rover.xml's own geometry - the straps already
    overlap the rover's own pack by 2.5 mm at rest - so a swapped pack ends up
    in exactly the interference state the rover ships with, and nothing new is
    introduced.  In the real machine the straps unbuckle; here they are
    `class="visual"` geoms and carry no contacts at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import mujoco
import numpy as np

try:  # normal package import
    from ..dock_mjcf import (
        CARRIAGE_BODY,
        CHASSIS_BODY,
        DOCK,
        DockLayout,
        LIFT_ACTUATOR,
        PACK_ATTACH_SITE,
        PACK_EMPTY_BODY,
        PACK_FULL_BODY,
        ROVER_BATTERY_GEOM,
        ROVER_MOUNT_SITE,
    )
except ImportError:  # pragma: no cover - direct execution
    from envs.swap.dock_mjcf import (  # type: ignore
        CARRIAGE_BODY,
        CHASSIS_BODY,
        DOCK,
        DockLayout,
        LIFT_ACTUATOR,
        PACK_ATTACH_SITE,
        PACK_EMPTY_BODY,
        PACK_FULL_BODY,
        ROVER_BATTERY_GEOM,
        ROVER_MOUNT_SITE,
    )

LEGACY_FULL_STROKE = 0.12   # swap_battery.py's hard-coded "raise fully"

_CAPTURE_EPS = 0.004        # carriage counts as "arrived" within 4 mm
_STOWED_EPS = 0.004         # carriage counts as "down" within 4 mm

VEHICLE = "vehicle"
CARRIAGE = "carriage"
RACK_FULL = "rack_full"
RACK_EMPTY = "rack_empty"


@dataclass
class _Pack:
    """One mocap pack and whatever is currently holding it."""

    name: str
    body_id: int
    mocap_id: int
    carrier: str
    shuttle_from: Optional[np.ndarray] = None
    shuttle_to: str = ""
    shuttle_t: float = 0.0
    shuttle_T: float = 1.0

    @property
    def shuttling(self) -> bool:
        return self.shuttle_from is not None


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class MuJoCoDockHardware:
    """B's `DockHardware`, realised against a compiled rover+dock MjModel.

    Structurally typed against the Protocol in `swap_battery.py`: the six
    methods below are the whole contract.  `sync(dt)` and the `events` hook are
    additions for the simulation loop and are invisible to the controller.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        layout: DockLayout = DOCK,
        *,
        lift_rate: float = 0.060,
        shuttle_seconds: float = 0.85,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.m = model
        self.d = data
        self.L = layout
        self.lift_rate = float(lift_rate)
        self.shuttle_seconds = float(shuttle_seconds)
        self._on_event = on_event
        self.events: list[Tuple[float, str]] = []

        nid = mujoco.mj_name2id
        self._lift_act = nid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
        self._carriage_body = nid(model, mujoco.mjtObj.mjOBJ_BODY, CARRIAGE_BODY)
        self._attach_site = nid(model, mujoco.mjtObj.mjOBJ_SITE, PACK_ATTACH_SITE)
        self._chassis_body = nid(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS_BODY)
        self._mount_site = nid(model, mujoco.mjtObj.mjOBJ_SITE, ROVER_MOUNT_SITE)
        bat = nid(model, mujoco.mjtObj.mjOBJ_GEOM, ROVER_BATTERY_GEOM)
        for name, idx in (
            (LIFT_ACTUATOR, self._lift_act), (CARRIAGE_BODY, self._carriage_body),
            (PACK_ATTACH_SITE, self._attach_site), (CHASSIS_BODY, self._chassis_body),
            (ROVER_MOUNT_SITE, self._mount_site), (ROVER_BATTERY_GEOM, bat),
        ):
            if idx < 0:
                raise RuntimeError(f"model is missing {name!r}: build it with dock_mjcf.build_model()")
        # The rover's battery bay, read from the model rather than restated.
        self._bay_local = np.array(model.geom_pos[bat], dtype=float)

        self._packs = {
            "full": self._make_pack(PACK_FULL_BODY, VEHICLE),
            "empty": self._make_pack(PACK_EMPTY_BODY, RACK_EMPTY),
        }

        self._lift_target = 0.0     # metres, already normalised to the 0.060 stroke
        self._lift_cmd = 0.0
        self._detach_full_armed = False
        self._stow_full_armed = False
        self._stage_empty_armed = False
        self._latch_empty_armed = False
        self._carriage_cleared = False
        self.sync(0.0)

    # ---- introspection -----------------------------------------------------
    def _make_pack(self, body_name: str, carrier: str) -> _Pack:
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"model is missing mocap body {body_name!r}")
        return _Pack(body_name, bid, int(self.m.body_mocapid[bid]), carrier)

    @property
    def carriage_top(self) -> float:
        """World z of the carriage top face (B's `pack_attach` frame)."""
        return float(self.d.site_xpos[self._attach_site][2])

    @property
    def lift_height(self) -> float:
        """Achieved stroke, 0.000 .. 0.060."""
        return self.carriage_top - self.L.carriage_top_stowed

    @property
    def mount_underside_z(self) -> float:
        """World z of the rover's `battery_mount` site - the pack underside."""
        return float(self.d.site_xpos[self._mount_site][2])

    @property
    def full_pack_carrier(self) -> str:
        return self._packs["full"].carrier

    @property
    def empty_pack_carrier(self) -> str:
        return self._packs["empty"].carrier

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
        if not self._detach_full_armed:
            self._detach_full_armed = True
            self._carriage_cleared = False
            self._event(f"full pack unlatched from {vehicle_id} - awaiting carriage")

    def stow_full_pack(self) -> None:
        if not self._stow_full_armed:
            self._stow_full_armed = True
            self._event("full pack routed to full_stow")

    def stage_empty_pack(self) -> None:
        if not self._stage_empty_armed:
            self._stage_empty_armed = True
            self._event("empty pack staged from empty_ready")

    def latch_empty_to_vehicle(self, vehicle_id: str) -> None:
        if not self._latch_empty_armed:
            self._latch_empty_armed = True
            self._event(f"empty pack armed to latch onto {vehicle_id}")

    def clear_carriage(self) -> None:
        if not self._carriage_cleared:
            self._carriage_cleared = True
            self._detach_full_armed = False
            self._latch_empty_armed = False
            self._event("carriage cleared")

    # ======================================================================
    # Simulation-side: request meets physics.
    # ======================================================================
    def sync(self, dt: float) -> None:
        """Advance the mechanism by `dt`. Call once per step, before mj_step."""
        self._advance_lift(dt)
        self._resolve_transfers()
        for pack in self._packs.values():
            self._advance_shuttle(pack, dt)
            self._write_pose(pack)

    def _advance_lift(self, dt: float) -> None:
        step = self.lift_rate * max(dt, 0.0)
        err = self._lift_target - self._lift_cmd
        self._lift_cmd += min(max(err, -step), step) if step > 0 else 0.0
        self._lift_cmd = min(max(self._lift_cmd, 0.0), self.L.lift_stroke)
        self.d.ctrl[self._lift_act] = self._lift_cmd

    def _resolve_transfers(self) -> None:
        full, empty = self._packs["full"], self._packs["empty"]
        top = self.carriage_top

        # 1. lift_full: the carriage has to REACH the pack before it takes it.
        if self._detach_full_armed and full.carrier == VEHICLE and not full.shuttling:
            if top >= self.mount_underside_z - _CAPTURE_EPS:
                full.carrier = CARRIAGE
                self._event("full pack captured by carriage")

        # 2. stow_full: only once the carriage is back down.
        if (self._stow_full_armed and full.carrier == CARRIAGE and not full.shuttling
                and top <= self.L.carriage_top_stowed + _STOWED_EPS):
            self._start_shuttle(full, RACK_FULL)
            self._stow_full_armed = False

        # 3. offer_empty: shuttle the fresh pack onto a lowered carriage.
        if (self._stage_empty_armed and empty.carrier == RACK_EMPTY and not empty.shuttling
                and top <= self.L.carriage_top_stowed + _STOWED_EPS):
            self._start_shuttle(empty, CARRIAGE)
            self._stage_empty_armed = False

        # 4. latch_empty: mirror of (1) - the pack top must reach the bay.
        if self._latch_empty_armed and empty.carrier == CARRIAGE and not empty.shuttling:
            if top >= self.mount_underside_z - _CAPTURE_EPS:
                empty.carrier = VEHICLE
                self._event("empty pack latched to vehicle")

    def _start_shuttle(self, pack: _Pack, to_carrier: str) -> None:
        pack.shuttle_from = np.array(self.d.mocap_pos[pack.mocap_id], dtype=float)
        pack.shuttle_to = to_carrier
        pack.shuttle_t = 0.0
        pack.shuttle_T = self.shuttle_seconds
        self._event(f"{pack.name} shuttling -> {to_carrier}")

    def _advance_shuttle(self, pack: _Pack, dt: float) -> None:
        if not pack.shuttling:
            return
        pack.shuttle_t += max(dt, 0.0)
        if pack.shuttle_t >= pack.shuttle_T:
            pack.carrier = pack.shuttle_to
            pack.shuttle_from = None
            self._event(f"{pack.name} seated at {pack.carrier}")

    # ---- pose sources ------------------------------------------------------
    def _carrier_pose(self, carrier: str) -> Tuple[np.ndarray, np.ndarray]:
        hz = self.L.pack_half[2]
        if carrier == VEHICLE:
            # Ride the chassis exactly where rover.xml puts its own battery.
            R = np.array(self.d.xmat[self._chassis_body], dtype=float).reshape(3, 3)
            pos = np.array(self.d.xpos[self._chassis_body], dtype=float) + R @ self._bay_local
            quat = np.array(self.d.xquat[self._chassis_body], dtype=float)
            return pos, quat
        if carrier == CARRIAGE:
            c = np.array(self.d.xpos[self._carriage_body], dtype=float)
            return np.array([c[0], c[1], self.carriage_top + hz]), np.array([1.0, 0, 0, 0])
        sgn = -1.0 if carrier == RACK_FULL else 1.0
        return (np.array([sgn * self.L.rack_x, 0.0, self.L.rack_pack_z]),
                np.array([1.0, 0, 0, 0]))

    def _write_pose(self, pack: _Pack) -> None:
        pos, quat = self._carrier_pose(pack.shuttle_to if pack.shuttling else pack.carrier)
        if pack.shuttling:
            u = _smoothstep(pack.shuttle_t / max(pack.shuttle_T, 1e-9))
            pos = pack.shuttle_from + u * (pos - pack.shuttle_from)
            quat = np.array([1.0, 0, 0, 0])
        self.d.mocap_pos[pack.mocap_id] = pos
        self.d.mocap_quat[pack.mocap_id] = quat


__all__ = ["MuJoCoDockHardware", "LEGACY_FULL_STROKE"]
