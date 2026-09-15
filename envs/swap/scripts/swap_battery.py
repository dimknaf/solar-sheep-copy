"""Scripted drive-over battery swap — CONTRACTS.md swap_battery().

Runs without Isaac Sim: phase timers + FAILURE reasons.
When Isaac is available, inject a DockHardware adapter that drives the
prismatic lift and re-parents packs; the phase machine stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from .status import Status


class Phase(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    LIFT_FULL = "lift_full"
    STOW_FULL = "stow_full"
    OFFER_EMPTY = "offer_empty"
    LATCH_EMPTY = "latch_empty"
    RELEASE = "release"
    DONE = "done"
    FAILED = "failed"


# Durations from SPEC.md (seconds of sim time).
PHASE_DURATION = {
    Phase.LIFT_FULL: 2.0,
    Phase.STOW_FULL: 2.0,
    Phase.OFFER_EMPTY: 2.0,
    Phase.LATCH_EMPTY: 2.5,
    Phase.RELEASE: 1.5,
}

APPROACH_TIMEOUT_S = 30.0
XY_TOL_M = 0.08
YAW_TOL_RAD = 0.157  # ~9°
HEIGHT_GAP_TOL_M = 0.03


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    z_mount: float = 0.16


@dataclass
class SwapRequest:
    vehicle_id: str
    dock_id: str = "dock0"
    vehicle_pose: Optional[Pose2D] = None
    berth_pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0, 0.16))
    empty_pack_count: int = 1
    force_aligned: bool = False


@dataclass
class SwapResult:
    status: Status
    phase: Phase
    reason: str = ""
    vehicle_id: str = ""
    dock_id: str = ""


class DockHardware(Protocol):
    """Optional Isaac / Omniverse side effects."""

    def set_lift_height(self, height_m: float) -> None: ...
    def detach_full_to_carriage(self, vehicle_id: str) -> None: ...
    def stow_full_pack(self) -> None: ...
    def stage_empty_pack(self) -> None: ...
    def latch_empty_to_vehicle(self, vehicle_id: str) -> None: ...
    def clear_carriage(self) -> None: ...


class NullHardware:
    def set_lift_height(self, height_m: float) -> None:
        return None

    def detach_full_to_carriage(self, vehicle_id: str) -> None:
        return None

    def stow_full_pack(self) -> None:
        return None

    def stage_empty_pack(self) -> None:
        return None

    def latch_empty_to_vehicle(self, vehicle_id: str) -> None:
        return None

    def clear_carriage(self) -> None:
        return None


def _aligned(vehicle: Pose2D, berth: Pose2D) -> bool:
    dx = vehicle.x - berth.x
    dy = vehicle.y - berth.y
    if (dx * dx + dy * dy) ** 0.5 > XY_TOL_M:
        return False
    yaw_err = abs(vehicle.yaw - berth.yaw)
    while yaw_err > 3.141592653589793:
        yaw_err -= 2 * 3.141592653589793
    yaw_err = abs(yaw_err)
    if yaw_err > YAW_TOL_RAD:
        return False
    if abs(vehicle.z_mount - berth.z_mount) > HEIGHT_GAP_TOL_M:
        return False
    return True


class SwapBatteryController:
    """One dock, one active swap. tick() every sim step."""

    def __init__(self, hardware: Optional[DockHardware] = None) -> None:
        self._hw: DockHardware = hardware or NullHardware()
        self._phase = Phase.IDLE
        self._t = 0.0
        self._vehicle_id = ""
        self._dock_id = ""
        self._empty_left = 0
        self._reason = ""

    @property
    def phase(self) -> Phase:
        return self._phase

    def busy(self) -> bool:
        return self._phase not in (Phase.IDLE, Phase.DONE, Phase.FAILED)

    def reset(self) -> None:
        self._phase = Phase.IDLE
        self._t = 0.0
        self._vehicle_id = ""
        self._dock_id = ""
        self._empty_left = 0
        self._reason = ""

    def start(self, req: SwapRequest) -> SwapResult:
        if self.busy():
            return SwapResult(
                status=Status.FAILURE,
                phase=self._phase,
                reason="busy",
                vehicle_id=req.vehicle_id,
                dock_id=req.dock_id,
            )

        self.reset()
        self._vehicle_id = req.vehicle_id
        self._dock_id = req.dock_id
        self._empty_left = req.empty_pack_count
        self._phase = Phase.APPROACH
        self._t = 0.0

        if req.force_aligned or (
            req.vehicle_pose is not None and _aligned(req.vehicle_pose, req.berth_pose)
        ):
            return self._enter_lift_full()

        if req.vehicle_pose is not None:
            return self._fail("misaligned")

        # No pose yet — wait in approach until a later tick supplies alignment
        # via tick(aligned=True) or we timeout.
        return self._running()

    def tick(
        self,
        dt: float,
        *,
        aligned: Optional[bool] = None,
        empty_pack_count: Optional[int] = None,
    ) -> SwapResult:
        if self._phase == Phase.DONE:
            return SwapResult(
                status=Status.SUCCESS,
                phase=self._phase,
                vehicle_id=self._vehicle_id,
                dock_id=self._dock_id,
            )
        if self._phase == Phase.FAILED:
            return SwapResult(
                status=Status.FAILURE,
                phase=self._phase,
                reason=self._reason,
                vehicle_id=self._vehicle_id,
                dock_id=self._dock_id,
            )
        if self._phase == Phase.IDLE:
            return SwapResult(status=Status.SUCCESS, phase=self._phase, reason="idle")

        if empty_pack_count is not None:
            self._empty_left = empty_pack_count

        self._t += dt

        if self._phase == Phase.APPROACH:
            if aligned is True:
                return self._enter_lift_full()
            if aligned is False:
                return self._fail("misaligned")
            if self._t >= APPROACH_TIMEOUT_S:
                return self._fail("timeout")
            return self._running()

        duration = PHASE_DURATION.get(self._phase)
        if duration is None:
            return self._fail("timeout")

        self._drive_hardware_for_phase()

        if self._t < duration:
            return self._running()

        return self._advance_phase()

    def _enter_lift_full(self) -> SwapResult:
        self._phase = Phase.LIFT_FULL
        self._t = 0.0
        self._hw.set_lift_height(0.12)
        self._hw.detach_full_to_carriage(self._vehicle_id)
        return self._running()

    def _advance_phase(self) -> SwapResult:
        order = [
            Phase.LIFT_FULL,
            Phase.STOW_FULL,
            Phase.OFFER_EMPTY,
            Phase.LATCH_EMPTY,
            Phase.RELEASE,
        ]
        idx = order.index(self._phase)
        nxt = order[idx + 1] if idx + 1 < len(order) else Phase.DONE

        if nxt == Phase.OFFER_EMPTY and self._empty_left <= 0:
            return self._fail("no_empty_pack")

        if nxt == Phase.DONE:
            self._hw.clear_carriage()
            self._phase = Phase.DONE
            self._t = 0.0
            self._reason = ""
            return SwapResult(
                status=Status.SUCCESS,
                phase=self._phase,
                vehicle_id=self._vehicle_id,
                dock_id=self._dock_id,
            )

        self._phase = nxt
        self._t = 0.0
        self._drive_hardware_for_phase()

        if self._phase == Phase.OFFER_EMPTY:
            self._hw.stage_empty_pack()
            self._empty_left = max(0, self._empty_left - 1)

        return self._running()

    def _drive_hardware_for_phase(self) -> None:
        if self._phase == Phase.LIFT_FULL:
            self._hw.set_lift_height(0.12)
            self._hw.detach_full_to_carriage(self._vehicle_id)
        elif self._phase == Phase.STOW_FULL:
            self._hw.set_lift_height(0.0)
            self._hw.stow_full_pack()
        elif self._phase == Phase.OFFER_EMPTY:
            self._hw.stage_empty_pack()
        elif self._phase == Phase.LATCH_EMPTY:
            self._hw.set_lift_height(0.12)
            self._hw.latch_empty_to_vehicle(self._vehicle_id)
        elif self._phase == Phase.RELEASE:
            self._hw.set_lift_height(0.0)
            self._hw.clear_carriage()

    def _running(self) -> SwapResult:
        return SwapResult(
            status=Status.RUNNING,
            phase=self._phase,
            vehicle_id=self._vehicle_id,
            dock_id=self._dock_id,
        )

    def _fail(self, reason: str) -> SwapResult:
        self._phase = Phase.FAILED
        self._reason = reason
        self._hw.set_lift_height(0.0)
        return SwapResult(
            status=Status.FAILURE,
            phase=self._phase,
            reason=reason,
            vehicle_id=self._vehicle_id,
            dock_id=self._dock_id,
        )


def swap_battery(
    controller: SwapBatteryController,
    vehicle_id: str,
    dock: str,
    *,
    dt: float = 0.0,
    request: Optional[SwapRequest] = None,
    aligned: Optional[bool] = None,
    empty_pack_count: Optional[int] = None,
) -> SwapResult:
    """CONTRACTS.md entry point — call every tick.

    If the dock is idle (or finished a prior job), starts a new swap for
    ``vehicle_id``. Otherwise advances the active swap by ``dt``.
    """
    if controller.phase in (Phase.DONE, Phase.FAILED):
        controller.reset()

    if not controller.busy():
        req = request or SwapRequest(vehicle_id=vehicle_id, dock_id=dock)
        started = controller.start(req)
        if started.status != Status.RUNNING or dt <= 0.0:
            return started

    return controller.tick(dt, aligned=aligned, empty_pack_count=empty_pack_count)
