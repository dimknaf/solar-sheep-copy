"""Pure-Python checks for swap phases — no Isaac required.

Run from repo root:
  python -m envs.swap.tests.test_swap_phases
or:
  python envs/swap/tests/test_swap_phases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.swap.scripts.status import Status
from envs.swap.scripts.swap_battery import (
    PHASE_DURATION,
    Phase,
    Pose2D,
    SwapBatteryController,
    SwapRequest,
)


def _drain(ctrl: SwapBatteryController, dt: float = 0.5) -> Status:
    last = Status.RUNNING
    guard = 0
    while last == Status.RUNNING and guard < 10_000:
        last = ctrl.tick(dt, aligned=True).status
        guard += 1
    return last


def test_happy_path_success() -> None:
    ctrl = SwapBatteryController()
    r = ctrl.start(
        SwapRequest(vehicle_id="sheep0", force_aligned=True, empty_pack_count=1)
    )
    assert r.status == Status.RUNNING
    assert _drain(ctrl) == Status.SUCCESS
    assert ctrl.phase == Phase.DONE


def test_misaligned_fails() -> None:
    ctrl = SwapBatteryController()
    r = ctrl.start(
        SwapRequest(
            vehicle_id="sheep0",
            vehicle_pose=Pose2D(1.0, 0.0, 0.0),
            berth_pose=Pose2D(0.0, 0.0, 0.0),
            empty_pack_count=1,
        )
    )
    assert r.status == Status.FAILURE
    assert r.reason == "misaligned"


def test_no_empty_pack_fails() -> None:
    ctrl = SwapBatteryController()
    ctrl.start(SwapRequest(vehicle_id="sheep0", force_aligned=True, empty_pack_count=0))
    # Run through lift + stow into offer_empty
    t = 0.0
    status = Status.RUNNING
    reason = ""
    while status == Status.RUNNING:
        res = ctrl.tick(0.5)
        status = res.status
        reason = res.reason
        t += 0.5
        assert t < 60.0
    assert status == Status.FAILURE
    assert reason == "no_empty_pack"


def test_busy_second_vehicle() -> None:
    ctrl = SwapBatteryController()
    ctrl.start(SwapRequest(vehicle_id="sheep0", force_aligned=True))
    r2 = ctrl.start(SwapRequest(vehicle_id="sheep1", force_aligned=True))
    assert r2.status == Status.FAILURE
    assert r2.reason == "busy"


def test_approach_timeout() -> None:
    ctrl = SwapBatteryController()
    ctrl.start(SwapRequest(vehicle_id="sheep0"))  # no pose → wait
    res = None
    for _ in range(61):
        res = ctrl.tick(0.5)  # 30.5 s
    assert res is not None
    assert res.status == Status.FAILURE
    assert res.reason == "timeout"


def test_phase_budget_matches_spec() -> None:
    total = sum(PHASE_DURATION.values())
    assert abs(total - 10.0) < 1e-9


def main() -> None:
    tests = [
        test_happy_path_success,
        test_misaligned_fails,
        test_no_empty_pack_fails,
        test_busy_second_vehicle,
        test_approach_timeout,
        test_phase_budget_matches_spec,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"FRAME SWAP LOGIC OK  ({len(tests)} tests)")


if __name__ == "__main__":
    main()
