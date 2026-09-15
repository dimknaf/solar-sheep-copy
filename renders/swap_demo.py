#!/usr/bin/env python3
"""renders/swap_demo.py -- one continuous shot of a real battery swap.

    approach  ->  berth on the pad  ->  lift takes the pack  ->  swap  ->  drive away

The phases in the video are B's phase machine, not an animation.  `renders/`
owns the camera and the drive commands; every state transition comes out of
`SwapBatteryController` in `envs/swap/scripts/swap_battery.py`, driven through
the `DockHardware` Protocol by `MuJoCoDockHardware`.  The controller is started
with NO pose, so it genuinely sits in `Phase.APPROACH` and waits; it is handed
`aligned=True` only once the rover has actually stopped inside B's own published
berth tolerances (`XY_TOL_M`, `YAW_TOL_RAD`, `HEIGHT_GAP_TOL_M`), measured off
the rover's `battery_mount` site.

robot/rover.xml is never written to.  The dock is grafted on in memory by
`envs.swap.dock_mjcf`.

Physics runs at rover.xml's native 1 ms timestep with rover.xml's own solver
settings.  No sub-stepping: the traverse workstream measured that at 4 ms with
the cheap 2/6 solver a full-command skid-steer spin diverges and flips the
rover, and there is no reason to take that risk for a 25 s clip.

Usage:
    python renders/swap_demo.py                # writes renders/swap_demo.mp4
    python renders/swap_demo.py --no-video     # assertions only, ~10 s
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from envs.swap.dock_mjcf import (  # noqa: E402
    DOCK,
    RF_FAN_Z,
    build_model,
    check_collision_bits,
    check_rangefinder_clearance,
)
from envs.swap.scripts.mujoco_hardware import MuJoCoDockHardware  # noqa: E402
from envs.swap.scripts.status import Status  # noqa: E402
from envs.swap.scripts.swap_battery import (  # noqa: E402
    HEIGHT_GAP_TOL_M,
    XY_TOL_M,
    YAW_TOL_RAD,
    Phase,
    SwapBatteryController,
    SwapRequest,
)

# ---- shot -----------------------------------------------------------------
WIDTH, HEIGHT, FPS = 1280, 720, 30
START_X = -1.75            # behind the dock, clear of the full_stow rack
APPROACH_MAX_S = 9.0
SETTLE_S = 1.2
DEPART_S = 5.0
SWAP_MAX_S = 20.0

# ---- drive ----------------------------------------------------------------
WHEEL_R = 0.090
V_CRUISE = 0.279           # measured top speed, robot/SPEC.md
KP_X = 1.9                 # m/s per m of range
KP_YAW = 2.2
KP_Y = 2.6
RAMP_S = 1.0               # SPEC: ramp ctrl over ~1 s, no real gearmotor snaps


def _quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Camera:
    """Free camera that eases toward a per-stage target. One continuous move."""

    TAU = 0.55

    def __init__(self) -> None:
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.state = np.array([START_X * 0.5, 0.0, 0.16, 3.4, 132.0, -13.0])
        self._apply()

    @staticmethod
    def target(stage: str, phase: Phase, rover_x: float) -> np.ndarray:
        if stage == "approach":
            return np.array([0.55 * rover_x, 0.0, 0.16, 3.30, 132.0, -13.0])
        if stage == "depart":
            return np.array([0.70 * rover_x, 0.0, 0.18, 3.25, 146.0, -15.0])
        # stage == "swap"/"settle": frame it off B's actual phase.
        by_phase = {
            Phase.APPROACH: ([0.00, 0.0, 0.14], 2.05, 118.0, -10.0),
            Phase.LIFT_FULL: ([0.00, 0.0, 0.075], 1.40, 104.0, -4.5),
            Phase.STOW_FULL: ([-0.42, 0.0, 0.085], 2.25, 112.0, -10.5),
            Phase.OFFER_EMPTY: ([0.42, 0.0, 0.085], 2.25, 119.0, -10.5),
            Phase.LATCH_EMPTY: ([0.00, 0.0, 0.075], 1.40, 104.0, -4.5),
            Phase.RELEASE: ([0.00, 0.0, 0.12], 2.05, 126.0, -10.0),
        }
        look, dist, az, el = by_phase.get(phase, ([0.0, 0.0, 0.14], 2.1, 120.0, -11.0))
        return np.array([look[0], look[1], look[2], dist, az, el])

    def step(self, dt: float, stage: str, phase: Phase, rover_x: float) -> None:
        tgt = self.target(stage, phase, rover_x)
        a = 1.0 - math.exp(-dt / self.TAU)
        self.state += a * (tgt - self.state)
        self._apply()

    def _apply(self) -> None:
        self.cam.lookat[:] = self.state[:3]
        self.cam.distance = float(self.state[3])
        self.cam.azimuth = float(self.state[4])
        self.cam.elevation = float(self.state[5])


class Demo:
    def __init__(self, write_video: bool, out_path: Path) -> None:
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)
        self.write_video = write_video
        self.out_path = out_path

        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[0] = START_X
        mujoco.mj_forward(self.m, self.d)

        self.hw = MuJoCoDockHardware(self.m, self.d, DOCK, on_event=self._hw_event)
        self.ctrl = SwapBatteryController(self.hw)

        nid = mujoco.mj_name2id
        self.act = [nid(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ("mot_fl", "mot_fr", "mot_rl", "mot_rr")]
        self.left = [self.act[0], self.act[2]]
        self.right = [self.act[1], self.act[3]]
        self.hi = float(self.m.actuator_ctrlrange[self.act[0], 1])
        self.chassis = nid(self.m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.wheels = {n: nid(self.m, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{n}_body")
                       for n in ("fl", "fr", "rl", "rr")}

        self.stage = "approach"
        self.stage_t = 0.0
        self.cam = Camera()
        self.phase_log: list[tuple[float, Phase, Phase]] = []
        self._last_phase = self.ctrl.phase

        # --- measurements the assertions are made against -------------------
        self.max_lift_top = -1e9
        self.max_lift_cmd = -1e9
        self.min_wheel_bottom_berthed = 1e9
        self.min_chassis_z_berthed = 1e9
        self.wheel_margin_berthed = 1e9
        self.wheel_xy_berthed: dict[str, tuple[float, float]] = {}
        self.berth_report = None
        self.frame_sum, self.n_frames = 0.0, 0

    # ---- logging -----------------------------------------------------------
    def _hw_event(self, text: str) -> None:
        print(f"[{self.d.time:7.3f} s]   hw: {text}")

    def _log_phase(self) -> None:
        p = self.ctrl.phase
        if p is not self._last_phase:
            print(f"[{self.d.time:7.3f} s] PHASE {self._last_phase.value} -> {p.value}")
            self.phase_log.append((float(self.d.time), self._last_phase, p))
            self._last_phase = p

    # ---- drive -------------------------------------------------------------
    def _drive(self, x_goal: float | None) -> None:
        if x_goal is None:
            l_cmd = r_cmd = 0.0
        else:
            x, y = float(self.d.qpos[0]), float(self.d.qpos[1])
            yaw = _quat_yaw(self.d.qpos[3:7])
            v = float(np.clip(KP_X * (x_goal - x), -V_CRUISE, V_CRUISE))
            if abs(x_goal - x) < 0.015:
                v = 0.0
            base = v / WHEEL_R
            diff = float(np.clip(-(KP_YAW * yaw + KP_Y * y), -0.9, 0.9))
            l_cmd, r_cmd = base - diff, base + diff
        slew = self.hi / RAMP_S * self.dt
        for idxs, cmd in ((self.left, l_cmd), (self.right, r_cmd)):
            for i in idxs:
                cur = float(self.d.ctrl[i])
                cur += float(np.clip(cmd - cur, -slew, slew))
                self.d.ctrl[i] = float(np.clip(cur, -self.hi, self.hi))

    def _stopped(self) -> bool:
        return (abs(float(self.d.qvel[0])) < 0.015
                and abs(float(self.d.qpos[0])) < 0.05)

    # ---- berth check, against B's own published tolerances -----------------
    def _berth_pose(self):
        x, y = float(self.d.qpos[0]), float(self.d.qpos[1])
        yaw = _quat_yaw(self.d.qpos[3:7])
        z_mount = self.hw.mount_underside_z
        return x, y, yaw, z_mount

    def _check_alignment(self) -> bool:
        x, y, yaw, z_mount = self._berth_pose()
        z_ref = DOCK.pad_top_z + DOCK.pack_underside_z
        xy_err = math.hypot(x - DOCK.berth_x, y - DOCK.berth_y)
        yaw_err = abs((yaw + math.pi) % (2 * math.pi) - math.pi)
        z_err = abs(z_mount - z_ref)
        ok = xy_err <= XY_TOL_M and yaw_err <= YAW_TOL_RAD and z_err <= HEIGHT_GAP_TOL_M
        self.berth_report = dict(x=x, y=y, yaw=yaw, z_mount=z_mount, z_ref=z_ref,
                                 xy_err=xy_err, yaw_err=yaw_err, z_err=z_err, ok=ok)
        print(f"[{self.d.time:7.3f} s] BERTH  xy_err={xy_err*1000:6.1f} mm "
              f"(tol {XY_TOL_M*1000:.0f})  yaw_err={math.degrees(yaw_err):5.2f} deg "
              f"(tol {math.degrees(YAW_TOL_RAD):.1f})  z_mount={z_mount:.4f} m "
              f"vs {z_ref:.4f} (tol {HEIGHT_GAP_TOL_M*1000:.0f} mm)  -> "
              f"{'ALIGNED' if ok else 'MISALIGNED'}")
        return ok

    # ---- per-step measurement ---------------------------------------------
    def _measure(self) -> None:
        self.max_lift_top = max(self.max_lift_top, self.hw.carriage_top)
        self.max_lift_cmd = max(self.max_lift_cmd, float(self.d.ctrl[self.hw._lift_act]))
        if self.stage in ("settle", "swap"):
            self.min_chassis_z_berthed = min(self.min_chassis_z_berthed,
                                             float(self.d.xpos[self.chassis][2]))
            for name, bid in self.wheels.items():
                p = self.d.xpos[bid]
                self.min_wheel_bottom_berthed = min(self.min_wheel_bottom_berthed,
                                                    float(p[2]) - 0.090)
                mx = DOCK.pad_half_x - abs(float(p[0]))
                my = DOCK.pad_half_y - abs(float(p[1]))
                self.wheel_margin_berthed = min(self.wheel_margin_berthed, mx, my)
                self.wheel_xy_berthed[name] = (float(p[0]), float(p[1]))

    # ---- main loop ---------------------------------------------------------
    def run(self) -> None:
        print(f"model: nq={self.m.nq} nu={self.m.nu} nbody={self.m.nbody} "
              f"ngeom={self.m.ngeom} nmocap={self.m.nmocap} dt={self.dt}")
        tallest = check_rangefinder_clearance(self.m)
        n_bits = check_collision_bits(self.m)
        print(f"dock: {n_bits} colliding geoms on the traverse arena's WB_BITS; "
              f"tallest dock geom z={tallest:.3f} m, rangefinder fan {RF_FAN_Z:.3f} m")

        renderer = writer = None
        if self.write_video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(self.m, height=HEIGHT, width=WIDTH)
            writer = imageio.get_writer(str(self.out_path), fps=FPS, codec="libx264")

        sub = max(1, int(round((1.0 / FPS) / self.dt)))
        self.ctrl.start(SwapRequest(vehicle_id="rover0", dock_id="dock0",
                                    empty_pack_count=1))
        print(f"[{self.d.time:7.3f} s] PHASE idle -> {self.ctrl.phase.value} "
              f"(controller waiting for berth)")
        self._last_phase = self.ctrl.phase

        k = 0
        while True:
            x_goal = self._advance_stage()
            if self.stage == "done":
                break
            self._drive(x_goal)
            self.hw.sync(self.dt)
            if self.stage != "depart":
                self.ctrl.tick(self.dt, aligned=self._aligned_flag)
                self._aligned_flag = None
            self._log_phase()
            mujoco.mj_step(self.m, self.d)
            self._measure()

            self.cam.step(self.dt, self.stage, self.ctrl.phase, float(self.d.qpos[0]))
            if k % sub == 0 and renderer is not None:
                renderer.update_scene(self.d, camera=self.cam.cam)
                frame = renderer.render()
                writer.append_data(frame)
                self.frame_sum += float(frame.mean())
                self.n_frames += 1
            k += 1

        if renderer is not None:
            renderer.close()
            writer.close()

    _aligned_flag = None

    def _advance_stage(self) -> float | None:
        """Returns the drive goal in x, or None to hold still. Advances stages."""
        self.stage_t += self.dt
        if self.stage == "approach":
            if self._stopped() or self.stage_t >= APPROACH_MAX_S:
                self.stage, self.stage_t = "settle", 0.0
                print(f"[{self.d.time:7.3f} s] STAGE approach -> settle "
                      f"(x={float(self.d.qpos[0]):+.4f} m)")
            return DOCK.berth_x
        if self.stage == "settle":
            if self.stage_t >= SETTLE_S:
                self._aligned_flag = self._check_alignment()
                self.stage, self.stage_t = "swap", 0.0
            return None
        if self.stage == "swap":
            res = self.ctrl.tick(0.0)
            if res.status is not Status.RUNNING or self.stage_t >= SWAP_MAX_S:
                print(f"[{self.d.time:7.3f} s] SWAP  {res.status.value}"
                      + (f" ({res.reason})" if res.reason else ""))
                self.swap_status = res.status
                self.stage, self.stage_t = "depart", 0.0
            return None
        if self.stage == "depart":
            if self.stage_t >= DEPART_S:
                self.stage = "done"
            return 3.0
        return None

    swap_status = Status.RUNNING

    # ---- assertions --------------------------------------------------------
    def assertions(self) -> None:
        print("\n" + "=" * 70)
        print("ASSERTIONS")
        print("=" * 70)

        # 1. rover.xml untouched on disk
        probe = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "rover.xml"))
        assert (probe.nq, probe.nu, probe.nbody) == (11, 4, 6), "rover.xml was modified!"
        print(f"PASS  rover.xml on disk still nq={probe.nq} nu={probe.nu} nbody={probe.nbody}")

        # 2. the swap actually completed, through B's machine
        assert self.swap_status is Status.SUCCESS, f"swap ended {self.swap_status}"
        assert self.ctrl.phase is Phase.DONE, f"controller ended in {self.ctrl.phase}"
        seq = [b.value for _, _, b in self.phase_log]
        expect = ["lift_full", "stow_full", "offer_empty", "latch_empty", "release", "done"]
        assert seq == expect, f"phase sequence {seq} != {expect}"
        print(f"PASS  B's phase machine ran {' -> '.join(expect)} and returned SUCCESS")

        # 3. berth tolerances (B's own numbers)
        b = self.berth_report
        assert b is not None and b["ok"], f"berth check failed: {b}"
        print(f"PASS  berthed inside B's tolerance: xy {b['xy_err']*1000:.1f}/"
              f"{XY_TOL_M*1000:.0f} mm, yaw {math.degrees(b['yaw_err']):.2f}/"
              f"{math.degrees(YAW_TOL_RAD):.1f} deg, mount z {b['z_err']*1000:.1f}/"
              f"{HEIGHT_GAP_TOL_M*1000:.0f} mm")

        # 4. wheels land on the pad, with margin
        assert self.wheel_margin_berthed > 0.05, (
            f"wheel margin {self.wheel_margin_berthed:.4f} m - wheels off the pad")
        ys = [abs(y) for _, y in self.wheel_xy_berthed.values()]
        assert all(abs(v - 0.40) < 0.02 for v in ys), f"wheel |y| = {ys}, expected 0.40"
        print(f"PASS  all 4 wheels on the pad: |y| = "
              f"{', '.join(f'{v:.3f}' for v in ys)} m, "
              f"min edge margin {self.wheel_margin_berthed*1000:.0f} mm "
              f"(pad {DOCK.pad_size_x:.2f} x {DOCK.pad_size_y:.2f} m)")
        for name, (wx, wy) in self.wheel_xy_berthed.items():
            assert abs(wx) > DOCK.slot_half_x, (
                f"wheel {name} at x={wx:.3f} sits over the {DOCK.slot_half_x:.3f} m lift well")
        print(f"PASS  no wheel over the lift well (|x| > {DOCK.slot_half_x:.3f} m)")

        # 5. lift reaches 0.060 and never exceeds it
        assert self.max_lift_top <= DOCK.lift_stroke + 1e-5, (
            f"lift top reached {self.max_lift_top:.5f} m > {DOCK.lift_stroke} m")
        assert self.max_lift_top >= DOCK.lift_stroke - 1e-3, (
            f"lift top only reached {self.max_lift_top:.5f} m")
        assert self.max_lift_cmd <= DOCK.lift_stroke + 1e-9, (
            f"lift was COMMANDED to {self.max_lift_cmd:.5f} m "
            f"(swap_battery.py's legacy 0.12 leaked through)")
        print(f"PASS  lift top max {self.max_lift_top:.5f} m == pack underside "
              f"{DOCK.pack_underside_z:.3f} m, never exceeded "
              f"(max command {self.max_lift_cmd:.5f} m, legacy 0.12 normalised)")

        # 6. the rover did not sink through the pad
        assert self.min_wheel_bottom_berthed >= DOCK.pad_top_z - 0.005, (
            f"wheel bottom sank to {self.min_wheel_bottom_berthed:.5f} m, "
            f"pad top is {DOCK.pad_top_z:.3f} m")
        nominal = 0.175 + DOCK.pad_top_z
        assert abs(self.min_chassis_z_berthed - nominal) < 0.010, (
            f"chassis at {self.min_chassis_z_berthed:.4f} m, expected ~{nominal:.4f} m")
        print(f"PASS  rover stands on the pad: wheel bottom min "
              f"{self.min_wheel_bottom_berthed*1000:+.2f} mm vs pad top "
              f"{DOCK.pad_top_z*1000:.1f} mm; chassis {self.min_chassis_z_berthed:.4f} m "
              f"(nominal {nominal:.4f})")

        # 7. the packs ended up where a SUCCESS claims they are
        assert self.hw.full_pack_carrier == "rack_full", (
            f"full pack is on {self.hw.full_pack_carrier!r}, not the stow rack")
        assert self.hw.empty_pack_carrier == "vehicle", (
            f"fresh pack is on {self.hw.empty_pack_carrier!r}, not the rover")
        assert self.hw.lift_height <= 0.004, (
            f"carriage not clear: {self.hw.lift_height:.4f} m")
        print("PASS  full pack at full_stow, fresh pack on the rover, carriage clear")

        # 8. video
        if self.write_video:
            size = self.out_path.stat().st_size
            mean = self.frame_sum / max(1, self.n_frames)
            assert size > 200_000, f"{self.out_path} is only {size} bytes"
            assert mean > 20.0, f"frames look black: mean pixel {mean:.2f}"
            print(f"PASS  {self.out_path.name}: {self.n_frames} frames, "
                  f"{size/1e6:.2f} MB, mean pixel value {mean:.2f}/255")
        print("=" * 70)
        print("ALL ASSERTIONS PASSED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-video", action="store_true", help="assertions only, no render")
    ap.add_argument("--out", default=str(REPO_ROOT / "renders" / "swap_demo.mp4"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    demo = Demo(write_video=not args.no_video, out_path=out)
    demo.run()
    demo.assertions()
    if not args.no_video:
        print(f"\nvideo: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
