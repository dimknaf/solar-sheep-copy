# Integration A ↔ B — rover vs battery-swap dock

**Date: 15 September 2026** · written by A (Dimitris) · reconciles `robot/rover.xml` against
`envs/swap/` at the state committed on branch `physical-ai-A2`.

**Reading rule for this document:** B's geometry is self-marked **PROPOSED**
(`envs/swap/SPEC.md:4`, `:50`, `scripts/frames.py:15`) pending `robot/SPEC.md`. A's geometry is
**measured** from a compiled, tuned MJCF. Wherever the two disagree, **A wins and B changes** — not
because A outranks B, but because `CONTRACTS.md:33` already assigned it:

> *"B and C import it **read-only**. If B models the battery mount against a different robot, the
> swap will never line up. This is the tightest coupling in the project — settle it first."*

That is exactly what happened. B guessed well and guessed wrong. **Nothing in B's Python needs to
change structurally** — the phase machine, the FAILURE taxonomy and the `DockHardware` seam are all
correct and should be kept verbatim. What changes is roughly fifteen numbers.

---

## 0 · Status corrections since the analysis agents ran

Three findings that were reported earlier in this session are **already fixed** and should not be
re-raised:

| Earlier finding | Current reality |
|---|---|
| "`robot/SPEC.md` does not exist — B is blocked" | It exists and is now **LOCKED**. B is unblocked. |
| "`panel_normal` +Z points down — latent bug" | **Fixed.** Declared with explicit `zaxis`; sensor `panel_normal_world` now reads `[0, 0, 1]`. Verified. |
| "`forcerange ±2.7 Nm`" | Now **±2.4 Nm**, with the 0.80 m track that made 2.4 viable. 30.2 W envelope corner. |
| "train.py `<include>`s rover.xml" | It uses **`mujoco.MjSpec`** surgery instead — a better hook, and the one this document's §5 relies on. |

---

## 1 · Mismatch table — both numbers side by side

| # | Item | **A measured** | **B proposed** | Δ | Who changes |
|---|---|---|---|---|---|
| **M1** | Forward axis at berth | **+X** | +Y (`SPEC.md:60`, approach `(0,−1.5,0)`) | **90°** | **B** (+ C places) |
| **M2** | Mount plane / `z_mount` | **0.060** | 0.16 (`swap_battery.py:49,57`; `pack_attach` `dock.usda:35`) | **+0.100** | **B** |
| **M3** | Pack X (fore-aft) | **0.200** | 0.30 | +0.100 | **B** |
| **M3** | Pack Y (lateral) | **0.450** | 0.20 | **−0.250** | **B** |
| **M3** | Pack Z (thickness) | **0.055** | 0.10 | +0.045 | **B** |
| **M4** | Dock housing top | must be ≤ **0.045** | **0.100** (`dock.usda:73-80`) | **+0.055** | **B** |
| **M5** | Carriage top, retracted | must be ≤ **0.045** | **0.070** (`dock.usda:93-100`) | **+0.025** | **B** |
| **M6** | Lift stroke | **≈ 0.070**, ending **−0.0075** below datum | 0.00 → 0.12, never below 0 | 1.7× too long, **and the wrong sign at the bottom** | **B** |
| **M7** | Carriage plate | **0.55 (Y) × ≤ 0.09 (X)** | 0.28 (X) × 0.18 (Y) | **−0.37 in Y** | **B** |
| **M8** | Berth XY tolerance | ≤ **0.035** | 0.08 (`XY_TOL_M`) | 2.3× too loose | **B** |
| **M8** | Berth yaw tolerance | 9° is **fine** | 0.157 rad (9°) | **match — keep** | — |
| **M9** | Drive-over pad width | ≥ **1.00** or ≤ **0.55** | 0.70 | **in the forbidden band** | **B** |
| **M10** | Rack poses | must clear the ±X drive line | `(±0.45, 0, 0.05)`, tops at 0.060 | on the drive line | **B** |
| **M11** | Assumed body width | 0.460 shell / **0.860** over tyres / **1.010** over panel | "~0.6 m wide body" (`SPEC.md:52`) | −0.26 … −0.55 | **B** |
| **M12** | Pack mate face | pack **BOTTOM** (`battery_mount` at z 0.060) | `mount_frame` at pack **TOP** (`pack.usda:38-43`) | mates the pack **inside** the chassis | **B** |
| — | Pack mass | **4.000 kg** | 4.0 kg | **match** | — |
| — | Pack capacity | 500 Wh | 500 Wh | **match** | — |
| — | Latch direction | pack underside, +Z down | "latch faces −Z toward dock" | **match — both sides converged independently** | — |
| — | Frame name | site `battery_mount` | `battery_mount` (path `/Sheep/…` TBD) | **name matches**; prefix is C's | — |
| — | Z-up, metres, at origin | yes | yes | **match** | — |

### ⚠️ The three that are hard blockers

**⚠️ M4 + M5 — with today's numbers the rover cannot reach the berth at all.**
The pack underside sits at **0.0600** and the straps at **0.0575**. B's housing tops out at
**0.100** and its retracted carriage plate at **0.070**. The rover would bulldoze the housing with
its battery 40 mm before the berth, and clip the carriage 10 mm before that. There is no dock state
and no approach angle that avoids this. This is not a tolerance problem, it is a solid-body overlap.

**⚠️ M2 — B's dock rejects A's rover 100 % of the time, in live code, on tick one.**
`_aligned()` (`swap_battery.py:113`) computes `abs(vehicle.z_mount − berth.z_mount)` against
`HEIGHT_GAP_TOL_M = 0.03`. Feed the real pose: `abs(0.060 − 0.16) = 0.100 > 0.03` →
`FAILURE reason="misaligned"`, forever. **The green test suite hides this**: five of the six tests
bypass `_aligned()` via `force_aligned=True` / `tick(aligned=True)`, and the sixth trips on a gross
1.0 m XY error. 6/6 passing is not evidence of compatibility.

**⚠️ M6 — the lift, as authored, cannot take the pack away even after everything else is fixed.**
See §2. This is the one item that needs a genuine redesign rather than a renumber.

---

## 2 · ⚠️ The finding nobody had: the lift must travel BELOW the wheel-standing plane

Every earlier analysis said "shrink the stroke from 0.12 to about 0.02–0.06". **All of them are
wrong**, because they costed the *capture* and not the *extraction*.

Datum `z = 0` = the plane the wheels stand on (ground, or pad top — identical either way, because
raising the rover on a pad raises the pack by exactly the same amount).

```
pack bay depth (datum → pack underside)      0.0600
pack thickness                               0.0550
                                             ------
drop available above the datum                0.0050   ← five millimetres
minus any carriage plate thickness            < 0
```

The carriage captures at **+0.060**. Carrying a 0.055 m pack it must then descend until the pack top
clears the rover's 0.0575 strap line with real margin: **carriage top ≤ −0.0075**. Stroke
≈ **0.070**, of which the last ~0.015–0.020 is **under the surface the wheels are standing on**.

| margin | capture top | release top (vs collidable 0.060) | release top (vs rendered 0.0575) | stroke |
|---|---|---|---|---|
| 0.005 | +0.060 | 0.0000 | −0.0025 | 0.0625 |
| **0.010** | **+0.060** | **−0.0050** | **−0.0075** | **0.0675** |
| 0.015 | +0.060 | −0.0100 | −0.0125 | 0.0725 |

**How B gets sub-datum travel without digging a hole:** make the rover **drive onto a pad** ≥ 0.025
thick and cut the lift well as a **slot in the pad**. The recess then costs nothing but pad
material, and no hole has to be modelled in C's world file. This — not clearance — is the real
argument for a full-width pad, and it is why M9 resolves toward "widen" rather than "narrow".

Ramp check: the rover **climbs 14°, stalls at 16°** — use **≤ 10°**. A 0.030 m pad needs a 0.170 m
ramp run; a 0.040 m pad needs 0.227 m. Break-over is `atan(0.060/0.110) = 28.6°`, so the short
0.220 m wheelbase makes the pad crest a non-issue.

---

## 3 · ⚠️ The pad is in the one width band that cannot work

B's pad is **0.70 m** across. A's track is **0.80 m** centre-to-centre.

| | |
|---|---|
| Tyre **inner** faces | \|y\| = **0.370** → 0.740 m clear span |
| Tyre **outer** faces | \|y\| = **0.430** → 0.860 m over tyres |
| B's pad half-width | 0.350 |
| **Margin to the tyre inner face** | **0.020 m** |
| B's own `XY_TOL_M` | **0.080 m** — four times the margin |

So a berth B calls "aligned" puts one wheel **60 mm up onto the pad edge** while the other stays on
grass. The rover berths tilted, the pack tilts with it, and the carriage misses.

**Two valid pads, nothing in between:**

| | width across the lane | rover sits | needs |
|---|---|---|---|
| **Widen** *(recommended)* | **≥ 1.00 m** (0.860 over tyres + 2 × 0.035 tol + 2 × 0.020 margin) | on the pad | a ≤ 10° ramp; gives the §2 slot for free |
| Narrow | **≤ 0.55 m** | on grade, straddling | a dug well below grade |

**The pad's length *along* the lane is already fine.** The 0.90 m B authored is generous — all four
wheel contacts lie within `x = ±0.110`, so 0.40 m of flat pad would do. **Keep 0.90 and add the
ramps outside it.**

---

## 4 · ⚠️ The berth tolerance is geometrically impossible, and the yaw number is the good one

The carriage must be **wide enough to catch a 0.450 m pack** yet **narrow enough to miss tyre inner
faces at 0.370**. That 0.290 m of difference *is* the whole tolerance budget, and a 0.450 m pack
swings long corners under yaw.

```
capture:    C ≥ 0.225·cos ε + 0.100·sin ε + e
clearance:  C ≤ 0.370·cos ε − 0.200·sin ε − e − m        (m = 0.015)
```

| yaw ε | max lateral error e | required carriage width |
|---|---|---|
| 0° | ±0.065 | 0.580 |
| 6° | ±0.049 | 0.566 |
| **9°** | **±0.041** | **0.557** |
| 12° | ±0.032 | 0.546 |

**B's ±0.08 m is infeasible at every yaw, including zero.** B's 9° yaw, by contrast, is a good
number and should be kept — the rover spins at 0.571 rad/s, so 9° is trivially reachable.

**Settle at `XY_TOL_M = 0.035`, `YAW_TOL_RAD = 0.157` (unchanged), carriage 0.55 (Y) × ≤ 0.09 (X).**
Hitting ±0.035 m is A's problem, at harvest creep (0.117 m/s), and **A accepts it.**

---

## 5 · ⚠️ There was nothing to detach — now there is. Decided, and verified.

The pack is a `<geom>` of `chassis`: no body, no joint, no weld, no equality. **As shipped, a swap
could not be represented at all** — B has been building a lift for a pack that is welded into the
chassis casting.

**DECISION (A): split the pack into its own welded body in the demo scene only, via `MjSpec`
surgery at load time. `rover.xml` is not edited.** This reuses the exact mechanism
`envs/traverse/train.py` already uses to inject terrain and sensors.

Verified on this machine, mujoco 3.13.0:

```
compiles:       nq 18 · nv 16 · nu 4 · nbody 7 · neq 1
mass 29.4000 kg · CoM z 0.186497   ← both unchanged from the monolithic model
welded, 2 s:    pack z = 0.08734  (authored 0.08750 → 0.16 mm drift)
eq_active = 0:  pack z = 0.02739  → detaches, falls, rests flat on the ground
```

**For B this means `detach_full_to_carriage()` / `latch_empty_to_vehicle()` are one line each:**
`data.eq_active[pack_weld] = 0` / `= 1`. The swap is **physically real**, not choreographed — which
is precisely what a Technological Implementation judge probes. The training model stays lean at
`nq 11`; only the demo model carries the extra 6 DoF.

---

## 6 · Format / engine decision

> **SUPERSEDED 24–25 Sep 2026** — the project moved to Omniverse (Isaac Lab on PhysX, checked on
> Newton MuJoCo-Warp). See [decisions.md](decisions.md) D1, D4–D6. Kept below as the historical
> record; the geometry sections above still hold.

> ### DECISION: **one simulator — MuJoCo. Hand-transcribe `dock.usda` to a ~40-line `dock.xml`.**
> Keep USD as a *render-only* option behind a pose-stream boundary, to be dropped the moment it
> costs a day.

**Cost of this decision:** ~half a day to transcribe five boxes and one prismatic joint, plus ~80
lines for a `MuJoCoDockHardware` adapter. **B loses the USD dock as the simulated artifact** and
keeps it as the visual one. B's Python is untouched.

**Why not the other way (rover MJCF → USD, everything in Isaac Sim):**

- The converter **does exist** (`mujoco-usd-converter`, vendored under
  `.agents/skills/omniverse-cad-to-simready/…/convert-to-usd/references/mujoco-usd-converter/`), and
  its own README calls itself **alpha**.
- But the geometry is not the problem — **the physics is.** `cone="elliptic"`, `impratio="1"`,
  `frictionloss="0.60"`, `solref`/`solimp` and the four `<velocity>` servos (kv 18, ±2.4 Nm) have
  **no UsdPhysics equivalent**. `rover.xml`'s header is explicit that those exact settings *are* the
  skid-steer behaviour: a pyramidal cone "made the rover turn flatteringly well for the wrong
  reason", and impratio 5 gave "about a third of the yaw".
- So **every headline number would have to be re-measured under PhysX** — 0.279 m/s, 0.571 rad/s,
  8.2 W, 14° climb, 33° slide, 65° tip. That is not a conversion, it is re-deriving the robot, and
  it invalidates every number in the pitch deck.
- **The reverse tool does not exist at all.** The converter router is strictly one-way
  (urdf→usd, mjcf→usd, gsplat→usd, cad→usd). There is no `usd2mjcf`. None is needed: `dock.usda` is
  five `Cube` prims and one `PhysicsPrismaticJoint`, zero meshes, zero textures.
- **A is already not on Isaac.** `envs/traverse` is MuJoCo MJX + Brax PPO, chosen deliberately.
- Local environment cannot iterate on Isaac: **no `pxr` installed**, Python 3.11.5 against the
  3.12 the toolchain requires, Isaac Sim unsupported on Windows/WSL. Every tuning cycle would run
  remote and blocking.
- Eligibility is "runs on Nebius AI Cloud **and** uses an NVIDIA open model". MJX on the rented
  RTX PRO 6000 satisfies the first; Nemotron in the orchestrator satisfies the second. **Isaac Sim
  is not an eligibility requirement and no judge scores the physics engine.**

**The format mismatch is not the problem.** Not one of M1–M12 is caused by MJCF-vs-USD; every one is
a number or a convention and would exist identically if both sides were USD. The format costs about
a day. The geometry costs far more.

**B's `DockHardware` seam is the right abstraction and survives the decision intact** — verified by
grep: zero `pxr`, `omni.`, or `isaacsim` imports anywhere in `envs/swap/*.py`; the entire type
vocabulary of the six-method Protocol is `float | str | None`.

*Two honest caveats on that seam, both additive:* (1) it is **pose-blind** — `tick(aligned: bool)`
takes alignment as a boolean the *caller* computes, so B's tolerance constants become advisory and
will drift. Adding `get_vehicle_mount_pose(vehicle_id) -> Pose2D` to the Protocol keeps `_aligned()`
authoritative. (2) It **leaks geometry into the logic file**: `z_mount = 0.16` appears twice and
`set_lift_height(0.12)` three times in `swap_battery.py`, so "format-independent" is slightly
overstated — five constants move when the frame is locked.

---

## 7 · B's TODO — ordered, with the arithmetic already done

All inside `envs/swap/` (B's folder). **A proposes; B applies.** Items 1–4 are the blockers.

| # | File | Change | From → To |
|---|---|---|---|
| **1** | `scripts/swap_battery.py:49,57` | `Pose2D.z_mount` default and `SwapRequest.berth_pose` z | `0.16` → **`0.060`** |
| **2** | `dock.usda:73-80` | `base/housing` — get it out of the drive lane | scale `(0.40,0.35,0.08)` @ z 0.06 → cap top at **≤ 0.045**, or move to **\|y\| ≥ 0.52** where height is unlimited *(preferred)* |
| **3** | `dock.usda:93-100` | `carriage/plate` — thin it and widen it | `(0.28, 0.18, 0.03)` → **`(0.09, 0.55, 0.012)`**, retracted top **≤ 0.045** |
| **4** | `dock.usda:103-111` | `lift_joint` — re-zero and allow sub-datum travel | `lower 0 / upper 0.12` → **`lower −0.020 / upper +0.060`** referenced to the pad top (see §2) |
| 5 | `pack.usda:24` | `body` scale, in A's X-forward frame | `(0.30, 0.20, 0.10)` → **`(0.200, 0.450, 0.055)`**; translate z `0.05` → **`0.0275`** |
| 6 | `pack.usda:38-43` | `mount_frame` — it is on the **bottom**, not the top (M12) | z `0.10` → **`0`**, co-located with `latch_frame`; or delete one of the two |
| 7 | `dock.usda:35` | `pack_attach` | z `0.16` → **`0.060`**, oriented **+Z up** to mate `battery_mount`'s +Z down |
| 8 | `dock.usda:55-64` | `pad` — widen across the lane, add ramps | `(0.90, 0.70, 0.04)` → **`(0.90, 1.00, 0.030)`** + a **≤ 10° ramp** (0.170 m run) on the −X approach side |
| 9 | `scripts/swap_battery.py:39` | `XY_TOL_M` | `0.08` → **`0.035`**. **Leave `YAW_TOL_RAD = 0.157` alone — it is correct.** |
| 10 | `scripts/swap_battery.py:227,269,277` | the three hardcoded `set_lift_height(0.12)` | → **`0.060`**; the three `set_lift_height(0.0)` → **`−0.020`** (stow) / `0.045` (clear) |
| 11 | `dock.usda:40-52,114-131` | `empty_ready` / `full_stow` + their markers | off the ±X drive line. The pack **can only exit along ±X** — the wheels block ±Y at ground level. Put both racks on the **far (+X)** side at **x = +0.55 and +0.85**, keeping the −X side free for the approach ramp |
| 12 | `stage.usda:21,30` | staged pack poses | `PackFull` z `0.16` → **`0.060`**; `PackEmpty` to match the new rack pose |
| 13 | `SPEC.md:60` | *"Vehicle +Y is forward when docked"* | → **"Vehicle +X is forward when docked"** |
| 14 | `SPEC.md:52` | *"a ~0.6 m wide body"* | → **"0.46 m shell, 0.86 m over tyres, 1.01 m over panel; 0.74 m clear between tyre inner faces"** |
| 15 | `SPEC.md:80` | reconcile with the code (internal, B-vs-B) | SPEC measures the gap against *carriage top*; `_aligned()` measures against `berth_pose.z_mount`. Two reference planes 0.09 m apart. Pick one. |
| 16 | `SPEC.md:50` | header | drop **PROPOSED** → **LOCKED against `robot/SPEC.md`, 15 Sep 2026** |

### Do not change

- The phase machine, the phase order, the 10.0 s budget, the four FAILURE reasons, the
  `DockHardware` six-method Protocol. All correct, all simulator-agnostic, 6/6 tests green.
- The drive-over-from-below mechanism and "latch faces −Z". Both sides converged on it independently.
- `YAW_TOL_RAD = 0.157`. Pack mass 4.0 kg. Capacity 500 Wh. Z-up, metres, authored at origin.

### Also worth B's time (not blockers)

- `tick()` on an **IDLE** controller returns `Status.SUCCESS` with reason `"idle"`
  (`swap_battery.py:196-197`). An orchestrator polling a dock that was never started reads SUCCESS.
  **C will hit this.**
- `stage_empty_pack()` is called **twice** on entering `OFFER_EMPTY` (`:259` via
  `_drive_hardware_for_phase`, then again at `:262`). Harmless with `NullHardware`; not harmless
  with a real adapter that instantiates a prim.
- **No hardware call can report failure** — all six Protocol methods return `None`. An adapter that
  detects "the latch did not engage" has no channel to say so.
- **A first integration test worth having:** feed A's real `battery_mount` world pose and assert
  `RUNNING`. Today that test would fail on tick one, and nothing in the current suite would tell you.

---

## 8 · Needs the human — genuinely undecidable here

1. **`swap_battery()`'s signature drift from the frozen contract.**
   `CONTRACTS.md:58` freezes `swap_battery(vehicle_id, dock) -> Status`.
   B ships `swap_battery(controller, vehicle_id, dock, *, dt, request, aligned, empty_pack_count)
   -> SwapResult` — an extra leading positional, four keyword-onlys, and a dataclass return. It is
   compatible *in spirit* (`SwapResult.status` is a `str`-Enum `Status`), but `CONTRACTS.md:6` says
   **"nobody changes this file alone."** Either B conforms or all three amend the contract. **C's
   orchestrator is the party that actually cares.** A has no standing to decide this.

2. **Who applies the 90° for M1.** Two valid fixes, different owners:
   (a) **C yaws the `/Dock` Xform −90° in `scene/world.usd`** — zero edits inside `envs/swap/`, and
   it is what `CONTRACTS.md`'s "authored at origin, placed by the world file" rule is *for*. ⚠️ But
   it does **not** fix the racks sitting on the drive line (item 11), so B has work either way.
   (b) **B re-authors the dock X-forward.** More edits, one coherent asset, no hidden world rotation.
   A's weak preference is (b) — but it is B's asset and C's world file, so it is theirs to call.
   **Whoever does it must say so out loud in `docs/decisions.md`**, or someone re-derives the wrong
   thing a week from now.

3. **Should A raise the battery straps 3 mm?** They are pure decoration (`contype=0 conaffinity=0
   mass=0`) hanging 2.5 mm below the pack, and they are the only reason the rendered floor (0.0575)
   differs from the collidable floor (0.0600). Raising them makes the two equal and deletes a whole
   class of "the carriage clips the strap on video" confusion, at the cost of touching a tuned file
   for a cosmetic reason. **A's call, low stakes, not blocking B.**

4. **Whether the dock is single-sided.** Item 11 assumes both racks go on the +X side so the −X
   approach ramp stays clear. A turntable, or a rack that the carriage rotates to, would also work
   and looks better on video. **B's mechanism, B's call** — A only constrains that the pack must
   exit along **±X**, because the wheels block ±Y at ground level.

---

## 9 · What is already right and must not be churned

Both sides independently converged on **drive-over, unlatch from below** — B's *"latch faces −Z
toward dock"* (`SPEC.md:56`) and A's `battery_mount` with `zaxis="0 0 -1"` on the pack underside
(`rover.xml:379-384`) are the same design, arrived at separately. Pack mass **4.0 kg** and capacity
**500 Wh** agree exactly. The frame **name** `battery_mount` was guessed correctly. Z-up, metres and
authored-at-origin discipline hold on both sides. The FAILURE taxonomy, the non-blocking
every-tick contract and the 12 s demo budget are all sound.

**B's scaffold is good work built against a robot that did not exist yet. Now it does.**
