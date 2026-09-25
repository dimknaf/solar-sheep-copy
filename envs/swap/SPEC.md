# Swap dock — SPEC

**Owner: B** · folder `envs/swap/` · layer `dock.usda` (composed via `stage.usda`).  
**Status (25 Sep 2026):** scripted phase machine + MuJoCo dock (`dock_mjcf.py`) working — `renders/swap_demo.py` passes all assertions. Geometry rows below are **LOCKED** against `robot/SPEC.md`. The `.usda` files are half-migrated (+X sizes, old +Y frames) and will be regenerated from `dock_mjcf.py` through the official Isaac Sim importer, not hand-edited (see `docs/decisions.md` D6).

---

## Contract

```
swap_battery(vehicle_id, dock) -> RUNNING | SUCCESS | FAILURE
```

Non-blocking. Called every sim tick. `FAILURE` is first-class (orchestrator recovers).  
See repo-root [`CONTRACTS.md`](../../CONTRACTS.md).

---

## Units & composition

| Rule | Value |
|---|---|
| Up axis | **Z** |
| Units | **metres** |
| Asset origin | Dock pad centre on ground (`z = 0`). **Never** bake world pose here. |
| World placement | C references this layer from `scene/world.usd` and sets the dock Xform. |
| Rover / pack mount | Dimitris publishes downward battery frame; B reads it **read-only**. |

---

## File map

```
envs/swap/
  SPEC.md                 this file
  pack.usda               single battery pack (rigid), at origin
  dock.usda               pad + prismatic underbody lift + named frames
  stage.usda              composition root (dock ⊃ pack references for authoring)
  scripts/
    status.py             RUNNING | SUCCESS | FAILURE
    frames.py             frame prim path constants
    swap_battery.py       scripted phase machine (Isaac-optional)
  tests/
    test_swap_phases.py   pure-Python phase / FAILURE tests (no Isaac)
  preview.html            laptop canvas stand-in (not Isaac) — open via static server
```

---

## Geometry (**LOCKED** — four rows corrected against `robot/SPEC.md`, 15 Sep 2026)

Provisional “slow cheap skid-steer” pack under a ~0.6 m wide body. Change only via this SPEC + Dimitris agreement.

The four rows marked **LOCKED** below were guessed here and have been replaced with the values A
measured on the compiled model. Source of truth: [`robot/SPEC.md`](../../robot/SPEC.md) §Geometry,
§Battery pack and §Constraints the dock must respect. Do not re-derive them from this file.

| Part | Size (m) | Notes |
|---|---|---|
| Pack outer | **`0.20 × 0.45 × 0.055`** (X×Y×Z) **LOCKED** | Was `0.30 × 0.20 × 0.10`. A wide flat slab, not a brick: the 0.22 m wheelbase leaves no fore-aft room. Underside at z = 0.060; latch faces −Z toward dock |
| Pack mass (viz) | 4.0 kg | Placeholder; not used by scripted controller |
| Dock pad | **`1.10 × 1.00 × 0.04`** **LOCKED** | Was `0.90 × 0.70`. Must span the **0.80 m track** — wheels at y = ±0.40 sit 0.10 m inside each lateral edge. A 0.70 m pad missed both wheels entirely |
| Lift stroke | **`0.00 → 0.060`** along +Z **LOCKED** | Was `0.00 → 0.12`. The pack underside is 60 mm off the ground; 120 mm drives the carriage into the chassis. `scripts/swap_battery.py` still passes its legacy `0.12` constant — `scripts/mujoco_hardware.py` reads that as “raise fully” and normalises it to 0.060 |
| Approach standoff | berth at origin; approach at **`(−1.50, 0, 0)`** **LOCKED** | Was `(0, −1.50, 0)`. **Vehicle +X is forward when docked**, not +Y: all four wheel hinges have axis `(0,1,0)`, and 8 s of full command measures dx = +2.216 m, dy = 0.000. The frames in `dock.usda` are authored for the old +Y convention and rotate 90° about Z — `(x, y) → (y, −x)` — when the dock is placed in the world |

### Named frames (on `dock.usda`)

| Frame prim | Role |
|---|---|
| `/Dock/frames/approach` | Where traverse hands off / swap may start aligning |
| `/Dock/frames/berth` | Drive-over seat; vehicle should be within tolerance here |
| `/Dock/frames/pack_attach` | On lift carriage top — mates to rover pack mount |
| `/Dock/frames/empty_ready` | Rack pose for the waiting empty pack |
| `/Dock/frames/full_stow` | Where the removed full pack is parked before plant handoff |

Rover-side frame (Dimitris owns): **`/Sheep/battery_mount`** (name TBD in `robot/SPEC.md`) — downward-facing, must align to `pack_attach` when berthing.

### Berth tolerance (LOCKED — see `docs/integration-A-B.md` §4)

Aligned enough to start lift if all hold:

- XY error ≤ **0.035 m** (was 0.08 m, which is geometrically impossible at any yaw for a
  0.450 m pack between tyre faces at ±0.370 m). The code constant `XY_TOL_M` in
  `scripts/swap_battery.py` is still 0.08 and changes with the dock rebuild (gate G6).
- Yaw error ≤ **9°** (0.157 rad) — matches published station misalignment class
- Pack mount height vs carriage top ≤ **0.03 m** gap before lift

---

## Mechanism (scripted — not learned)

Drive-over underbody lift only. No manipulator arm.

```
carry pack on rover
    → berth on pad
    → lift rises, unlatches full pack onto carriage
    → carriage lowers, stows full pack at full_stow
    → empty pack offered at empty_ready → raised and latched
    → lift clears → rover free
```

Plant conveyor / discharge units stay in the JS economics sim; Physical AI dock only proves **one swap cycle** for the video. Empty-pack availability is an input flag / rack count on the dock API.

---

## Phases & timing

Demo budget ~**12 s** wall-clock at 1× (JS plant uses `swapSec = 120` for economics — do not use that here).

| Phase | Id | Duration (s) | Effect |
|---|---|---|---|
| 0 | `approach` | — | Wait until berth OK or fail misaligned / timeout |
| 1 | `lift_full` | 2.0 | Carriage ↑; detach full pack from vehicle → carriage |
| 2 | `stow_full` | 2.0 | Carriage ↓ to `full_stow`; rack accepts full |
| 3 | `offer_empty` | 2.0 | Stage empty at `empty_ready` (fails if none) |
| 4 | `latch_empty` | 2.5 | Carriage ↑; attach empty to vehicle mount |
| 5 | `release` | 1.5 | Carriage ↓ clear of chassis; done |
| — | `SUCCESS` | — | Terminal |
| — | `FAILURE` | — | Terminal — see below |

Total scripted motion after berth ≈ **10 s**. Approach timeout default **30 s**.

---

## FAILURE paths (first-class)

| Code | When | Orchestrator hint |
|---|---|---|
| `misaligned` | Start requested but pose outside berth tolerance | Re-traverse to dock |
| `no_empty_pack` | Entering `offer_empty` with rack empty | Hold / send another rover later |
| `timeout` | Approach or any phase exceeds its budget | Abort; clear berth |
| `busy` | Dock already RUNNING another vehicle | Queue / pick other dock later |

Controller returns `FAILURE` (and a reason string on the result object). It never raises for these cases.

---

## Success criteria

`SUCCESS` only when:

1. Vehicle previously carrying pack A (full or “outgoing”)  
2. Vehicle now carrying pack B from the empty rack  
3. Pack A is on the dock stow / plant-handoff pose  
4. Lift carriage is clear (lowered)  
5. No phase timed out  

---

## Open locks (HITL)

1. Exact `/Sheep/battery_mount` prim path + local transform — **Dimitris**  
2. World dock pose name in `scene/SPEC.md` — **C**  
3. Whether demo uses 1 empty pre-staged on rack or a tiny FIFO of N — default **1 empty pre-staged**  

Log closures in `docs/decisions.md` when agreed.
