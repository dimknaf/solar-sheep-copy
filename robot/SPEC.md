# Rover — SPEC

**Owner: A (Dimitris)** · folder `robot/` · model `rover.xml` (MuJoCo MJCF)
**Status: LOCKED** — 15 September 2026. B may build against every number below.

Every value here was **measured from the compiled model** (`mj_forward`, mujoco 3.13.0), not read
off the XML text and not copied from a comment. Where a header comment in `rover.xml` disagrees
with this file, **this file is right** — see *Known comment drift* at the end.

---

## The one line B was waiting for

```
battery_mount    world pos (0, 0, 0.060)    +Z = (0, 0, -1)    POINTS DOWN AT THE DOCK
```

MJCF site `battery_mount` on body `chassis`, local `pos = (0, 0, -0.115)`, `zaxis = "0 0 -1"`.
Frame is **x = vehicle forward, y = vehicle right, z = down**. A dock rising from below sees this
frame's +Z coming at it. Verified after `mj_forward` and again under a +90° body yaw: +Z stays
`(0,0,-1)`.

B currently guesses the prim path `/Sheep/battery_mount`. **The site name `battery_mount` is
correct.** The `/Sheep` prefix is whatever C names the articulation root in `scene/world.usd`; it is
not A's to fix and not a blocker.

---

## 1 · Axes, units, composition

| Rule | Value |
|---|---|
| Up axis | **Z** |
| Units | **metres**, radians, kg |
| Authored at | **origin** — world pose set by the scene, never baked here |
| **FORWARD** | **+X** ⚠️ |
| Left / Right | +Y / −Y |
| Handedness | right-handed; +yaw is CCW seen from above (nose swings toward +Y) |

⚠️ **Forward is +X, not +Y.** `envs/swap/SPEC.md:60` says *"Vehicle +Y is forward when docked"*.
That is wrong by 90°. Evidence, any one of which is sufficient:

- All four wheel hinge axes are `(0,1,0)` — the wheels spin about **Y**, so the machine can only
  translate along **±X**.
- Tyre cylinders are `fromto="0 -0.03 0  0 0.03 0"` — axle axis is Y.
- Wheels at `x = ±0.110` (= the 0.22 m **wheelbase**) and `y = ±0.400` (= the 0.80 m **track**).
- Bodywork: sensor pod + lens at `x = +0.264…+0.346`; rear carry handle at `x = −0.317`.
- Measured: `ctrl = +3.1416` on all four for 8 s → `dx = +2.2159 m`, `dy = 0.0000 m`, quat
  unchanged. Body-frame velocimeter reads `(0.2792, 0, 0)`.

**This is not negotiable on A's side.** The axis is baked into four hinge axes, four actuators, the
f/r body names and the bodywork. The fix belongs on the dock side or in C's world placement — see
`docs/integration-A-B.md` §M1.

---

## 2 · Geometry

| Part | Value |
|---|---|
| Chassis shell (collidable) | 0.560 (X) × 0.460 (Y) × 0.120 (Z), world z **0.115 … 0.235** |
| PV frame (collidable) | 0.910 × 1.010 × 0.020, world z 0.270 … 0.290 |
| PV glass (collidable) | 0.890 × 0.990 × 0.016, world z 0.290 … 0.306 → **0.8811 m²** |
| Overall plan footprint | **0.910 (X) × 1.010 (Y)** — the PV frame is the widest and longest part |
| Overall height | **0.306** |
| **Track** (wheel centres) | **0.800** (y = ±0.400) |
| **Wheelbase** (axles) | **0.220** (x = ±0.110) |
| track / wheelbase | 3.64 |
| Wheel | radius **0.090**, width 0.060 (half 0.030) |
| Tyre inner faces | \|y\| = **0.370** → **0.740 m of clear span between them** |
| Tyre outer faces | \|y\| = 0.430 → **0.860 m over tyres** |
| Outermost detail (spoke) | \|y\| = 0.446 — 49 mm inside the glass edge; nothing pokes out |

### Ground clearances (flat ground, unladen, settled)

| Feature | \|y\| band | world z |
|---|---|---|
| Battery **straps** (visual) — true rendered floor | ≤ 0.227 | **0.0575** |
| Battery **pack** (collidable) — lowest collidable point | ≤ 0.225 | **0.0600** |
| Outboard hub faces (visual) | 0.430 … 0.440 | 0.0400 |
| Stub axles (visual) | 0.162 … 0.388 | 0.0720 |
| Wheel spokes (visual) | 0.436 … 0.446 | 0.0770 |
| Sill band (visual) | ≤ 0.232 | 0.1130 |
| Chassis shell (collidable) | ≤ 0.230 | **0.1150** |
| Axle centreline | — | 0.0900 |

Settling under load costs 0.14 mm of tyre sink — negligible, ignore it.

---

## 3 · Battery pack — what the dock must fit

| Part | Value |
|---|---|
| **Pack outer** | **0.200 (X, fore-aft) × 0.450 (Y, lateral) × 0.055 (Z)** |
| Pack centre (world) | (0, 0, 0.0875) |
| Pack AABB (world) | x ±0.100 · y ±0.225 · z **0.0600 … 0.1150** |
| **Pack underside — lowest collidable z** | **0.0600** |
| Pack top face | 0.1150 — flush with the shell underside, **no gap** |
| Pack mass | **4.000 kg** · 500 Wh · 4.95 L |
| Clear air to each tyre inner face | 0.145 |
| Retaining straps (visual only) | x = ±0.048…±0.072, y ±0.227, z **0.0575** … 0.1175 |

⚠️ **The straps hang 2.5 mm below the pack and you cannot thread a carriage between them.** The gap
between the two straps is only **0.096 m** (inner faces at x = ±0.048), and a berth with ±0.035 m of
fore-aft error moves the carriage right onto a strap. Treat **0.0575** as the hard rendered ceiling
under the pack, not 0.0600. The straps carry `contype=0 conaffinity=0 mass=0`, so they will not
*block* a carriage in simulation — they will simply be seen passing through it on video.

---

## 4 · Drive-over tunnel — the envelope a ground dock must live inside

Max height of any **ground-fixed** structure the rover passes over, swept along its X travel.
Datum `z = 0` is the plane the wheels stand on (true ground, or pad top if the rover drives onto a
pad — the numbers are identical either way).

| \|y\| band | all geometry (what you see) | collidable only |
|---|---|---|
| 0.000 … 0.227 | **0.0575** straps | 0.0600 battery |
| 0.227 … 0.232 | 0.1130 sill | 0.1150 shell |
| 0.232 … 0.370 | 0.0720 stub axles | 0.2700 PV frame |
| **0.370 … 0.430** | **0 — TYRE CONTACT PATCH. Nothing may be here.** | 0 |
| 0.430 … 0.440 | 0.0400 hubs | 0.2700 |
| 0.440 … 0.446 | 0.0770 spokes | 0.2700 |
| 0.446 … 0.505 | 0.2700 PV frame — free air under the panel overhang | 0.2700 |
| > 0.505 | unlimited — outside the machine's silhouette | unlimited |

> **Working rule for B: nothing in the drive lane above `z = 0.045`.**
> That is the 0.0575 rendered floor minus 12.5 mm for pitch transients, tyre sink and berth error.
> Anything taller than that in `|y| ≤ 0.37` stops the rover dead — it rams it with the pack.
>
> **A mast or electronics housing belongs at `|y| ≥ 0.52`**, beside the lane and outside the panel
> silhouette, where it can be any height at all.

---

## 5 · Lift stroke — the number that is NOT 0.060

Capturing the pack is the easy half. **Taking it away is what sets the stroke**, and it needs the
carriage to travel *below* the plane the wheels stand on.

| Step | Carriage top z | Why |
|---|---|---|
| Retracted (rover drives over) | ≤ **0.045** | drive-over tunnel, §4 |
| Capture (touching the pack) | **0.060** | pack underside |
| Released, carrying the pack | ≤ **−0.0075** | pack top = top + 0.055 must clear the 0.0575 strap line with 10 mm of margin |
| **Total stroke** | **≈ 0.070** | 0.060 − (−0.0075), round up |

Arithmetic: the bay is 0.060 deep and the pack is 0.055 thick, so **only 5 mm of drop is available
above the datum** before the pack fouls the rover it just came out of. Every millimetre of carriage
plate thickness comes straight out of that 5 mm. There is no version of this that works with the
carriage sitting on the surface.

**Two ways to buy the sub-datum travel:**

| Option | How | Cost |
|---|---|---|
| **P — drive onto a pad** *(recommended)* | Pad ≥ 0.025 thick spanning the full 0.860 over-tyres width; the lift well is a **slot in the pad**, not a hole in the ground | needs a ramp (below) and a pad ≥ 1.00 m wide |
| S — straddle | Pad narrow (≤ 0.55) between the wheels, rover stays on grade; the well must be **dug below grade** | needs a real hole modelled in the world file |

Option P is strongly preferred: the recess comes free inside pad material, and the rover's
break-over angle is `atan(0.060 / 0.110) = 28.6°`, so a short wheelbase makes the pad crest a
non-issue.

**Ramp, if B uses option P:** the rover **climbs 14° and stalls at 16°**. Use **≤ 10°**.

| Pad thickness | ramp run @ 10° | @ 14° (do not) |
|---|---|---|
| 0.020 | 0.113 | 0.080 |
| 0.030 | 0.170 | 0.120 |
| 0.040 | 0.227 | 0.160 |

⚠️ **A drive-over pad must either span the whole 0.860 m over-tyres width, or fit inside the
0.740 m clear span — never in between.** B's current 0.70 m pad is in the worst place: half-width
0.350 against a tyre inner face at 0.370 leaves **20 mm of margin**, which B's own ±0.08 m berth
tolerance blows through four times over. One wheel climbs the pad edge, the other does not, and the
rover berths tilted.

---

## 6 · Berth tolerance — the budget, with the arithmetic

The carriage must be **wide enough to catch a 0.450 m pack** and **narrow enough to miss tyre inner
faces at \|y\| = 0.370**. The 0.290 m of difference is the entire tolerance budget, and yaw eats it
fast because a 0.450 m pack has long corners.

Capture: `C ≥ 0.225·cos ε + 0.100·sin ε + e`
Clearance: `C ≤ 0.370·cos ε − 0.200·sin ε − e − m`
(`C` = carriage half-width, `e` = lateral berth error, `ε` = yaw error, `m` = 0.015 safety margin)

| Yaw error ε | max lateral error e | required carriage width |
|---|---|---|
| 0° | ±0.065 | 0.580 |
| 3° | ±0.057 | 0.574 |
| 6° | ±0.049 | 0.566 |
| **9°** | **±0.041** | **0.557** |
| 12° | ±0.032 | 0.546 |

**B's ±0.08 m XY tolerance is geometrically impossible at any yaw.** Its 9° yaw tolerance is fine
and is comfortably achievable (0.571 rad/s turn-in-place).

**A's recommendation to B:** keep `YAW_TOL_RAD = 0.157` (9°); change `XY_TOL_M` **0.08 → 0.035**;
size the carriage **0.55 (Y) × ≤ 0.09 (X)**. A 0.035 m / 9° berth is reachable at harvest creep
speed (0.117 m/s) — that is A's job, not B's, and A accepts it.

---

## 7 · Sites and sensors

| Site | Local pos (in `chassis`) | World pos | +Z | Purpose |
|---|---|---|---|---|
| `battery_mount` | (0, 0, −0.115) | (0, 0, **0.060**) | **(0,0,−1) DOWN** | dock interface |
| `panel_normal` | (0, 0, 0.181) | (0, 0, 0.356) | **(0,0,+1) UP** | `dot(+Z, sun_dir)` |
| `imu` | (0, 0, 0) | (0, 0, 0.175) | (0,0,+1) | body-frame sensors |

> `panel_normal` was originally declared with `fromto` and compiled with its frame **Z pointing
> down**, silently negating every exposure reading. It is now set with an explicit `zaxis` and the
> sensor reads `[0, 0, 1]`. **Do not revert it to `fromto`** — in MuJoCo 3.13 a `fromto` frame's +Z
> runs from the *second* point back to the *first*.

| Sensor | Type | Width | Reads |
|---|---|---|---|
| `panel_normal_world` | framezaxis | 3 | panel aim, world unit vector |
| `panel_pos_world` | framepos | 3 | panel centre, world |
| `chassis_vel` | velocimeter @ `imu` | 3 | body-frame linear velocity |
| `chassis_gyro` | gyro @ `imu` | 3 | body-frame angular rate; close the turn loop on `[2]` |

`nsensordata = 12`. `envs/traverse` adds a 9-ray rangefinder fan at runtime via `MjSpec` — that is
A's training observation, **not** part of this interface.

---

## 8 · Articulation, action space, solver

| | |
|---|---|
| Joints, in order | `root` (free, 7 qpos / 6 dof), `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr` (hinge, axis `0 1 0`) |
| Model size | `nq 11 · nv 10 · nu 4 · nbody 6 · nsite 3 · nsensor 4` |
| Actuators | `mot_fl`, `mot_fr`, `mot_rl`, `mot_rr` — `<velocity>` servos |
| Action space | **4 wheel velocity commands**, in the actuator order above |
| ctrlrange | **±3.1416 rad/s** (a stock 30 rpm gearmotor) |
| forcerange | **±2.4 Nm** |
| kv | 18 |
| Keyframe `home` | `qpos = 0 0 0.175  1 0 0 0  0 0 0 0`, `ctrl = 0 0 0 0` |

Derived: `v_max = 3.1416 × 0.090 = 0.2827 m/s` · max tractive force `4 × 2.4 / 0.09 = 106.7 N` ·
envelope corner `4 × 2.4 × 3.1416 = 30.2 W`.

**Solver context any co-simulated dock inherits** — `timestep 0.001` · `integrator implicitfast` ·
`cone ELLIPTIC` · `impratio 1` · `gravity −9.81`. ⚠️ **Do not "fix" the cone or raise impratio.**
The rover only scrub-steers because it slips; a pyramidal cone made it turn well for the wrong
reason, and `impratio 5` cut yaw to about a third.

Contact: there is **no `<contact>` block** — `nexclude = 0`, `npair = 0`. Wheel geoms carry
`priority = 1`, so in any tyre-vs-pad contact the tyre's friction (`0.65 0.005 0.0001`) wins and the
pad's own friction value is ignored. The battery geom is collidable (`condim 3`,
`friction 0.4 0.005 0.0001`) — and because it is a geom of `chassis`, **any force on it moves the
whole 29.4 kg rover.**

---

## 9 · Measured performance

| | |
|---|---|
| Mass | **29.400 kg** · loaded CoM z = **0.186497** |
| Top speed | **0.279 m/s** at 8.2 W mech (~16 W electrical) |
| Harvest speed | 0.117 m/s at 3.3 W mech |
| Turn rate, spin in place | **0.571 rad/s** at 20.6 W mech (~41 W electrical) |
| **Climb limit** | **14°** at 0.271 m/s — **stalls at 16°** |
| Nose-over (pitch) | 30.5° — the weak axis, by a factor of two |
| Break-over | 28.6° |
| Roll-tip | 65° — but it **slides at 33°** on μ = 0.65, so it never reaches tip |
| Envelope corner | **30.2 W mechanical** |

**Product thesis:** ~30 W of motion buys up to ~196 W of aimed collection. A spin costs ~2.5× cruise
— **the energy model must bill turns separately**, not assume a flat 30 W. A 90° turn takes 2.7 s
and costs 0.031 Wh.

**Note for the behaviour stack:** 0.571 rad/s consumes the **full** ctrl range, so approach must be
a straight-line creep, and `wheel_cmd ≈ 5.5 × desired_yaw_rate`.

---

## 10 · Detach mechanism — how the swap is actually realised

**The pack is currently a `<geom>` of `chassis`.** There is no pack body, no joint, no weld and no
equality constraint, so as shipped **there is physically nothing to detach.**

**DECISION (A, 15 Sep 2026): the pack splits into its own welded body in the *demo scene only*, via
`MjSpec` surgery at load time. `rover.xml` does not change.**

This is the same mechanism `envs/traverse/train.py` already uses to add terrain and sensors without
writing to `rover.xml`. **Verified working, this machine, mujoco 3.13.0:**

```
delete geom 'battery' from chassis → add worldbody body 'pack' (freejoint, same box, mass 4.0)
  → add mjEQ_WELD chassis↔pack, data = [0,0,0, 0,0,-0.0875, 1,0,0,0, 1]
compiles:  nq 18 · nv 16 · nu 4 · nbody 7 · neq 1 · total mass 29.4000 · CoM z 0.186497 (unchanged)
welded, 2 s:    pack z = 0.08734   (authored 0.08750 — 0.16 mm of drift)
eq_active = 0:  pack z = 0.02739   → detaches and drops to the ground. THE SWAP IS REAL.
```

Consequences for B: `detach_full_to_carriage()` and `latch_empty_to_vehicle()` are
`data.eq_active[pack_weld] = 0 / 1`. **Nothing is teleported and nothing is faked.** The training
model stays lean at `nq 11`; only the demo model carries the extra 6 DoF.

---

## 11 · Frozen vs open

**FROZEN — B may build against these and A will not move them:**

forward axis (+X) · `battery_mount` pose, name and direction · pack outer size, mass and 0.060
underside · the 0.045 drive-over ceiling · track 0.800, wheelbase 0.220, wheel radius 0.090 ·
0.740 clear span · actuator names, order, ctrlrange and forcerange · climb 14° / stall 16° · mass,
CoM, top speed, turn rate.

**OPEN — check before depending on:**

chassis styling and colours · panel tilt (flat today; a single pitch actuator is still under
consideration — it would move `panel_normal`, never `battery_mount`) · μ assumption (0.65 pasture;
0.80 also verified to turn) · whether the straps get raised 3 mm to make the rendered floor equal
the collidable floor (cosmetic, A's call, see below).

---

## 12 · Constraints the dock must respect — the short list

1. **Approach along ±X.** Forward is +X.
2. **Nothing above `z = 0.045` anywhere in `|y| ≤ 0.370`.** Put tall structure at `|y| ≥ 0.52`.
3. **Lift stroke ≈ 0.070**, from a retracted top ≤ 0.045, through capture at 0.060, down to
   **−0.0075 relative to the wheel-standing plane**. Budget sub-datum travel or the pack cannot come
   out.
4. **Pad either ≥ 1.00 m wide (wheels drive on) or ≤ 0.55 m wide (wheels straddle).** Never between
   0.55 and 1.00.
5. **Ramp onto the pad ≤ 10°.** Hard stall at 16°.
6. **Carriage 0.55 (Y) × ≤ 0.09 (X)**, with berth tolerance ±0.035 m / ±9°.
7. Nothing in `0.370 ≤ |y| ≤ 0.430` — that is the tyre contact patch.
8. Approach speed ≤ 0.279 m/s; a 90° reorientation takes ≈ 2.7 s; 1.5 m of run-in is 5.4 s.

---

## Known comment drift in `rover.xml` (A's to fix, harmless today)

| Header comment says | Measured | Note |
|---|---|---|
| glass "= 0.900 m² exactly" | **0.8811 m²** | 2.1 % short → 195.6 W peak at 222 W/m², not 200 W |
| "overall length … is 0.90 m" | **0.910** | PV frame outer |
| pack bottom "the lowest point on the machine" | straps at **0.0575** | true for collidable geometry only |

None of these are load-bearing for B. The pack, the mount frame and the clearances are all correct
in the file.
