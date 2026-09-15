# `envs/traverse` — pose + heading tracking with obstacle avoidance

One file, `train.py`. Trains the rover's `traverse()` skill: **drive to a commanded (x, y),
settle at a commanded yaw, do not hit anything, do not tip** — over ground that gets rougher and
more cluttered as training proceeds. Targets resample as they are reached, so one episode is
several commands.

> The reward contains **nothing about the sun**. "Face the sun" has a degenerate optimum — park
> facing the sun and never move. Sun-seeking is the planner's decision; this policy only executes
> "go here, end up pointing this way". Because skid steer makes heading == panel orientation,
> that is all the planner needs from us.

`robot/rover.xml` is owned by another workstream and is **never written to**. Sites, rangefinder
sensors, terrain, obstacles and the target marker are injected into an **in-memory `MjSpec`**
built from it. The file on disk is untouched.

---

## Stack: MuJoCo MJX + Brax PPO

- **Thousands of envs on one GPU.** MJX runs the physics step as XLA kernels, so 4096 rovers
  step in lockstep on the RTX PRO 6000. Brax PPO keeps rollout, GAE and SGD on-device.
- **Published precedent at this scale.** The Go1 joystick policy trains in ~7 minutes on a single
  RTX 4090 with this exact stack.
- **No conversion step.** The env is written directly against MJX and the bare `brax.envs.Env`
  interface, *not* `brax.PipelineEnv`, so brax's MJCF importer never has to understand
  `rover.xml` — only MuJoCo does. Same approach MuJoCo Playground takes.
- **The renderer is the same engine.** `--rollout` replays on plain CPU MuJoCo, which doubles as
  a sim-to-sim check (it also prints the max disagreement between the real MuJoCo rangefinder and
  the analytic ray model).

### ⚠️ The Blackwell (sm_120) risk

JAX CUDA wheels have historically lagged on new architectures, and the failure mode is silent:
JAX quietly runs on CPU and a 20-minute run becomes a 20-hour one. So the script's **first
action** is to print `jax.devices()`, the backend, device kind and compute capability, then
**hard-exit(2)** if the backend is not `gpu`. `--allow-cpu` overrides. Two further tripwires:

1. A 4096³ matmul benchmark at startup — below 5 TFLOP/s it warns loudly.
2. **The first jitted step is timed separately from steady state.** A first compile measured in
   minutes rather than seconds is the signature of runtime PTX→SASS JIT, i.e. a wheel with no
   Blackwell binary.

---

## Install (resolve latest, verify, pin what you got)

**Nothing here is pinned from memory.** Install current versions, read what the script prints at
startup, pin *those*.

```bash
pip install -U "jax[cuda12]" mujoco mujoco-mjx brax "imageio[ffmpeg]"
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

`train.py` prints resolved `jax / jaxlib / mujoco / mujoco-mjx / brax / flax / optax` on every
run. Copy that block into `requirements.txt` once a run succeeds. If `jax[cuda12]` has no sm_120
support, fall back to NVIDIA's JAX container (`nvcr.io/nvidia/jax`) or a nightly jaxlib — **not**
to CPU.

---

## Commands

```bash
# 1. pipeline gate — builds, steps, runs 2 tiny PPO iterations, exits
python train.py --smoke

# 2. the real run (~15–25 min on one RTX PRO 6000)
python train.py

# 3. ESCAPE HATCH: the simple pose-tracking task we know trains in ~10 min
python train.py --rough 0 --obstacles 0 --rubble 0

# 4. the deliverable
python train.py --rollout --params runs/params.pkl
python train.py --rollout --cam hero --rollout-difficulty 1.0   # model camera, max difficulty
```

Outputs: `runs/params.pkl`, `runs/meta.json`, `runs/rollout.mp4` (PNG + ffmpeg-line fallback).

Progress line per eval:

```
[  7]   21,000,000 ( 35.0%)  R=  184.3 len= 1250 dist= 0.71 head= 0.38 reach= 0.021 hit= 0.004 up= 0.99 diff= 0.58   980k sps   402.1s  eta 12.4m
```

`dist`/`head` should fall, `reach` rise, `hit` stay low, `up` near 1.0, `diff` ramp 0→1. If `len`
collapses the rover is tipping; if `hit` climbs while `reach` stalls, the collision penalty is
losing to progress and wants raising.

---

## Observation — 32 dims (30 with `--action-mode two`)

Body frame throughout: relative targets generalise, world coordinates do not.

| block | dims | notes |
|---|---|---|
| chassis linear velocity | 3 | `Rᵀ · qvel[lin]` — MuJoCo stores free-joint linear velocity in the **world** frame |
| chassis angular velocity | 3 | straight from `qvel[ang]`, which is **already body-local** |
| up-vector (tilt) | 3 | `Rᵀ·[0,0,1]` = third row of R |
| **slope (downhill)** | **2** | `sinθ · [cos ψ, sin ψ]` in body frame — what an IMU reads on a slope |
| wheel joint velocities | 4 | normalised by max ctrlrange |
| target offset dx, dy | 2 | body frame, clipped |
| heading error | 2 | `sin`, `cos` of `wrap(yaw_target − yaw)` |
| **rangefinders** | **9** | ±90° fan, 3 m, normalised to [0,1]; **1.0 = clear**, MuJoCo's −1 no-hit → max |
| previous action | 4 | keeps the action-rate penalty Markovian |

Both velocity-frame conventions were verified numerically against `mj_objectVelocity`, not
assumed. `--rough 0 --obstacles 0` **keeps all 32 dims** (the fan just reads constant-clear), so
params transfer between the simple and full tasks in both directions.

### The rangefinders are real MuJoCo sensors

9 `rangefinder` sensors on sites injected into the chassis at `(0.34, 0, 0.005)` — the existing
sensor pod. Deliberately cheap: IR/ultrasonic, not lidar, which suits a £370 machine and keeps
the observation small.

That mounting point matters. MuJoCo excludes only geoms in the *site's own body* from a ray cast,
so a fan mounted at the chassis centre would see the rover's own wheels at y = ±0.40. At x = 0.34
it does not — the front wheels only reach x = 0.20. **Verified: on an empty arena all 9 rays
return −1.** Verified too that the target marker and rubble at maximum amplitude are never seen
(rays sit at z ≈ 0.18 m; the marker tops out at 0.055 m and rubble at 0.07 m). That is why
`TARGET_Z` carries a "do not raise this" comment — `contype=0` kills contacts but **not** rays.

`--rangefinder auto` (default) probes MJX's sensordata against CPU MuJoCo at startup and falls
back to a closed-form ray-vs-cylinder model if MJX cannot evaluate the sensor. The fallback is
*exact* here, not an approximation: horizontal rays against vertical posts collapse to 2-D.

---

## Reward

Weights live in `REWARD_W` and are printed at startup. Rescaled for the **measured** 0.279 m/s
top speed — at that speed a per-step distance delta is ~5 mm, so the progress term is expressed
as a closing **speed** and weighted to dominate.

| term | weight | form |
|---|---|---|
| `progress` | **+2.0** | × closing speed in m/s, clipped ±0.5 → up to **+1.0/step**. The driving signal. |
| `pos` | +1.0 | `exp(−(d/0.6)²)` |
| `head` | +1.5 | `½(1+cos θ)·exp(−(d/0.8)²)` — **gated by proximity** |
| `reach` | +20.0 | one-off: `d < 0.35 m` **and** `\|θ\| < 0.5 rad`, then resample |
| `alive` | +0.05 | deliberately tiny — freezing must not pay |
| `ctrl` / `rate` / `energy` | −0.02 / −0.05 / −0.01 | `mean(a²)`, `mean(Δa²)`, `mean(\|cmd\|·\|wheel speed\|)` ≈ power (the 8 W story) |
| `tip` | −4.0 | `max(0, 0.90 − up_z)`, plus **−10.0** one-off on tip-over |
| `spin` | −0.5 | `max(0, \|ω_z\| − 0.5)²`. Light: measured max turn rate is 0.571 rad/s |
| **`collide`** | **−1.0** | per step overlapping an obstacle |
| **`collide_depth`** | **−2.0** | × overlap depth in metres |
| **`idle`** | **−0.3** | speed < 0.05 m/s while still > 0.8 m from the target. The anti-freeze term. |
| `bounds` | −1.0 | × metres outside the arena |

Clipped to [−25, +40] and NaN-guarded. **Collision is deliberately below progress+pos**: a rover
in contact loses at most ~1.5/step while driving earns up to 2.0/step plus 20 per target.

**Termination:** up-vector z below 0.40, non-finite state, or the 25 s limit.

**Target rejection.** Eight candidate targets are sampled at once and the one with the best
clearance to the nearest obstacle surface wins — loop-free, so it stays jittable. Targets inside
or immediately behind an obstacle lose, which is what keeps the task solvable.

Collision is measured geometrically (rover radius 0.51 m, computed from the model, vs each
obstacle's radius) rather than by introspecting MJX's contact arrays. That is version-proof and
gives a graded depth for free; the physics still does the real stopping.

---

## Curriculum

`u = rough · clip(t_env / (curriculum_frac · steps_per_env), 0, 1)`, where `t_env` is **each
env's own cumulative step count**, kept in `state.info`.

This works precisely because **brax's auto-reset never calls `reset()`** — it restores
`pipeline_state` and leaves `info` alone. So the counter survives episode boundaries and is a
faithful proxy for global training progress with no plumbing into the PPO loop.

| at difficulty `u` | 0.0 | 1.0 |
|---|---|---|
| rubble proud height | `rough · 0.015 m` | `rough · 0.070 m` |
| rubble density | 25 % of a 16-box pool | 100 % |
| obstacles active | 0 (they start at `u = 0.15`) | 8 |
| slope | 0° | **12°** |
| push disturbance | 0.002 /step | 0.008 /step |

Default ramp is the first **60 %** of training (`--curriculum-frac`). `--rough` caps the top end;
`--rough 0` is flat ground forever.

### Everything variable is a MOCAP body

Model geometry is shared across all envs and cannot ramp. **Mocap poses live in `mjx.Data`**, so
they are per-env and settable inside a jitted step. Rubble amplitude is controlled by **burial
depth** (fixed 0.05 m half-height boxes, z set to `−0.05 + proud`); inactive bodies are parked at
z = −50. The layout is re-applied from `info` on **every** step, because auto-reset would
otherwise restore the very first episode's layout forever.

### ⚠️ Slopes are a body force, not tilted geometry — and it is a 2 % approximation

No cliff-free ≤12° ramp can be built from box primitives at a 0.22 m wheelbase: every finite
tilted box has a vertical face, and a tile long enough to hide it is metres across. So a slope is
simulated as a constant horizontal force `m·g·sin θ` on the chassis, with the downhill direction
fed to the policy in body frame.

**This is dynamically exact apart from the normal-force term.** On a real slope the normal force
is `m·g·cos θ`; here it stays `m·g`. At the 12° cap that is a **2 % error in available traction**,
which flatters the rover slightly. Everything else — the downhill pull, the friction budget, the
IMU reading — is right. Do not later mistake this for a geometrically modelled slope.

**The 12° cap is a hard limit, not a tuning knob.** The rover climbs 14°, stalls at 16° and noses
over at 30.5°. Steeper is not "hard", it is impossible, and it teaches helplessness.

---

## Physics settings — two deliberate, measured trades

### 1. `--physics-dt 0.004` (rover.xml authors 0.001)

A 4× cut in MJX substeps per control step, 20 → 5. **This is a fidelity trade, taken knowingly.**
Measured on flat ground against the vehicle's own numbers:

| timestep | solver | substeps | top speed | turn rate | verdict |
|---|---|---|---|---|---|
| 0.001 | 2/6 | 20 | 0.276 | 0.475 | stable, slow |
| 0.002 | 2/6 | 10 | 0.276 | 0.567 | stable |
| **0.004** | **2/6** | **5** | 0.276 | **2.433** | **UNSTABLE — the rover flips during a spin** |
| **0.004** | **4/10** | **5** | **0.276** | **0.536** | **stable — the default** |
| 0.005 | 2/6 | 4 | — | 0.847 | diverges, rover ends up inverted |

*(spec: 0.279 m/s, 0.571 rad/s)*

### 2. Solver `4/10`, not `2/6`

That table is the whole reason. At 4 ms substeps the cheap solver makes a full-command spin
**diverge to 2.4 rad/s and flip the rover** — which would have silently taught the policy that
turning is fatal. Skid-steer yaw is friction-scrub dominated and is by far the most
timestep-sensitive thing in this model. 4/10 reproduces the measured turn rate within 7 %, is
stable through 60 s of aggressive random control over max-amplitude rubble, and is still
**3.6× faster** than 20 substeps at 2/6. The script warns if you lower these with a large
timestep. **Re-run a spin test before changing either.**

Also applied: cone `elliptic → pyramidal` (markedly faster and steadier in MJX).

### 3. Contact filtering (`--no-contact-filter` to disable)

Rover geoms get `contype=ROVER/conaffinity=WORLD`, static geoms the mirror image. Rover-vs-ground
and rover-vs-terrain are unaffected; **terrain-vs-terrain and terrain-vs-ground stop being tested
at all** — with 24 terrain bodies that is ~300 dead pairs per step. Rover self-collision also
goes, but the parent/child rule already excluded it (the wheels are children of the chassis).

Verified physics-neutral: filter on vs off gives an identical settled height (0.00 mm difference)
and identical speed. `--no-contact-filter` exists so contact behaviour can be bisected if
anything ever looks wrong.

---

## Honest training time, and what to cut

**~15–25 minutes on one RTX PRO 6000** for good pose tracking plus *cautious* obstacle behaviour
(slows, drifts around posts, occasional clip) — not confident avoidance. Three costs stacked at
once: 5 MJX substeps per control step, ~8 rover geoms × 25 terrain geoms of collision pairs, and
a genuinely harder task than pose tracking alone.

**Cut list, in priority order.** This is the thing to reach for under deadline pressure:

1. `--rubble 8 --obstacles 6` — fewer collision pairs, the single biggest lever
2. `--physics-dt 0.005 --episode-seconds 20` — *only with `--solver-iters 6 --ls-iters 12`;*
   0.005 at 2/6 diverges (see the table)
3. `--action-mode two` — 2-dim action space, genuinely easier and skid-steer-correct
4. Set `SLOPE_MAX_DEG = 0` — least visible thing in the video
5. Last resort: `--rough 0 --obstacles 0 --rubble 0` and ship the pose tracker that trains in
   ~10 minutes

---

## Degenerate solutions, and the term that blocks each

| behaviour | why it is tempting | what blocks it |
|---|---|---|
| **freeze** | never moving means zero collisions, and collision is the only large negative | `alive` cut to +0.05, plus the explicit **`idle` −0.3** when stopped > 0.8 m out. Driving pays up to +2.0/step and +20/target — an order of magnitude more. |
| **circle / pirouette** | heading reward looks farmable by spinning | `head` is gated by `exp(−(d/0.8)²)` so it only pays near the target; `progress` pays nothing for circling; `spin` on top |
| **wall-hug** | rangefinders read clear along a boundary | no walls exist; `bounds` penalises leaving the arena; targets are always sampled inside |
| **creep** | crawling avoids both collisions and tipping | `progress` is a **speed**, so crawling earns proportionally less; the +20 reach bonus rewards throughput |
| **farm the reach bonus in place** | +20 repeatedly at one spot | on reach the target immediately resamples 0.8–2.5 m away |
| **sun-parking** | — | no sun term exists anywhere, by design |

---

## Verified on this machine (CPU, against the real `rover.xml`)

`python -m py_compile` passes; `pyflakes` clean. No GPU here, so MJX/JAX paths are unverified.

- **MjSpec surgery** on the real model: +9 rangefinders, +1 marker, +16 rubble, +8 obstacles.
  `rover.xml` untouched. (Gotcha found: rangefinder sensors need `intprm[0] ≥ 1` or `compile()`
  raises `data spec (intprm[0]) must be positive`.)
- **Contact filter does not break rover-vs-ground**: `ncon = 8`, all eight are ground-vs-tyre
  (two per wheel), settled z = 0.1750 against a 0.175 spawn. No sinking, no fall-through.
- **Rangefinders**: post at (1.9, 0) r=0.13 → centre ray 1.433 m vs 1.43 predicted; second post
  at (1.3, −0.9) → the −45° ray at 1.012 m. Empty arena → all −1. Marker invisible at 15 tested
  placements; rubble at max amplitude invisible.
- **Escape hatch** `--rough 0 --obstacles 0 --rubble 0` builds, `nmocap = 1`, all 9 rays read
  3.0 m, **obs dim still 32**.
- **Timestep/solver sweep** as tabled above, plus 3 × 20 s of aggressive random control over
  max-amplitude rubble at the chosen settings: no flips, peak yaw 0.61 rad/s.
- **Render**: 1280×720, non-black (mean 99), chase camera and the model's `hero` camera both work.

## Known risks

1. **sm_120 wheels.** The script fails fast rather than training on CPU.
2. **MJX rangefinder support is version-dependent.** `--rangefinder auto` probes it against CPU
   MuJoCo at startup and falls back to the analytic model, which is exact for this geometry.
3. **MJX collision coverage** for cylinder-vs-box pairs. `--rubble 0 --obstacles 0` is the escape
   hatch; the startup banner prints the geom inventory.
4. **Headless rendering.** `MUJOCO_GL=egl` is set before MuJoCo imports; `osmesa` is the
   fallback. The script warns if every frame is black rather than shipping a black video.
5. **The 2 % slope traction approximation** above — flatters the rover very slightly.
