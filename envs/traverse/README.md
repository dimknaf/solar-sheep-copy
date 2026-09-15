# `envs/traverse` — pose + heading tracking policy

One file, `train.py`. Trains the rover's `traverse()` skill: **drive to a commanded
(x, y) and settle at a commanded yaw**, over slightly rough ground, without tipping.
Targets resample as they are reached, so one episode is several commands.

> The reward contains **nothing about the sun**. "Face the sun" has a degenerate optimum —
> park facing the sun and never move. Sun-seeking is the planner's decision; this policy only
> executes "go here, end up pointing this way". Because skid steer makes heading == panel
> orientation, that is all the planner needs from us.

`robot/rover.xml` is owned by another workstream and is **never written to**. Everything the
training and rendering setup needs (ground plane if absent, lights, bumps, a target marker) is
injected by compiling a small wrapper MJCF that `<include>`s it.

---

## Stack: MuJoCo MJX + Brax PPO

Chosen over Isaac Lab / SB3-on-CPU because it is the only route that fits the budget:

- **Thousands of envs on one GPU.** MJX runs the whole physics step as XLA kernels, so 4096
  rovers step in lockstep on the RTX PRO 6000. Brax PPO keeps rollout, GAE and SGD on-device —
  no host round-trip per step.
- **Published precedent at this scale.** The Go1 joystick policy trains in roughly 7 minutes on
  a single RTX 4090 with this exact stack. A 4-wheel skid-steer rover has a far smaller action
  space than a 12-DoF quadruped, so ~10 minutes is a comfortable target, not a stretch.
- **No conversion step.** The env is written directly against MJX and the bare `brax.envs.Env`
  interface, *not* `brax.PipelineEnv`. That means brax's MJCF importer never has to understand
  `rover.xml` — only MuJoCo does. Same approach MuJoCo Playground takes, and it removes the
  most common source of "works in MuJoCo, fails in Brax".
- **The renderer is the same engine.** `--rollout` replays on plain CPU MuJoCo with the offscreen
  renderer, which doubles as a free sim-to-sim check: if the MJX-trained policy also works under
  the reference C solver, it is not exploiting an MJX artefact.

### ⚠️ The Blackwell (sm_120) risk

JAX CUDA wheels have historically lagged on new architectures, and the failure mode is silent:
JAX quietly runs on CPU and a 10-minute run becomes a 10-hour one. So the script's **first
action**, before touching MuJoCo, is to print `jax.devices()`, the backend, the device kind and
the compute capability, then **hard-exit(2)** if the backend is not `gpu`. `--allow-cpu` overrides.

Two further tripwires, because "a GPU is visible" is not the same as "sm_120 code was emitted":

1. A 4096³ matmul benchmark at startup. A healthy RTX PRO 6000 is far above 5 TFLOP/s; below
   that the script warns loudly and tells you to suspect the jaxlib CUDA build.
2. **The first jitted step is timed separately from steady state.** A pathologically slow first
   compile (minutes, not seconds) is the signature of runtime PTX→SASS JIT, i.e. the wheel
   shipped no Blackwell binary. The script prints `FIRST reset`, `FIRST step` and `steady step`
   on their own lines and warns above 240 s.

---

## Install (resolve latest, verify, pin what you got)

**Nothing in this repo is pinned from memory.** Install the current versions, read what the
script prints at startup, and pin *those* — not whatever a model remembered.

```bash
pip install -U "jax[cuda12]" mujoco mujoco-mjx brax "imageio[ffmpeg]"
```

Then verify before trusting it:

```bash
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

`train.py` prints the resolved `jax / jaxlib / mujoco / mujoco-mjx / brax / flax / optax`
versions on every run. Copy that block into `requirements.txt` once a run succeeds.

If `jax[cuda12]` has no sm_120 support yet, the fallbacks in order are: NVIDIA's JAX container
(`nvcr.io/nvidia/jax`), or a nightly jaxlib. Do not "fix" it by training on CPU.

---

## Commands

```bash
# 1. pipeline gate — builds the env, steps it, runs 2 tiny PPO iterations, exits (~1 min)
python train.py --smoke

# 2. the real run (~10 min on one RTX PRO 6000)
python train.py

# 3. the deliverable: one episode, MP4 with a chase camera following the rover
python train.py --rollout --params runs/params.pkl
```

Useful knobs: `--steps 50000000 --num-envs 4096 --episode-seconds 20 --bumps 12
--action-mode two --model /path/to/rover.xml --cam-distance 3.2`.

Run `--smoke` first, every time. It is the gate.

### Outputs

| path | what |
|---|---|
| `runs/params.pkl` | trained PPO params (normalizer + policy + value), via `brax.io.model` |
| `runs/meta.json` | obs/act dims, reward weights, resolved versions, wall clock — everything `--rollout` needs to reproduce the policy |
| `runs/rollout.mp4` | 1280×720 chase-cam video of one episode |
| `runs/rollout/frame_*.png` | fallback if no ffmpeg; the script prints the exact `ffmpeg` line to run |

Per-iteration progress goes to stdout:

```
[  7]   17,825,792 steps ( 35.7%)  R=  184.3 +/-31.2   len= 1000 dist= 0.71 head= 0.38 reach= 0.021 up= 0.99   1,430k sps   231.4s  eta 4.1m
```

`dist` should fall, `head` should fall, `reach` (fraction of steps at a satisfied target) should
rise, `up` should stay near 1.0. If `len` collapses, the rover is tipping.

---

## Observation / action / reward

**Observation (21 floats for 4 wheels + 4 actions)** — everything in the rover's own body frame,
because relative targets generalise and world coordinates do not:

| block | dims | notes |
|---|---|---|
| chassis linear velocity | 3 | `Rᵀ · qvel[lin]` — MuJoCo stores free-joint linear velocity in the **world** frame |
| chassis angular velocity | 3 | taken straight from `qvel[ang]`, which is **already body-local** |
| up-vector (tilt) | 3 | `Rᵀ·[0,0,1]`, i.e. the third row of R |
| wheel joint velocities | 4 | normalised by max ctrlrange |
| target offset dx, dy | 2 | body frame, clipped |
| heading error | 2 | `sin`, `cos` of `wrap(yaw_target − yaw)` |
| previous action | 4 | keeps the action-rate penalty Markovian |

(Both velocity-frame conventions above were verified numerically against `mj_objectVelocity`,
not assumed.)

**Action** — one command per wheel actuator in `[-1, 1]`, affine-mapped onto each actuator's
`ctrlrange`. `--action-mode two` collapses it to a left/right pair (true skid steer; usually
learns faster and cannot fight itself).

**Reward** — weights live in one dict, `REWARD_W` in `train.py`, and are printed at startup:

| term | weight | form |
|---|---|---|
| `progress` | **+8.0** | `clip(d_prev − d, ±0.5)` — shaped closing speed on the target |
| `pos` | **+1.5** | `exp(−(d/0.6)²)` — distance-to-target bump |
| `head` | **+1.5** | `½(1+cos θ_err) · exp(−(d/0.8)²)` — heading alignment, **gated by proximity** so it only pays near the target |
| `reach` | **+15.0** | one-off when `d < 0.35 m` **and** `\|θ_err\| < 0.5 rad`, then a new target is sampled |
| `alive` | **+0.2** | per step |
| `ctrl` | **−0.02** | `mean(a²)` |
| `rate` | **−0.02** | `mean((a − a_prev)²)` — anti-chatter |
| `energy` | **−0.005** | `mean(\|cmd\| · \|wheel speed\|)` ≈ mechanical power. This is the 30 W-vs-300 W product argument, in the loss |
| `tip` | **−4.0** | `max(0, 0.90 − up_z)`, plus **−10.0** one-off on tip-over |
| `spin` | **−0.3** | `max(0, \|ω_z\| − 2.5)²` — no gratuitous pirouettes |

Reward is clipped to `[−25, +35]` and NaN-guarded.

**Termination:** up-vector z below **0.40** (tipped), non-finite state, or the 20 s time limit.

**Why `head` is gated by distance:** an ungated heading term is satisfiable by spinning on the
spot anywhere in the arena, which competes with driving. Gating it behind `exp(−(d/0.8)²)` means
"get there first, then point" is the only way to collect it.

**Rough ground** comes from three places, none of which touch `rover.xml`: a field of `--bumps`
small static boxes (1–3 cm proud) injected by the arena wrapper; a random roll/pitch and yaw at
every reset; and random shoves (0.4 % of steps) to the chassis during the episode. Set
`--bumps 0` if MJX complains about an unsupported collision pair with the real wheel geometry —
the disturbances alone still produce a robust policy.

---

## Defaults, and why

`4096` envs · `512` batch · `32` minibatches · `20`-step unroll · `4` updates/batch ·
`γ = 0.995` · `lr 3e-4` · entropy `5e-3` · policy `(128,128,128)` / value `(256,256,256)` ·
`50 M` steps.

The discount is the one to leave alone: at 50 Hz a 3 m drive is ~400 steps, so `γ = 0.99`
(≈100-step horizon) cannot see the target it is being asked to reach. Everything else is
routine; `batch_size × num_minibatches` must stay a multiple of `--num-envs` and the script
exits with a clear message if you break that.

## Known risks

1. **sm_120 wheels.** Covered above; the script fails fast rather than training on CPU.
2. **MJX collision coverage.** If the real wheels are cylinders and MJX has no cylinder-vs-box
   pair, `--bumps 0` is the escape hatch. The startup banner prints the geom inventory.
3. **`rover.xml` contents are assumed, not seen.** The script resolves the chassis by the free
   joint (preferring a body literally named `chassis`) and the wheels/actuators by name, with
   loud fallbacks to "first free joint" and "all hinge joints". Check those two banner lines on
   the first run.
4. **Headless rendering.** `MUJOCO_GL=egl` is set before MuJoCo is imported. If the VM has no
   EGL, use `MUJOCO_GL=osmesa python train.py --rollout`. The script warns if every frame is
   black rather than silently shipping a black video.
