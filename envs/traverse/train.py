#!/usr/bin/env python3
"""
traverse/train.py -- pose + heading tracking with obstacle avoidance, for the
4-wheel skid-steer solar rover.

ONE file. MuJoCo MJX (GPU physics) + Brax PPO. Target box: a single NVIDIA
RTX PRO 6000 (Blackwell, sm_120).

The task the policy learns:
    "drive to (x, y), end up pointing at yaw, do not hit anything, do not tip"
    -- over ground that gets rougher and more cluttered as training proceeds.
    Targets resample as they are reached, so one episode is several commands.

    NOT in the reward: anything about the sun. "Face the sun" has a degenerate
    optimum (park facing the sun, never move). Sun-seeking is the planner's
    decision; this policy only executes the resulting command.

robot/rover.xml is owned by another workstream and is NEVER written to. Sites,
rangefinder sensors, terrain and obstacles are injected into an in-memory
MjSpec built from it. The file on disk is untouched.

Observation (32 floats with 4 wheels + 4 actions) -- body frame throughout,
because relative targets generalise and world coordinates do not:
    chassis linear velocity        3
    chassis angular velocity       3
    up-vector / gravity tilt       3
    slope tilt (downhill, body)    2
    wheel joint velocities         4
    target offset dx, dy           2
    sin/cos of heading error       2
    rangefinder fan                9   <- 9 rays, +/-90 deg, 3 m, normalised
    previous action                4

Usage:
    python train.py --smoke                     # pipeline gate, exits fast
    python train.py                             # the real run
    python train.py --rough 0 --obstacles 0     # escape hatch: the simple task
    python train.py --rollout --params runs/params.pkl   # writes an MP4

Nothing here is pinned from memory. The script prints the versions it actually
resolved at startup -- see README.md.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# MUST be set before mujoco is imported anywhere, or the offscreen renderer on a
# headless VM either dies or hands back black frames.
# ---------------------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent.parent / "robot" / "rover.xml")

# --- hard limits measured on the real vehicle. Do not exceed these. ---------
SLOPE_MAX_DEG = 12.0   # rover climbs 14 deg, stalls at 16 deg, noses over at 30.5 deg.
                       # 12 deg is "hard but possible". Steeper teaches helplessness.
TOP_SPEED = 0.279      # m/s, measured
TURN_RATE = 0.571      # rad/s, measured

# --- perception ------------------------------------------------------------
RF_N = 9               # rangefinders
RF_SPAN = math.pi / 2  # +/-90 deg fan
RF_MAX = 3.0           # m. Cheap IR/ultrasonic, not lidar -- suits a GBP 370 machine
RF_POS = (0.34, 0.0, 0.005)   # chassis-local: the existing sensor pod, ahead of the
                              # wheels so the fan never sees the rover's own tyres

# --- task tolerances -------------------------------------------------------
REACH_R = 0.35         # m
REACH_YAW = 0.50       # rad (~29 deg)
TIP_TERM = 0.40        # terminate when body up-vector z drops below this
TIP_SOFT = 0.90        # start penalising tilt below this
SPIN_MAX = 0.50        # rad/s before the spin penalty bites

# --- reward weights. One place, printed at startup, quoted in the README ----
REWARD_W = dict(
    progress=2.0,     # + closing speed on the target, in m/s. THE driving signal.
    pos=1.0,          # + Gaussian bump on distance-to-target
    head=1.5,         # + heading alignment, gated by proximity to the target
    reach=20.0,       # + one-off when a target is reached (then resample)
    alive=0.05,       # + per step. Deliberately tiny: freezing must not pay.
    ctrl=0.02,        # - control magnitude
    rate=0.05,        # - action rate (jerk / chatter)
    energy=0.01,      # - |cmd| * |wheel speed| ~ mechanical power (the 8 W story)
    tip=4.0,          # - tilt, from the body up-vector z component
    tipover=10.0,     # - one-off, on the tip-over termination
    spin=0.5,         # - yaw rate beyond SPIN_MAX
    collide=1.0,      # - per step in contact with an obstacle  }  kept BELOW the
    collide_depth=2.0,  # - overlap depth, metres                }  progress reward
    idle=0.3,         # - parked while still far from the target. The anti-freeze term.
    bounds=1.0,       # - leaving the arena
)


# ===========================================================================
# 1. CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pose+heading tracking with obstacle avoidance for a skid-steer rover (MJX + Brax PPO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="Build env, step it, run 2 tiny PPO iterations, print shapes, exit fast.")
    p.add_argument("--rollout", action="store_true",
                   help="Load --params, roll out one episode on CPU MuJoCo, write an MP4.")
    p.add_argument("--allow-cpu", action="store_true", help="Do not hard-fail when no GPU is visible.")

    # model / arena
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to rover.xml. READ ONLY; never written.")
    p.add_argument("--arena", type=float, default=6.0, help="Half-size of the drivable area, metres.")
    p.add_argument("--ctrl-dt", type=float, default=0.02, help="Control period, seconds.")
    p.add_argument("--physics-dt", type=float, default=0.004,
                   help="Override the model's timestep for MJX throughput. 0 = keep the model's.")
    p.add_argument("--episode-seconds", type=float, default=25.0, help="Episode time limit.")
    p.add_argument("--action-mode", choices=["four", "two"], default="four",
                   help="'four' = one command per wheel. 'two' = left/right pair (true skid steer).")
    # Measured on this rover: at 4 ms substeps, solver 2/6 makes a full-command
    # spin blow up (2.4 rad/s and the rover flips; the real limit is 0.571).
    # 4/10 is stable, reproduces the measured turn rate, and is still faster
    # than 10 substeps at 2/6. Do not lower these without re-running a spin test.
    p.add_argument("--solver-iters", type=int, default=4)
    p.add_argument("--ls-iters", type=int, default=10)

    # curriculum / terrain / obstacles
    p.add_argument("--rough", type=float, default=1.0, metavar="0..1",
                   help="MAX terrain difficulty. Ramps 0 -> this over training. 0 = flat ground.")
    p.add_argument("--rubble", type=int, default=16, help="Size of the rubble-box pool (density ramps).")
    p.add_argument("--obstacles", type=int, default=8,
                   help="Size of the obstacle pool: rocks/posts the rover must avoid. 0 = none.")
    p.add_argument("--curriculum-frac", type=float, default=0.6,
                   help="Fraction of training over which difficulty ramps 0 -> --rough.")
    p.add_argument("--rangefinder", choices=["auto", "sensor", "analytic"], default="auto",
                   help="'sensor' = MuJoCo rangefinders via mjx sensordata. 'analytic' = closed-form "
                        "ray-vs-obstacle in JAX. 'auto' probes MJX against CPU MuJoCo and picks.")
    p.add_argument("--no-contact-filter", action="store_true",
                   help="Disable the contype/conaffinity filter that stops terrain colliding with "
                        "terrain and with the ground (~300 wasted static pairs per step). Slower; "
                        "here so contact behaviour can be bisected against the unfiltered model.")

    # PPO
    p.add_argument("--steps", type=int, default=60_000_000, help="Total environment steps.")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=5e-3)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--evals", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)

    # io
    p.add_argument("--out", default="runs")
    p.add_argument("--params", default=None, help="Params file for --rollout (default: <out>/params.pkl).")
    p.add_argument("--video", default=None, help="MP4 path for --rollout (default: <out>/rollout.mp4).")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--cam", default="chase", help="'chase', or the name of a camera in rover.xml (e.g. hero).")
    p.add_argument("--cam-distance", type=float, default=3.4)
    p.add_argument("--cam-elevation", type=float, default=-20.0)
    p.add_argument("--rollout-difficulty", type=float, default=-1.0,
                   help="Fix curriculum difficulty for --rollout (0..1). -1 = use --rough.")
    p.add_argument("--gl", default=None, help="Override MUJOCO_GL (egl | osmesa | glfw).")
    return p.parse_args(argv)


# ===========================================================================
# 2. The loud GPU / version banner. Deliberately the first thing that happens:
#    a silent CPU fallback is the most expensive failure available to us.
# ===========================================================================
def _version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        try:
            return __import__(pkg).__version__
        except Exception:
            return "?"


def banner_and_gpu_check(allow_cpu):
    t0 = time.time()
    import jax
    import jax.numpy as jnp

    print("=" * 78)
    print("RESOLVED VERSIONS (nothing is pinned from memory -- pin what you see here)")
    for pkg in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "flax", "optax"):
        print(f"    {pkg:<12} {_version(pkg)}")
    print(f"    python       {sys.version.split()[0]}")
    print(f"    MUJOCO_GL    {os.environ.get('MUJOCO_GL')}")
    print("-" * 78)

    devices = jax.devices()
    backend = jax.default_backend()
    print(f"jax.default_backend() = {backend!r}")
    print(f"jax.devices()         = {devices}")
    for d in devices:
        cc = getattr(d, "compute_capability", None)
        print(f"    device[{d.id}] platform={d.platform} kind={d.device_kind!r}"
              + (f"  compute_capability={cc}" if cc else ""))
    print(f"(jax import + device discovery took {time.time() - t0:.1f}s)")

    if backend != "gpu":
        print("\n" + "!" * 78 +
              f"\n!! NO GPU VISIBLE TO JAX. default_backend() == {backend!r}\n"
              "!! Training on CPU would take HOURS, not the ~10 minutes budgeted.\n"
              "!! Most likely causes, in order:\n"
              "!!   1. jax installed without CUDA:  pip install -U 'jax[cuda12]'\n"
              "!!   2. the CUDA jaxlib wheel has no sm_120 (Blackwell) code -- look for\n"
              "!!      'ptxas' / 'no kernel image is available' messages above.\n"
              "!!   3. CUDA_VISIBLE_DEVICES is empty, or the driver predates the wheel.\n"
              "!! Re-run with --allow-cpu ONLY if you know you want the slow path.\n"
              + "!" * 78 + "\n", file=sys.stderr)
        if not allow_cpu:
            sys.exit(2)

    if backend == "gpu":
        n = 4096
        a = jnp.ones((n, n), jnp.float32)
        f = jax.jit(lambda x: x @ x.T)
        t = time.time(); jax.block_until_ready(f(a)); compile_s = time.time() - t
        t = time.time()
        for _ in range(5):
            r = f(a)
        jax.block_until_ready(r); run_s = (time.time() - t) / 5
        tflops = 2 * n ** 3 / run_s / 1e12
        print(f"GPU sanity matmul {n}x{n}: first(compile)={compile_s:.2f}s  "
              f"steady={run_s * 1e3:.2f}ms  ~{tflops:.1f} TFLOP/s")
        if tflops < 5.0:
            print("!! WARNING: <5 TFLOP/s on a 4096^3 matmul. That is NOT a healthy RTX PRO 6000.\n"
                  "!! Strongly suspect a bad sm_120 code path. Check the jaxlib CUDA build first.",
                  file=sys.stderr)
    print("=" * 78, flush=True)


# ===========================================================================
# 3. Model building -- MjSpec surgery on an in-memory copy of rover.xml.
#
#    rover.xml is owned by another workstream. We load it into an MjSpec, add
#    what training and rendering need, and compile. Nothing is written to disk.
#
#    Everything that has to change per episode (terrain height, obstacle
#    layout, target pose) lives on MOCAP bodies, because mocap poses live in
#    mjx.Data -- so they are per-env and settable from inside a jitted step.
#    Geometry in the Model is shared by all envs and cannot ramp.
# ===========================================================================
def _zaxis_quat(phi):
    """Quaternion rotating +z onto the horizontal direction (cos phi, sin phi, 0)."""
    s = math.sin(math.pi / 4)
    return [math.cos(math.pi / 4), -math.sin(phi) * s, math.cos(phi) * s, 0.0]


def rf_angles():
    return [-RF_SPAN + 2 * RF_SPAN * i / (RF_N - 1) for i in range(RF_N)]


# Collision filter bits. Terrain must not collide with terrain or with the
# ground plane: 16 rubble boxes + 8 obstacles is ~300 needless static-static
# pairs, and MJX pays for every one of them on every step.
BIT_ROVER, BIT_WORLD = 1, 2

RUBBLE_HALF_Z = 0.05      # rubble boxes are buried to control how proud they stand
OBST_HALF_Z = 0.30        # obstacles stand 0.60 m tall: above the ray height, not drivable
FAR_XY, FAR_Z = 1.0e3, -50.0   # where inactive mocap bodies are parked
TARGET_Z = 0.04           # target marker height. MUST stay well below the ray fan
                          # (~0.18 m) or it reads as a phantom obstacle.


def build_model(args, video_wh=(1280, 720)):
    """Compile the training model. Returns (MjModel, info dict). Never writes to disk."""
    import mujoco
    import numpy as np

    path = Path(args.model).resolve()
    if not path.exists():
        sys.exit(f"ERROR: model not found at {path}\n"
                 f"       (owned by another workstream -- pass --model if it moved)")

    probe = mujoco.MjModel.from_xml_path(str(path))
    print(f"[model] {path}")
    print(f"[model] rover.xml provides: nq={probe.nq} nv={probe.nv} nu={probe.nu} "
          f"nbody={probe.nbody} ngeom={probe.ngeom} nsensor={probe.nsensor} "
          f"nlight={probe.nlight} nkey={probe.nkey} timestep={probe.opt.timestep}")

    rng = np.random.default_rng(args.seed)
    n_rub, n_obs = max(0, args.rubble), max(0, args.obstacles)
    surgery_ok = True
    try:
        spec = mujoco.MjSpec.from_file(str(path))
        chassis = spec.body("chassis")
        if chassis is None:
            raise RuntimeError("no body named 'chassis'")

        # --- 1. the rangefinder fan, on the existing sensor pod -------------
        for i, phi in enumerate(rf_angles()):
            st = chassis.add_site()
            st.name = f"tv_rf{i}"
            st.pos = list(RF_POS)
            st.quat = _zaxis_quat(phi)
            st.size = [0.004, 0.004, 0.004]
            st.group = 4
            st.rgba = [1.0, 0.25, 0.1, 0.35]
            sen = spec.add_sensor()
            sen.name = f"tv_rf{i}"
            sen.type = mujoco.mjtSensor.mjSENS_RANGEFINDER
            sen.objtype = mujoco.mjtObj.mjOBJ_SITE
            sen.objname = f"tv_rf{i}"
            try:    # newer MuJoCo wants a positive ray count in intprm[0]
                ip = list(sen.intprm); ip[0] = 1; sen.intprm = ip
            except Exception:
                pass

        # --- 2. collision filtering ----------------------------------------
        # Rover geoms get contype=ROVER/conaffinity=WORLD, everything static gets
        # the mirror image. Rover-vs-ground and rover-vs-terrain are unaffected;
        # terrain-vs-terrain and terrain-vs-ground stop being tested at all. With
        # 24 terrain bodies that is ~300 dead static pairs per step in MJX.
        # (Rover self-collision also goes, but the parent/child rule already
        # excluded it -- the wheels are children of the chassis.)
        filt = not args.no_contact_filter
        n_filtered = 0
        if filt:
            for b in spec.bodies:
                static = (b.name == spec.worldbody.name)
                for g in b.geoms:
                    if g.contype or g.conaffinity:   # leave visual-only geoms alone
                        g.contype, g.conaffinity = ((BIT_WORLD, BIT_ROVER) if static
                                                    else (BIT_ROVER, BIT_WORLD))
                        n_filtered += 1
        wb = spec.worldbody
        ground = next((g for g in wb.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE), None)
        print(f"[model] contact filter {'ON' if filt else 'OFF'} "
              f"({n_filtered} colliding geoms re-bitted; --no-contact-filter disables)")
        WB_BITS = (BIT_WORLD, BIT_ROVER) if filt else (1, 1)
        if ground is None:      # rover.xml has no floor: give it one
            g = wb.add_geom()
            g.name = "tv_ground"; g.type = mujoco.mjtGeom.mjGEOM_PLANE
            g.size = [0, 0, 0.05]; g.rgba = [0.30, 0.32, 0.34, 1.0]
            g.friction = [0.65, 0.005, 0.0001]; g.condim = 3
            g.contype, g.conaffinity = WB_BITS
            try:    # a checker texture if this MuJoCo's material API cooperates
                tex = spec.add_texture()
                tex.name = "tv_grid"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D
                tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
                tex.rgb1 = [0.24, 0.26, 0.28]; tex.rgb2 = [0.33, 0.35, 0.37]
                tex.width = tex.height = 300
                mat = spec.add_material(); mat.name = "tv_grid"
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tv_grid"
                mat.texrepeat = [10, 10]; mat.texuniform = True
                g.material = "tv_grid"
            except Exception:
                pass            # plain colour is fine; do not lose the whole surgery
            print("[model] rover.xml had no ground plane -- added one")

        if probe.nlight == 0:   # never ship a black video
            for nm, pos, dr, dif, shadow in (
                    ("tv_key", [3, -3, 6], [-0.4, 0.4, -1], [0.75, 0.72, 0.65], True),
                    ("tv_fill", [-5, 4, 5], [0.5, -0.4, -1], [0.28, 0.30, 0.34], False)):
                li = wb.add_light()
                li.name = nm; li.pos = pos; li.dir = dr
                li.diffuse = dif; li.castshadow = shadow
                try:
                    li.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
                except Exception:
                    li.directional = True
            print("[model] rover.xml had no lights -- added key + fill")

        # --- 3. the commanded-pose marker (mocap, no collision) -------------
        # Kept DELIBERATELY FLAT. contype/conaffinity 0 removes contacts, but
        # rangefinder rays do not respect those bits -- a tall marker would read
        # as a phantom obstacle straight into the observation. Top of the marker
        # is 0.055 m; the ray fan sits at ~0.18 m. Do not raise it.
        tb = wb.add_body(); tb.name = "tv_target"; tb.mocap = True; tb.pos = [2, 0, TARGET_Z]
        g = tb.add_geom()
        g.name = "tv_tgt_ring"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [REACH_R, 0.015, 0]; g.rgba = [0.15, 0.95, 0.45, 0.55]
        g.contype = g.conaffinity = 0; g.group = 1; g.mass = 1e-6
        g = tb.add_geom()
        g.name = "tv_tgt_dir"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [REACH_R + 0.28, 0, 0.0]; g.size = [0.28, 0.07, 0.015]
        g.rgba = [1.0, 0.72, 0.10, 0.95]
        g.contype = g.conaffinity = 0; g.group = 1; g.mass = 1e-6

        # --- 4. rubble pool: buried boxes. Burial depth = terrain amplitude -
        rub_size = []
        for i in range(n_rub):
            sx, sy = rng.uniform(0.10, 0.34), rng.uniform(0.08, 0.26)
            rub_size.append((sx, sy))
            b = wb.add_body(); b.name = f"tv_rub{i}"; b.mocap = True; b.pos = [0, 0, FAR_Z]
            g = b.add_geom()
            g.name = f"tv_rub{i}_g"; g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [sx, sy, RUBBLE_HALF_Z]
            g.rgba = [0.42, 0.40, 0.36, 1.0]
            g.friction = [0.9, 0.02, 0.001]; g.condim = 3
            g.contype, g.conaffinity = WB_BITS
            g.mass = 1e-6

        # --- 5. obstacle pool: rocks and posts, varied radius ---------------
        obst_r = []
        for i in range(n_obs):
            r = float(rng.uniform(0.10, 0.30))
            obst_r.append(r)
            b = wb.add_body(); b.name = f"tv_obs{i}"; b.mocap = True; b.pos = [FAR_XY, FAR_XY, FAR_Z]
            g = b.add_geom()
            g.name = f"tv_obs{i}_g"
            if i % 2 == 0:
                g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                g.size = [r, OBST_HALF_Z, 0]
                g.rgba = [0.46, 0.42, 0.38, 1.0]
            else:
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [r, r, OBST_HALF_Z]
                g.rgba = [0.38, 0.34, 0.30, 1.0]
            g.friction = [0.9, 0.02, 0.001]; g.condim = 3
            g.contype, g.conaffinity = WB_BITS
            g.mass = 1e-6

        m = spec.compile()
        print(f"[model] MjSpec surgery OK: +{RF_N} rangefinders +1 target marker "
              f"+{n_rub} rubble +{n_obs} obstacles (all mocap; rover.xml untouched)")
    except Exception as e:  # noqa: BLE001 -- any failure must degrade, not crash
        print("!" * 78, file=sys.stderr)
        print(f"!! MjSpec surgery failed ({type(e).__name__}: {e})", file=sys.stderr)
        print("!! Falling back to the bare rover.xml: NO rangefinders, NO obstacles,", file=sys.stderr)
        print("!! NO rubble. Pose tracking still trains; obstacle avoidance does not.", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        m = mujoco.MjModel.from_xml_path(str(path))
        surgery_ok = False
        n_rub = n_obs = 0
        rub_size, obst_r = [], []

    # --- solver / renderer settings in Python, not XML (lower risk) ---------
    if args.physics_dt > 0 and abs(m.opt.timestep - args.physics_dt) > 1e-9:
        print(f"[model] timestep {m.opt.timestep} -> {args.physics_dt} "
              f"(--physics-dt; {args.physics_dt/m.opt.timestep:.0f}x fewer MJX substeps)")
        m.opt.timestep = args.physics_dt
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations = args.solver_iters
    m.opt.ls_iterations = args.ls_iters
    if m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        print("[model] cone elliptic -> pyramidal (MJX is markedly faster and steadier with it)")
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    # MJX refuses implicitfast the moment any fluid drag is active
    # ("implicitfast not implemented for fluid drag"), and rover.xml authors
    # density="1.2" (air).  At 0.279 m/s the dynamic pressure on the panel is
    # ~0.05 Pa against a 106.7 N tractive force -- six orders of magnitude down,
    # so dropping it costs nothing physical and keeps the fast integrator.
    if m.opt.density != 0.0 or m.opt.viscosity != 0.0:
        print(f"[model] fluid drag density={m.opt.density} viscosity={m.opt.viscosity} -> 0 "
              f"(MJX cannot pair implicitfast with fluid drag; negligible at 0.28 m/s)")
        m.opt.density = 0.0
        m.opt.viscosity = 0.0
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, video_wh[0])
    m.vis.global_.offheight = max(m.vis.global_.offheight, video_wh[1])
    m.vis.map.znear = max(m.vis.map.znear, 0.01)

    n_frames = max(1, int(round(args.ctrl_dt / m.opt.timestep)))
    real_dt = n_frames * m.opt.timestep
    print(f"[model] timestep={m.opt.timestep}  n_frames={n_frames}  control dt={real_dt:.4f}s "
          f"({1/real_dt:.0f} Hz)  solver=NEWTON/{args.solver_iters}/{args.ls_iters}")
    if n_frames > 10:
        print(f"[model] WARNING: {n_frames} physics substeps per control step is expensive in MJX. "
              f"Consider --physics-dt {args.ctrl_dt/5:.4f} (with --solver-iters 4 --ls-iters 10).")
    if n_frames <= 6 and (args.solver_iters < 4 or args.ls_iters < 10):
        print(f"!! WARNING: {n_frames} substeps with solver {args.solver_iters}/{args.ls_iters}. "
              f"Measured on this rover, a large timestep with few solver iterations makes a\n"
              f"!! full-command SPIN diverge (2.4 rad/s vs the real 0.571, and the rover flips).\n"
              f"!! Use --solver-iters 4 --ls-iters 10, or drop --physics-dt to 0.002.",
              file=sys.stderr)

    # --- index everything, by name where possible, robustly otherwise ------
    free_joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joints:
        sys.exit("ERROR: no free joint in the model -- the chassis must be on a free joint.")
    fj = free_joints[0]
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    for j in free_joints:
        if cid >= 0 and m.jnt_bodyid[j] == cid:
            fj = j
            break
    chassis_body = int(m.jnt_bodyid[fj])
    chassis_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, chassis_body)

    wj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
          for n in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")]
    if any(j < 0 for j in wj):
        wj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        print(f"[model] WARNING: wheel joints not all found by name; using all {len(wj)} hinge joints.")
    wheel_dof = [int(m.jnt_dofadr[j]) for j in wj]
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

    ctrl_lo = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 0], -10.0).astype(np.float32)
    ctrl_hi = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 1], 10.0).astype(np.float32)
    bad = ~np.isfinite(ctrl_lo) | ~np.isfinite(ctrl_hi) | (ctrl_hi <= ctrl_lo)
    ctrl_lo[bad], ctrl_hi[bad] = -10.0, 10.0

    def _mocap(name):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return int(m.body_mocapid[b]) if b >= 0 else -1

    rf_adr = [int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"tv_rf{i}")])
              for i in range(RF_N)] if surgery_ok else []

    # circumscribed radius of the rover in the xy plane, from the model itself
    rover_bodies = {chassis_body} | {int(m.jnt_bodyid[j]) for j in wj}
    rad = 0.0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in rover_bodies:
            off = m.body_pos[int(m.geom_bodyid[g])][:2] if int(m.geom_bodyid[g]) != chassis_body else np.zeros(2)
            rad = max(rad, float(np.linalg.norm(off + m.geom_pos[g][:2]) + m.geom_rbound[g]))
    rover_r = 0.75 * rad     # circumscribed is too pessimistic; 0.75x ~ the inscribed hull

    mass = float(m.body_subtreemass[chassis_body])
    print(f"[model] chassis={chassis_name!r} (qpos@{m.jnt_qposadr[fj]} dof@{m.jnt_dofadr[fj]}) "
          f"mass={mass:.2f} kg  collision radius={rover_r:.2f} m")
    print(f"[model] wheel dofs={wheel_dof}  actuators={act_names}")
    print(f"[model] ctrlrange lo={ctrl_lo.tolist()} hi={ctrl_hi.tolist()}")
    print(f"[model] compiled: nbody={m.nbody} ngeom={m.ngeom} nmocap={m.nmocap} "
          f"nsensor={m.nsensor} nsensordata={m.nsensordata}")

    info = dict(
        n_frames=n_frames, ctrl_dt=float(real_dt), chassis_body=chassis_body,
        qadr=int(m.jnt_qposadr[fj]), vadr=int(m.jnt_dofadr[fj]),
        wheel_dof=wheel_dof, act_names=act_names, ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi,
        target_mocap=_mocap("tv_target"),
        rub_mocap=[_mocap(f"tv_rub{i}") for i in range(n_rub)],
        obs_mocap=[_mocap(f"tv_obs{i}") for i in range(n_obs)],
        rub_size=rub_size, obst_r=obst_r, rf_adr=rf_adr, surgery_ok=surgery_ok,
        rover_r=rover_r, mass=mass, n_rub=n_rub, n_obs=n_obs,
    )
    return m, info


def action_mixer(act_names, nu, mode):
    """(nu, act_dim) matrix mapping policy output -> per-actuator command."""
    import numpy as np
    if mode == "four" or nu != 4:
        return np.eye(nu, dtype=np.float32)
    left = [i for i, n in enumerate(act_names) if n and n.endswith(("_fl", "_rl"))]
    right = [i for i, n in enumerate(act_names) if n and n.endswith(("_fr", "_rr"))]
    if len(left) != 2 or len(right) != 2:
        left, right = [0, 2], [1, 3]
        print("[model] WARNING: could not split actuators L/R by name; assuming [0,2]=left [1,3]=right.")
    A = np.zeros((nu, 2), dtype=np.float32)
    A[left, 0] = 1.0
    A[right, 1] = 1.0
    return A


# ===========================================================================
# 4. Small math helpers (jnp, but they accept numpy too, so the CPU rollout
#    path reuses the exact same observation code as training).
# ===========================================================================
def quat_to_mat(q):
    import jax.numpy as jnp
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_quat(roll, pitch, yaw):
    import jax.numpy as jnp
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    return jnp.array([cr * cp * cy + sr * sp * sy,
                      sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy])


def wrap_pi(a):
    import jax.numpy as jnp
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def analytic_rangefinders(pos_xy, R, obs_xy, obs_r):
    """Closed-form ray-vs-vertical-cylinder for the 9-ray fan. Exact for this
    setup: the rays are horizontal and the obstacles are vertical posts, so the
    3-D cast collapses to 2-D. Used when MJX cannot evaluate the real sensor."""
    import jax.numpy as jnp
    o = pos_xy + (R @ jnp.array([RF_POS[0], RF_POS[1], RF_POS[2]]))[:2]
    dirs = []
    for phi in rf_angles():
        v = R @ jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
        v2 = v[:2]
        dirs.append(v2 / (jnp.linalg.norm(v2) + 1e-9))
    u = jnp.stack(dirs)                       # (RF_N, 2)
    mvec = obs_xy[None, :, :] - o[None, None, :]      # (1, N, 2)
    t_ca = jnp.sum(mvec * u[:, None, :], axis=-1)     # (RF_N, N)
    d2 = jnp.sum(mvec ** 2, axis=-1) - t_ca ** 2
    r2 = (obs_r ** 2)[None, :]
    inside = d2 < r2
    t_hc = jnp.sqrt(jnp.maximum(r2 - d2, 0.0))
    t = t_ca - t_hc
    hit = inside & (t > 0.0)
    return jnp.min(jnp.where(hit, t, RF_MAX), axis=1)


# ===========================================================================
# 5. The environment. Written directly against MJX and the bare brax Env
#    interface, so brax's MJCF importer never has to understand rover.xml.
# ===========================================================================
def make_env(m, mi, args, use_sensor_rf):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.envs.base import Env, State
    from mujoco import mjx

    mx = mjx.put_model(m)
    A = action_mixer(mi["act_names"], m.nu, args.action_mode)
    act_dim = int(A.shape[1])
    Aj = jnp.asarray(A)
    ctrl_mid = jnp.asarray((mi["ctrl_lo"] + mi["ctrl_hi"]) * 0.5)
    ctrl_half = jnp.asarray((mi["ctrl_hi"] - mi["ctrl_lo"]) * 0.5)
    wheel_dof = jnp.asarray(mi["wheel_dof"], dtype=jnp.int32)
    wheel_scale = float(max(1.0, np.abs(mi["ctrl_hi"]).max()))
    qadr, vadr, n_frames = mi["qadr"], mi["vadr"], mi["n_frames"]
    dt, n_wheel = mi["ctrl_dt"], len(mi["wheel_dof"])
    chassis_body, rover_r, mass = mi["chassis_body"], mi["rover_r"], mi["mass"]
    n_rub, n_obs = mi["n_rub"], mi["n_obs"]
    arena, W = args.arena, REWARD_W
    obs_dim = 3 + 3 + 3 + 2 + n_wheel + 2 + 2 + RF_N + act_dim

    tgt_mc = mi["target_mocap"]
    rub_mc = jnp.asarray(mi["rub_mocap"], dtype=jnp.int32) if n_rub else None
    obs_mc = jnp.asarray(mi["obs_mocap"], dtype=jnp.int32) if n_obs else None
    obst_r = jnp.asarray(mi["obst_r"], dtype=jnp.float32) if n_obs else jnp.zeros((0,))
    rf_adr = jnp.asarray(mi["rf_adr"], dtype=jnp.int32) if mi["rf_adr"] else None

    # curriculum: difficulty ramps with each env's OWN accumulated step count.
    # info survives brax's auto-reset, so this is a faithful proxy for training
    # progress and needs no plumbing from the PPO loop.
    steps_per_env = max(1.0, args.steps / max(1, args.num_envs))
    ramp_steps = max(1.0, args.curriculum_frac * steps_per_env)
    rough_max = float(np.clip(args.rough, 0.0, 1.0))
    slope_max = math.radians(SLOPE_MAX_DEG)

    import mujoco as _mj
    qpos_ref = np.array(m.key_qpos[0] if m.nkey > 0 else m.qpos0, dtype=np.float64)
    d0 = _mj.MjData(m)
    d0.qpos[:] = qpos_ref
    _mj.mj_forward(m, d0)
    dx0 = mjx.put_data(m, d0)
    qpos_ref = jnp.asarray(qpos_ref, dtype=jnp.float32)
    spawn_z = float(qpos_ref[qadr + 2])
    mocap_pos0 = jnp.asarray(np.array(d0.mocap_pos, dtype=np.float32))
    mocap_quat0 = jnp.asarray(np.array(d0.mocap_quat, dtype=np.float32))

    # ---------------- curriculum schedule ----------------
    def difficulty(t_total):
        return rough_max * jnp.clip(t_total / ramp_steps, 0.0, 1.0)

    def layout(key, u):
        """Sample a whole episode: terrain, obstacles, slope. Returns mocap
        arrays plus the obstacle centres/radii the reward and rays need."""
        k_r, k_o, k_s, k_a = jax.random.split(key, 4)
        # --- rubble: burial depth sets how proud each box stands -----------
        if n_rub:
            kk = jax.random.split(k_r, 4)
            amp = rough_max * (0.015 + 0.055 * u)
            frac = rough_max * (0.25 + 0.75 * u)
            xy = jax.random.uniform(kk[0], (n_rub, 2), minval=-arena, maxval=arena)
            proud = jax.random.uniform(kk[1], (n_rub,)) * amp
            live = jax.random.uniform(kk[2], (n_rub,)) < frac
            yaw = jax.random.uniform(kk[3], (n_rub,), minval=-jnp.pi, maxval=jnp.pi)
            rz = jnp.where(live, -RUBBLE_HALF_Z + proud, FAR_Z)
            rub_p = jnp.concatenate([xy, rz[:, None]], axis=1)
            rub_q = jnp.stack([jnp.cos(yaw / 2), jnp.zeros(n_rub), jnp.zeros(n_rub),
                               jnp.sin(yaw / 2)], axis=1)
        else:
            rub_p = jnp.zeros((0, 3)); rub_q = jnp.zeros((0, 4))
        # --- obstacles: count ramps, and none within 1.2 m of the spawn ----
        if n_obs:
            kk = jax.random.split(k_o, 3)
            k_live = jnp.clip((u - 0.15) / 0.85, 0.0, 1.0)      # drive first, dodge later
            idx = jnp.arange(n_obs)
            live = idx < jnp.round(n_obs * k_live)
            # 1.6 m floor: the rover spawns within 0.4 m of the origin and its
            # collision radius is ~0.51 m, so anything closer can overlap at reset.
            r = jax.random.uniform(kk[0], (n_obs,), minval=1.6, maxval=arena * 0.95)
            th = jax.random.uniform(kk[1], (n_obs,), minval=-jnp.pi, maxval=jnp.pi)
            oxy = jnp.stack([r * jnp.cos(th), r * jnp.sin(th)], axis=1)
            oxy = jnp.where(live[:, None], oxy, FAR_XY)
            oz = jnp.where(live, OBST_HALF_Z, FAR_Z)
            obs_p = jnp.concatenate([oxy, oz[:, None]], axis=1)
            yaw = jax.random.uniform(kk[2], (n_obs,), minval=-jnp.pi, maxval=jnp.pi)
            obs_q = jnp.stack([jnp.cos(yaw / 2), jnp.zeros(n_obs), jnp.zeros(n_obs),
                               jnp.sin(yaw / 2)], axis=1)
        else:
            oxy = jnp.zeros((0, 2)); obs_p = jnp.zeros((0, 3)); obs_q = jnp.zeros((0, 4))
        # --- slope: a downhill body force. Capped at SLOPE_MAX_DEG --------
        ang = jax.random.uniform(k_s, (), minval=0.0, maxval=slope_max * rough_max * u)
        psi = jax.random.uniform(k_a, (), minval=-jnp.pi, maxval=jnp.pi)
        f = mass * 9.81 * jnp.sin(ang)
        slope_f = jnp.array([f * jnp.cos(psi), f * jnp.sin(psi)])
        slope_vec = jnp.sin(ang) * jnp.array([jnp.cos(psi), jnp.sin(psi)])

        mp, mq = mocap_pos0, mocap_quat0
        if tgt_mc >= 0:
            mp = mp.at[tgt_mc].set(jnp.array([0.0, 0.0, TARGET_Z]))
        if n_rub:
            mp = mp.at[rub_mc].set(rub_p); mq = mq.at[rub_mc].set(rub_q)
        if n_obs:
            mp = mp.at[obs_mc].set(obs_p); mq = mq.at[obs_mc].set(obs_q)
        return mp, mq, oxy, slope_f, slope_vec

    # ---------------- shared kinematics + observation ----------------
    def decode(qpos, qvel):
        pos = qpos[qadr:qadr + 3]
        R = quat_to_mat(qpos[qadr + 3:qadr + 7])
        # MuJoCo free joint: qvel linear is WORLD frame, angular is BODY frame.
        v_body = R.T @ qvel[vadr:vadr + 3]
        w_body = qvel[vadr + 3:vadr + 6]
        up_body = R[2, :]                       # == R.T @ [0,0,1]
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return pos, R, v_body, w_body, up_body, yaw, qvel[wheel_dof]

    def rangefinders(dx, pos, R, obs_xy):
        if use_sensor_rf and rf_adr is not None:
            r = dx.sensordata[rf_adr]
            r = jnp.where(r < 0.0, RF_MAX, r)           # MuJoCo returns -1 for "no hit"
        else:
            r = analytic_rangefinders(pos[:2], R, obs_xy, obst_r) if n_obs \
                else jnp.full((RF_N,), RF_MAX)
        return jnp.clip(r, 0.0, RF_MAX)

    def observe(qpos, qvel, target, last_action, rf, slope_vec, R=None):
        pos, Rm, v_body, w_body, up_body, yaw, wv = decode(qpos, qvel)
        R = Rm if R is None else R
        d_body = R.T @ jnp.array([target[0] - pos[0], target[1] - pos[1], 0.0])
        he = wrap_pi(target[2] - yaw)
        slope_b = R[:2, :2].T @ slope_vec       # downhill direction, body frame
        o = jnp.concatenate([
            v_body / 0.5,
            w_body / 1.0,
            up_body,
            slope_b / max(1e-6, math.sin(slope_max)),
            wv / wheel_scale,
            jnp.clip(d_body[:2] / 2.0, -2.0, 2.0),
            jnp.array([jnp.sin(he), jnp.cos(he)]),
            rf / RF_MAX,
            last_action,
        ])
        return jnp.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)

    def sample_target(key, pos_xy, obs_xy):
        """8 candidates at once; keep the one with the best obstacle clearance.
        Rejection sampling without a loop -- targets stay reachable."""
        k1, k2, k3 = jax.random.split(key, 3)
        NC = 8
        r = jax.random.uniform(k1, (NC,), minval=0.8, maxval=2.5)
        th = jax.random.uniform(k2, (NC,), minval=-jnp.pi, maxval=jnp.pi)
        yaw = jax.random.uniform(k3, (NC,), minval=-jnp.pi, maxval=jnp.pi)
        xy = pos_xy[None, :] + jnp.stack([r * jnp.cos(th), r * jnp.sin(th)], axis=1)
        n = jnp.linalg.norm(xy, axis=1, keepdims=True) + 1e-6
        xy = xy * jnp.minimum(1.0, (arena * 0.85) / n)     # stay inside the arena
        if n_obs:
            # clearance = distance to the nearest obstacle surface. A target
            # inside, or immediately behind, an obstacle scores badly and loses.
            dd = jnp.linalg.norm(xy[:, None, :] - obs_xy[None, :, :], axis=-1) - obst_r[None, :]
            clear = jnp.min(dd, axis=1) - (rover_r + 0.25)
            best = jnp.argmax(jnp.minimum(clear, 0.6))
        else:
            best = 0
        return jnp.array([xy[best, 0], xy[best, 1], yaw[best]])

    def obstacle_overlap(pos_xy, obs_xy):
        if not n_obs:
            return jnp.zeros(())
        d = jnp.linalg.norm(pos_xy[None, :] - obs_xy, axis=1)
        return jnp.max(jnp.maximum(0.0, (obst_r + rover_r) - d))

    def zero_metrics():
        keys = ["r_progress", "r_pos", "r_head", "r_reach", "c_ctrl", "c_rate", "c_energy",
                "c_tip", "c_spin", "c_collide", "c_idle", "dist", "heading_err", "upright",
                "reached", "speed", "rf_min", "difficulty", "slope_deg", "collided"]
        return {k: jnp.zeros(()) for k in keys}

    class RoverEnv(Env):
        @property
        def observation_size(self):
            return obs_dim

        @property
        def action_size(self):
            return act_dim

        @property
        def backend(self):
            return "mjx"

        # ---- a fresh episode: pose, terrain, obstacles, slope, target ----
        def _spawn(self, rng, t_total):
            rng, k_pose, k_yaw, k_vel, k_lay, k_tgt = jax.random.split(rng, 6)
            u = difficulty(t_total)
            mp, mq, obs_xy, slope_f, slope_vec = layout(k_lay, u)

            kp = jax.random.uniform(k_pose, (5,), minval=-1.0, maxval=1.0)
            qpos = qpos_ref
            qpos = qpos.at[qadr + 0].set(kp[0] * 0.4)
            qpos = qpos.at[qadr + 1].set(kp[1] * 0.4)
            qpos = qpos.at[qadr + 2].set(spawn_z + 0.02 + 0.01 * kp[2])
            quat = euler_to_quat(kp[3] * 0.05, kp[4] * 0.05,
                                 jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi))
            qpos = qpos.at[qadr + 3:qadr + 7].set(quat)
            qvel = jax.random.normal(k_vel, (m.nv,)) * 0.02
            target = sample_target(k_tgt, qpos[qadr:qadr + 2], obs_xy)
            if tgt_mc >= 0:
                mp = mp.at[tgt_mc].set(jnp.array([target[0], target[1], TARGET_Z]))
                mq = mq.at[tgt_mc].set(jnp.array([jnp.cos(target[2] / 2), 0.0, 0.0,
                                                  jnp.sin(target[2] / 2)]))
            return rng, qpos, qvel, mp, mq, obs_xy, slope_f, slope_vec, target, u

        def reset(self, rng):
            t0 = jnp.zeros(())
            rng, qpos, qvel, mp, mq, obs_xy, slope_f, slope_vec, target, u = self._spawn(rng, t0)
            dx = dx0.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(m.nu),
                             mocap_pos=mp, mocap_quat=mq)
            dx = mjx.forward(mx, dx)
            pos, R = dx.qpos[qadr:qadr + 3], quat_to_mat(dx.qpos[qadr + 3:qadr + 7])
            rf = rangefinders(dx, pos, R, obs_xy)
            la = jnp.zeros(act_dim)
            obs = observe(dx.qpos, dx.qvel, target, la, rf, slope_vec)
            info = {"rng": rng, "target": target, "last_action": la,
                    "prev_dist": jnp.linalg.norm(target[:2] - pos[:2]),
                    "t_total": t0, "mocap_pos": mp, "mocap_quat": mq,
                    "obs_xy": obs_xy, "slope_f": slope_f, "slope_vec": slope_vec}
            return State(pipeline_state=dx, obs=obs, reward=jnp.zeros(()),
                         done=jnp.zeros(()), metrics=zero_metrics(), info=info)

        def step(self, state, action):
            dx = state.pipeline_state
            rng, k_push, k_mag, k_tgt, k_new = jax.random.split(state.info["rng"], 5)
            t_total = state.info["t_total"] + 1.0
            u = difficulty(t_total)

            # brax's auto-reset restores pipeline_state from the FIRST state of
            # the run, so the per-episode layout and slope have to be re-applied
            # from info on every step. This is what makes the curriculum work.
            dx = dx.replace(mocap_pos=state.info["mocap_pos"],
                            mocap_quat=state.info["mocap_quat"])
            slope_f = state.info["slope_f"]
            xfrc = jnp.zeros_like(dx.xfrc_applied).at[chassis_body, 0:2].set(slope_f)

            action = jnp.clip(action, -1.0, 1.0)
            ctrl = ctrl_mid + (Aj @ action) * ctrl_half

            push = jax.random.bernoulli(k_push, 0.002 + 0.006 * u)
            kick = jax.random.normal(k_mag, (3,)) * jnp.array([0.25, 0.25, 0.5])
            qvel = dx.qvel
            qvel = qvel.at[vadr:vadr + 2].add(jnp.where(push, kick[:2], 0.0))
            qvel = qvel.at[vadr + 5].add(jnp.where(push, kick[2], 0.0))
            dx = dx.replace(qvel=qvel, ctrl=ctrl, xfrc_applied=xfrc)

            def one(carry, _):
                return mjx.step(mx, carry), None
            dx, _ = jax.lax.scan(one, dx, (), n_frames)

            target, obs_xy = state.info["target"], state.info["obs_xy"]
            pos, R, v_body, w_body, up_body, yaw, wv = decode(dx.qpos, dx.qvel)
            upright = R[2, 2]
            dist = jnp.linalg.norm(target[:2] - pos[:2])
            he = wrap_pi(target[2] - yaw)
            speed = jnp.linalg.norm(v_body[:2])
            rf = rangefinders(dx, pos, R, obs_xy)
            overlap = obstacle_overlap(pos[:2], obs_xy)
            collided = (overlap > 0.0).astype(jnp.float32)

            # ---------------- reward ----------------
            closing = jnp.clip((state.info["prev_dist"] - dist) / dt, -0.5, 0.5)
            r_progress = W["progress"] * closing
            r_pos = W["pos"] * jnp.exp(-(dist / 0.6) ** 2)
            head_q = 0.5 * (1.0 + jnp.cos(he))
            r_head = W["head"] * head_q * jnp.exp(-(dist / 0.8) ** 2)
            reached = ((dist < REACH_R) & (jnp.abs(he) < REACH_YAW)).astype(jnp.float32)
            r_reach = W["reach"] * reached

            c_ctrl = W["ctrl"] * jnp.mean(action ** 2)
            c_rate = W["rate"] * jnp.mean((action - state.info["last_action"]) ** 2)
            c_energy = W["energy"] * jnp.mean(jnp.abs(Aj @ action) * jnp.abs(wv) / wheel_scale)
            tipped = upright < TIP_TERM
            c_tip = W["tip"] * jnp.maximum(0.0, TIP_SOFT - upright) + W["tipover"] * tipped
            c_spin = W["spin"] * jnp.maximum(0.0, jnp.abs(w_body[2]) - SPIN_MAX) ** 2
            c_collide = W["collide"] * collided + W["collide_depth"] * overlap
            # the anti-freeze term: parked, far from the target, is worse than trying
            c_idle = W["idle"] * ((speed < 0.05) & (dist > 0.8)).astype(jnp.float32)
            c_bounds = W["bounds"] * jnp.maximum(0.0, jnp.linalg.norm(pos[:2]) - arena)

            reward = (r_progress + r_pos + r_head + r_reach + W["alive"]
                      - c_ctrl - c_rate - c_energy - c_tip - c_spin - c_collide
                      - c_idle - c_bounds)
            reward = jnp.clip(jnp.nan_to_num(reward), -25.0, 40.0)

            # ---------------- target resampling ----------------
            new_tgt = sample_target(k_tgt, pos[:2], obs_xy)
            target = jnp.where(reached > 0, new_tgt, target)
            prev_dist = jnp.where(reached > 0, jnp.linalg.norm(target[:2] - pos[:2]), dist)
            mp, mq = state.info["mocap_pos"], state.info["mocap_quat"]
            if tgt_mc >= 0:
                mp = mp.at[tgt_mc].set(jnp.array([target[0], target[1], TARGET_Z]))
                mq = mq.at[tgt_mc].set(jnp.array([jnp.cos(target[2] / 2), 0.0, 0.0,
                                                  jnp.sin(target[2] / 2)]))

            nan_guard = ~(jnp.isfinite(dx.qpos).all() & jnp.isfinite(dx.qvel).all())
            done = (tipped | nan_guard).astype(jnp.float32)

            # On termination, draw the NEXT episode's layout now: brax's auto-reset
            # will restore the physics but never calls our reset() again, so the
            # curriculum has to be advanced here.
            _, _, _, mp_n, mq_n, oxy_n, sf_n, sv_n, tgt_n, _ = self._spawn(k_new, t_total)
            fresh = done > 0
            mp = jnp.where(fresh, mp_n, mp)
            mq = jnp.where(fresh, mq_n, mq)
            obs_xy_next = jnp.where(fresh, oxy_n, obs_xy) if n_obs else obs_xy
            slope_f = jnp.where(fresh, sf_n, slope_f)
            slope_vec = jnp.where(fresh, sv_n, state.info["slope_vec"])
            target = jnp.where(fresh, tgt_n, target)
            prev_dist = jnp.where(fresh, jnp.linalg.norm(tgt_n[:2] - pos[:2]), prev_dist)

            obs = observe(dx.qpos, dx.qvel, target, action, rf, slope_vec)
            metrics = dict(r_progress=r_progress, r_pos=r_pos, r_head=r_head, r_reach=r_reach,
                           c_ctrl=c_ctrl, c_rate=c_rate, c_energy=c_energy, c_tip=c_tip,
                           c_spin=c_spin, c_collide=c_collide, c_idle=c_idle, dist=dist,
                           heading_err=jnp.abs(he), upright=upright, reached=reached,
                           speed=speed, rf_min=jnp.min(rf), difficulty=u, collided=collided,
                           slope_deg=jnp.degrees(jnp.arcsin(jnp.clip(
                               jnp.linalg.norm(slope_vec), 0.0, 1.0))))
            info = {"rng": rng, "target": target, "last_action": action, "prev_dist": prev_dist,
                    "t_total": t_total, "mocap_pos": mp, "mocap_quat": mq,
                    "obs_xy": obs_xy_next, "slope_f": slope_f, "slope_vec": slope_vec}
            # brax's EpisodeWrapper scans step() over action_repeat, and lax.scan
            # demands the carry pytree be identical in and out.  The wrappers add
            # their own keys (steps, truncation, first_obs, episode_metrics, ...)
            # and AutoResetWrapper reads them, so MERGE into what we were handed
            # rather than replacing it -- a fresh dict drops their keys and scan
            # dies with a carry-structure mismatch.
            return state.replace(pipeline_state=dx, obs=obs, reward=reward, done=done,
                                 metrics={**state.metrics, **metrics},
                                 info={**state.info, **info})

    env = RoverEnv()
    env.obs_dim, env.act_dim, env.mixer = obs_dim, act_dim, A
    env.observe_fn, env.sample_target_fn, env.layout_fn = observe, sample_target, layout
    env.difficulty_fn, env.decode_fn = difficulty, decode
    env.obstacle_overlap_fn = obstacle_overlap
    return env


# ===========================================================================
# 6. Rangefinder probe: does MJX actually evaluate the sensor? MJX's sensor
#    coverage is version-dependent, so ask rather than assume. CPU MuJoCo is
#    the ground truth; the analytic model is the fallback.
# ===========================================================================
def probe_rangefinders(m, mi, args):
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    if not mi["surgery_ok"] or not mi["rf_adr"]:
        print("[rf] no rangefinders in the model -> observation carries a constant clear fan.")
        return False
    if args.rangefinder != "auto":
        print(f"[rf] mode forced to {args.rangefinder!r}")
        return args.rangefinder == "sensor"
    if mi["n_obs"] == 0:
        print("[rf] --obstacles 0: nothing to see, using the analytic path (identical result).")
        return False

    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mc, r_test = mi["obs_mocap"][0], mi["obst_r"][0]
    d.mocap_pos[mc] = [1.8, 0.35, OBST_HALF_Z]
    mujoco.mj_forward(m, d)
    cpu = np.array([d.sensordata[a] for a in mi["rf_adr"]])
    cpu = np.where(cpu < 0, RF_MAX, np.clip(cpu, 0, RF_MAX))

    R = np.asarray(quat_to_mat(jnp.asarray(d.qpos[mi["qadr"] + 3:mi["qadr"] + 7], jnp.float32)))
    ana = np.asarray(analytic_rangefinders(
        jnp.asarray(d.qpos[mi["qadr"]:mi["qadr"] + 2], jnp.float32), jnp.asarray(R),
        jnp.asarray([[1.8, 0.35]], jnp.float32), jnp.asarray([r_test], jnp.float32)))
    ana_err = float(np.abs(ana - cpu).max())

    gpu_err, ok = float("inf"), False
    try:
        mx = mjx.put_model(m)
        dxp = mjx.forward(mx, mjx.put_data(m, d))
        gpu = np.asarray(dxp.sensordata)[np.asarray(mi["rf_adr"])]
        gpu = np.where(gpu < 0, RF_MAX, np.clip(gpu, 0, RF_MAX))
        gpu_err = float(np.abs(gpu - cpu).max())
        ok = gpu_err < 0.05
    except Exception as e:  # noqa: BLE001
        print(f"[rf] MJX rejected the model or the sensor ({type(e).__name__}: {e})")

    print(f"[rf] CPU MuJoCo (truth) : {np.round(cpu, 3).tolist()}")
    print(f"[rf] MJX sensordata     : max err {gpu_err:.3f} m -> {'USABLE' if ok else 'NOT usable'}")
    print(f"[rf] analytic ray model : max err {ana_err:.3f} m")
    if ok:
        print("[rf] using the real MuJoCo rangefinder via MJX sensordata.")
    else:
        print("[rf] using the analytic ray model (exact here: horizontal rays, vertical posts).")
    return ok


# ===========================================================================
# 7. Training
# ===========================================================================
def ppo_config(args, episode_length, smoke):
    import functools
    from brax.training.agents.ppo import networks as ppo_networks

    if smoke:
        num_envs, batch, mb, unroll = 64, 16, 4, 10
        steps, evals, ep_len = num_envs * unroll * mb * 4, 2, min(episode_length, 100)
    else:
        num_envs, batch, mb, unroll = (args.num_envs, args.batch_size,
                                       args.num_minibatches, args.unroll_length)
        steps, evals, ep_len = args.steps, args.evals, episode_length

    if (batch * mb) % num_envs != 0:
        sys.exit(f"ERROR: batch_size*num_minibatches ({batch*mb}) must be a multiple "
                 f"of num_envs ({num_envs}).")

    net = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=(128, 128, 128),
                            value_hidden_layer_sizes=(256, 256, 256))
    cfg = dict(num_timesteps=steps, num_envs=num_envs, episode_length=ep_len,
               batch_size=batch, num_minibatches=mb, unroll_length=unroll,
               num_updates_per_batch=args.updates_per_batch, learning_rate=args.lr,
               entropy_cost=args.entropy, discounting=args.gamma, gae_lambda=0.95,
               clipping_epsilon=0.2, reward_scaling=1.0, normalize_observations=True,
               max_grad_norm=1.0, action_repeat=1, num_evals=evals, num_eval_envs=128,
               seed=args.seed, network_factory=net)
    per_iter = batch * unroll * mb
    print(f"[ppo] {steps:,} steps | {num_envs} envs | episode {ep_len} steps "
          f"| {per_iter:,} steps/iteration | ~{max(1, steps // per_iter)} iterations")
    return cfg


def progress_printer(total_steps, t_start):
    def fmt(v, n=2):
        try:
            return f"{float(v):.{n}f}"
        except Exception:
            return "  -  "
    state = {"i": 0}

    def progress(step, metrics):
        state["i"] += 1
        el = time.time() - t_start
        g = metrics.get
        sps = g("training/sps") or g("eval/sps") or (step / max(el, 1e-6))
        line = (f"[{state['i']:>3}] {step:>12,} ({100.0*step/max(1,total_steps):5.1f}%)"
                f"  R={fmt(g('eval/episode_reward', float('nan')), 1):>8}"
                f" len={fmt(g('eval/avg_episode_length', float('nan')), 0):>5}"
                f" dist={fmt(g('eval/episode_dist', float('nan'))):>5}"
                f" head={fmt(g('eval/episode_heading_err', float('nan'))):>5}"
                f" reach={fmt(g('eval/episode_reached', float('nan')), 3):>6}"
                f" hit={fmt(g('eval/episode_collided', float('nan')), 3):>6}"
                f" up={fmt(g('eval/episode_upright', float('nan'))):>5}"
                f" diff={fmt(g('eval/episode_difficulty', float('nan'))):>5}"
                f"  {float(sps)/1e3:,.0f}k sps  {el:6.1f}s")
        eta = (total_steps - step) / max(float(sps), 1.0)
        print(line + (f"  eta {eta/60:.1f}m" if step > 0 else ""), flush=True)
    return progress


def timed_jit_check(env, seed):
    """Time the FIRST jitted reset/step separately from steady state. A
    pathologically slow first step is the signature of a bad sm_120 fallback."""
    import jax
    import jax.numpy as jnp

    jreset, jstep = jax.jit(env.reset), jax.jit(env.step)
    t = time.time(); s = jreset(jax.random.PRNGKey(seed)); jax.block_until_ready(s.obs)
    t_reset = time.time() - t
    a = jnp.zeros(env.action_size)
    t = time.time(); s = jstep(s, a); jax.block_until_ready(s.obs); t_step1 = time.time() - t
    t = time.time()
    for _ in range(50):
        s = jstep(s, a)
    jax.block_until_ready(s.obs); t_rest = (time.time() - t) / 50

    print(f"[jit] FIRST reset  (compile+run): {t_reset:8.2f} s")
    print(f"[jit] FIRST step   (compile+run): {t_step1:8.2f} s   <-- watch this one")
    print(f"[jit] steady step  (1 env)      : {t_rest*1e3:8.3f} ms")
    if t_step1 > 240:
        print("!! WARNING: first jitted step took > 4 minutes. On a healthy sm_120 build this\n"
              "!! is seconds to tens of seconds. Suspect PTX JIT / missing Blackwell SASS.",
              file=sys.stderr)
    return s


def train_main(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.agents.ppo import train as ppo_train

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m, mi = build_model(args, (args.video_width, args.video_height))
    use_sensor = probe_rangefinders(m, mi, args)
    env = make_env(m, mi, args, use_sensor)
    episode_length = int(round(args.episode_seconds / mi["ctrl_dt"]))
    steps_per_env = args.steps / max(1, args.num_envs)

    print("-" * 78)
    print(f"[env] obs_dim={env.obs_dim} act_dim={env.act_dim} mode={args.action_mode}  "
          f"(3 v + 3 w + 3 up + 2 slope + {len(mi['wheel_dof'])} wheels + 2 dxdy + 2 head "
          f"+ {RF_N} rangefinders + {env.act_dim} prev act)")
    print(f"[env] curriculum: difficulty 0 -> {min(1.0, max(0.0, args.rough))} over the first "
          f"{args.curriculum_frac:.0%} of training (~{args.curriculum_frac*steps_per_env:,.0f} "
          f"steps per env, of {steps_per_env:,.0f})")
    print(f"[env]   rubble proud height  {0.015:.3f} -> {0.070:.3f} m, density 25% -> 100% "
          f"of a {mi['n_rub']}-box pool")
    print(f"[env]   obstacles            0 -> {mi['n_obs']} (start at difficulty 0.15)")
    print(f"[env]   slope                0 -> {SLOPE_MAX_DEG:.0f} deg "
          f"(HARD CAP: rover climbs 14 deg, stalls at 16 deg)")
    print(f"[env] reward weights: {json.dumps(REWARD_W)}")
    print(f"[env] reach: dist<{REACH_R}m and |heading err|<{REACH_YAW}rad -> "
          f"+{REWARD_W['reach']} and resample")
    print(f"[env] terminate: up-vector z < {TIP_TERM}, or {args.episode_seconds}s limit")
    print("-" * 78, flush=True)

    s = timed_jit_check(env, args.seed)
    print(f"[env] obs shape {tuple(np.asarray(s.obs).shape)}  reward {float(s.reward):+.3f}  "
          f"done {float(s.done):.0f}  rf_min {float(s.metrics['rf_min']):.2f}  "
          f"target {np.asarray(s.info['target']).round(2).tolist()}")
    if args.smoke:
        jstep = jax.jit(env.step)
        for i in range(5):
            a = jnp.asarray(np.random.default_rng(i).uniform(-1, 1, env.action_size), jnp.float32)
            s = jstep(s, a)
            print(f"   smoke step {i}: r={float(s.reward):+7.3f} dist={float(s.metrics['dist']):.2f} "
                  f"up={float(s.metrics['upright']):.3f} rf_min={float(s.metrics['rf_min']):.2f} "
                  f"hit={float(s.metrics['collided']):.0f} done={float(s.done):.0f}")
        print("-" * 78, flush=True)

    cfg = ppo_config(args, episode_length, args.smoke)
    t0 = time.time()
    make_inference_fn, params, _ = ppo_train.train(
        environment=env, progress_fn=progress_printer(cfg["num_timesteps"], t0), **cfg)
    wall = time.time() - t0
    print("-" * 78)
    print(f"[done] training wall clock: {wall/60:.2f} min  "
          f"({cfg['num_timesteps']/max(wall,1e-9)/1e3:,.0f}k env steps/s)")

    pfile = out / ("params_smoke.pkl" if args.smoke else "params.pkl")
    brax_model.save_params(str(pfile), params)
    meta = dict(obs_dim=env.obs_dim, act_dim=env.act_dim, action_mode=args.action_mode,
                model=str(Path(args.model).resolve()), rough=args.rough, rubble=args.rubble,
                obstacles=args.obstacles, curriculum_frac=args.curriculum_frac,
                rangefinder="sensor" if use_sensor else "analytic", rf_n=RF_N, rf_max=RF_MAX,
                arena=args.arena, ctrl_dt=mi["ctrl_dt"], episode_seconds=args.episode_seconds,
                seed=args.seed, reward_weights=REWARD_W, slope_max_deg=SLOPE_MAX_DEG,
                wall_clock_s=wall,
                versions={p: _version(p) for p in ("jax", "jaxlib", "mujoco", "brax")})
    (out / ("meta_smoke.json" if args.smoke else "meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[done] params -> {pfile}")
    if args.smoke:
        print("\nSMOKE TEST PASSED -- env builds, steps, and PPO runs end to end.")
    else:
        print(f"\nNext: python {Path(__file__).name} --rollout --params {pfile}")


# ===========================================================================
# 8. Rollout + video. Plain CPU MuJoCo (not MJX): that is what the offscreen
#    renderer needs, and it doubles as a sim-to-sim check on the policy.
# ===========================================================================
def rollout_main(args):
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pfile = Path(args.params) if args.params else out / "params.pkl"
    if not pfile.exists():
        sys.exit(f"ERROR: no params at {pfile}. Train first, or pass --params.")
    video = Path(args.video) if args.video else out / "rollout.mp4"

    m, mi = build_model(args, (args.video_width, args.video_height))
    use_sensor = probe_rangefinders(m, mi, args)
    env = make_env(m, mi, args, use_sensor)
    observe, sample_target, layout = env.observe_fn, env.sample_target_fn, env.layout_fn

    net = ppo_networks.make_ppo_networks(
        env.obs_dim, env.act_dim, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128, 128), value_hidden_layer_sizes=(256, 256, 256))
    params = brax_model.load_params(str(pfile))
    policy = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))
    print(f"[rollout] params {pfile}  obs_dim={env.obs_dim} act_dim={env.act_dim} "
          f"rf={'sensor' if use_sensor else 'analytic'}")

    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)

    # one fixed episode at the requested difficulty
    u = args.rough if args.rollout_difficulty < 0 else args.rollout_difficulty
    rng = jax.random.PRNGKey(args.seed + 1234)
    rng, k_lay, k_tgt = jax.random.split(rng, 3)
    mp, mq, obs_xy, slope_f, slope_vec = layout(k_lay, jnp.asarray(u, jnp.float32))
    d.mocap_pos[:] = np.asarray(mp)
    d.mocap_quat[:] = np.asarray(mq)
    mujoco.mj_forward(m, d)
    print(f"[rollout] difficulty={float(u):.2f}  active obstacles="
          f"{int((np.asarray(obs_xy)[:, 0] < 900).sum()) if mi['n_obs'] else 0}  "
          f"slope={math.degrees(math.asin(min(1.0, float(np.linalg.norm(slope_vec))))):.1f} deg")

    qadr, A = mi["qadr"], env.mixer
    lo, hi = mi["ctrl_lo"], mi["ctrl_hi"]
    mid, half = (lo + hi) * 0.5, (hi - lo) * 0.5
    target = np.asarray(sample_target(k_tgt, jnp.asarray(d.qpos[qadr:qadr + 2], jnp.float32), obs_xy))
    last_action = np.zeros(env.act_dim, dtype=np.float32)
    slope_world = np.asarray(slope_f, dtype=np.float64)

    # ---- renderer: an explicit camera, never the implicit default ----
    try:
        renderer = mujoco.Renderer(m, height=args.video_height, width=args.video_width)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: could not open the offscreen renderer ({type(e).__name__}: {e}).\n"
                 f"       On a headless box try MUJOCO_GL=egl (default) or MUJOCO_GL=osmesa.")
    named_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if args.cam != "chase" and named_cam >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = named_cam
        print(f"[rollout] camera: model camera {args.cam!r}")
    else:
        if args.cam != "chase":
            print(f"[rollout] camera {args.cam!r} not in the model; falling back to chase.")
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mi["chassis_body"]
        cam.distance, cam.elevation, cam.azimuth = args.cam_distance, args.cam_elevation, 135.0
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)

    steps = int(round(args.episode_seconds / mi["ctrl_dt"]))
    render_every = max(1, int(round((1.0 / mi["ctrl_dt"]) / args.video_fps)))
    fps = 1.0 / (mi["ctrl_dt"] * render_every)
    frames, reached_n, hits, min_up, rf_err = [], 0, 0, 1.0, 0.0

    for i in range(steps):
        qpos = jnp.asarray(d.qpos, jnp.float32)
        qvel = jnp.asarray(d.qvel, jnp.float32)
        R = quat_to_mat(qpos[qadr + 3:qadr + 7])
        if mi["rf_adr"]:
            rf_s = np.array([d.sensordata[a] for a in mi["rf_adr"]])
            rf_s = np.where(rf_s < 0, RF_MAX, np.clip(rf_s, 0, RF_MAX))
        else:
            rf_s = np.full(RF_N, RF_MAX)
        rf_a = np.asarray(analytic_rangefinders(qpos[qadr:qadr + 2], R, obs_xy,
                                                jnp.asarray(mi["obst_r"], jnp.float32))) \
            if mi["n_obs"] else np.full(RF_N, RF_MAX)
        rf_err = max(rf_err, float(np.abs(rf_s - rf_a).max()))
        rf = jnp.asarray(rf_s if use_sensor or mi["rf_adr"] else rf_a, jnp.float32)

        obs = observe(qpos, qvel, jnp.asarray(target, jnp.float32),
                      jnp.asarray(last_action), rf, slope_vec, R)
        rng, ka = jax.random.split(rng)
        act, _ = policy(obs, ka)
        act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
        d.ctrl[:] = mid + (A @ act) * half
        last_action = act
        if mi["target_mocap"] >= 0:
            mc = mi["target_mocap"]
            d.mocap_pos[mc] = [target[0], target[1], TARGET_Z]
            d.mocap_quat[mc] = [math.cos(target[2] / 2), 0.0, 0.0, math.sin(target[2] / 2)]
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[mi["chassis_body"], 0:2] = slope_world

        for _ in range(mi["n_frames"]):
            mujoco.mj_step(m, d)

        Rn = np.asarray(quat_to_mat(jnp.asarray(d.qpos[qadr + 3:qadr + 7], jnp.float32)))
        up = float(Rn[2, 2]); min_up = min(min_up, up)
        yaw = math.atan2(Rn[1, 0], Rn[0, 0])
        pos = d.qpos[qadr:qadr + 2].copy()
        dist = float(np.linalg.norm(target[:2] - pos))
        he = (target[2] - yaw + math.pi) % (2 * math.pi) - math.pi
        if mi["n_obs"] and float(env.obstacle_overlap_fn(jnp.asarray(pos, jnp.float32), obs_xy)) > 0:
            hits += 1
        if dist < REACH_R and abs(he) < REACH_YAW:
            reached_n += 1
            rng, k = jax.random.split(rng)
            target = np.asarray(sample_target(k, jnp.asarray(pos, jnp.float32), obs_xy))
            print(f"    t={i*mi['ctrl_dt']:5.1f}s  TARGET {reached_n} REACHED -> new target "
                  f"({target[0]:+.2f}, {target[1]:+.2f}, yaw {math.degrees(target[2]):+.0f} deg)")
        if up < TIP_TERM:
            print(f"    t={i*mi['ctrl_dt']:5.1f}s  TIPPED OVER (up={up:.2f}) -- ending episode")
            break

        if i % render_every == 0:
            if cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                want = math.degrees(yaw) + 210.0
                cam.azimuth += 0.05 * ((want - cam.azimuth + 180.0) % 360.0 - 180.0)
            renderer.update_scene(d, camera=cam, scene_option=vopt)
            frames.append(renderer.render())

    print(f"[rollout] {len(frames)} frames | {reached_n} targets reached | {hits} obstacle-contact "
          f"steps | min up-vector z {min_up:.3f}")
    if mi["n_obs"] and mi["rf_adr"]:
        print(f"[rollout] rangefinder sim-to-sim: max |MuJoCo sensor - analytic model| = {rf_err:.3f} m")
    if not frames:
        sys.exit("ERROR: no frames rendered.")
    if int(np.asarray(frames).max()) == 0:
        print("!! WARNING: every frame is black. Check MUJOCO_GL (egl/osmesa) and that lights exist.",
              file=sys.stderr)

    wrote = False
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(str(video), fps=fps, codec="libx264", quality=8, macro_block_size=None)
        for f in frames:
            w.append_data(f)
        w.close()
        wrote = True
        print(f"[rollout] MP4 -> {video}  ({fps:.1f} fps, {args.video_width}x{args.video_height})")
    except Exception as e:  # noqa: BLE001
        print(f"[rollout] imageio failed ({type(e).__name__}: {e}); trying mediapy...", file=sys.stderr)
        try:
            import mediapy
            mediapy.write_video(str(video), frames, fps=fps)
            wrote = True
            print(f"[rollout] MP4 -> {video}")
        except Exception as e2:  # noqa: BLE001
            print(f"[rollout] mediapy failed too ({type(e2).__name__}: {e2}).", file=sys.stderr)

    if not wrote:
        fdir = video.with_suffix("")
        fdir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        for i, f in enumerate(frames):
            imageio.imwrite(str(fdir / f"frame_{i:05d}.png"), f)
        print(f"[rollout] wrote {len(frames)} PNGs -> {fdir}\n[rollout] make the MP4 with:\n"
              f"    ffmpeg -y -framerate {fps:.0f} -i {fdir}/frame_%05d.png "
              f"-c:v libx264 -pix_fmt yuv420p -crf 18 {video}")


# ===========================================================================
def main():
    args = parse_args()
    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl
        os.environ["PYOPENGL_PLATFORM"] = args.gl
    banner_and_gpu_check(allow_cpu=args.allow_cpu or args.rollout)
    if args.rollout:
        rollout_main(args)
    else:
        train_main(args)


if __name__ == "__main__":
    main()
