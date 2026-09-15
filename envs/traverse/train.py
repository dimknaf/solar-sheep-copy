#!/usr/bin/env python3
"""
traverse/train.py -- pose + heading tracking policy for a 4-wheel skid-steer rover.

ONE file. MuJoCo MJX (GPU physics) + Brax PPO. Target box: a single NVIDIA
RTX PRO 6000 (Blackwell, sm_120). Full run is meant to finish in ~10 minutes.

The task the policy learns:
    "drive to (x, y) and end up pointing at yaw" -- over slightly rough ground,
    without tipping. Targets are resampled mid-episode as they are reached, so
    one episode contains several go-here-point-this-way commands.

    NOT in the reward: anything about the sun. "Face the sun" has a degenerate
    optimum (park facing the sun, never move). The sun-seeking decision belongs
    to the high-level planner; this policy only executes the resulting command.

Observation (all in the rover's own body frame -- relative targets generalise,
world coordinates do not):
    chassis linear velocity      (3)
    chassis angular velocity     (3)
    up-vector / gravity tilt     (3)
    wheel joint velocities       (n_wheels, normally 4)
    target offset dx, dy         (2)
    sin/cos of heading error     (2)
    previous action              (n_act)   <- needed for the action-rate penalty

Action: one command per wheel actuator in [-1, 1], affine-mapped onto each
actuator's ctrlrange (`--action-mode two` collapses it to left/right, which is
the honest skid-steer parameterisation and usually learns faster).

Usage:
    python train.py --smoke                 # pipeline gate, ~1 min, exits
    python train.py                         # the real run
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
# headless VM either dies or hands back black frames. EGL is the right default
# for a cloud GPU box with no X server.
# ---------------------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# XLA flag that measurably helps MJX on a big single GPU.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent.parent / "robot" / "rover.xml")


# ===========================================================================
# 1. CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a pose+heading tracking controller for a skid-steer rover (MJX + Brax PPO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # what to run
    p.add_argument("--smoke", action="store_true",
                   help="Build env, step it, run 2 tiny PPO iterations, print shapes, exit fast.")
    p.add_argument("--rollout", action="store_true",
                   help="Load --params, roll out one episode on CPU MuJoCo, write an MP4.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Do not hard-fail when no GPU is visible. Training on CPU takes hours; "
                        "this exists for laptop debugging only.")

    # model
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to rover.xml (owned by another workstream; read-only).")
    p.add_argument("--bumps", type=int, default=12,
                   help="Number of small static bumps scattered on the ground = 'slightly rough'. "
                        "0 disables (use that if MJX complains about an unsupported collision pair).")
    p.add_argument("--arena", type=float, default=8.0, help="Half-size of the drivable area, metres.")
    p.add_argument("--ctrl-dt", type=float, default=0.02, help="Control period, seconds. Physics substeps are derived.")
    p.add_argument("--episode-seconds", type=float, default=20.0, help="Episode time limit.")
    p.add_argument("--action-mode", choices=["four", "two"], default="four",
                   help="'four' = one command per wheel. 'two' = left/right pair (true skid steer, learns faster).")
    p.add_argument("--solver-iters", type=int, default=2, help="MJX solver iterations (low = fast).")
    p.add_argument("--ls-iters", type=int, default=6, help="MJX line-search iterations.")

    # PPO
    p.add_argument("--steps", type=int, default=50_000_000, help="Total environment steps.")
    p.add_argument("--num-envs", type=int, default=4096, help="Parallel MJX envs.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=5e-3)
    p.add_argument("--gamma", type=float, default=0.995, help="Discount. Needs to be high: a 3 m drive is ~400 steps.")
    p.add_argument("--evals", type=int, default=20, help="Number of eval/progress lines over the run.")
    p.add_argument("--seed", type=int, default=0)

    # io
    p.add_argument("--out", default="runs", help="Output directory for params / metadata / video.")
    p.add_argument("--params", default=None, help="Params file to load for --rollout (default: <out>/params.pkl).")
    p.add_argument("--video", default=None, help="MP4 path for --rollout (default: <out>/rollout.mp4).")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--cam-distance", type=float, default=3.2, help="Chase camera distance, metres.")
    p.add_argument("--cam-elevation", type=float, default=-20.0, help="Chase camera elevation, degrees.")
    p.add_argument("--gl", default=None, help="Override MUJOCO_GL (egl | osmesa | glfw).")
    return p.parse_args(argv)


# ===========================================================================
# 2. The loud GPU / version banner. This is deliberately the first thing that
#    happens, because a silent CPU fallback is the single most expensive
#    failure mode available to us.
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
    """Print resolved versions + jax devices. Hard-fail if there is no GPU."""
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
        extra = f"  compute_capability={cc}" if cc else ""
        print(f"    device[{d.id}] platform={d.platform} kind={d.device_kind!r}{extra}")
    print(f"(jax import + device discovery took {time.time() - t0:.1f}s)")

    if backend != "gpu":
        msg = (
            "\n" + "!" * 78 +
            "\n!! NO GPU VISIBLE TO JAX. default_backend() == %r\n"
            "!! Training on CPU would take HOURS, not the 10 minutes budgeted.\n"
            "!! Most likely causes, in order:\n"
            "!!   1. jax installed without CUDA:  pip install -U 'jax[cuda12]'\n"
            "!!   2. the CUDA jaxlib wheel has no sm_120 (Blackwell) code -- check for\n"
            "!!      'ptxas' / 'no kernel image is available' messages above.\n"
            "!!   3. CUDA_VISIBLE_DEVICES is empty, or the driver is older than the wheel.\n"
            "!! Re-run with --allow-cpu ONLY if you know you want the slow path.\n"
            % backend + "!" * 78 + "\n"
        )
        print(msg, file=sys.stderr)
        if not allow_cpu:
            sys.exit(2)

    # A bad sm_120 fallback is usually visible as a ridiculous matmul time
    # (JIT-to-PTX-to-SASS at runtime, or silent host execution). Catch it now
    # rather than 40 minutes into a run.
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
        print(f"GPU sanity matmul {n}x{n}: first(compile)={compile_s:.2f}s  steady={run_s * 1e3:.2f}ms  ~{tflops:.1f} TFLOP/s")
        if tflops < 5.0:
            print("!! WARNING: <5 TFLOP/s on a 4096^3 matmul. That is NOT a healthy RTX PRO 6000.\n"
                  "!! Strongly suspect a bad sm_120 code path. Check the jaxlib CUDA build before training.",
                  file=sys.stderr)
    print("=" * 78, flush=True)


# ===========================================================================
# 3. Model building.
#
#    robot/rover.xml is owned by another workstream and is READ-ONLY for us.
#    So everything the training/rendering setup needs -- a ground plane if the
#    rover file has none, lights, bumps, and a mocap marker showing the
#    commanded pose -- is injected by compiling a small wrapper MJCF that
#    <include>s rover.xml. rover.xml itself is never written to.
# ===========================================================================
_BUMP_TMPL = ('    <geom name="tv_bump{i}" type="box" pos="{x:.3f} {y:.3f} 0" '
              'euler="0 0 {yaw:.3f}" size="{sx:.3f} {sy:.3f} {sz:.3f}" '
              'rgba="0.42 0.40 0.36 1" friction="1.0 0.02 0.001" condim="3"/>')

_PLANE_XML = """  <asset>
    <texture name="tv_grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.28" rgb2="0.33 0.35 0.37"
             width="300" height="300"/>
    <material name="tv_grid" texture="tv_grid" texrepeat="10 10" texuniform="true" reflectance="0.04"/>
  </asset>
"""


def _bump_field(n, arena, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = rng.uniform(1.3, arena * 0.92)
        th = rng.uniform(-math.pi, math.pi)
        out.append(_BUMP_TMPL.format(
            i=i, x=r * math.cos(th), y=r * math.sin(th), yaw=rng.uniform(-math.pi, math.pi),
            sx=rng.uniform(0.12, 0.34), sy=rng.uniform(0.10, 0.26), sz=rng.uniform(0.010, 0.026)))
    return "\n".join(out)


def build_model(path, bumps, arena, seed, ctrl_dt, solver_iters, ls_iters, video_wh=(1280, 720)):
    """Compile the MuJoCo model. Returns (MjModel, info dict)."""
    import mujoco
    import numpy as np

    path = Path(path).resolve()
    if not path.exists():
        sys.exit(f"ERROR: model not found at {path}\n"
                 f"       (that file is owned by another workstream -- pass --model if it moved)")

    # probe compile: what does rover.xml already provide?
    probe = mujoco.MjModel.from_xml_path(str(path))
    has_plane = bool((np.asarray(probe.geom_type) == mujoco.mjtGeom.mjGEOM_PLANE).any())
    print(f"[model] {path}")
    print(f"[model] rover.xml provides: ground plane={has_plane}  lights={probe.nlight}  "
          f"nq={probe.nq} nv={probe.nv} nu={probe.nu} nbody={probe.nbody} ngeom={probe.ngeom}")

    extra = []
    if not has_plane:
        extra.append('    <geom name="tv_ground" type="plane" size="0 0 0.05" material="tv_grid" '
                     'friction="1.0 0.02 0.001" condim="3"/>')
    # Always add lights: a headlight-only scene renders flat and shadowless, and
    # "the video is the deliverable".
    extra.append('    <light name="tv_key" directional="true" pos="3 -3 6" dir="-0.4 0.4 -1" '
                 'diffuse="0.75 0.72 0.65" specular="0.25 0.25 0.25" castshadow="true"/>')
    extra.append('    <light name="tv_fill" directional="true" pos="-5 4 5" dir="0.5 -0.4 -1" '
                 'diffuse="0.28 0.30 0.34" specular="0.0 0.0 0.0" castshadow="false"/>')
    # Mocap marker for the commanded pose: a ring at (x,y) plus a bar showing the
    # commanded heading. No collisions (contype/conaffinity 0), so it is free.
    extra.append('    <body name="tv_target" mocap="true" pos="2 0 0.015">\n'
                 '      <geom name="tv_tgt_ring" type="cylinder" size="0.35 0.008" rgba="0.15 0.95 0.45 0.40"\n'
                 '            contype="0" conaffinity="0" group="1"/>\n'
                 '      <geom name="tv_tgt_dir" type="box" pos="0.46 0 0.004" size="0.22 0.045 0.008"\n'
                 '            rgba="1.0 0.72 0.10 0.95" contype="0" conaffinity="0" group="1"/>\n'
                 '    </body>')
    if bumps > 0:
        extra.append(_bump_field(bumps, arena, seed))

    wrapper = (
        '<mujoco model="traverse_arena">\n'
        f'  <include file="{path.name}"/>\n'
        + (_PLANE_XML if not has_plane else "")
        + "  <worldbody>\n" + "\n".join(extra) + "\n  </worldbody>\n</mujoco>\n"
    )

    # chdir to rover.xml's directory so <include> and any relative meshdir/
    # texturedir inside rover.xml resolve exactly as if it were the main file.
    cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        m = mujoco.MjModel.from_xml_string(wrapper)
        print(f"[model] arena wrapper compiled: +plane={not has_plane} +2 lights "
              f"+target marker +{bumps} bumps")
    except Exception as e:  # noqa: BLE001 - we genuinely want any failure to degrade gracefully
        print("!" * 78, file=sys.stderr)
        print(f"!! Arena wrapper failed to compile ({type(e).__name__}: {e})", file=sys.stderr)
        print("!! Falling back to the bare rover.xml: no bumps, no target marker,", file=sys.stderr)
        print("!! and no extra lights. Training still works; the video will be plainer.", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        m = mujoco.MjModel.from_xml_path(str(path))
    finally:
        os.chdir(cwd)

    # --- solver / renderer settings applied in Python, not XML (lower risk) ---
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations = solver_iters
    m.opt.ls_iterations = ls_iters
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL   # MJX is much happier with pyramidal
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, video_wh[0])
    m.vis.global_.offheight = max(m.vis.global_.offheight, video_wh[1])
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.45
    m.vis.map.znear = 0.02

    n_frames = max(1, int(round(ctrl_dt / m.opt.timestep)))
    real_dt = n_frames * m.opt.timestep
    print(f"[model] timestep={m.opt.timestep}s  n_frames={n_frames}  control dt={real_dt:.4f}s "
          f"({1/real_dt:.0f} Hz)  solver=NEWTON/{solver_iters}/{ls_iters} cone=PYRAMIDAL")

    # --- index everything we need, by name where possible, robustly otherwise ---
    def _jid(name):
        try:
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        except Exception:
            return -1

    free_joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joints:
        sys.exit("ERROR: no free joint in the model -- the chassis must be on a free joint.")
    fj = free_joints[0]
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    if cid >= 0:
        for j in free_joints:
            if m.jnt_bodyid[j] == cid:
                fj = j
                break
    chassis_body = int(m.jnt_bodyid[fj])
    chassis_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, chassis_body)

    wheel_names = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
    wj = [_jid(n) for n in wheel_names]
    if any(j < 0 for j in wj):
        wj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        print(f"[model] WARNING: wheel joints not all found by name; using all {len(wj)} hinge joints instead.")
    wheel_dof = [int(m.jnt_dofadr[j]) for j in wj]

    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print(f"[model] chassis body = {chassis_name!r} (free joint qpos@{m.jnt_qposadr[fj]} dof@{m.jnt_dofadr[fj]})")
    print(f"[model] wheel dofs = {wheel_dof}   actuators = {act_names}")

    ctrl_lo = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 0], -10.0).astype(np.float32)
    ctrl_hi = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 1], 10.0).astype(np.float32)
    bad = ~np.isfinite(ctrl_lo) | ~np.isfinite(ctrl_hi) | (ctrl_hi <= ctrl_lo)
    ctrl_lo[bad], ctrl_hi[bad] = -10.0, 10.0
    print(f"[model] ctrlrange lo={ctrl_lo.tolist()} hi={ctrl_hi.tolist()}")

    tgt_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tv_target")
    tgt_mocap = int(m.body_mocapid[tgt_body]) if tgt_body >= 0 else -1

    info = dict(
        n_frames=n_frames, ctrl_dt=float(real_dt), chassis_body=chassis_body,
        qadr=int(m.jnt_qposadr[fj]), vadr=int(m.jnt_dofadr[fj]),
        wheel_dof=wheel_dof, act_names=act_names,
        ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, target_mocap=tgt_mocap,
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
# 4. Small math helpers (jnp; they also accept numpy arrays, so the CPU rollout
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


# ===========================================================================
# 5. The environment.
#
#    Written directly against MJX and the bare brax Env interface rather than
#    brax.PipelineEnv, so that brax's MJCF importer never has to understand
#    rover.xml -- only MJX does. Same approach MuJoCo Playground takes.
# ===========================================================================
# reward weights -- one place, printed at startup, quoted in the README
REWARD_W = dict(
    progress=8.0,     # + shaped closing speed on the target
    pos=1.5,          # + Gaussian bump on distance-to-target
    head=1.5,         # + heading alignment, gated by proximity to the target
    reach=15.0,       # + one-off bonus when a target is reached (then resample)
    alive=0.2,        # + per-step alive bonus
    ctrl=0.02,        # - control magnitude
    rate=0.02,        # - action rate (jerk / chatter)
    energy=0.005,     # - |command| * |wheel speed| ~ mechanical power (the 30 W story)
    tip=4.0,          # - tilt, from the body up-vector z component
    tipover=10.0,     # - one-off, on the tip-over termination
    spin=0.3,         # - yaw rate beyond SPIN_MAX
)
REACH_R = 0.35        # m   -- position tolerance
REACH_YAW = 0.50      # rad -- heading tolerance (~29 deg)
TIP_TERM = 0.40       # terminate when up-vector z drops below this
TIP_SOFT = 0.90       # start penalising tilt below this
SPIN_MAX = 2.5        # rad/s


def make_env(m, mi, args):
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
    qadr, vadr = mi["qadr"], mi["vadr"]
    n_frames = mi["n_frames"]
    n_wheel = len(mi["wheel_dof"])
    obs_dim = 3 + 3 + 3 + n_wheel + 2 + 2 + act_dim

    # reference pose: keyframe 0 if the rover file ships one, else qpos0
    import mujoco as _mj
    qpos_ref = np.array(m.key_qpos[0] if m.nkey > 0 else m.qpos0, dtype=np.float64)
    d0 = _mj.MjData(m)
    d0.qpos[:] = qpos_ref
    _mj.mj_forward(m, d0)
    dx0 = mjx.put_data(m, d0)
    qpos_ref = jnp.asarray(qpos_ref, dtype=jnp.float32)
    spawn_z = float(qpos_ref[qadr + 2])

    arena = args.arena
    W = REWARD_W

    # ---------------- shared kinematics + observation ----------------
    def decode(qpos, qvel):
        """-> pos(3), R(3,3), v_body(3), w_body(3), up_body(3), yaw, wheel_vel"""
        pos = qpos[qadr:qadr + 3]
        R = quat_to_mat(qpos[qadr + 3:qadr + 7])
        # MuJoCo free joint: qvel linear is WORLD frame, angular is BODY frame.
        v_body = R.T @ qvel[vadr:vadr + 3]
        w_body = qvel[vadr + 3:vadr + 6]
        up_body = R[2, :]                       # == R.T @ [0,0,1]
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return pos, R, v_body, w_body, up_body, yaw, qvel[wheel_dof]

    def observe(qpos, qvel, target, last_action):
        pos, R, v_body, w_body, up_body, yaw, wv = decode(qpos, qvel)
        d_world = jnp.array([target[0] - pos[0], target[1] - pos[1], 0.0])
        d_body = R.T @ d_world
        he = wrap_pi(target[2] - yaw)
        o = jnp.concatenate([
            v_body / 2.0,
            w_body / 5.0,
            up_body,
            wv / wheel_scale,
            jnp.clip(d_body[:2] / 3.0, -2.0, 2.0),
            jnp.array([jnp.sin(he), jnp.cos(he)]),
            last_action,
        ])
        return jnp.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)

    def sample_target(key, pos_xy):
        k1, k2, k3 = jax.random.split(key, 3)
        r = jax.random.uniform(k1, (), minval=1.0, maxval=4.0)
        th = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)
        yaw = jax.random.uniform(k3, (), minval=-jnp.pi, maxval=jnp.pi)
        xy = pos_xy + jnp.array([r * jnp.cos(th), r * jnp.sin(th)])
        # keep targets inside the (bumpy) arena
        n = jnp.linalg.norm(xy) + 1e-6
        xy = xy * jnp.minimum(1.0, (arena * 0.8) / n)
        return jnp.array([xy[0], xy[1], yaw])

    def zero_metrics():
        keys = ["r_progress", "r_pos", "r_head", "r_reach", "r_alive",
                "c_ctrl", "c_rate", "c_energy", "c_tip", "c_spin",
                "dist", "heading_err", "upright", "reached", "speed"]
        return {k: jnp.zeros(()) for k in keys}

    class RoverEnv(Env):
        """Pose + heading tracking. Targets are RELATIVE in the observation."""

        # ---- brax Env interface ----
        @property
        def observation_size(self):
            return obs_dim

        @property
        def action_size(self):
            return act_dim

        @property
        def backend(self):
            return "mjx"

        # ---- reset ----
        def reset(self, rng):
            rng, k_pose, k_vel, k_tgt, k_keep = jax.random.split(rng, 5)
            kp = jax.random.uniform(k_pose, (5,), minval=-1.0, maxval=1.0)
            qpos = qpos_ref
            qpos = qpos.at[qadr + 0].set(kp[0] * 0.5)
            qpos = qpos.at[qadr + 1].set(kp[1] * 0.5)
            qpos = qpos.at[qadr + 2].set(spawn_z + 0.01 * (kp[2] + 1.0))
            # small random roll/pitch: "slightly rough ground" as an initial condition
            quat = euler_to_quat(kp[3] * 0.06, kp[4] * 0.06,
                                 jax.random.uniform(k_keep, (), minval=-jnp.pi, maxval=jnp.pi))
            qpos = qpos.at[qadr + 3:qadr + 7].set(quat)
            qvel = jax.random.normal(k_vel, (m.nv,)) * 0.02

            dx = dx0.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(m.nu))
            dx = mjx.forward(mx, dx)

            target = sample_target(k_tgt, qpos[qadr:qadr + 2])
            last_action = jnp.zeros(act_dim)
            obs = observe(dx.qpos, dx.qvel, target, last_action)
            d = jnp.linalg.norm(target[:2] - dx.qpos[qadr:qadr + 2])
            info = {"rng": rng, "target": target, "prev_dist": d, "last_action": last_action}
            return State(pipeline_state=dx, obs=obs, reward=jnp.zeros(()),
                         done=jnp.zeros(()), metrics=zero_metrics(), info=info)

        # ---- step ----
        def step(self, state, action):
            dx = state.pipeline_state
            rng, k_push, k_mag, k_tgt = jax.random.split(state.info["rng"], 4)
            action = jnp.clip(action, -1.0, 1.0)
            ctrl = ctrl_mid + (Aj @ action) * ctrl_half

            # Random shoves: cheap stand-in for terrain we did not model, and it
            # is what keeps the policy from being a fragile open-loop trajectory.
            push = jax.random.bernoulli(k_push, 0.004)
            kick = jax.random.normal(k_mag, (3,)) * jnp.array([0.35, 0.35, 0.8])
            qvel = dx.qvel
            qvel = qvel.at[vadr:vadr + 2].add(jnp.where(push, kick[:2], 0.0))
            qvel = qvel.at[vadr + 5].add(jnp.where(push, kick[2], 0.0))
            dx = dx.replace(qvel=qvel, ctrl=ctrl)

            def one(carry, _):
                return mjx.step(mx, carry), None
            dx, _ = jax.lax.scan(one, dx, (), n_frames)

            target = state.info["target"]
            pos, R, v_body, w_body, up_body, yaw, wv = decode(dx.qpos, dx.qvel)
            upright = R[2, 2]
            dist = jnp.linalg.norm(target[:2] - pos[:2])
            he = wrap_pi(target[2] - yaw)
            speed = jnp.linalg.norm(v_body[:2])

            # ---------------- reward ----------------
            progress = jnp.clip(state.info["prev_dist"] - dist, -0.5, 0.5)
            r_progress = W["progress"] * progress
            r_pos = W["pos"] * jnp.exp(-(dist / 0.6) ** 2)
            head_q = 0.5 * (1.0 + jnp.cos(he))                 # 0..1
            r_head = W["head"] * head_q * jnp.exp(-(dist / 0.8) ** 2)
            reached = ((dist < REACH_R) & (jnp.abs(he) < REACH_YAW)).astype(jnp.float32)
            r_reach = W["reach"] * reached
            r_alive = W["alive"] * jnp.ones(())

            c_ctrl = W["ctrl"] * jnp.mean(action ** 2)
            c_rate = W["rate"] * jnp.mean((action - state.info["last_action"]) ** 2)
            c_energy = W["energy"] * jnp.mean(jnp.abs(Aj @ action) * jnp.abs(wv) / wheel_scale)
            tipped = (upright < TIP_TERM)
            c_tip = W["tip"] * jnp.maximum(0.0, TIP_SOFT - upright) + W["tipover"] * tipped
            c_spin = W["spin"] * jnp.maximum(0.0, jnp.abs(w_body[2]) - SPIN_MAX) ** 2

            reward = (r_progress + r_pos + r_head + r_reach + r_alive
                      - c_ctrl - c_rate - c_energy - c_tip - c_spin)
            reward = jnp.clip(jnp.nan_to_num(reward), -25.0, 35.0)

            # ---------------- target resampling ----------------
            new_tgt = sample_target(k_tgt, pos[:2])
            target = jnp.where(reached > 0, new_tgt, target)
            prev_dist = jnp.where(reached > 0, jnp.linalg.norm(target[:2] - pos[:2]), dist)

            # ---------------- termination ----------------
            nan_guard = ~(jnp.isfinite(dx.qpos).all() & jnp.isfinite(dx.qvel).all())
            done = (tipped | nan_guard).astype(jnp.float32)

            obs = observe(dx.qpos, dx.qvel, target, action)
            metrics = dict(r_progress=r_progress, r_pos=r_pos, r_head=r_head, r_reach=r_reach,
                           r_alive=r_alive, c_ctrl=c_ctrl, c_rate=c_rate, c_energy=c_energy,
                           c_tip=c_tip, c_spin=c_spin, dist=dist, heading_err=jnp.abs(he),
                           upright=upright, reached=reached, speed=speed)
            info = {"rng": rng, "target": target, "prev_dist": prev_dist, "last_action": action}
            return state.replace(pipeline_state=dx, obs=obs, reward=reward, done=done,
                                 metrics=metrics, info=info)

    env = RoverEnv()
    env.obs_dim, env.act_dim, env.mixer = obs_dim, act_dim, A
    env.observe_fn, env.sample_target_fn = observe, sample_target
    return env


# ===========================================================================
# 6. Training
# ===========================================================================
def ppo_config(args, episode_length, smoke):
    import functools
    from brax.training.agents.ppo import networks as ppo_networks

    if smoke:
        num_envs, batch, mb, unroll = 64, 16, 4, 10
        steps = num_envs * unroll * mb * 2 * 2
        evals, ep_len = 2, min(episode_length, 100)
    else:
        num_envs, batch, mb, unroll = args.num_envs, args.batch_size, args.num_minibatches, args.unroll_length
        steps, evals, ep_len = args.steps, args.evals, episode_length

    if (batch * mb) % num_envs != 0:
        sys.exit(f"ERROR: batch_size*num_minibatches ({batch*mb}) must be a multiple of num_envs ({num_envs}).")

    net = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=(128, 128, 128),
                            value_hidden_layer_sizes=(256, 256, 256))
    cfg = dict(
        num_timesteps=steps, num_envs=num_envs, episode_length=ep_len,
        batch_size=batch, num_minibatches=mb, unroll_length=unroll,
        num_updates_per_batch=args.updates_per_batch, learning_rate=args.lr,
        entropy_cost=args.entropy, discounting=args.gamma, gae_lambda=0.95,
        clipping_epsilon=0.2, reward_scaling=1.0, normalize_observations=True,
        max_grad_norm=1.0, action_repeat=1, num_evals=evals, num_eval_envs=128,
        seed=args.seed, network_factory=net,
    )
    per_iter = batch * unroll * mb
    print(f"[ppo] {steps:,} steps | {num_envs} envs | episode {ep_len} steps "
          f"({ep_len * args.ctrl_dt:.0f}s) | {per_iter:,} steps/iteration "
          f"| ~{max(1, steps // per_iter)} iterations")
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
        pct = 100.0 * step / max(1, total_steps)
        sps = metrics.get("training/sps") or metrics.get("eval/sps") or (step / max(el, 1e-6))
        g = metrics.get
        line = (f"[{state['i']:>3}] {step:>12,} steps ({pct:5.1f}%)  "
                f"R={fmt(g('eval/episode_reward', g('episode/sum_reward', float('nan'))), 1):>8}"
                f" +/-{fmt(g('eval/episode_reward_std', float('nan')), 1):<7}"
                f" len={fmt(g('eval/avg_episode_length', float('nan')), 0):>5}"
                f" dist={fmt(g('eval/episode_dist', float('nan'))):>5}"
                f" head={fmt(g('eval/episode_heading_err', float('nan'))):>5}"
                f" reach={fmt(g('eval/episode_reached', float('nan')), 3):>6}"
                f" up={fmt(g('eval/episode_upright', float('nan'))):>5}"
                f"  {float(sps)/1e3:,.0f}k sps  {el:6.1f}s")
        eta = (total_steps - step) / max(float(sps), 1.0)
        print(line + (f"  eta {eta/60:.1f}m" if step > 0 else ""), flush=True)

    return progress


def timed_jit_check(env, seed):
    """Time the FIRST jitted reset/step separately from steady state.
    A pathologically slow first step is the signature of a bad sm_120 fallback."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    jreset, jstep = jax.jit(env.reset), jax.jit(env.step)

    t = time.time(); s = jreset(key); jax.block_until_ready(s.obs); t_reset = time.time() - t
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
        print("!! WARNING: first jitted step took > 4 minutes. On a healthy sm_120 build this is\n"
              "!! seconds to tens of seconds. Suspect PTX JIT / missing Blackwell SASS.", file=sys.stderr)
    return s


def train_main(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.agents.ppo import train as ppo_train

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m, mi = build_model(args.model, args.bumps, args.arena, args.seed,
                        args.ctrl_dt, args.solver_iters, args.ls_iters,
                        (args.video_width, args.video_height))
    env = make_env(m, mi, args)
    episode_length = int(round(args.episode_seconds / mi["ctrl_dt"]))

    print("-" * 78)
    print(f"[env] obs_dim={env.obs_dim}  act_dim={env.act_dim}  action_mode={args.action_mode}")
    print(f"[env] reward weights: {json.dumps(REWARD_W)}")
    print(f"[env] reach: dist<{REACH_R}m and |heading err|<{REACH_YAW}rad -> +{REWARD_W['reach']} and resample")
    print(f"[env] terminate: up-vector z < {TIP_TERM}, or {args.episode_seconds}s limit")
    print("-" * 78, flush=True)

    s = timed_jit_check(env, args.seed)
    print(f"[env] obs shape {tuple(np.asarray(s.obs).shape)}  reward {float(s.reward):+.3f}  "
          f"done {float(s.done):.0f}  target {np.asarray(s.info['target']).round(2).tolist()}")
    if args.smoke:
        jstep = jax.jit(env.step)
        for i in range(5):
            a = jnp.asarray(np.random.default_rng(i).uniform(-1, 1, env.action_size), jnp.float32)
            s = jstep(s, a)
            print(f"   smoke step {i}: r={float(s.reward):+7.3f} dist={float(s.metrics['dist']):.2f} "
                  f"up={float(s.metrics['upright']):.3f} done={float(s.done):.0f}")
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
                model=str(Path(args.model).resolve()), bumps=args.bumps, arena=args.arena,
                ctrl_dt=mi["ctrl_dt"], episode_seconds=args.episode_seconds, seed=args.seed,
                reward_weights=REWARD_W, wall_clock_s=wall,
                versions={p: _version(p) for p in ("jax", "jaxlib", "mujoco", "brax")})
    (out / ("meta_smoke.json" if args.smoke else "meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[done] params -> {pfile}")
    print(f"[done] meta   -> {out / ('meta_smoke.json' if args.smoke else 'meta.json')}")
    if args.smoke:
        print("\nSMOKE TEST PASSED -- env builds, steps, and PPO runs end to end.")
    else:
        print(f"\nNext: python {Path(__file__).name} --rollout --params {pfile}")


# ===========================================================================
# 7. Rollout + video. Runs on plain CPU MuJoCo (not MJX) because that is what
#    the offscreen renderer needs, and it doubles as a sim-to-sim check.
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

    m, mi = build_model(args.model, args.bumps, args.arena, args.seed,
                        args.ctrl_dt, args.solver_iters, args.ls_iters,
                        (args.video_width, args.video_height))
    env = make_env(m, mi, args)
    observe, sample_target = env.observe_fn, env.sample_target_fn

    # rebuild the policy exactly as it was built for training
    net = ppo_networks.make_ppo_networks(
        env.obs_dim, env.act_dim,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256))
    params = brax_model.load_params(str(pfile))
    policy = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))
    print(f"[rollout] params {pfile}  obs_dim={env.obs_dim} act_dim={env.act_dim}")

    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    qadr, A = mi["qadr"], env.mixer
    lo, hi = mi["ctrl_lo"], mi["ctrl_hi"]
    mid, half = (lo + hi) * 0.5, (hi - lo) * 0.5

    rng = jax.random.PRNGKey(args.seed + 1234)
    rng, k = jax.random.split(rng)
    target = np.asarray(sample_target(k, jnp.asarray(d.qpos[qadr:qadr + 2], jnp.float32)))
    last_action = np.zeros(env.act_dim, dtype=np.float32)

    # ---- renderer: an explicit tracking camera, never the implicit default ----
    if args.gl:
        print(f"[rollout] NOTE: --gl must be set before launch; MUJOCO_GL is currently "
              f"{os.environ.get('MUJOCO_GL')!r}. Use MUJOCO_GL={args.gl} python {Path(__file__).name} ...")
    try:
        renderer = mujoco.Renderer(m, height=args.video_height, width=args.video_width)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: could not open the offscreen renderer ({type(e).__name__}: {e}).\n"
                 f"       On a headless box try MUJOCO_GL=egl (default) or MUJOCO_GL=osmesa.")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mi["chassis_body"]
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = 135.0
    scn = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scn)
    scn.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    steps = int(round(args.episode_seconds / mi["ctrl_dt"]))
    render_every = max(1, int(round((1.0 / mi["ctrl_dt"]) / args.video_fps)))
    fps = 1.0 / (mi["ctrl_dt"] * render_every)
    frames, reached_n, min_up, t0 = [], 0, 1.0, time.time()

    for i in range(steps):
        qpos = jnp.asarray(d.qpos, jnp.float32)
        qvel = jnp.asarray(d.qvel, jnp.float32)
        obs = observe(qpos, qvel, jnp.asarray(target, jnp.float32), jnp.asarray(last_action))
        rng, ka = jax.random.split(rng)
        act, _ = policy(obs, ka)
        act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
        d.ctrl[:] = mid + (A @ act) * half
        last_action = act

        mc = mi["target_mocap"]
        if mc >= 0:
            d.mocap_pos[mc] = [target[0], target[1], 0.015]
            cy, sy = math.cos(target[2] / 2), math.sin(target[2] / 2)
            d.mocap_quat[mc] = [cy, 0.0, 0.0, sy]

        for _ in range(mi["n_frames"]):
            mujoco.mj_step(m, d)

        R = np.asarray(quat_to_mat(jnp.asarray(d.qpos[qadr + 3:qadr + 7], jnp.float32)))
        up = float(R[2, 2]); min_up = min(min_up, up)
        yaw = math.atan2(R[1, 0], R[0, 0])
        pos = d.qpos[qadr:qadr + 2].copy()
        dist = float(np.linalg.norm(target[:2] - pos))
        he = (target[2] - yaw + math.pi) % (2 * math.pi) - math.pi
        if dist < REACH_R and abs(he) < REACH_YAW:
            reached_n += 1
            rng, k = jax.random.split(rng)
            target = np.asarray(sample_target(k, jnp.asarray(pos, jnp.float32)))
            print(f"    t={i*mi['ctrl_dt']:5.1f}s  TARGET {reached_n} REACHED -> new target "
                  f"({target[0]:+.2f}, {target[1]:+.2f}, yaw {math.degrees(target[2]):+.0f} deg)")
        if up < TIP_TERM:
            print(f"    t={i*mi['ctrl_dt']:5.1f}s  TIPPED OVER (up={up:.2f}) -- ending episode")
            break

        if i % render_every == 0:
            # gentle chase: lag the azimuth behind the rover so the view never spins
            want = math.degrees(yaw) + 180.0 + 30.0
            delta = (want - cam.azimuth + 180.0) % 360.0 - 180.0
            cam.azimuth += 0.05 * delta
            renderer.update_scene(d, camera=cam, scene_option=scn)
            frames.append(renderer.render())

    print(f"[rollout] {len(frames)} frames, {reached_n} targets reached, min up-vector z = {min_up:.3f}, "
          f"{time.time()-t0:.1f}s")
    if not frames:
        sys.exit("ERROR: no frames rendered.")
    arr = np.asarray(frames)
    if int(arr.max()) == 0:
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
        import imageio.v2 as imageio  # PNG writing needs far less than video writing
        for i, f in enumerate(frames):
            imageio.imwrite(str(fdir / f"frame_{i:05d}.png"), f)
        print(f"[rollout] wrote {len(frames)} PNGs -> {fdir}\n"
              f"[rollout] make the MP4 with:\n"
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
