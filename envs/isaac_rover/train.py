"""envs/isaac_rover/train.py -- gate G3: train the rover's traverse skill with Isaac Lab's own trainer.

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    bash scripts/gpu/isaac.sh envs/isaac_rover/train.py --task SolarSheep-Traverse-Flat-Rover --video
    bash scripts/gpu/isaac.sh envs/isaac_rover/train.py --task SolarSheep-Traverse-Rough-Rover --video
    smoke:  ... --task SolarSheep-Traverse-Flat-Rover --num_envs 64 --max_iterations 2
    resume: ... --task SolarSheep-Traverse-Rough-Rover --checkpoint latest   (or a model_<N>.pt path)
    other rover USD: ... --usd /data/runs/usd/<other>.usda   (default: rover_cfg.ROVER_USD)

This file only registers our tasks (envs/isaac_rover/solar_sheep_tasks) and hands the command line to
Isaac Lab's unified training entrypoint, run_train_cli -- the same one IsaacLab's
scripts/reinforcement_learning/train.py calls -- so every flag of `isaaclab train` works here. The task
names RSL-RL as its default agent, so --rl_library can be left out. Logs, checkpoints and TensorBoard
events go to logs/rsl_rl/<experiment>/<date>/ under the container's working directory, which is
/data/runs on the box (scripts/gpu/isaac.sh). --video records from the task's headless Kit view into
<run>/videos/train: an 8 s clip (400 steps) every 100 PPO iterations (2400 env steps) unless
--video_length / --video_interval say otherwise. Without --video nothing is recorded.

--usd is ours, not Isaac Lab's: it becomes ROVER_USD (read by rover_cfg.py) and is removed from the
command line. It exists because scripts/gpu/isaac.sh does not forward host environment variables.

Watching: the task's Viser view listens on the box's 127.0.0.1:8080 and TensorBoard
(scripts/gpu/tensorboard.sh) on 127.0.0.1:6006, both reached only through scripts/gpu/watch.sh. Safety: before anything starts, this refuses any flag that
would open a listener beyond localhost or send data off the box (--viz other than kit/viser/none,
--livestream, --logger wandb/neptune, a Kit livestream extension in --kit_args), and any Hydra override
(agent.logger=..., env.sim.visualizer_cfgs...=...) of the logger or visualizer settings the task fixes.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAFE_VIZ = {"kit", "viser", "none"}
# config fields the safety rule fixes: the RSL-RL logger and the visualizers' network settings
GUARDED_FIELDS = {"logger", "wandb_project", "neptune_project", "bind_address", "display_address", "share",
                  "open_browser"}
# --video without these flags: one 400-step clip every 100 PPO iterations of 24 steps
VIDEO_DEFAULTS = {"--video_length": "400", "--video_interval": str(24 * 100)}


def refuse_unsafe(argv: list[str]) -> None:
    """Exit if the command line asks for a listener beyond localhost or an off-box logger."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualizer", "--viz", default=None)
    parser.add_argument("--livestream", default="0")
    parser.add_argument("--logger", default=None)
    args, _ = parser.parse_known_args(argv)
    viz = {v.strip().lower() for v in (args.visualizer or "").split(",") if v.strip()}
    problems = []
    if viz - SAFE_VIZ:
        problems.append(f"--viz {','.join(sorted(viz - SAFE_VIZ))}: only kit, viser (bound to 127.0.0.1) or none")
    if str(args.livestream) != "0" or os.environ.get("LIVESTREAM", "0") not in ("", "0"):
        problems.append("--livestream / LIVESTREAM: the Kit stream server would listen on the network")
    if args.logger in ("wandb", "neptune"):
        problems.append(f"--logger {args.logger}: uploads to a third party; use tensorboard")
    # Hydra overrides (env.<path>=v, agent.<path>=v) are set straight onto the configs by Isaac Lab
    # (isaaclab_tasks/utils/hydra.py), past every flag above; --kit_args goes straight to Kit.
    for tok in argv:
        key, sep, _ = tok.partition("=")
        key = key.lstrip("-")
        if sep and "." in key and (key.rsplit(".", 1)[-1] in GUARDED_FIELDS or "visualizer_cfg" in key):
            problems.append(f"{tok}: overrides a logger/visualizer setting the safety rule fixes")
        if "livestream" in tok.lower() and not tok.startswith("--livestream"):
            problems.append(f"{tok}: a Kit livestream extension would listen on the network")
    if problems:
        raise SystemExit("refusing to run:\n  " + "\n  ".join(problems))


def take_usd(argv: list[str]) -> list[str]:
    """Move ``--usd PATH`` (or ``--usd=PATH``) into ROVER_USD for rover_cfg.py; return argv without it."""
    for i, tok in enumerate(argv):
        if tok == "--usd" or tok.startswith("--usd="):
            path = tok.partition("=")[2] if "=" in tok else (argv[i + 1] if i + 1 < len(argv) else "")
            if not os.path.isfile(path):
                raise SystemExit(f"--usd {path!r}: no such file")
            os.environ["ROVER_USD"] = os.path.abspath(path)
            return argv[:i] + argv[i + (1 if "=" in tok else 2):]
    return argv


def with_video_defaults(argv: list[str]) -> list[str]:
    """With --video, add VIDEO_DEFAULTS for the clip flags the command line leaves out."""
    if "--video" not in argv:
        return argv
    given = lambda flag: any(a == flag or a.startswith(flag + "=") for a in argv)  # noqa: E731
    return argv + [x for flag, value in VIDEO_DEFAULTS.items() if not given(flag) for x in (flag, value)]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    refuse_unsafe(argv)
    argv = take_usd(argv)             # before the task configs (and rover_cfg.py) are imported
    # Warp reads enable_backward when a kernel module is built, so it goes first (as in IsaacLab's train.py)
    import warp as wp

    wp.config.enable_backward = False
    sys.path.insert(0, HERE)
    import solar_sheep_tasks  # noqa: F401  registers SolarSheep-Traverse-{Flat,Rough}-Rover
    from isaaclab_rl.entrypoints import run_train_cli

    return run_train_cli(with_video_defaults(argv))


if __name__ == "__main__":
    raise SystemExit(main())
