"""envs/isaac_rover/play.py -- watch a trained traverse policy drive, and record it.

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box, from the box:

    bash scripts/gpu/isaac.sh envs/isaac_rover/play.py --task SolarSheep-Traverse-Rough-Rover \
        --checkpoint latest --video --video_length 1200
    add --real-time to watch it at real speed in the live view (http://localhost:8080 via watch.sh)

Same arrangement as train.py: register our tasks, then Isaac Lab's unified playback entrypoint
(run_play_cli, as in IsaacLab's scripts/reinforcement_learning/play.py). Play applies the task's
play_mode(): at most 50 envs, every terrain row, no pushes. --checkpoint latest takes the newest run of
this task; with --video, clips land in <run>/videos/play. --usd and the safety refusals are train.py's.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import HERE, refuse_unsafe, take_usd  # noqa: E402  (this folder's train.py: first on sys.path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    refuse_unsafe(argv)
    argv = take_usd(argv)
    import warp as wp

    wp.config.enable_backward = False
    sys.path.insert(0, HERE)
    import solar_sheep_tasks  # noqa: F401  registers SolarSheep-Traverse-{Flat,Rough}-Rover
    from isaaclab_rl.entrypoints import run_play_cli

    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
