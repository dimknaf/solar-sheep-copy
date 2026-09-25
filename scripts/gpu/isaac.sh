#!/usr/bin/env bash
# Run a Python script from this repo INSIDE NVIDIA's Isaac Lab container, on the GPU box.
#
#   bash scripts/gpu/isaac.sh robot/usd/convert_rover.py --out /data/runs/usd
#   bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd /data/runs/usd/rover_train/rover_train.usda
#
# The repo is mounted read-only at /work; write outputs under /data/runs (persistent disk,
# mounted at the same path). The image has no uv: Isaac Lab lives in Isaac Sim's bundled
# Python 3.12, so the environment is set up the way /isaac-sim/python.sh does it.
# Running the container accepts NVIDIA's Isaac Sim / Omniverse licence terms (ACCEPT_EULA=Y);
# telemetry consent is not given.

set -euo pipefail
IMAGE="${IMAGE:-solar/isaac-lab:3.0.0-rc1-video}"   # built by smoke.sh from scripts/gpu/Dockerfile
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
C=/data/isaac-sim
[ $# -ge 1 ] || { echo "usage: $0 <script.py relative to the repo> [args...]" >&2; exit 2; }
SCRIPT="$1"; shift

exec docker run --rm --gpus all --network=host -e ACCEPT_EULA=Y --entrypoint bash \
  -v "$C/cache/kit:/isaac-sim/kit/cache:rw" -v "$C/cache/ov:/root/.cache/ov:rw" \
  -v "$C/cache/pip:/root/.cache/pip:rw" -v "$C/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$C/cache/computecache:/root/.nv/ComputeCache:rw" -v "$C/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$C/data:/root/.local/share/ov/data:rw" -v "$C/documents:/root/Documents:rw" \
  -v /data/runs:/data/runs:rw -v /data/runs:/workspace/isaaclab/logs:rw -v "$REPO:/work:ro" \
  "$IMAGE" -c '
    export CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps
    source /isaac-sim/setup_python_env.sh
    export PATH=/isaac-sim/kit/python/bin:$PATH
    cd /workspace/isaaclab
    exec /isaac-sim/kit/python/bin/python3 "/work/$0" "$@"
  ' "$SCRIPT" "$@"
