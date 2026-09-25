#!/usr/bin/env bash
# Gate G1, run ON THE GPU BOX:  bash smoke.sh
#   1. host checks: GPU, driver, EGL/GL, Vulkan, OptiX, NVENC (Isaac needs graphics even headless)
#   2. NVIDIA's pre-built Isaac Lab container trains Cartpole for a few iterations (PhysX),
#      then plays the checkpoint and records a video
# Everything lands on the persistent data disk: /data/runs (logs, checkpoints, videos).
#
# By running NVIDIA's Isaac container we accept NVIDIA's Isaac Sim / Omniverse licence
# terms (ACCEPT_EULA=Y; skill: third-party-eula-preflight). Telemetry/privacy consent is
# NOT given (PRIVACY_CONSENT is left unset).

set -uo pipefail
IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-rc1}"
MIN_DRIVER="580.95.05"
ITER="${ITER:-30}"
fails=0
pass() { echo "PASS  $*"; }
warn() { echo "WARN  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }

echo "=== 1. host ==="
[ -f /var/lib/solar-setup.done ] && pass "first-boot setup finished" \
  || { fail "first-boot setup not finished - see /var/log/solar-setup.log"; exit 1; }
mountpoint -q /data && pass "data disk mounted at /data ($(df -h /data | awk 'NR==2{print $2}'))" \
  || warn "/data is not a separate disk - runs will be lost on teardown"
Q=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader) \
  && pass "GPU: $Q" || { fail "nvidia-smi"; exit 1; }
DRV=$(echo "$Q" | cut -d, -f2 | tr -d ' ')
[ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$DRV" | sort -V | head -1)" = "$MIN_DRIVER" ] \
  && pass "driver $DRV >= $MIN_DRIVER" || warn "driver $DRV < $MIN_DRIVER (Isaac Lab EA recommendation) - the Cartpole run decides"
ldconfig -p | grep -q libEGL_nvidia.so.0 && pass "EGL: libEGL_nvidia" || fail "EGL: libEGL_nvidia missing"
ldconfig -p | grep -q libGLX_nvidia.so.0 && pass "GLX: libGLX_nvidia" || warn "GLX: libGLX_nvidia missing"
vulkaninfo --summary 2>/dev/null | grep -qi nvidia && pass "Vulkan sees the NVIDIA GPU" || fail "Vulkan: no NVIDIA device"
[ -e /usr/share/nvidia/nvoptix.bin ] && pass "OptiX: nvoptix.bin" || warn "OptiX: /usr/share/nvidia/nvoptix.bin missing"
ldconfig -p | grep -q libnvidia-encode && pass "NVENC: libnvidia-encode" || warn "NVENC: libnvidia-encode missing (video encode)"
docker info >/dev/null 2>&1 && pass "docker" || { fail "docker not usable (log out/in for the docker group?)"; exit 1; }
[ "$fails" -eq 0 ] || { echo "host checks failed ($fails) - fix before pulling the 16 GB image"; exit 1; }

echo "=== 2. Isaac Lab container: $IMAGE ==="
docker pull -q "$IMAGE" && pass "pulled $(docker image inspect "$IMAGE" --format '{{.Id}}' | cut -c1-19)" \
  || { fail "docker pull"; exit 1; }

C=/data/isaac-sim
RUN=(docker run --rm --gpus all --network=host -e ACCEPT_EULA=Y --entrypoint bash
  -v "$C/cache/kit:/isaac-sim/kit/cache:rw" -v "$C/cache/ov:/root/.cache/ov:rw"
  -v "$C/cache/pip:/root/.cache/pip:rw" -v "$C/cache/glcache:/root/.cache/nvidia/GLCache:rw"
  -v "$C/cache/computecache:/root/.nv/ComputeCache:rw" -v "$C/logs:/root/.nvidia-omniverse/logs:rw"
  -v "$C/data:/root/.local/share/ov/data:rw" -v "$C/documents:/root/Documents:rw"
  -v /data/runs:/workspace/isaaclab/logs:rw "$IMAGE")

"${RUN[@]}" -lc '
  cd /workspace/isaaclab 2>/dev/null || cd /workspace/IsaacLab
  echo "isaaclab at: $(pwd)  version: $(cat VERSION 2>/dev/null)  commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  uv run isaaclab train --help 2>&1 | grep -E -- "--(max_iterations|video|viz)" | head -5
' || fail "container start"

echo "=== 3. train Cartpole ($ITER iterations, headless) ==="
T0=$(date +%s)
"${RUN[@]}" -lc "
  cd /workspace/isaaclab 2>/dev/null || cd /workspace/IsaacLab
  timeout 3600 uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --run_name smoke \
    --max_iterations $ITER --viz none
" > /data/runs/smoke_train.log 2>&1 && pass "trained in $(( $(date +%s) - T0 )) s" \
  || fail "training (see /data/runs/smoke_train.log)"
grep -iE "physics|backend|Mean reward|mean_reward|Learning iteration $((ITER - 1))" /data/runs/smoke_train.log | tail -5

echo "=== 4. play the checkpoint and record a video ==="
"${RUN[@]}" -lc "
  cd /workspace/isaaclab 2>/dev/null || cd /workspace/IsaacLab
  timeout 1800 uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole --num_envs 16 \
    --checkpoint latest --video --video_length 300 --viz none
" > /data/runs/smoke_play.log 2>&1 && pass "played the checkpoint" \
  || fail "play/video (see /data/runs/smoke_play.log)"
VIDEOS=$(find /data/runs -name '*.mp4' -newermt "@$T0" 2>/dev/null)
if [ -n "$VIDEOS" ]; then
  for v in $VIDEOS; do pass "video $(du -h "$v" | cut -f1)  $v"; done
else
  fail "no video written"
fi

echo
[ "$fails" -eq 0 ] && echo "G1 SMOKE PASSED" || echo "G1 SMOKE FAILED ($fails)"
echo "Copy results to the laptop:  scp -r solar@<ip>:/data/runs ./runs/gpu"
echo "WHEN FINISHED (from the laptop):  bash scripts/teardown.sh --yes"
[ "$fails" -eq 0 ]
