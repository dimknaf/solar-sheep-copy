#!/usr/bin/env bash
# Start TensorBoard ON THE GPU BOX, bound to the box's localhost only (never --bind_all).
# The owner sees it at http://localhost:6006 on the laptop through scripts/gpu/watch.sh.
#   bash scripts/gpu/tensorboard.sh          # (re)start; survives until the VM is deleted
set -euo pipefail
IMAGE="${IMAGE:-solar/isaac-lab:3.0.0-rc1-video}"
docker rm -f tb >/dev/null 2>&1 || true
docker run -d --name tb --restart unless-stopped --network=host -v /data/runs:/runs:ro \
  --entrypoint bash "$IMAGE" -c \
  'source /isaac-sim/setup_python_env.sh; exec /isaac-sim/kit/python/bin/python3 -m tensorboard.main \
     --logdir /runs --host 127.0.0.1 --port 6006 --reload_interval 30' >/dev/null
for _ in $(seq 30); do curl -fs -o /dev/null http://127.0.0.1:6006/ && break; sleep 1; done
ss -tln | grep -q '127.0.0.1:6006' && echo "TensorBoard up on 127.0.0.1:6006 (box-local only)" \
  || { echo "TensorBoard did not come up"; docker logs tb | tail -5; exit 1; }
