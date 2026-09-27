#!/usr/bin/env bash
# Copy PASSIVE results from the GPU box to the laptop (run in WSL), then rebuild the local
# progress page.
#   bash scripts/gpu/pull_runs.sh            # once
#   bash scripts/gpu/pull_runs.sh --loop     # every 5 minutes until Ctrl+C
#
# Safety: only these file types come back - videos (.mp4), images (.png), metrics and results
# (.json, TensorBoard event files) and text logs (.log, .yaml). Model checkpoints (.pt) and
# anything executable stay on the box: a .pt is a pickle and loading one can run code.
# Nothing pulled is ever executed. Destination: runs/gpu/runs (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../.."
IP="${VM_IP:-$(cat "$HOME/.solar/vm_ip")}"
KEY="${SSH_KEY:-$HOME/.ssh/solar_nebius}"
DEST="runs/gpu/runs"
WIN_REPO="${WIN_REPO:-/mnt/c/Users/dimkn/source/repos/random/nebius-hackathon}"

pull_once() {
  mkdir -p "$WIN_REPO/$DEST"
  ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes "solar@$IP" \
    "cd /data/runs && find . -type f -size -300M \\( -name '*.mp4' -o -name '*.png' -o -name '*.json' \
       -o -name '*.log' -o -name '*.yaml' -o -name 'events.out.tfevents.*' \\) -print0 \
     | tar czf - --null -T -" \
    | tar xzf - -C "$WIN_REPO/$DEST" --no-same-owner --no-same-permissions
  python3 "$WIN_REPO/scripts/progress_page.py" --runs "$WIN_REPO/$DEST" >/dev/null
  echo "$(date +%H:%M:%S) pulled -> $DEST ; progress page rebuilt"
}

if [ "${1:-}" = "--loop" ]; then
  while true; do pull_once || echo "$(date +%H:%M:%S) pull failed (box busy or down) - retrying"; sleep 300; done
else
  pull_once
fi
