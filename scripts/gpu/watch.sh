#!/usr/bin/env bash
# THE ONE WAY IN: an outbound SSH tunnel from the laptop (run in WSL) to the GPU box.
#   bash scripts/gpu/watch.sh        # keep it running; Ctrl+C closes it
# Then, in the laptop's browser:
#   http://localhost:8080   the live 3D Omniverse world (Isaac Lab Viser visualizer)
#   http://localhost:6006   training charts (TensorBoard)
#
# Safety: both ends bind 127.0.0.1 only. The laptop opens no listening port to any network
# (WSL forwards these to Windows loopback only); the box's services also listen on its
# localhost only, and the box accepts nothing but SSH from this laptop's IP.
# Strict host-key checking: the key was pinned on the first connection after up.sh.
set -euo pipefail
IP="${VM_IP:-$(cat "$HOME/.solar/vm_ip")}"
KEY="${SSH_KEY:-$HOME/.ssh/solar_nebius}"
exec ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 \
  -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  -L 127.0.0.1:6006:127.0.0.1:6006 \
  "solar@$IP"
