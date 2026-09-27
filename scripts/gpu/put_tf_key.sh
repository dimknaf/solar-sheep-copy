#!/usr/bin/env bash
# Give the GPU box the Token Factory key for the fleet brain (orchestrator/), run IN WSL on the laptop:
#   ENV_FILE=/mnt/c/.../nebius-hackathon/.env bash scripts/gpu/put_tf_key.sh
# The key (NEBIUS_TOKEN_FACTORY_KEY in the gitignored .env) travels only on the SSH connection's
# stdin into ~/.config/solar/tf.env on the box (mode 600, outside the repo and outside /data/runs,
# so pull_runs.sh never copies it back). scripts/gpu/isaac.sh hands that file to the container with
# --env-file. The key is never put on a command line, echoed, or written to a log.
set -euo pipefail
cd "$(dirname "$0")/../.."
IP="${VM_IP:-$(cat "$HOME/.solar/vm_ip")}"
KEY="${SSH_KEY:-$HOME/.ssh/solar_nebius}"
ENV_FILE="${ENV_FILE:-.env}"
line="$(grep -E '^NEBIUS_TOKEN_FACTORY_KEY=' "$ENV_FILE" | tail -1 | tr -d '\r')" \
  || { echo "no NEBIUS_TOKEN_FACTORY_KEY in $ENV_FILE" >&2; exit 1; }
value="${line#*=}"; value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
case "$value" in v1.*) ;; *) echo "NEBIUS_TOKEN_FACTORY_KEY in $ENV_FILE does not look like a Token Factory key (v1.)" >&2; exit 1 ;; esac
printf 'NEBIUS_TOKEN_FACTORY_KEY=%s\n' "$value" | ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes "solar@$IP" \
  'umask 077; mkdir -p ~/.config/solar; cat > ~/.config/solar/tf.env; chmod 600 ~/.config/solar/tf.env
   echo "key stored on the box: ~/.config/solar/tf.env, $(wc -c < ~/.config/solar/tf.env) bytes, mode $(stat -c %a ~/.config/solar/tf.env)"'
