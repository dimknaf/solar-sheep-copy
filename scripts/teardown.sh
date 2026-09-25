#!/usr/bin/env bash
# TEAR DOWN THE GPU BOX. Run this when you are done. Costs money every hour it is up.
#
#   bash scripts/teardown.sh          # show what exists in every project, and what it costs
#   bash scripts/teardown.sh --yes    # delete every instance, and every disk except the data disk
#
# Scope: every project in NEBIUS_TENANT_ID, so a box in uk-south2 or eu-south1 is
# found as well as one in us-central1. Falls back to NEBIUS_PROJECT_ID alone.
# IDs come from the environment or .env, never from this file.
#
# The persistent data disk named $KEEP_DISK (default: solar-data) is never deleted;
# it holds checkpoints and caches between sessions. Set KEEP_DISK= to delete it too.
#
# RTX PRO 6000 on-demand is ~$1.80/hr => ~$43/day if left running.
# Deleting an instance does NOT delete its boot disk, so this removes both.

set -euo pipefail
export PATH="$PATH:$HOME/.nebius/bin"
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
KEEP_DISK="${KEEP_DISK-solar-data}"

# Print one line per item of a nebius JSON list: id, then the requested fields.
rows() {
  python3 -c '
import json,sys
fields=sys.argv[1:]
raw=sys.stdin.read()  # empty when nebius itself failed; its error is already on stderr
for i in (json.loads(raw) if raw.strip() else {}).get("items",[]):
    m,sp,st=i.get("metadata",{}),i.get("spec",{}),i.get("status",{})
    out=[m.get("id","")]
    for f in fields:
        out.append(str({"name":m.get("name"),"state":st.get("state"),
                        "size":sp.get("size_gibibytes")}[f]))
    print("\t".join(out))
' "$@"
}

if [ -n "${NEBIUS_TENANT_ID:-}" ]; then
  PROJECTS=$(nebius iam project list --parent-id "$NEBIUS_TENANT_ID" --format json | rows name)
elif [ -n "${NEBIUS_PROJECT_ID:-}" ]; then
  PROJECTS=$(printf '%s\t(from NEBIUS_PROJECT_ID)\n' "$NEBIUS_PROJECT_ID")
else
  echo "Set NEBIUS_TENANT_ID (preferred) or NEBIUS_PROJECT_ID, in the environment or .env." >&2
  exit 2
fi

list_instances() { nebius compute instance list --parent-id "$1" --format json | rows name state; }
list_disks()     { nebius compute disk list     --parent-id "$1" --format json | rows name size; }

total_i=0; total_d=0
while IFS=$'\t' read -r proj pname; do
  [ -n "$proj" ] || continue
  inst=$(list_instances "$proj"); disks=$(list_disks "$proj")
  [ -z "$inst$disks" ] && continue
  echo "=== $pname ($proj) ==="
  [ -n "$inst" ]  && sed 's/^/  instance  /' <<< "$inst"
  [ -n "$disks" ] && sed 's/^/  disk      /' <<< "$disks"
  total_i=$((total_i + $(printf '%s' "$inst"  | grep -c . || true)))
  total_d=$((total_d + $(printf '%s' "$disks" | grep -c . || true)))
done <<< "$PROJECTS"
echo "Total: $total_i instance(s), $total_d disk(s). Kept disk name: ${KEEP_DISK:-(none)}"

if [ "${1:-}" != "--yes" ]; then
  echo
  echo "Nothing deleted. Re-run with --yes to actually tear down."
  exit 0
fi

while IFS=$'\t' read -r proj pname; do
  [ -n "$proj" ] || continue
  inst=$(list_instances "$proj")
  if [ -n "$inst" ]; then
    echo "=== $pname: deleting instances ==="
    while IFS=$'\t' read -r id name _; do
      echo "  deleting instance $name ($id)"
      nebius compute instance delete --id "$id"
    done <<< "$inst"
    echo "  waiting for instances to clear (up to 5 min)..."
    for _ in $(seq 60); do
      [ -z "$(list_instances "$proj")" ] && break
      sleep 5
    done
  fi
  while IFS=$'\t' read -r id name _; do
    [ -n "$id" ] || continue
    if [ -n "$KEEP_DISK" ] && [ "$name" = "$KEEP_DISK" ]; then
      echo "  keeping data disk $name ($id)"; continue
    fi
    echo "  deleting disk $name ($id)"
    nebius compute disk delete --id "$id"
  done <<< "$(list_disks "$proj")"
done <<< "$PROJECTS"

echo
echo "=== final check — only the kept data disk may remain ==="
while IFS=$'\t' read -r proj pname; do
  [ -n "$proj" ] || continue
  left_i=$(list_instances "$proj"); left_d=$(list_disks "$proj")
  [ -n "$left_i$left_d" ] || continue
  echo "  $pname:"
  [ -n "$left_i" ] && sed 's/^/    instance  /' <<< "$left_i"
  [ -n "$left_d" ] && sed 's/^/    disk      /' <<< "$left_d"
done <<< "$PROJECTS"
echo "done."
