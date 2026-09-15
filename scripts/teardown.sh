#!/usr/bin/env bash
# TEAR DOWN THE GPU BOX. Run this when you are done. Costs money every hour it is up.
#
#   bash scripts/teardown.sh          # show what is running and what it costs
#   bash scripts/teardown.sh --yes    # actually delete it
#
# RTX PRO 6000 on-demand is ~$1.80/hr => ~$43/day if left running.
# Deleting the instance does NOT delete its boot disk, so this removes both.

set -uo pipefail
export PATH="$PATH:/root/.nebius/bin:$HOME/.nebius/bin"
PROJECT="project-u00vpbp7kc00vhag08bn1s"

echo "=== instances in $PROJECT ==="
nebius compute instance list --parent-id "$PROJECT" --format json 2>/dev/null \
  | python3 -c '
import json,sys
d=json.load(sys.stdin); items=d.get("items",[])
if not items: print("  (none — nothing is billing)"); raise SystemExit
for i in items:
    m=i.get("metadata",{}); s=i.get("status",{})
    print("  %-22s %-38s %s" % (m.get("name"), m.get("id"), s.get("state")))
'

echo
echo "=== disks ==="
nebius compute disk list --parent-id "$PROJECT" --format json 2>/dev/null \
  | python3 -c '
import json,sys
d=json.load(sys.stdin); items=d.get("items",[])
if not items: print("  (none)"); raise SystemExit
for i in items:
    m=i.get("metadata",{}); s=i.get("spec",{})
    print("  %-22s %-38s %s GiB" % (m.get("name"), m.get("id"), s.get("size_gibibytes")))
'

if [ "${1:-}" != "--yes" ]; then
  echo
  echo "Nothing deleted. Re-run with --yes to actually tear down."
  exit 0
fi

echo
echo "=== deleting instances ==="
for id in $(nebius compute instance list --parent-id "$PROJECT" --format json 2>/dev/null \
            | python3 -c 'import json,sys;[print(i["metadata"]["id"]) for i in json.load(sys.stdin).get("items",[])]'); do
  echo "  deleting $id"
  nebius compute instance delete --id "$id" 2>&1 | tail -2
done

echo "=== waiting for instances to clear, then deleting disks ==="
sleep 25
for id in $(nebius compute disk list --parent-id "$PROJECT" --format json 2>/dev/null \
            | python3 -c 'import json,sys;[print(i["metadata"]["id"]) for i in json.load(sys.stdin).get("items",[])]'); do
  echo "  deleting $id"
  nebius compute disk delete --id "$id" 2>&1 | tail -2
done

echo
echo "=== final check — this should be empty ==="
nebius compute instance list --parent-id "$PROJECT" --format json 2>/dev/null \
  | python3 -c 'import json,sys;print("  instances left:", len(json.load(sys.stdin).get("items",[])))'
nebius compute disk list --parent-id "$PROJECT" --format json 2>/dev/null \
  | python3 -c 'import json,sys;print("  disks left:    ", len(json.load(sys.stdin).get("items",[])))'
