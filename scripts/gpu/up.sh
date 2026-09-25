#!/usr/bin/env bash
# CREATE THE GPU BOX. Costs money from the moment Nebius places it (~$1.80/h on-demand,
# ~$0.95/h preemptible). Tear it down with:  bash scripts/teardown.sh --yes
#
#   bash scripts/gpu/up.sh               # show the plan: region, capacity, project, disk, image
#   bash scripts/gpu/up.sh --yes         # actually create (data disk if missing, then the VM)
#
#   REGION=eu-south1 bash scripts/gpu/up.sh --yes    # fallback region
#   PREEMPTIBLE=1    bash scripts/gpu/up.sh --yes    # cheaper; Nebius may stop it at any time
#
# Why this shape (docs/decisions.md D8): on 15 Sep the us-central1 box never booted -
# 4x NotEnoughResources. So: check capacity first, create with --async, then READ the
# operation result; on NotEnoughResources delete at once and move region, never retry-loop.
# IDs come from the environment or .env (NEBIUS_TENANT_ID), never from this file.
# The boot disk is managed (deleted with the VM); the data disk "solar-data" is kept.

set -euo pipefail
export PATH="$PATH:$HOME/.nebius/bin"
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . <(tr -d '\r' < .env); set +a; fi

REGION="${REGION:-uk-south2}"
PLATFORM="${PLATFORM:-gpu-rtx6000-a}"
PRESET="${PRESET:-1gpu-24vcpu-218gb}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}"
VM_NAME="${VM_NAME:-solar-isaac}"   # not NAME: WSL sets NAME to the PC name
BOOT_GIB="${BOOT_GIB:-200}"
DATA_DISK="${KEEP_DISK:-solar-data}"
DATA_GIB="${DATA_GIB:-200}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
KEY="${SSH_KEY:-$HOME/.ssh/solar_nebius}"
: "${NEBIUS_TENANT_ID:?Set NEBIUS_TENANT_ID in the environment or .env}"

json() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }

echo "=== capacity: $PLATFORM / $PRESET in $REGION ==="
nebius capacity resource-advice list --parent-id "$NEBIUS_TENANT_ID" --format json | json "
tier = 'preemptible' if '$PREEMPTIBLE' == '1' else 'on_demand'
hits = [i for i in d.get('items', []) if i['spec'].get('region') == '$REGION'
        and i['spec'].get('compute_instance', {}).get('platform') == '$PLATFORM'
        and i['spec']['compute_instance'].get('preset', {}).get('name') == '$PRESET']
if not hits: sys.exit('no capacity advice for this region/platform/preset')
s = hits[0]['status'].get(tier, {})
print(f\"  {tier}: {s.get('availability_level')}  available {s.get('available')}/{s.get('limit')}\")
if s.get('availability_level') not in ('AVAILABILITY_LEVEL_HIGH', 'AVAILABILITY_LEVEL_MEDIUM'):
    sys.exit('capacity is not HIGH/MEDIUM here - try REGION=eu-south1')
"

PROJECT=$(nebius iam project list --parent-id "$NEBIUS_TENANT_ID" --format json | json "
m = [i for i in d.get('items', []) if i.get('status', {}).get('region') == '$REGION'
     or i.get('spec', {}).get('region') == '$REGION' or i['metadata'].get('name', '').endswith('$REGION')]
if not m: sys.exit('no project in region $REGION')
print(m[0]['metadata']['id'])")
SUBNET=$(nebius vpc subnet list --parent-id "$PROJECT" --format json | json "print(d['items'][0]['metadata']['id'])")
DISK_ID=$(nebius compute disk list --parent-id "$PROJECT" --format json | json "
print(next((i['metadata']['id'] for i in d.get('items', []) if i['metadata'].get('name') == '$DATA_DISK'), ''))")
EXISTING=$(nebius compute instance list --parent-id "$PROJECT" --format json | json "
print(' '.join(i['metadata']['name'] for i in d.get('items', [])))")

echo "=== plan ==="
echo "  project    (the $REGION project of the tenant)"
echo "  instance   $VM_NAME  $PLATFORM $PRESET  $([ "$PREEMPTIBLE" = 1 ] && echo preemptible || echo on-demand)"
echo "  boot disk  ${BOOT_GIB} GiB network_ssd from $IMAGE_FAMILY (deleted with the VM)"
echo "  data disk  $DATA_DISK ${DATA_GIB} GiB $([ -n "$DISK_ID" ] && echo '(exists, reattached)' || echo '(will be created, kept on teardown)')"
echo "  ssh key    $KEY"
[ -n "$EXISTING" ] && echo "  NOTE: instances already in this project: $EXISTING"

if [ "${1:-}" != "--yes" ]; then
  echo; echo "Nothing created. Re-run with --yes to create (starts billing)."; exit 0
fi
case " $EXISTING " in *" $VM_NAME "*) echo "An instance named $VM_NAME already exists - tear it down first." >&2; exit 1;; esac

[ -f "$KEY" ] || { mkdir -p "$(dirname "$KEY")"; ssh-keygen -q -t ed25519 -N '' -C solar-nebius -f "$KEY"; }

if [ -z "$DISK_ID" ]; then
  echo "=== creating data disk $DATA_DISK ==="
  DISK_ID=$(nebius compute disk create --parent-id "$PROJECT" --name "$DATA_DISK" \
    --size-gibibytes "$DATA_GIB" --type network_ssd --format json | json "print(d['metadata']['id'])")
fi

USER_DATA=$(sed "s|__SSH_PUBKEY__|$(cat "$KEY.pub")|" scripts/gpu/cloud-init.yaml)
PREEMPT_ARGS=()
[ "$PREEMPTIBLE" = 1 ] && PREEMPT_ARGS=(--preemptible-on-preemption STOP --preemptible-priority 1)

echo "=== creating instance $VM_NAME (async) ==="
OP=$(nebius compute instance create --async --format json --parent-id "$PROJECT" --name "$VM_NAME" \
  --resources-platform "$PLATFORM" --resources-preset "$PRESET" \
  --boot-disk-attach-mode read_write \
  --boot-disk-managed-disk-name "$VM_NAME-boot" --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes "$BOOT_GIB" \
  --boot-disk-managed-disk-source-image-family-image-family "$IMAGE_FAMILY" \
  --secondary-disks "[{\"attach_mode\":\"read_write\",\"device_id\":\"solar-data\",\"existing_disk\":{\"id\":\"$DISK_ID\"}}]" \
  --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET\",\"ip_address\":{},\"public_ip_address\":{}}]" \
  --cloud-init-user-data "$USER_DATA" "${PREEMPT_ARGS[@]}" | json "print(d.get('id') or d['metadata']['id'])")

echo "  operation $OP - waiting (placement can take ~6 min; NotEnoughResources shows up here)"
nebius compute instance operation wait "$OP" >/dev/null || true
RESULT=$(nebius compute instance operation get "$OP" --format json)
CODE=$(echo "$RESULT" | json "print(d.get('status', {}).get('code', 0) or 0)")
INST=$(echo "$RESULT" | json "print(d.get('resource_id', ''))")
if [ "$CODE" != "0" ]; then
  echo "$RESULT" | json "print('  FAILED:', d.get('status'))"
  echo "  deleting the failed instance so it does not bill for its boot disk"
  [ -n "$INST" ] && nebius compute instance delete --id "$INST" || true
  echo "  try the next region:  REGION=eu-south1 bash scripts/gpu/up.sh --yes"
  exit 1
fi

IP=$(nebius compute instance get --id "$INST" --format json | json "
nics = d.get('status', {}).get('network_interfaces', [])
print(nics[0].get('public_ip_address', {}).get('address', '').split('/')[0] if nics else '')")
echo
echo "=== UP: $VM_NAME in $REGION ==="
echo "  ssh -i $KEY solar@$IP"
echo "  first-boot setup log: ssh ... 'tail -f /var/log/solar-setup.log'  (done: /var/lib/solar-setup.done)"
echo "  then: scp scripts/gpu/smoke.sh solar@$IP: && ssh ... 'bash smoke.sh'"
echo "  WHEN FINISHED: bash scripts/teardown.sh --yes"
