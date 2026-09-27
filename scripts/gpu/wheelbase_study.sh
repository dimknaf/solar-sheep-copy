#!/usr/bin/env bash
# Wheelbase study, ON THE GPU BOX (owner, 27 Sep: "the two axles are so close"):
#   bash scripts/gpu/wheelbase_study.sh            # converted USDs must exist (robot/usd/convert_rover.py)
# Runs measure_rover.py on the 0.22 / 0.30 / 0.36 m rovers, one after the other, at real speed and
# repeated, so the owner can watch each in the live view (http://localhost:18080 via watch.sh).
# Results: /data/runs/measure/measure_physx_240hz_wbXXX.{json,mp4}; logs run_wbXXX.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOOPS="${LOOPS:-3}"
declare -A USD=(
  [wb022]=/data/runs/usd/rover_train/rover_train/rover_train.usda
  [wb030]=/data/runs/usd/rover_train_wb030/rover_train_wb030/rover_train_wb030.usda
  [wb036]=/data/runs/usd/rover_train_wb036/rover_train_wb036/rover_train_wb036.usda
)
for tag in wb022 wb030 wb036; do
  usd="${USD[$tag]}"
  [ -f "$usd" ] || { echo "missing $usd - run robot/usd/convert_rover.py first"; exit 1; }
  echo "=== $tag: $usd"
  bash scripts/gpu/isaac.sh envs/isaac_rover/measure_rover.py --usd "$usd" --tag "$tag" \
    --realtime --loops "$LOOPS" > "/data/runs/measure/run_$tag.log" 2>&1
  grep -E "^\s+(straight|spin|climb_1[46]|hold_3[36]|pitch_)|PASS|FAIL|Traceback" "/data/runs/measure/run_$tag.log"
done
echo "WHEELBASE STUDY DONE"
