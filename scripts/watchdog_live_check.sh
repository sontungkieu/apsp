#!/usr/bin/env bash
# Live check of the cost watchdog on a real Vast instance. No model, no workload. See docs/CALIBRATION_PLAN.md §2.
#
#   bash scripts/watchdog_live_check.sh            # on the instance, right after login
#
# 1. read-only API probe (GET own instance) through vast_watchdog.py --check-status; abort if it fails
# 2. arm the watchdog with a deadline 6 min ahead (save margin 60 s), detached from this SSH session
# 3. the operator logs out; the instance should stop by itself at about +5 min
# 4. EXTERNAL confirmation is required: the Vast console must show the instance as Stopped
#    (SSH loss alone is not proof). After the next Start, state.json on the kept disk shows what the watchdog saw.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CONTAINER_ID:?not a Vast instance (CONTAINER_ID unset)}"
STATE="${WATCHDOG_STATE_DIR:-/workspace/apsp/watchdog_live_check}"
mkdir -p "$(dirname "$STATE")"
echo "== 1. API probe (read-only)"
if ! python3 scripts/vast_watchdog.py --instance-id "$CONTAINER_ID" --state-dir "$STATE" --check-status; then
  echo "API probe failed: the watchdog cannot be trusted on this provider; do NOT start a long session" >&2
  exit 4
fi
DEADLINE=$(( $(date +%s) + 360 ))
echo "== 2. arming: instance $CONTAINER_ID, deadline $(date -u -d "@$DEADLINE" +%FT%TZ)"
bash scripts/arm_watchdog.sh "$CONTAINER_ID" "$DEADLINE" "$STATE" --save-margin-s 60 --poll-s 5 --retry-s 15 \
  --save-cmd "date -u +%FT%TZ > $STATE/save_ran.txt"
python3 - "$STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/state.json"))
print({k: s[k] for k in ("phase", "instance_id", "deadline_epoch", "pid")})
PY
echo "== 3. log out now. Expected: stop request at about $(date -u -d "@$(( DEADLINE - 60 ))" +%TZ)."
echo "== 4. confirm 'Stopped' in the Vast console, then Start again and run:"
echo "     python3 -c \"import json;print(json.load(open('$STATE/state.json')))\""
