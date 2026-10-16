#!/usr/bin/env bash
# Live check of the cost watchdog on a real Vast instance. No model, no workload. See docs/CALIBRATION_PLAN.md §2.
#
#   bash scripts/watchdog_live_check.sh            # on the instance, right after login
#
# 1. read-only API probe (GET own instance) through vast_watchdog.py --check-status; abort if it fails
# 2. default MODE=no_deadline: arm in the no-financial-deadline mode P1 uses, around a 90 s dummy workload
#    (MODE=deadline: the old check, deadline 6 min ahead, save margin 60 s); detached from this SSH session
# 3. the operator logs out; the instance should stop by itself at about +2.5 min (+5 min with MODE=deadline)
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
if [[ "${MODE:-no_deadline}" == "no_deadline" ]]; then
  # the mode P1 uses (D47): a dummy workload in its own session runs 90 s, then writes the finished marker; the
  # watchdog ends it by process group if needed, waits the 60 s idle grace and stops -> stop request at about +2.5 min
  rm -f "$STATE/workload.finished" "$STATE/workload.pid"
  setsid nohup bash -c "echo \$\$ > '$STATE/workload.pid'; sleep 90; date -u +%FT%TZ > '$STATE/workload.finished'; sleep 600" \
    > /dev/null 2>&1 < /dev/null &
  for _ in $(seq 1 20); do [[ -s "$STATE/workload.pid" ]] && break; sleep 0.5; done
  echo "== 2. arming (no-financial-deadline mode): instance $CONTAINER_ID, dummy workload pid $(cat "$STATE/workload.pid")"
  bash scripts/arm_watchdog.sh "$CONTAINER_ID" none "$STATE" --poll-s 5 --retry-s 15 --idle-grace-s 60 \
    --finished-marker "$STATE/workload.finished" --workload-pidfile "$STATE/workload.pid" --stall-s 900 \
    --save-cmd "date -u +%FT%TZ > $STATE/save_ran.txt"
  EXPECT=$(( $(date +%s) + 150 ))
else
  DEADLINE=$(( $(date +%s) + 360 ))
  echo "== 2. arming: instance $CONTAINER_ID, deadline $(date -u -d "@$DEADLINE" +%FT%TZ)"
  bash scripts/arm_watchdog.sh "$CONTAINER_ID" "$DEADLINE" "$STATE" --save-margin-s 60 --poll-s 5 --retry-s 15 \
    --save-cmd "date -u +%FT%TZ > $STATE/save_ran.txt"
  EXPECT=$(( DEADLINE - 60 ))
fi
python3 - "$STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/state.json"))
print({k: s.get(k) for k in ("phase", "instance_id", "deadline_epoch", "financial_deadline", "pid")})
PY
echo "== 3. log out now. Expected: stop request at about $(date -u -d "@${EXPECT}" +%TZ)."
echo "== 4. confirm 'Stopped' in the Vast console, then Start again and run:"
echo "     python3 -c \"import json;print(json.load(open('$STATE/state.json')))\""
