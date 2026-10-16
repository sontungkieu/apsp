#!/usr/bin/env bash
# Arm the cost watchdog on THIS instance before any workload, detached from the SSH session (setsid + nohup,
# stdin from /dev/null), then wait until it reports "armed". See docs/COST_WATCHDOG.md.
#
#   bash scripts/arm_watchdog.sh <instance_id> <deadline_epoch> <state_dir> [extra vast_watchdog.py args...]
#   e.g. bash scripts/arm_watchdog.sh "$CONTAINER_ID" $(( $(date +%s) + 115*60 )) /workspace/apsp/watchdog \
#          --finished-marker /workspace/apsp/diag2.finished --idle-grace-s 900 \
#          --save-cmd "pkill -TERM -f '[s]cripts/gpu_smoke.sh'" \
#          --save-cmd 'tar --exclude="*.lock" -czf /workspace/apsp/diag2_artifacts.tgz -C /workspace/apsp/runs diag2'
#
# The API key stays in the environment (CONTAINER_API_KEY on Vast); never pass it as an argument.
set -euo pipefail
IID="$1"; DEADLINE="$2"; STATE="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${WATCHDOG_PYTHON:-python3}"
mkdir -p "${STATE}"
if "${PY}" "${HERE}/vast_watchdog.py" --check-armed --instance-id "${IID}" --state-dir "${STATE}" >/dev/null 2>&1; then
  echo "watchdog already armed for ${IID} in ${STATE}"; exit 0
fi
# DEADLINE "none" = explicit no-financial-cap mode (D47): vast_watchdog.py then requires --finished-marker,
# --workload-pidfile and --stall-s, and stops on completion, workload death or a stall instead of a clock.
if [[ "${DEADLINE}" == "none" ]]; then DL=(--no-financial-deadline); else DL=(--deadline-epoch "${DEADLINE}"); fi
setsid nohup "${PY}" "${HERE}/vast_watchdog.py" --instance-id "${IID}" "${DL[@]}" \
  --state-dir "${STATE}" "$@" > "${STATE}/watchdog.out" 2>&1 < /dev/null &
for _ in $(seq 1 30); do
  if "${PY}" "${HERE}/vast_watchdog.py" --check-armed --instance-id "${IID}" --state-dir "${STATE}" >/dev/null 2>&1; then
    echo "watchdog armed: instance ${IID}, deadline $(date -u -d "@${DEADLINE%.*}" +%FT%TZ 2>/dev/null || echo "${DEADLINE} (completion/death/stall triggers)")"
    exit 0
  fi
  sleep 1
done
echo "watchdog failed to arm; see ${STATE}/watchdog.out" >&2
exit 1
