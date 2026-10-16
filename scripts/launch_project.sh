#!/usr/bin/env bash
# Start the project orchestrator on THIS Vast instance, detached and resumable.
#
# Plan v1.3 / user order 2026-10-16: the instance must NOT be stopped before the whole project is complete. The
# watchdog is therefore armed --monitor-only: it creates no provider client, has no stop call, and only records
# alerts (run_project pid gone, no progress for STALL_MIN). run_project runs in its own session / process group
# (setsid); jobs are ended or retried by their PID/PGID, never by stopping the machine.
#
#   APSP_STORE=/workspace/apsp bash scripts/launch_project.sh [extra run_project args, e.g. --only p1_env]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?set APSP_STORE}"
: "${CONTAINER_ID:?not a Vast instance}"
PLAN="${PLAN:-configs/full_experiments.yaml}"
PLAN_IID=$(grep -E '^instance_id:' "$PLAN" | awk '{print $2}')
if [[ "$PLAN_IID" != "$CONTAINER_ID" ]]; then
  echo "refusing: plan instance_id $PLAN_IID != this instance $CONTAINER_ID" >&2; exit 3
fi
PROJECT_RUN=$(grep -E '^project_run:' "$PLAN" | awk '{print $2}')
STATE="${APSP_STORE}/watchdog"
PIDF="${APSP_STORE}/run_project.pid"
STALL_MIN="${STALL_MIN:-60}"
mkdir -p "${APSP_STORE}/project_runs/${PROJECT_RUN}/logs" "${APSP_STORE}/runs" "$STATE"
TS=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -s "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "run_project already running (pid $(cat "$PIDF")); not starting a second one" >&2; exit 3
fi
rm -f "$PIDF"
bash scripts/arm_watchdog.sh "$CONTAINER_ID" monitor "$STATE" \
  --workload-pidfile "$PIDF" --stall-s $(( STALL_MIN * 60 )) --idle-grace-s 600 --poll-s 30 \
  --progress-path "${APSP_STORE}/project_runs/${PROJECT_RUN}" --progress-path "${APSP_STORE}/runs/calib1/logs" \
  --progress-path "${APSP_STORE}/runs/calib1/summary.tsv" --progress-path "${APSP_STORE}/runs/pilot1/logs"
python3 - "$STATE/state.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s.get("monitor_only") is True and s.get("stop_path", "").startswith("disabled"), s
print("watchdog:", {k: s.get(k) for k in ("instance_id", "monitor_only", "stop_path", "deadline_epoch", "pid")})
PY
# fresh disk: the pinned generator venv must exist before the orchestrator itself can run
if [[ ! -x "${APSP_STORE}/venv-gpu/bin/python" ]]; then
  VENV="${APSP_STORE}/venv-gpu" HF_HOME="${APSP_STORE}/hf" bash env/setup_gpu_linux.sh > "${APSP_STORE}/setup_${TS}.log" 2>&1 \
    || { echo "setup failed; see ${APSP_STORE}/setup_${TS}.log" >&2; exit 1; }
fi
PY="${APSP_STORE}/venv-gpu/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
# the orchestrator writes its own pid into $PIDF; setsid gives it a new session and process group
setsid nohup "$PY" -m apsp.run_project --plan "$PLAN" --resume "$@" \
  > "${APSP_STORE}/run_project_${TS}.out" 2>&1 < /dev/null &
for _ in $(seq 1 30); do [[ -s "$PIDF" ]] && break; sleep 1; done
[[ -s "$PIDF" ]] || { echo "run_project did not start; see ${APSP_STORE}/run_project_${TS}.out" >&2; exit 1; }
echo "run_project pid $(cat "$PIDF") (own session); log ${APSP_STORE}/run_project_${TS}.out"
