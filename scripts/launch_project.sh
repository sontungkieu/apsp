#!/usr/bin/env bash
# Start the project orchestrator on THIS Vast instance in the explicit no-financial-cap mode (DECISIONS D47):
#   1. arm the watchdog first (detached), without a money deadline: it stops the instance when run_project writes
#      its finished marker (no eligible job left) + idle grace, when the run_project pid dies without the marker, or
#      after STALL_MIN without progress (watched logs unchanged and GPU idle);
#   2. launch run_project in its own session/process group (setsid), so the watchdog ends it by PGID, never by
#      command-line pattern.
#
#   APSP_STORE=/workspace/apsp bash scripts/launch_project.sh [extra run_project args, e.g. --only p1_env]
#
# Packaging (tar of the run directories) is a bounded save command; if it fails the stop is still sent.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?set APSP_STORE}"
: "${CONTAINER_ID:?not a Vast instance}"
PLAN="${PLAN:-configs/full_experiments.yaml}"
PROJECT_RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$PLAN'))['project_run'])" 2>/dev/null || echo apsp_full_sd15_v1)
PLAN_IID=$(grep -E '^instance_id:' "$PLAN" | awk '{print $2}')
if [[ "$PLAN_IID" != "$CONTAINER_ID" ]]; then
  echo "refusing: plan instance_id $PLAN_IID != this instance $CONTAINER_ID" >&2; exit 3
fi
STATE="${APSP_STORE}/watchdog"
PIDF="${APSP_STORE}/run_project.pid"; FIN="${APSP_STORE}/run_project.finished"
STALL_MIN="${STALL_MIN:-60}"; IDLE_GRACE_S="${IDLE_GRACE_S:-900}"
rm -f "$FIN" "$PIDF"
mkdir -p "${APSP_STORE}/project_runs/${PROJECT_RUN}/logs" "${APSP_STORE}/runs" "$STATE"
TS=$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/arm_watchdog.sh "$CONTAINER_ID" none "$STATE" \
  --finished-marker "$FIN" --workload-pidfile "$PIDF" --stall-s $(( STALL_MIN * 60 )) --idle-grace-s "$IDLE_GRACE_S" \
  --progress-path "${APSP_STORE}/project_runs/${PROJECT_RUN}" --progress-path "${APSP_STORE}/runs/calib1/logs" \
  --progress-path "${APSP_STORE}/runs/calib1/summary.tsv" --progress-path "${APSP_STORE}/runs/pilot1/logs" \
  --save-cmd-timeout-s 900 \
  --save-cmd "tar --exclude='*.lock' --exclude='banks' -czf ${APSP_STORE}/project_artifacts_${TS}.tgz -C ${APSP_STORE} project_runs runs"
# fresh disk: the pinned generator venv must exist before the orchestrator itself can run (watchdog already armed;
# its never-started grace covers this)
if [[ ! -x "${APSP_STORE}/venv-gpu/bin/python" ]]; then
  VENV="${APSP_STORE}/venv-gpu" HF_HOME="${APSP_STORE}/hf" bash env/setup_gpu_linux.sh > "${APSP_STORE}/setup_${TS}.log" 2>&1 \
    || { echo "setup failed; see ${APSP_STORE}/setup_${TS}.log (the watchdog stops the instance after its grace)" >&2; exit 1; }
fi
PY="${APSP_STORE}/venv-gpu/bin/python"
# the orchestrator writes its own pid into $PIDF; setsid gives it a new session and process group
setsid nohup "$PY" -m apsp.run_project --plan "$PLAN" --resume "$@" \
  > "${APSP_STORE}/run_project_${TS}.out" 2>&1 < /dev/null &
for _ in $(seq 1 30); do [[ -s "$PIDF" ]] && break; sleep 1; done
[[ -s "$PIDF" ]] || { echo "run_project did not start; see ${APSP_STORE}/run_project_${TS}.out" >&2; exit 1; }
echo "run_project pid $(cat "$PIDF") (own session); log ${APSP_STORE}/run_project_${TS}.out; watchdog ${STATE}/state.json"
