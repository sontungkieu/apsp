#!/usr/bin/env bash
# One-shot: once job $1 is recorded done, end the current run_project pass (started with older code that has no
# RELOAD_PLAN check) so the supervisor relaunches it with the deployed plan/code. SIGKILL of the whole session at once
# so the job that has just started (CPU) is not recorded as failed; it is re-run as a new attempt.
#   APSP_STORE=/workspace/apsp setsid nohup bash scripts/switch_after_job.sh p2_bank > $APSP_STORE/switch.log 2>&1 &
set -uo pipefail
: "${APSP_STORE:?}"
JOB="$1"
MAN="${APSP_STORE}/project_runs/${PROJECT_RUN:-apsp_full_sd15_v1}/job_manifest.jsonl"
until python3 -c "import json,sys; sys.exit(0 if any(json.loads(l).get('id')=='$JOB' and json.loads(l).get('status') in ('done','failed') for l in open('$MAN')) else 1)"; do
  sleep 10
done
PID=$(cat "${APSP_STORE}/run_project.pid")
SID=$(ps -o sid= -p "$PID" | tr -d ' ')
echo "$(date -u +%FT%TZ) $JOB finished; ending run_project session $SID (pid $PID) for relaunch with the deployed plan"
[[ -n "$SID" && "$SID" == "$PID" ]] && pkill -KILL -s "$SID"
