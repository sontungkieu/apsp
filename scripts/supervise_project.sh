#!/usr/bin/env bash
# Keeps the project pipeline moving (plan v1.3): whenever run_project is not running, relaunch it with --resume so that
# newly eligible jobs (new code/plan deployed, a marker written by a previous job) start without waiting for a human.
# Never stops or destroys the instance. Stops itself only when $APSP_STORE/PROJECT_COMPLETE exists.
#   APSP_STORE=/workspace/apsp setsid nohup bash scripts/supervise_project.sh > $APSP_STORE/supervisor.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?}"
PIDF="${APSP_STORE}/run_project.pid"
echo $$ > "${APSP_STORE}/supervisor.pid"
while [[ ! -f "${APSP_STORE}/PROJECT_COMPLETE" ]]; do
  if ! { [[ -s "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }; then
    echo "$(date -u +%FT%TZ) relaunch run_project (HEAD $(git rev-parse --short HEAD))"
    bash scripts/launch_project.sh 2>&1 | tail -2
  fi
  sleep "${SUPERVISE_POLL_S:-30}"
done
echo "$(date -u +%FT%TZ) PROJECT_COMPLETE present; supervisor exits (instance left running)"
