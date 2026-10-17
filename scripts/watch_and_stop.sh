#!/usr/bin/env bash
# Laptop side, on the owner's instruction (2026-10-18): watch the pipeline; when everything is done, or when a decision
# from the owner is needed and they have not answered within 30 min, pull everything to the laptop and STOP (never
# destroy) the instance with the Vast CLI on the box and its instance-scoped key, as in /etc/vast-agents-guide.md.
#   watch      poll every POLL_S; run pull_full.sh every PULL_EVERY_S in the background; exit when attention is needed:
#              10 = final report written (all done), 11 = a job failed, 12 = no compute for IDLE_MIN minutes,
#              13 = box unreachable for 30 min
#   countdown  wait WAIT_MIN (30) minutes; stop early with no action if $DEST/CANCEL_STOP appears; then do `stop`
#   stop       pull_full.sh until VERIFIED (3 tries), then stop the instance and print its intended status
#   HOST=root@IP PORT=62992 DEST=/d/Workspace/DMRL_artifacts/run_56353913/full bash scripts/watch_and_stop.sh watch
set -uo pipefail
: "${HOST:?}" "${PORT:?}" "${DEST:?}"
MODE="${1:?watch|countdown|stop}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST")
LOG="$DEST/watch.log"
mkdir -p "$DEST"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

status() {   # one line: final=<0|1> running=<job|-> failed=<a,b|-> idle=<consecutive non-compute heartbeats>
  "${SSH[@]}" 'python3 - <<"PY"
import json, os
S = "/workspace/apsp"; R = S + "/project_runs/apsp_full_sd15_v1/job_manifest.jsonl"
last = {}
for l in open(R):
    d = json.loads(l); last[d["id"]] = d["status"]
hb = [json.loads(l)["state"] for l in open(S + "/heartbeat.jsonl").readlines()[-60:]]
idle = 0
for s in reversed(hb):
    if s in ("gpu_compute", "cpu_progress", "cpu_setup", "upload"):
        break
    idle += 1
run = [k for k, v in last.items() if v == "running"]
fail = [k for k, v in last.items() if v == "failed"]
print("final=%d running=%s failed=%s idle=%d hb=%s" % (os.path.exists(S + "/final/final_report.md"),
      ",".join(run) or "-", ",".join(fail) or "-", idle, hb[-1] if hb else "-"))
PY' 2>/dev/null
}

do_stop() {
  local ok=0
  for i in 1 2 3; do
    if bash "$HERE/pull_full.sh"; then ok=1; break; fi
    log "pull attempt $i not verified"
  done
  (( ok )) && log "pull VERIFIED; stopping instance" || log "pull NOT verified after 3 tries (see $DEST/.pull/missing.tsv); stopping anyway as instructed"
  "${SSH[@]}" 'vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY; sleep 5
    vastai show instance $CONTAINER_ID --api-key $CONTAINER_API_KEY --raw | python3 -c "import sys,json; d=json.load(sys.stdin); print(\"intended_status=\", d.get(\"intended_status\"), \"actual_status=\", d.get(\"actual_status\"))"' \
    2>/dev/null | tee -a "$LOG"
  log "stop requested"
}

case "$MODE" in
  watch)
    POLL_S="${POLL_S:-120}"; PULL_EVERY_S="${PULL_EVERY_S:-1800}"; IDLE_MIN="${IDLE_MIN:-10}"
    last_pull=0; pull_pid=""; unreach=0
    while true; do
      now=$(date +%s)
      if (( now - last_pull >= PULL_EVERY_S )) && { [[ -z "$pull_pid" ]] || ! kill -0 "$pull_pid" 2>/dev/null; }; then
        bash "$HERE/pull_full.sh" > /dev/null 2>&1 & pull_pid=$!; last_pull=$now
      fi
      s=$(status)
      if [[ -z "$s" ]]; then
        unreach=$((unreach + POLL_S)); log "box unreachable (${unreach}s)"
        (( unreach >= 1800 )) && { log "EXIT 13 unreachable"; exit 13; }
      else
        unreach=0
        echo "$(date -u +%FT%TZ) $s" >> "$LOG"
        idle=$(sed -E 's/.*idle=([0-9]+).*/\1/' <<< "$s")
        if [[ "$s" == *"final=1 running=-"* ]]; then log "EXIT 10 all done: $s"; wait; exit 10; fi
        if [[ "$s" != *"failed=-"* ]]; then log "EXIT 11 failed job: $s"; exit 11; fi
        # a running CPU job whose outputs do not grow for a while is not idle unless the heartbeat calls it dead
        if (( idle >= IDLE_MIN )) && { [[ "$s" == *"running=-"* ]] || [[ "$s" == *"hb=job_dead"* ]]; }; then log "EXIT 12 no compute for ${idle} min: $s"; exit 12; fi
      fi
      sleep "$POLL_S"
    done ;;
  countdown)
    WAIT_MIN="${WAIT_MIN:-30}"; rm -f "$DEST/CANCEL_STOP"
    log "countdown ${WAIT_MIN} min before pull+stop (touch $DEST/CANCEL_STOP to cancel)"
    bash "$HERE/pull_full.sh" > /dev/null 2>&1 &          # pull now: an idle box may be destroyed by the mentor at 30 min
    end=$(( $(date +%s) + WAIT_MIN * 60 ))
    while (( $(date +%s) < end )); do
      [[ -f "$DEST/CANCEL_STOP" ]] && { log "countdown cancelled"; exit 0; }
      sleep 20
    done
    wait
    do_stop ;;
  stop) do_stop ;;
esac
