#!/usr/bin/env bash
# Off-instance backup that runs on the instance (independent of the laptop). /workspace is NOT a volume on Vast
# instances without one: everything not copied off is lost on destroy.
#   APSP_STORE=/workspace/apsp APSP_BACKUP_REMOTE=<rclone-remote>:<path> setsid nohup bash scripts/backup_daemon.sh &
# The rclone remote is configured by the owner (rclone config on the instance); the daemon never reads or prints
# credentials. Every BACKUP_POLL_S (default 60 s) it copies new/changed files (rclone copy = incremental: finished
# shards, checkpoints, job outputs, frozen files, PREREG snapshots written on the box) and syncs logs, heartbeat and
# state. A shard/job is "backed up" when its files are listed on the remote; the daemon records the last successful
# pass in backup_status.json and appends failures to ALERTS.jsonl.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?}" "${APSP_BACKUP_REMOTE:?set APSP_BACKUP_REMOTE=remote:path}"
POLL="${BACKUP_POLL_S:-60}"
INST="${CONTAINER_ID:-unknown}"
DST="${APSP_BACKUP_REMOTE%/}/${INST}"
echo $$ > "${APSP_STORE}/backup.pid"
FILTER="${APSP_STORE}/.backup_filter"
cat > "$FILTER" <<'EOF'
- venv-*/**
- hf/**
- evaluators/**
- xfer/**
- **/.tmp_*/**
- .heartbeat.tmp
+ **
EOF
RC=(rclone --config "${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}" --transfers 8 --checkers 16 --retries 5
    --low-level-retries 10 --stats 0 --log-level ERROR)
while true; do
  t0=$(date -u +%FT%TZ)
  ok=1
  "${RC[@]}" copy "${APSP_STORE}" "${DST}/store" --filter-from "$FILTER" || ok=0
  # repo-side files written by pipeline jobs (PREREG snapshots, locked evaluation config)
  "${RC[@]}" copy docs/PREREG_SIGNED "${DST}/repo/docs/PREREG_SIGNED" || ok=0
  "${RC[@]}" copy docs/PREREG_PROPOSALS "${DST}/repo/docs/PREREG_PROPOSALS" || ok=0
  [[ -f configs/evaluation_locked.yaml ]] && { "${RC[@]}" copyto configs/evaluation_locked.yaml "${DST}/repo/configs/evaluation_locked.yaml" || ok=0; }
  if [[ $ok -eq 1 ]]; then
    printf '{"last_ok_start": "%s", "last_ok_end": "%s", "remote": "%s"}\n' "$t0" "$(date -u +%FT%TZ)" "$DST" \
      > "${APSP_STORE}/backup_status.json"
    "${RC[@]}" copyto "${APSP_STORE}/backup_status.json" "${DST}/store/backup_status.json" || true
  else
    echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"state\": \"backup_failed\", \"why\": \"rclone copy to ${DST} failed\"}" \
      >> "${APSP_STORE}/ALERTS.jsonl"
  fi
  sleep "$POLL"
done
