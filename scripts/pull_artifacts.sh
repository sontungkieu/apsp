#!/usr/bin/env bash
# Laptop side: pull finished light artifacts (json/jsonl/csv/md/yaml/log/txt, small parquet, PREREG files written on
# the box) every PULL_S seconds. Never touches the workload; a failed pull is retried at the next pass.
#   HOST=root@IP PORT=62992 DEST=/d/Workspace/DMRL_artifacts/run_56353913 bash scripts/pull_artifacts.sh
set -uo pipefail
: "${HOST:?}" "${PORT:?}" "${DEST:?}"
PULL_S="${PULL_S:-600}"
mkdir -p "$DEST"
while true; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  if ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" \
      'cd /workspace && find apsp DMRL/docs/PREREG_SIGNED DMRL/docs/PREREG_PROPOSALS DMRL/configs/evaluation_locked.yaml \
         \( -path apsp/venv-\* -o -path apsp/hf -o -path apsp/evaluators -o -path apsp/xfer -o -path "*/images" \
            -o -path "*/shards" -o -path "*_layout" \) -prune -o -type f \
         \( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" -o -name "*.md" -o -name "*.yaml" -o -name "*.log" \
            -o -name "*.out" -o -name "*.txt" -o -name "*.tsv" -o -name "*.parquet" -o -name "*.pt" \) -size -200M -print 2>/dev/null \
       | tar czf - -T - 2>/dev/null' > "$DEST/.pull_${ts}.tgz" 2>/dev/null; then
    tar xzf "$DEST/.pull_${ts}.tgz" -C "$DEST" 2>/dev/null && echo "$ts ok $(du -sh "$DEST/.pull_${ts}.tgz" | cut -f1)" >> "$DEST/pull.log"
  else
    echo "$ts pull failed" >> "$DEST/pull.log"
  fi
  rm -f "$DEST/.pull_${ts}.tgz"
  sleep "$PULL_S"
done
