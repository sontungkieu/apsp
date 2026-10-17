#!/usr/bin/env bash
# Laptop side, run OUTSIDE the Claude session (own window/process): repeat the verified pull every PULL_S seconds
# until $DEST/STOP_PULL exists. Results go to $DEST/pull.log.
#   HOST=root@IP PORT=62992 DEST=/d/Workspace/DMRL_artifacts/run_56353913/full ACTIVE='runs/pilot1/bench_val/' \
#     bash scripts/pull_loop.sh
set -uo pipefail
: "${DEST:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PULL_S="${PULL_S:-900}"
mkdir -p "$DEST"; echo $$ > "$DEST/pull_loop.pid"
while [[ ! -f "$DEST/STOP_PULL" ]]; do
  bash "$HERE/pull_full.sh"
  for _ in $(seq 1 $((PULL_S / 10))); do [[ -f "$DEST/STOP_PULL" ]] && break; sleep 10; done
done
echo "$(date -u +%FT%TZ) STOP_PULL found; pull loop exits" >> "$DEST/pull.log"
