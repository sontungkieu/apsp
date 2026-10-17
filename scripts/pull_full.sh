#!/usr/bin/env bash
# Laptop side: incremental mirror of everything the study produced on the box: all of /workspace/apsp except the
# venvs and the HF model cache (both reinstallable from pinned versions), plus the box repo's untracked files (PREREG
# locks written on the box) and its git state. Missing or size-changed files are fetched over STREAMS parallel
# ssh/tar streams (one stream is ~2 MB/s on this link). Ends with "VERIFIED" or "MISSING n" against a fresh remote list
# (files rewritten every minute by the heartbeat/supervisor are not counted).
#   HOST=root@IP PORT=62992 DEST=/d/Workspace/DMRL_artifacts/run_56353913/full bash scripts/pull_full.sh
set -uo pipefail
: "${HOST:?}" "${PORT:?}" "${DEST:?}"
STREAMS="${STREAMS:-4}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$PORT" "$HOST")
W="$DEST/.pull"
mkdir -p "$W"
VOLATILE='(heartbeat\.(json|jsonl|log)|supervisor\.log|run_project_[0-9TZ]+\.out|project_state\.json|budget\.json|box_repo_state\.txt|ALERTS\.jsonl)$'

remote_list() {
  "${SSH[@]}" 'cd /workspace/DMRL && { git log -1 --format=%H; git status --porcelain=v1; git diff; } > /workspace/apsp/box_repo_state.txt 2>&1
    cd /workspace && find apsp \( -path "apsp/venv-*" -o -path apsp/hf \) -prune -o -type f -printf "%s\t%p\n"
    cd /workspace/DMRL && git ls-files -o --exclude-standard | while IFS= read -r f; do printf "%s\tDMRL/%s\n" "$(stat -c %s "$f")" "$f"; done' \
    2>/dev/null | LC_ALL=C sort > "$W/remote.tsv"
  [[ -s "$W/remote.tsv" ]]
}
local_list() {
  (cd "$DEST" && find apsp DMRL -type f -printf "%s\t%p\n" 2>/dev/null) | LC_ALL=C sort > "$W/local.tsv"
}

remote_list || { echo "$(date -u +%FT%TZ) remote list failed" | tee -a "$DEST/pull.log"; exit 2; }
local_list
LC_ALL=C comm -23 "$W/remote.tsv" "$W/local.tsv" | sort -t$'\t' -k1,1nr > "$W/need.tsv"
n=$(wc -l < "$W/need.tsv"); mb=$(awk -F'\t' '{s+=$1} END {printf "%.0f", s/1e6}' "$W/need.tsv")
echo "$(date -u +%FT%TZ) fetching $n files, ${mb} MB, $STREAMS streams" | tee -a "$DEST/pull.log"
if (( n > 0 )); then
  rm -f "$W"/chunk.*
  awk -F'\t' -v n="$STREAMS" -v w="$W" '{print $2 > (w "/chunk." (NR % n))}' "$W/need.tsv"
  for c in "$W"/chunk.*; do
    ( "${SSH[@]}" 'cd /workspace && tar cf - --ignore-failed-read -T - 2>/dev/null' < "$c" 2>/dev/null | tar xf - -C "$DEST" 2>/dev/null ) &
  done
  wait
fi
remote_list || { echo "$(date -u +%FT%TZ) remote list failed after fetch" | tee -a "$DEST/pull.log"; exit 2; }
local_list
LC_ALL=C comm -23 "$W/remote.tsv" "$W/local.tsv" | grep -Ev "$VOLATILE" > "$W/missing.tsv"
m=$(wc -l < "$W/missing.tsv")
tot=$(awk -F'\t' '{s+=$1} END {printf "%.0f", s/1e6}' "$W/remote.tsv")
if (( m == 0 )); then
  echo "$(date -u +%FT%TZ) VERIFIED $(wc -l < "$W/remote.tsv") files ${tot} MB" | tee -a "$DEST/pull.log"
else
  echo "$(date -u +%FT%TZ) MISSING $m (see $W/missing.tsv)" | tee -a "$DEST/pull.log"
  exit 1
fi
