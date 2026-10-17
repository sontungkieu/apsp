#!/usr/bin/env bash
# Laptop side: resumable, verified mirror of everything the study produced on the box.
#   Kept: all of /workspace/apsp (banks, parquet, embeddings, latents, images, runs, RL checkpoints/logs, tuned
#   baselines, PREREG/decision files, orchestrator state, cost tables, env freezes), /workspace/xfer, and the box repo's
#   untracked files + git state (commit, status, diff, submodule SHA/status) and an inventory of the HF cache
#   (file sizes + pinned refs, never token files).
#   Left out: apsp/venv-* and apsp/hf (foundation model weights, re-downloadable from the pinned revisions).
# Pass 1 fetches missing or size-changed files (STREAMS parallel ssh/tar streams; files already correct are skipped).
# Pass 2 compares SHA256 remote vs local for every file except those still being written (ACTIVE regex) and the
# per-minute heartbeat/supervisor files; hash mismatches are re-fetched once and re-checked.
# Prints "VERIFIED files=N missing=0 hash_mismatch=0" only when both are zero; otherwise "NOT VERIFIED ...".
#   HOST=root@IP PORT=62992 DEST=/d/Workspace/DMRL_artifacts/run_56353913/full [ACTIVE='runs/pilot1/bench_val/'] \
#     bash scripts/pull_full.sh
set -uo pipefail
: "${HOST:?}" "${PORT:?}" "${DEST:?}"
STREAMS="${STREAMS:-4}"
ACTIVE="${ACTIVE:-^$}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$PORT" "$HOST")
W="$DEST/.pull"
mkdir -p "$W"
VOLATILE='(heartbeat\.(json|jsonl|log)|supervisor\.log|run_project_[0-9TZ]+\.out|project_state\.json|budget\.json|box_repo_state\.txt|hf_inventory\.txt|ALERTS\.jsonl|watchdog/.*)$'
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$DEST/pull.log"; }

remote_list() {
  "${SSH[@]}" 'cd /workspace/DMRL && { echo "## commit"; git log -1 --format="%H %s"; echo "## branch"; git rev-parse --abbrev-ref HEAD
      echo "## submodules"; git submodule status; echo "## status"; git status --porcelain=v1
      echo "## submodule third_party/psp status"; git -C third_party/psp status --porcelain=v1 | head -200
      echo "## diff"; git diff; } > /workspace/apsp/box_repo_state.txt 2>&1
    cd /workspace/apsp && { echo "## files"; find hf -type f ! -name "*token*" -printf "%s\t%p\n"
      echo "## refs"; find hf -path "*/refs/*" -type f ! -name "*token*" | while read -r f; do echo "$f $(cat "$f")"; done; } > hf_inventory.txt 2>&1
    cd /workspace && find apsp xfer \( -path "apsp/venv-*" -o -path apsp/hf \) -prune -o -type f -printf "%s\t%p\n"
    cd /workspace/DMRL && git ls-files -o --exclude-standard | while IFS= read -r f; do printf "%s\tDMRL/%s\n" "$(stat -c %s "$f")" "$f"; done' \
    2>/dev/null | LC_ALL=C sort > "$W/remote.tsv"
  [[ -s "$W/remote.tsv" ]]
}
local_list() { (cd "$DEST" && find apsp xfer DMRL -type f -printf "%s\t%p\n" 2>/dev/null) | LC_ALL=C sort > "$W/local.tsv"; }
fetch() {   # $1 = file with one relative path per line
  local n; n=$(wc -l < "$1"); (( n == 0 )) && return 0
  rm -f "$W"/chunk.*
  awk -v n="$STREAMS" -v w="$W" '{print > (w "/chunk." (NR % n))}' "$1"
  for c in "$W"/chunk.*; do
    ( "${SSH[@]}" 'cd /workspace && tar cf - --ignore-failed-read -T - 2>/dev/null' < "$c" 2>/dev/null | tar xf - -C "$DEST" 2>/dev/null ) &
  done
  wait
}

# ---- pass 1: size-based resume
remote_list || { log "remote list failed"; exit 2; }
local_list
LC_ALL=C comm -23 "$W/remote.tsv" "$W/local.tsv" | sort -t$'\t' -k1,1nr | cut -f2 > "$W/need.txt"
log "pass1: fetching $(wc -l < "$W/need.txt") files, $(LC_ALL=C comm -23 "$W/remote.tsv" "$W/local.tsv" | awk -F'\t' '{s+=$1} END {printf "%.0f", s/1e6}') MB"
fetch "$W/need.txt"

# ---- pass 2: SHA256 remote vs local (finished files only)
remote_list || { log "remote list failed"; exit 2; }
cut -f2 "$W/remote.tsv" | grep -Ev "$VOLATILE" | grep -Ev "$ACTIVE" > "$W/check.txt"
"${SSH[@]}" 'cd /workspace && tr "\n" "\0" | xargs -0 -r sha256sum' < "$W/check.txt" 2>/dev/null | LC_ALL=C sort -k2 > "$W/remote.sha"
local_sha() {   # Git-for-Windows sha256sum marks binary mode with "*" before the path: drop it
  (cd "$DEST" && tr '\n' '\0' < "$W/check.txt" | xargs -0 -r sha256sum 2>/dev/null) | sed 's/^\([0-9a-f]*\) \*/\1  /' \
    | LC_ALL=C sort -k2 > "$W/local.sha"
}
local_sha
compare() {
  awk 'NR==FNR {r[$2]=$1; next} {l[$2]=$1} END {for (p in r) { if (!(p in l)) print "MISSING\t" p; else if (l[p]!=r[p]) print "MISMATCH\t" p }}' \
    "$W/remote.sha" "$W/local.sha" > "$W/diff.tsv"
}
compare
if grep -q MISMATCH "$W/diff.tsv" || grep -q MISSING "$W/diff.tsv"; then
  cut -f2 "$W/diff.tsv" > "$W/refetch.txt"
  log "pass2: re-fetching $(wc -l < "$W/refetch.txt") files with missing/different hash"
  fetch "$W/refetch.txt"
  local_sha
  compare
fi
nr=$(wc -l < "$W/remote.sha"); miss=$(grep -c ^MISSING "$W/diff.tsv"); bad=$(grep -c ^MISMATCH "$W/diff.tsv")
nchk=$(wc -l < "$W/check.txt"); nact=$(cut -f2 "$W/remote.tsv" | grep -Ec "$ACTIVE")
cp "$W/remote.sha" "$DEST/SHA256SUMS.remote.txt"
mb=$(awk -F'\t' '{s+=$1} END {printf "%.0f", s/1e6}' "$W/remote.tsv")
if (( miss == 0 && bad == 0 && nr == nchk )); then
  log "VERIFIED files=$nr missing=0 hash_mismatch=0 total_MB=$mb (excluded as still being written: $nact)"
else
  log "NOT VERIFIED hashed=$nr/$nchk missing=$miss hash_mismatch=$bad (list: $W/diff.tsv; still being written: $nact)"
  exit 1
fi
