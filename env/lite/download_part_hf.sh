#!/usr/bin/env bash
# Adapted from DSH research_vai/PROXY_DOWNLOAD_GUIDE.md section6 and DPCA0b.2.
# Primary company-node path: curl + explicit proxy + pinned HF revision; no SDK.
# Invoke with bash (not source). All exits remain inside the subshell below.
set +x
(
set -euo pipefail
MANIFEST=${1:?parts manifest required}
PART=${2:?part 1 or 2 required}
DEST=${3:?persistent destination required}
HF_REPO=${4:?HF dataset owner/name required}
REV=${5:?pinned HF commit required}
N=${APSP_DOWNLOAD_THREADS:-12}
[[ "$PART" =~ ^[12]$ && "$N" =~ ^[1-9][0-9]?$ && "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'TU_CHOI: part/threads/revision khong hop le'; exit 2; }
PROXY=${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}
[ -n "$PROXY" ] || { echo 'TU_CHOI: THIEU proxy; nap env theo PROXY_DOWNLOAD_NOTE.md'; exit 2; }
command -v curl >/dev/null; command -v python3 >/dev/null
# Only shell-quoted manifest fields are evaluated. No hardcoded asset byte count.
eval "$(python3 - "$MANIFEST" "$PART" <<'PY'
import json,re,shlex,sys
p=json.load(open(sys.argv[1]))['parts'][int(sys.argv[2])-1]
assert re.fullmatch(r'[A-Za-z0-9_.-]+',p['file']) and re.fullmatch(r'[0-9a-f]{64}',p['sha256'])
assert isinstance(p['bytes'],int) and p['bytes']>0
for k,v in [('F',p['file']),('WANT',str(p['bytes'])),('WANT_SHA',p['sha256'])]: print(k+'='+shlex.quote(v))
PY
)"
mkdir -p "$DEST"; cd "$DEST"
if [ -e "$F" ]; then
  [ "$(stat -c%s "$F")" = "$WANT" ] && printf '%s  %s\n' "$WANT_SHA" "$F" | sha256sum -c - && { echo 'DA_CO_DUNG_SHA'; exit 0; }
  echo 'TU_CHOI: file dich da co nhung khong dung size/SHA'; exit 2
fi
LOCK="$F.download.lock"
mkdir "$LOCK" 2>/dev/null || { echo 'TU_CHOI: download lock da co; khong tu cuop lock'; exit 2; }
KEEP_LOCK=0
trap 'if [ "$KEEP_LOCK" = 0 ]; then rmdir "$LOCK" 2>/dev/null || true; fi' EXIT
trap 'KEEP_LOCK=1; echo "BI_NGAT: giu lock de operator kiem tra worker"; exit 130' INT TERM
RANGES="$F.ranges"; mkdir -p "$RANGES"
CONTRACT="$F $WANT $WANT_SHA $N"
if [ -e "$RANGES/contract" ]; then
  [ "$(cat "$RANGES/contract")" = "$CONTRACT" ] || { echo 'TU_CHOI: range contract khac'; exit 2; }
else printf '%s\n' "$CONTRACT" > "$RANGES/contract"; fi
AUTH=()
if [ -n "${HF_TOKEN_FILE:-}" ]; then
  HF_TOKEN=$(grep '^HF_TOKEN=' "$HF_TOKEN_FILE" | cut -d= -f2-)
  [ -n "$HF_TOKEN" ] || { echo 'TU_CHOI: HF token field rong'; exit 2; }
  AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi
URL="https://huggingface.co/datasets/$HF_REPO/resolve/$REV/$F"
echo "PART=$PART FILE=$F WANT=$WANT N=$N HF_REPO=$HF_REPO REV=$REV"
echo 'proxy: co bien (khong in gia tri)'
# A server that ignores Range must fail before any multi-GB request.
PROBE="$RANGES/probe.$BASHPID"
if HTTP=$(curl --silent --fail --location --connect-timeout 20 --max-filesize 1048576 \
  --proxy "$PROXY" --noproxy '' "${AUTH[@]}" -H 'Range: bytes=0-0' \
  "$URL" -o "$PROBE" -w '%{http_code}' 2>/dev/null); then RC=0; else RC=$?; fi
[ "$RC" = 0 ] && [ "$HTTP" = 206 ] && [ "$(stat -c%s "$PROBE")" = 1 ] || { echo "TU_CHOI: Range probe HTTP=$HTTP curl_rc=$RC"; exit 2; }
rm -- "$PROBE"
CHUNK=$(( (WANT + N - 1) / N ))
download_range() (
  trap - EXIT INT TERM
  I=$1; S=$((I*CHUNK)); E=$(((I+1)*CHUNK-1)); [ "$E" -lt "$WANT" ] || E=$((WANT-1))
  NEED=$((E-S+1)); FILE=$(printf '%s/range.%02d' "$RANGES" "$I")
  HAVE=0; [ ! -f "$FILE" ] || HAVE=$(stat -c%s "$FILE")
  [ "$HAVE" -le "$NEED" ] || { echo "SAI_SIZE range=$I"; exit 2; }
  echo "RANGE=$I start=$S end=$E have=$HAVE expected=$NEED"
  for ATTEMPT in $(seq 1 8); do
    [ "$HAVE" -lt "$NEED" ] || { echo "RANGE_OK=$I bytes=$HAVE"; exit 0; }
    START=$((S+HAVE)); LEFT=$((NEED-HAVE)); TMP="$FILE.attempt.$BASHPID.$ATTEMPT"
    LIMIT=$LEFT; [ "$LIMIT" -ge 1048576 ] || LIMIT=1048576
    # Deliberately no --retry and no curl >>. Append only a bounded 206 response.
    if HTTP=$(curl --silent --fail --location --connect-timeout 20 --max-filesize "$LIMIT" \
      --proxy "$PROXY" --noproxy '' "${AUTH[@]}" -H "Range: bytes=$START-$E" \
      "$URL" -o "$TMP" -w '%{http_code}' 2>/dev/null); then RC=0; else RC=$?; fi
    GOT=0; [ ! -f "$TMP" ] || GOT=$(stat -c%s "$TMP")
    if [ "$HTTP" = 206 ] && [ "$GOT" -gt 0 ] && [ "$GOT" -le "$LEFT" ]; then
      cat "$TMP" >> "$FILE"; HAVE=$(stat -c%s "$FILE")
      rm -- "$TMP"
    fi
    echo "RANGE=$I attempt=$ATTEMPT have=$HAVE/$NEED HTTP=$HTTP curl_rc=$RC"
    [ "$HAVE" -eq "$NEED" ] || sleep 2
  done
  [ "$HAVE" -eq "$NEED" ] || { echo "RANGE_CHUA_XONG=$I; rerun de resume"; exit 1; }
)
PIDS=()
for I in $(seq 0 $((N-1))); do download_range "$I" & PIDS+=("$!"); done
BAD=0
for PID in "${PIDS[@]}"; do if wait "$PID"; then :; else BAD=1; fi; done
[ "$BAD" = 0 ] || { echo 'CHUA_XONG: giu ranges/log, khong ghep'; exit 1; }
JOINED="$F.joined.$BASHPID.tmp"; : > "$JOINED"
for I in $(seq 0 $((N-1))); do
  FILE=$(printf '%s/range.%02d' "$RANGES" "$I")
  S=$((I*CHUNK)); NEED=$CHUNK; [ "$I" -lt $((N-1)) ] || NEED=$((WANT-S))
  [ "$(stat -c%s "$FILE")" = "$NEED" ] || { echo "SAI_SIZE range=$I"; exit 2; }
  cat "$FILE" >> "$JOINED"
done
[ "$(stat -c%s "$JOINED")" = "$WANT" ]
printf '%s  %s\n' "$WANT_SHA" "$JOINED" | sha256sum -c -
[ ! -e "$F" ] || { echo 'TU_CHOI: dich vua xuat hien'; exit 2; }
mv -- "$JOINED" "$F"
printf '%s  %s\n' "$WANT_SHA" "$F" | sha256sum -c -
unset HF_TOKEN
echo 'PART_DOWNLOAD_PASS'
)
DOWNLOAD_RC=$?
echo '--- het Block APSP-HF12, shell con song ---'
(exit "$DOWNLOAD_RC")
