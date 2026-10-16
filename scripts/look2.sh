#!/usr/bin/env bash
# Look 2 of the confirmatory val protocol (PREREG-B snapshot): runs only if look 1 left a primary contrast INC.
#   APSP_STORE=... RUN_ID=pilot1 COSTS=... FROZEN=... bash scripts/look2.sh run|report
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="${APSP_STORE}/runs/${RUN_ID}"
PY="${APSP_STORE}/venv-gpu/bin/python"
L1="${RUN}/confirm_look1.json"
ST="${RUN}/look2_status.json"
N=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['look2_val_ext_prompts'] if d['need_look2'] else 0)" "$L1") || exit 2
case "${1:-}" in
  run)
    if [[ "$N" -eq 0 ]]; then
      echo '{"look2": "not needed (no primary contrast INC after look 1)"}' > "$ST"; exit 0
    fi
    VAL_EXT_N="$N" bash scripts/gpu_smoke.sh 44b_bench_val_ext || exit 1
    echo "{\"look2\": \"run\", \"val_ext_prompts\": $N}" > "$ST" ;;
  report)
    if [[ "$N" -eq 0 ]]; then cp "$L1" "${RUN}/confirm_final.json"; exit 0; fi
    "$PY" -m apsp.reports.confirm --runs "${RUN}/bench_val" "${RUN}/bench_val_ext" --frozen "${FROZEN}" \
      --config configs/evaluation_locked.yaml --look 2 --val-log "${RUN}/val_access.jsonl" \
      --out "${RUN}/confirm_look2.json" && cp "${RUN}/confirm_look2.json" "${RUN}/confirm_final.json" ;;
  *) echo "usage: look2.sh run|report" >&2; exit 2 ;;
esac
