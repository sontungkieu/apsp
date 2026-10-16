#!/usr/bin/env bash
# GenEval scoring of a geneval benchmark run with the OFFICIAL evaluator (env/setup_evaluators.sh geneval), then the
# PREREG-C report. Evaluation only: runs after delivery, never part of inference cost.
#   APSP_STORE=... RUN=STORE/runs/pilot1/geneval PROMPTS=STORE/geneval/prompts.jsonl SNAP=docs/.../PREREG_C_LOCKED.json \
#     bash scripts/geneval_eval.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?}" "${RUN:?}" "${PROMPTS:?}" "${SNAP:?}"
EV="${APSP_STORE}/evaluators"
VG="${APSP_STORE}/venv-geneval/bin/python"
PY="${APSP_STORE}/venv-gpu/bin/python"
L="${RUN}_layout"
export HF_HOME="${APSP_STORE}/hf"
"$PY" scripts/geneval_tools.py layout --run "$RUN" --prompts "$PROMPTS" --out "$L"
# mmdet installed from the wheel ships its configs under mmdet/.mim/configs (the official default path assumes a
# source checkout next to the package); same config file, explicit path.
CFG=$("$VG" -c "import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), '.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'))")
mkdir -p "$L/results"
for h in $("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))))" "$L/methods.json"); do
  if [[ -s "$L/results/$h.jsonl" ]]; then echo "[skip] $h"; continue; fi
  "$VG" "$EV/geneval/evaluation/evaluate_images.py" "$L/$h" --outfile "$L/results/$h.tmp.jsonl" \
    --model-config "$CFG" --model-path "$EV/geneval_models"
  mv "$L/results/$h.tmp.jsonl" "$L/results/$h.jsonl"
  "$VG" "$EV/geneval/evaluation/summary_scores.py" "$L/results/$h.jsonl" > "$L/results/$h.summary.txt" || true
done
"$PY" scripts/geneval_tools.py report --layout "$L" --results "$L/results" --snapshot "$SNAP" --out "${RUN}/geneval_report.json"
