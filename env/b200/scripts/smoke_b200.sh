#!/usr/bin/env bash
# First run on a B200 node: environment + GPU check, then one small inference shard per generator.
#   cd <repo>/env/b200 && bash scripts/smoke_b200.sh /path/to/out [sd15 sdxl sd35m flux_dev]
# Needs: uv sync --locked done; HF_HOME on a large disk; for sd35m / flux_dev an HF account that accepted the model
# licences, logged in on the node (`uv run hf auth login` or HF_TOKEN in the job environment; never commit a token).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:?usage: smoke_b200.sh OUT_DIR [models...]}"; shift
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(sd15 sdxl sd35m flux_dev); fi
mkdir -p "$OUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1
uv sync --locked
uv run --locked python scripts/check_gpu.py --out "$OUT/gpu_check.json" --vram-gb "${VRAM_GB:-40}" | tail -1
cat > "$OUT/smoke_prompts.txt" <<'EOF'
a red bicycle leaning against a brick wall, morning light
a watercolor painting of a fox in a snowy forest
two cups of coffee on a wooden table, top view
a futuristic city skyline at night with neon signs
EOF
for m in "${MODELS[@]}"; do
  echo "== $m"
  uv run --locked python scripts/run_shard.py --model "$m" --prompts "$OUT/smoke_prompts.txt" --k 4 --banks 0 \
    --out "$OUT/shards" --save-preview-png 2>&1 | tee "$OUT/smoke_$m.log"
done
echo "smoke done: $OUT"
