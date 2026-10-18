#!/usr/bin/env bash
# FIRST COMMAND ON A B200 NODE (one GPU is enough): runtime check, then one real shard per generator with every
# checkpoint decoded and scored by both verifiers, then output validation (incl. re-decode of stored latents).
#   cd <repo>/env/b200 && HF_HOME=/shared/hf bash scripts/b200_first_run.sh /scratch/$USER/b200_first [sd15 sdxl sd35m flux_dev]
# Needs: uv >= 0.11; NVIDIA driver >= 570; for sd35m / flux_dev an HF account that accepted the licences, logged in on
# the node (`uv run hf auth login` or HF_TOKEN in the job environment; never commit a token).
# Exit 0 only if the GPU check and every shard + validation pass. Log: OUT/first_run.log, summary: OUT/summary.json.
# When it passes, deploy workers with the same commit and the same uv.lock (uv sync --locked).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="$(mkdir -p "${1:?usage: b200_first_run.sh OUT_DIR [models...]}" && cd "$1" && pwd)"; shift
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(sd15 sdxl sd35m flux_dev); fi
exec > >(tee -a "$OUT/first_run.log") 2>&1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUTF8=1
PY="${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python"
NAMES=(); RCS=()
step() { local n=$1; shift; echo "=== [$n] $(date -u +%FT%TZ): $*"; "$@"; local rc=$?; echo "=== [$n] rc=$rc"; NAMES+=("$n"); RCS+=("$rc"); }

echo "commit $(git rev-parse HEAD 2>/dev/null) host $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
step lock_check uv lock --check
step sync_locked uv sync --locked
step pip_check uv pip check --python "$PY"
step gpu_check "$PY" scripts/check_gpu.py --out "$OUT/gpu_check.json" --vram-gb "${VRAM_GB:-40}" --expect-sm "${EXPECT_SM:-100}"
cat > "$OUT/first_prompts.txt" <<'P'
a red bicycle leaning against a brick wall, morning light
a watercolor painting of a fox in a snowy forest
two cups of coffee on a wooden table, top view
a futuristic city skyline at night with neon signs
P
for m in "${MODELS[@]}"; do
  step "shard_$m" "$PY" scripts/run_shard.py --model "$m" --prompts "$OUT/first_prompts.txt" --k 4 --banks 0 \
       --out "$OUT/shards" --save-preview-png
  step "validate_$m" "$PY" scripts/validate_shard.py "$OUT/shards/$m/shard00000of00001" --require-scores \
       --redecode 1 --device cuda --pixel-tol 2 --out "$OUT/validate_$m.json"
done
fail=0
{ echo "{\"commit\": \"$(git rev-parse HEAD 2>/dev/null)\", \"steps\": {"
  for i in "${!NAMES[@]}"; do [[ ${RCS[$i]} -ne 0 ]] && fail=1; s=","; [[ $i -eq $((${#NAMES[@]}-1)) ]] && s=""; echo "  \"${NAMES[$i]}\": ${RCS[$i]}$s"; done
  echo "}, \"pass\": $([[ $fail -eq 0 ]] && echo true || echo false)}"; } > "$OUT/summary.json"
cat "$OUT/summary.json"
echo "B200 FIRST RUN $([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit $fail
