#!/usr/bin/env bash
# Clean-Linux validation of env/b200 (no GPU needed): install from the lock, consistency, imports, CPU tests.
#   bash env/b200/scripts/linux_cpu_validation.sh LOG_DIR [V1_REF_DIR]
# Runs unattended (network retries included); every step is logged with its exit code; LOG_DIR/summary.json lists
# them; exit code 0 only if every step passed. Model downloads go to $HF_HOME (default LOG_DIR/hf).
# Steps:
#   uv_version, lock_check (uv lock --check), sync_locked (uv sync --locked, retried), pip_check (uv pip check),
#   env_imports (check_gpu.py --allow-cpu: versions + collector imports), latent_roundtrip,
#   tiny_<family> + validate_<family> for SD1.5 / SDXL / SD3 / FLUX tiny random pipelines on CPU (x0 at every
#   checkpoint, VAE decode, safetensors latents, PNG, parquet, re-decode of stored latents == PNG),
#   ir_cpu (real ImageReward), pickscore_cpu (real PickScore), prompt_emb_cpu (real SD1.5 text encoder),
#   e2e_tiny_scored (tiny SD1.5 shard scored by the real IR + PickScore + prompt encoder, validated).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."            # env/b200
LOG="$(mkdir -p "${1:?usage: linux_cpu_validation.sh LOG_DIR [V1_REF_DIR]}" && cd "$1" && pwd)"
REF="${2:-}"
exec > >(tee -a "$LOG/validation.log") 2>&1
export PYTHONUTF8=1 HF_HOME="${HF_HOME:-$LOG/hf}" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python"
REFARG=(); [[ -n "$REF" ]] && REFARG=(--v1-ref "$REF")
NAMES=(); RCS=()
step() {
  local name=$1; shift
  echo "=== [$name] start $(date -u +%FT%TZ): $*"
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== [$name] rc=$rc ($((SECONDS - t0)) s)"
  NAMES+=("$name"); RCS+=("$rc")
  return 0
}
retry_sync() {
  local i
  for i in 1 2 3 4 5 6; do
    uv sync --locked && return 0
    echo "uv sync attempt $i failed; retrying in 60 s"; sleep 60
  done
  return 1
}

echo "env/b200 validation at $(git rev-parse HEAD 2>/dev/null || echo 'no git'), host $(uname -a)"
step uv_version uv --version
step lock_check uv lock --check
step sync_locked retry_sync
step pip_check uv pip check --python "$PY"
step env_imports "$PY" scripts/check_gpu.py --allow-cpu --expect-sm "" --out "$LOG/env_imports.json"
step latent_roundtrip "$PY" scripts/cpu_checks.py roundtrip --out "$LOG/latent_roundtrip.json"

printf '%s\n' "a red bicycle leaning against a brick wall" "a watercolor painting of a fox in the snow" > "$LOG/tiny_prompts.txt"
for fam in sd15 sdxl sd3 flux; do
  step "tiny_$fam" "$PY" scripts/run_shard.py --model "tiny_$fam" --device cpu --prompts "$LOG/tiny_prompts.txt" --k 2 \
       --banks 0 1 --no-ir --no-pickscore --no-prompt-emb --out "$LOG/tiny" --save-preview-png
  step "validate_$fam" "$PY" scripts/validate_shard.py "$LOG/tiny/tiny_$fam/shard00000of00001" --redecode 2 \
       --out "$LOG/validate_tiny_$fam.json"
done

step ir_cpu "$PY" scripts/cpu_checks.py ir --out "$LOG/ir_cpu.json" "${REFARG[@]}"
step pickscore_cpu "$PY" scripts/cpu_checks.py pickscore --out "$LOG/pickscore_cpu.json" "${REFARG[@]}"
step prompt_emb_cpu "$PY" scripts/cpu_checks.py prompt_emb --out "$LOG/prompt_emb_cpu.json" "${REFARG[@]}"
step e2e_tiny_scored "$PY" scripts/run_shard.py --model tiny_sd15 --device cpu --prompts "$LOG/tiny_prompts.txt" --k 2 \
     --banks 0 --out "$LOG/e2e"
step validate_e2e "$PY" scripts/validate_shard.py "$LOG/e2e/tiny_sd15/shard00000of00001" --redecode 1 --require-scores \
     --out "$LOG/validate_e2e.json"

fail=0
{
  echo "{\"commit\": \"$(git rev-parse HEAD 2>/dev/null)\", \"finished\": \"$(date -u +%FT%TZ)\", \"steps\": {"
  for i in "${!NAMES[@]}"; do
    [[ ${RCS[$i]} -ne 0 ]] && fail=1
    sep=","; [[ $i -eq $((${#NAMES[@]} - 1)) ]] && sep=""
    echo "  \"${NAMES[$i]}\": ${RCS[$i]}$sep"
  done
  echo "}, \"pass\": $([[ $fail -eq 0 ]] && echo true || echo false)}"
} > "$LOG/summary.json"
cat "$LOG/summary.json"
echo "VALIDATION $([[ $fail -eq 0 ]] && echo PASS || echo FAIL) $(date -u +%FT%TZ)"
exit $fail
