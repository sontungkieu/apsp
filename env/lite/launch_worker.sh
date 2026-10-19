#!/usr/bin/env bash
# Run in tmux or the company's allocated scheduler. Explicit assigned UUID only.
set -euo pipefail
: "${APSP_GPU_UUID:?set the GPU UUID allocated to this worker}"
: "${APSP_SOURCE:?set the exact Lite checkout path}"
: "${APSP_PYTHON:?set the qualified runtime Python path}"
: "${APSP_PROFILE:?set sdxl_lightning8, hyper_flux8 or sdxl_base50}"
: "${APSP_TASKS:?set the immutable task JSONL path}"
: "${APSP_OUTPUT:?set persistent release root}"
: "${APSP_BACKUP:?set independent backup root}"
: "${APSP_QUALIFICATION:?set the passing profile receipt}"
: "${HF_HOME:?set the persistent pinned asset cache}"
[[ "$APSP_GPU_UUID" == GPU-* ]] || { echo 'Expected an allocated GPU UUID' >&2; exit 2; }
nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$APSP_GPU_UUID"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$APSP_GPU_UUID"; then
  echo 'Assigned GPU is busy; no processes were changed.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$APSP_GPU_UUID" PYTHONPATH="$APSP_SOURCE" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$APSP_SOURCE"
exec "$APSP_PYTHON" -m apsp.lite.collector --tasks "$APSP_TASKS" --profile "$APSP_PROFILE" \
  --out "$APSP_OUTPUT" --backup-root "$APSP_BACKUP" --qualification "$APSP_QUALIFICATION" \
  --workers "${APSP_WORKERS:-1}" --worker-index "${APSP_WORKER_INDEX:-0}" --max-wave "${APSP_MAX_WAVE:-2000}"
