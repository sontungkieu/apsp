#!/usr/bin/env bash
# CPU-only analysis of the operator-supplied export; no runtime activation/install.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=/workspace/storage-shared/nlp/tungks/apsp_all/apsp-lite2-fast2/lite2-r1
COLLECTION="${APSP_EDA_COLLECTION:-$RUN/release}"
EXPORT="${APSP_EDA_EXPORT:-$COLLECTION/exports/1792706875990190337}"
MANIFESTS="${APSP_EDA_MANIFESTS:-$COLLECTION/manifests}"
BACKUP="${APSP_EDA_BACKUP:-$RUN/backup}"
OUT="${APSP_EDA_OUT:-/workspace/storage-shared/nlp/tungks/apsp_all/eda-v2/$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${APSP_EDA_PYTHON:-/opt/apsp-runtime/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
flags=()
if [[ "${APSP_EDA_FULL_HASH:-1}" == 1 ]]; then flags+=(--full-hash); fi
if [[ "${APSP_EDA_BACKUP_FULL_HASH:-0}" == 1 ]]; then flags+=(--backup-full-hash); fi
exec "$PYTHON" "$SCRIPT_DIR/eda_v2.py" \
  --export "$EXPORT" --collection "$COLLECTION" --manifests "$MANIFESTS" \
  --backup "$BACKUP" --out "$OUT" --native-values "${APSP_EDA_NATIVE_VALUES:-4}" \
  "${flags[@]}" "$@"
