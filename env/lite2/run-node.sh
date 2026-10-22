#!/usr/bin/env bash
set -euo pipefail
set +x
: "${APSP_SOURCE:?}" "${APSP_PYTHON:?}" "${HF_HOME:?}"
test -x "$APSP_PYTHON"
export PYTHONPATH="$APSP_SOURCE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$APSP_PYTHON" "$APSP_SOURCE/env/lite2/run_node.py" "$@"
