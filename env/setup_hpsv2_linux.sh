#!/usr/bin/env bash
# Separate evaluation venv for HPS v2.1 (hpsv2 1.2.0 pins protobuf<4 and pytest==7.2.0, which conflict with the
# generator env). Evaluation-only: never imported by env/features/policies. Pins: configs/evaluators.yaml.
#   VENV=/workspace/apsp/venv-hps bash env/setup_hpsv2_linux.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
VENV="${VENV:-.venv-hps}"
WHEEL_URL="https://files.pythonhosted.org/packages/46/f3/4ff37b7e5c5079bdc2880c9a72300c5b962e62ac0ebdcda2af358c3ddc4f/hpsv2-1.2.0-py3-none-any.whl"
WHEEL_SHA=3ee638b06847bef24e1de43d0b5e3890a4a850034b4bf618e4bb3e78e2d806a1
BPE_URL="https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_SHA=924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a
[[ -x "${VENV}/bin/python" ]] || uv venv --python 3.10 "${VENV}"
PY="${VENV}/bin/python"
TMP=$(mktemp -d)
curl -sSfL -o "${TMP}/hpsv2-1.2.0-py3-none-any.whl" "${WHEEL_URL}"
echo "${WHEEL_SHA}  ${TMP}/hpsv2-1.2.0-py3-none-any.whl" | sha256sum -c -
uv pip install --python "${PY}" "setuptools==69.5.1" wheel
uv pip install --python "${PY}" torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "${PY}" "${TMP}/hpsv2-1.2.0-py3-none-any.whl" pyyaml pandas pyarrow
uv pip install --python "${PY}" --no-deps -e .
DEST=$("${PY}" -c "import hpsv2.src.open_clip.tokenizer as t; print(t.default_bpe())")
[[ -f "${DEST}" ]] || curl -sSfL -o "${DEST}" "${BPE_URL}"
echo "${BPE_SHA}  ${DEST}" | sha256sum -c -
uv pip freeze --python "${PY}" > env/env.lock.linux-hps.txt
echo "hpsv2 venv ready: ${VENV} (lock: env/env.lock.linux-hps.txt)"
