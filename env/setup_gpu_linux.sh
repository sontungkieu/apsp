#!/usr/bin/env bash
# Create the Linux GPU environment for Adaptive PSP. Run from the repo root.
#   bash env/setup_gpu_linux.sh            # creates .venv-gpu with Python 3.10
# Requires: git, an NVIDIA driver supporting CUDA 12.1 wheels, and either python3.10 or uv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
VENV="${VENV:-.venv-gpu}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.10 "${VENV}"
  PIP="uv pip install --python ${VENV}/bin/python"
  FREEZE="uv pip freeze --python ${VENV}/bin/python"
else
  python3.10 -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip wheel
  PIP="${VENV}/bin/python -m pip install"
  FREEZE="${VENV}/bin/python -m pip freeze"
fi
PY="${VENV}/bin/python"

# ImageReward build compatibility (same as the PSP base setup)
${PIP} "setuptools==69.5.1" wheel
${PIP} torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
${PIP} -r env/requirements-gpu-linux.txt
${PIP} --no-deps --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
${PIP} --no-deps --no-build-isolation "git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3"
${PIP} --no-deps -e .

# hpsv2 tokenizer asset (same fix as third_party/psp/setup/setup.sh)
"${PY}" - <<'PY'
import importlib.util, pathlib, urllib.request
spec = importlib.util.find_spec("hpsv2")
if spec and spec.origin:
    p = pathlib.Path(spec.origin).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz", str(p))
PY

"${PY}" -c "import torch, diffusers, transformers, ImageReward, clip, apsp; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
${FREEZE} > env/env.lock.linux-gpu.txt
echo "[setup] done. Resolved packages written to env/env.lock.linux-gpu.txt"
