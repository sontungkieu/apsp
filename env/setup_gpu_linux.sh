#!/usr/bin/env bash
# Create (or update in place) the Linux GPU environment for Adaptive PSP. Run from anywhere:
#   bash env/setup_gpu_linux.sh            # .venv-gpu with Python 3.10
#   VENV=/workspace/venv-gpu bash env/setup_gpu_linux.sh
# Re-runnable: an existing venv is reused only if it is Python 3.10; otherwise the script stops (it never
# deletes a venv). Any failing step exits non-zero (set -euo pipefail). Scientific pins are never changed here:
# if a pin does not resolve, the script fails and the error must go to docs/QUESTIONS.md.
# CPU-only development does not need this script: `pip install -e ".[dev,rl]"` (see README).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
VENV="${VENV:-.venv-gpu}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
CLIP_PIN="git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
IR_PIN="git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3"
export PYTHONUTF8=1   # ImageReward's setup.py reads its README with the locale codec

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }
trap 'echo "[setup] FAILED at line ${LINENO} (exit $?)" >&2' ERR

command -v git >/dev/null || { echo "git is required" >&2; exit 2; }

if [[ -x "${VENV}/bin/python" ]]; then
  ver="$("${VENV}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [[ "${ver}" != "3.10" ]]; then
    echo "existing ${VENV} is Python ${ver}, expected 3.10; move it away or set VENV=... (not deleting it)" >&2
    exit 2
  fi
  log "reusing existing ${VENV} (Python ${ver})"
else
  if command -v uv >/dev/null 2>&1; then
    log "creating ${VENV} with uv (Python 3.10)"
    uv venv --python 3.10 "${VENV}"
  elif command -v python3.10 >/dev/null 2>&1; then
    log "creating ${VENV} with python3.10 -m venv"
    python3.10 -m venv "${VENV}"
  else
    echo "need uv or python3.10 (install uv: curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    exit 2
  fi
fi
PY="${VENV}/bin/python"

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install --python "${PY}")
  FREEZE=(uv pip freeze --python "${PY}")
else
  "${PY}" -m pip install --upgrade pip wheel
  PIP=("${PY}" -m pip install)
  FREEZE=("${PY}" -m pip freeze)
fi

log "build tooling (setuptools 69.5.1 as in the PSP base, needed for CLIP/ImageReward builds)"
"${PIP[@]}" "setuptools==69.5.1" wheel
log "torch 2.4.0 / torchvision 0.19.0 (CUDA 12.1 wheels)"
"${PIP[@]}" torch==2.4.0 torchvision==0.19.0 --index-url "${TORCH_INDEX}"
log "pinned requirements"
"${PIP[@]}" -r env/requirements-gpu-linux.txt
log "CLIP and ImageReward at pinned commits (no deps, no build isolation)"
"${PIP[@]}" --no-deps --no-build-isolation "${CLIP_PIN}"
"${PIP[@]}" --no-deps --no-build-isolation "${IR_PIN}"
log "this repo (editable, no deps)"
"${PIP[@]}" --no-deps -e .

log "hpsv2 tokenizer asset (same fix as third_party/psp/setup/setup.sh)"
"${PY}" - <<'PY'
import importlib.util, pathlib, urllib.request
spec = importlib.util.find_spec("hpsv2")
if spec and spec.origin:
    p = pathlib.Path(spec.origin).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz", str(p))
PY

log "import + version check (fails if a scientific pin drifted)"
"${PY}" - <<'PY'
import sys, torch, diffusers, transformers, ImageReward, clip, apsp
want = {"torch": "2.4.0", "diffusers": "0.31.0.dev0", "transformers": "4.38.2"}
got = {"torch": torch.__version__.split("+")[0], "diffusers": diffusers.__version__, "transformers": transformers.__version__}
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "|", got)
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
if sys.version_info[:2] != (3, 10) or bad:
    raise SystemExit(f"version mismatch: python={sys.version.split()[0]} {bad}")
PY
"${FREEZE[@]}" > env/env.lock.linux-gpu.txt
log "done. Resolved packages written to env/env.lock.linux-gpu.txt"
