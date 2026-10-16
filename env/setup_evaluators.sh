#!/usr/bin/env bash
# Evaluator environments, separate from the generator venv (configs/evaluators.yaml). Evaluation only: nothing here
# generates images. Re-runnable; existing venvs are reused.
#   APSP_STORE=/workspace/apsp bash env/setup_evaluators.sh hps|geneval|all
#
# hps      venv-hps: torch 2.4.0+cu121, hpsv2==1.2.0 (wheel sha256 pinned), HPS v2.1 checkpoint + OpenCLIP ViT-H-14 base
#          at the pinned HF revisions, sha256-checked.
# geneval  venv-geneval: official GenEval (djghosh13/geneval) at GENEVAL_COMMIT, mmdet 2.x + mmcv-full 1.7 (needs
#          torch 1.13.1+cu117; sm_86 kernels run on the 4090), Mask2Former Swin-S COCO detector and the CLIP ViT-L/14
#          colour classifier that the official evaluate_images.py loads. Prompt metadata only; no images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${APSP_STORE:?}"
export HF_HOME="${APSP_STORE}/hf" PYTHONUTF8=1
EV="${APSP_STORE}/evaluators"
mkdir -p "$EV"
log() { echo "[eval-setup $(date -u +%H:%M:%S)] $*"; }
cfg() { python3 -c "import yaml,sys; print(yaml.safe_load(open('configs/evaluators.yaml'))[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
GENEVAL_REPO="https://github.com/djghosh13/geneval.git"
GENEVAL_COMMIT="${GENEVAL_COMMIT:-af4902f}"

setup_hps() {
  local V="${APSP_STORE}/venv-hps"
  [[ -x "$V/bin/python" ]] || uv venv --python 3.10 "$V"
  uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0
  "$V/bin/python" -m pip --version >/dev/null 2>&1 || uv pip install --python "$V/bin/python" pip
  "$V/bin/python" -m pip download --no-deps -d "$EV/wheels" hpsv2==1.2.0 -q
  local whl; whl=$(ls "$EV"/wheels/hpsv2-1.2.0*.whl)
  echo "$(cfg hpsv2 package_wheel_sha256)  $whl" | sha256sum -c -
  uv pip install --python "$V/bin/python" "$whl" huggingface_hub pyyaml pandas pillow
  # the 1.2.0 wheel ships without the tokenizer BPE file; fetch the pinned file into the package and verify it
  local bpe; bpe="$("$V/bin/python" -c 'import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), "src/open_clip/bpe_simple_vocab_16e6.txt.gz"))')"
  [[ -s "$bpe" ]] || curl -fsSL "$(cfg hpsv2 tokenizer_bpe_url)" -o "$bpe"
  echo "$(cfg hpsv2 tokenizer_bpe_sha256)  $bpe" | sha256sum -c -
  "$V/bin/python" - <<PY
import hashlib, yaml
from huggingface_hub import hf_hub_download
c = yaml.safe_load(open("configs/evaluators.yaml"))["hpsv2"]
for repo, f, rev, sha in ((c["checkpoint_repo"], c["checkpoint_file"], c["checkpoint_revision"], c["checkpoint_sha256"]),
                          (c["base_repo"], c["base_file"], c["base_revision"], c["base_sha256"])):
    p = hf_hub_download(repo, f, revision=rev)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == sha, (repo, f, h.hexdigest())
    print("ok", repo, f, p)
PY
  log "hps ready: $V"
}

setup_geneval() {
  local V="${APSP_STORE}/venv-geneval" R="$EV/geneval"
  [[ -d "$R/.git" ]] || git clone -q "$GENEVAL_REPO" "$R"
  git -C "$R" fetch -q origin
  git -C "$R" -c advice.detachedHead=false checkout -q "$GENEVAL_COMMIT"
  log "geneval at $(git -C "$R" rev-parse HEAD)"
  [[ -x "$V/bin/python" ]] || uv venv --python 3.10 "$V"
  uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchvision==0.14.1
  uv pip install --python "$V/bin/python" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html \
    mmcv-full==1.7.2 mmdet==2.28.2 "open-clip-torch==2.20.0" clip-benchmark "numpy<1.24" pandas pillow tqdm \
    huggingface_hub pyyaml "setuptools<70"
  mkdir -p "$EV/geneval_models"
  if [[ ! -s "$EV/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]]; then
    bash "$R/evaluation/download_models.sh" "$EV/geneval_models/"
  fi
  sha256sum "$EV"/geneval_models/*.pth | tee "$EV/geneval_models/SHA256SUMS"
  "$V/bin/python" -c "import mmdet, mmcv, open_clip, torch; print('mmdet', mmdet.__version__, 'mmcv', mmcv.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
  log "geneval ready: $V"
}

case "${1:-all}" in
  hps) setup_hps ;;
  geneval) setup_geneval ;;
  all) setup_hps; setup_geneval ;;
  *) echo "usage: $0 hps|geneval|all" >&2; exit 2 ;;
esac
