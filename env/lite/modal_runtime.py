"""Remote build/CPU validation, asset staging, B200 qualification and portable export.

Run with a pre-authorized Modal profile and cost guard. No local dependency install.
Image dependency inputs are isolated from the live, filtered source overlay.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/apsp")
APP_NAME = "apsp-lite-b200-runtime-v1"
BASE = "/opt/apsp-python/cpython-3.11.6-linux-x86_64-gnu"
VPY = "/opt/apsp-runtime/bin/python"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("apsp-lite-runtime-v1", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ca-certificates", "libgl1", "libglib2.0-0", "libgomp1", "build-essential")
    .pip_install("uv==0.11.17")
    .env({"UV_PYTHON_INSTALL_DIR": "/opt/apsp-python", "UV_PROJECT_ENVIRONMENT": "/opt/apsp-runtime",
          "LANG": "C.UTF-8", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(ROOT / "pyproject.toml"), "/opt/apsp/pyproject.toml", copy=True)
    .add_local_file(str(ROOT / "env/lite/pyproject.toml"), "/opt/apsp/env/lite/pyproject.toml", copy=True)
    .add_local_file(str(ROOT / "env/lite/uv.lock"), "/opt/apsp/env/lite/uv.lock", copy=True)
    .run_commands("uv python install 3.11.6",
                  f"cd /opt/apsp/env/lite && uv sync --locked --default-index https://pypi.org/simple --no-install-package adaptive-psp --python {BASE}/bin/python3.11 --verbose",
                  "uv pip check --python /opt/apsp-runtime/bin/python")
)
source_image = image.add_local_dir(str(ROOT), "/opt/apsp", copy=False,
    ignore=modal.FilePatternMatcher(".git", ".venv", "data", "runs", "outputs", "remote_artifacts", "__pycache__", ".pytest_cache", "*.pyc"))


def run(args, timeout=1200):
    env = {**os.environ, "PYTHONPATH": "/opt/apsp", "HF_HOME": "/persist/hf",
           "PYTHONNOUSERSITE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    subprocess.run([VPY, *args], cwd="/opt/apsp", env=env, check=True, timeout=timeout)


@app.function(image=source_image, cpu=2, memory=16384, timeout=1800, volumes={"/persist": volume},
              max_containers=1, scaledown_window=2)
def cpu():
    code = """import importlib,importlib.metadata as m,json,pathlib,torch
from apsp.lite.compat import imagereward_import_bridge
bridge=imagereward_import_bridge()
names=['torch','torchvision','diffusers','transformers','accelerate','peft','ImageReward','clip','safetensors','pandas','pyarrow']
for name in names: importlib.import_module(name)
from diffusers import StableDiffusionXLPipeline,FluxPipeline
from ImageReward.models.BLIP.med import BertConfig,BertModel
cfg=BertConfig(vocab_size=128,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64)
cfg.encoder_width=32
cfg.add_cross_attention=True
tiny=BertModel(cfg).eval()
with torch.inference_mode():
    check=tiny(input_ids=torch.zeros((1,4),dtype=torch.long),attention_mask=torch.ones((1,4)),encoder_hidden_states=torch.zeros((1,4,32)),encoder_attention_mask=torch.ones((1,4)),return_dict=True)
assert torch.isfinite(check.last_hidden_state).all()
from diffusers import FlowMatchEulerDiscreteScheduler,EulerDiscreteScheduler
from apsp.lite.runtime import CleanTap
for cls in [FlowMatchEulerDiscreteScheduler,EulerDiscreteScheduler]:
    scheduler=cls()
    scheduler.set_timesteps(8)
    tap=CleanTap(scheduler,[1])
    x=torch.ones((1,4,8,8))
    t=scheduler.timesteps[0]
    v=torch.full_like(x,0.25)
    if cls is EulerDiscreteScheduler: scheduler.scale_model_input(x,t)
    sigma=scheduler.sigmas[0].clone()
    scheduler.step(v,t,x)
    assert torch.allclose(tap.x0[1],x-sigma*v)
from apsp.lite.contracts import load_registry,file_hash
registry=load_registry()
report={'pass':True,'bert_forward_finite':True,'clean_prediction_formula_checked':True,'compatibility_bridge':bridge,'python':__import__('sys').version,'torch_cuda':torch.version.cuda,
        'versions':{n:m.version(n) for n in ['torch','torchvision','diffusers','transformers','peft','accelerate']},
        'lock_sha256':file_hash('/opt/apsp/env/lite/uv.lock'),'stage':'cpu_imports',
        'qualification':'imports_only_not_model_or_gpu_qualification'}
pathlib.Path('/persist/receipts').mkdir(exist_ok=True)
pathlib.Path('/persist/receipts/cpu.json').write_text(json.dumps(report,indent=2))
print('APSP_CPU_JSON='+json.dumps(report))
"""
    run(["-c", code])
    volume.commit()
    return Path("/persist/receipts/cpu.json").read_text()


@app.function(image=source_image, cpu=2, memory=32768, timeout=1800, volumes={"/persist": volume},
              max_containers=1, scaledown_window=2)
def assets():
    try:
        run(["-m", "apsp.lite.assets", "--out", "/persist/assets.json"], timeout=1700)
    finally:
        volume.commit()
    return Path("/persist/assets.json").read_text()


@app.function(image=source_image, gpu="B200", cpu=4, memory=98304, timeout=1500,
              volumes={"/persist": volume}, max_containers=1, scaledown_window=2)
def qualify():
    # Assets are staged on CPU first; GPU does not pay to download model weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        run(["-m", "apsp.lite.qualify", "--out", "/persist/qualification"], timeout=1400)
    finally:
        volume.commit()
    return Path("/persist/qualification/summary.json").read_text()


@app.function(image=source_image, cpu=2, memory=8192, timeout=900, volumes={"/persist": volume},
              max_containers=1, scaledown_window=2)
def package():
    import tarfile
    import hashlib
    cpu_receipt = json.loads(Path("/persist/receipts/cpu.json").read_text())
    summary = json.loads(Path("/persist/qualification/summary.json").read_text())
    if not cpu_receipt.get("pass") or not summary.get("pass"):
        raise RuntimeError("portable export requires CPU and all B200 qualification gates")
    dest = Path("/persist/export")
    dest.mkdir(exist_ok=True)
    archive = dest / "runtime.tar.gz"
    with tarfile.open(archive.with_suffix(".tmp"), "w:gz", compresslevel=1) as tf:
        for path in ("/opt/apsp-runtime", "/opt/apsp-python"):
            tf.add(path, arcname=path.lstrip("/"))
    archive.with_suffix(".tmp").replace(archive)
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""): h.update(chunk)
    manifest = {"archive_sha256": h.hexdigest(), "archive_bytes": archive.stat().st_size,
                "prefixes": ["/opt/apsp-runtime", "/opt/apsp-python"], "cpu": cpu_receipt,
                "qualification": summary, "weights_included": False, "source_included": False}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return json.dumps(manifest)


@app.local_entrypoint()
def prepare_assets():
    # Use only the known HF_TOKEN field of the established project-local credential source.
    # No whole-file environment load and no persistent Modal Secret object.
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        source = Path("/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env")
        if source.is_file():
            for line in source.read_text().splitlines():
                key, sep, value = line.removeprefix("export ").partition("=")
                if sep and key.strip() == "HF_TOKEN":
                    token = value.strip().strip("\"'")
                    break
    secret = modal.Secret.from_dict({"HF_TOKEN": token}) if token else None
    result = assets.with_options(secrets=[secret]).remote() if secret else assets.remote()
    print(result)
