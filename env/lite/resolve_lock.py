"""Regenerate the Lite lock on remote CPU with the bundle's exact uv version."""
from pathlib import Path
import modal

root = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
app = modal.App("apsp-lite-resolve-lock")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("uv==0.11.17")
    .add_local_file(str(root / "pyproject.toml"), "/work/pyproject.toml", copy=True)
    .add_local_file(str(root / "env/lite/pyproject.toml"), "/work/env/lite/pyproject.toml", copy=True)
    .add_local_file(str(root / "env/lite/uv.lock"), "/work/env/lite/uv.lock", copy=True))

@app.function(image=image, cpu=1, memory=2048, timeout=180, scaledown_window=2)
def resolve():
    import subprocess
    subprocess.run(["uv", "lock", "--python", "/usr/local/bin/python"], cwd="/work/env/lite", check=True, timeout=120)
    return Path("/work/env/lite/uv.lock").read_text()
