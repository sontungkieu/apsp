"""Use the existing TTS publish pattern: ephemeral token, CPU upload, Hub readback."""
import json,os,shlex,subprocess
from pathlib import Path
import modal
import sys
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/apsp')
if modal.is_local():
    sys.path.insert(0,str(ROOT/'env/lite'))
    import modal_runtime
    image=modal_runtime.source_image
else:image=modal.Image.debian_slim(python_version='3.11')
app=modal.App('apsp-lite-hf-publication-v1')
volume=modal.Volume.from_name('apsp-lite-runtime-v1')


@app.function(image=image,cpu=2,memory=8192,timeout=5400,retries=0,volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def upload():
    env={**os.environ,'PYTHONPATH':'/opt/apsp','HF_HOME':'/persist/hf-publication-cache','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_VERBOSITY':'error'}
    child=subprocess.Popen(['/opt/apsp-runtime/bin/python','/opt/apsp/env/lite/publish_companion.py'],env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    try:
        for line in child.stdout:
            if line.startswith('HF_PUBLIC_STAGE='):print(line.rstrip(),flush=True)
        if child.wait()!=0:raise RuntimeError('publication failed; sanitized stage marker above')
    finally:volume.commit()
    return Path('/persist/export/hf-publication-receipt.json').read_text()


@app.local_entrypoint()
def publish():
    # Same established WSL project-local source as TTS, field-only read.
    source=Path('/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env')
    token=None
    for line in source.read_text().splitlines():
        key,sep,value=line.removeprefix('export ').partition('=')
        if sep and key.strip()=='HF_TOKEN':token=shlex.split(value)[0];break
    if not token:raise RuntimeError('HF_TOKEN field missing')
    result=upload.with_options(secrets=[modal.Secret.from_dict({'HF_TOKEN':token})]).remote()
    print(result)


@app.function(image=image,cpu=1,memory=2048,timeout=300,retries=0,volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def update_download():
    env={**os.environ,'HF_HOME':'/persist/hf-publication-cache','HF_HUB_DISABLE_PROGRESS_BARS':'1'}
    child=subprocess.run(['/opt/apsp-runtime/bin/python','/opt/apsp/env/lite/update_hf_download.py'],env=env,capture_output=True,text=True)
    for line in child.stdout.splitlines():
        if line.startswith('HF_PUBLIC_STAGE='):print(line,flush=True)
    volume.commit()
    if child.returncode:raise RuntimeError('small helper update failed; no large upload attempted')


@app.local_entrypoint()
def fix_download():
    source=Path('/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env')
    token=None
    for line in source.read_text().splitlines():
        key,sep,value=line.removeprefix('export ').partition('=')
        if sep and key.strip()=='HF_TOKEN':token=shlex.split(value)[0];break
    if not token:raise RuntimeError('HF_TOKEN field missing')
    update_download.with_options(secrets=[modal.Secret.from_dict({'HF_TOKEN':token})]).remote()
