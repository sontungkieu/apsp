"""Reuses the existing locked Modal runtime and cache. No rebuild or model download."""
import os
import subprocess
import sys
import time
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/apsp')
if modal.is_local():
    sys.path.insert(0,str(ROOT/'env/lite'))
    from modal_runtime import source_image as image
else:
    image=modal.Image.debian_slim(python_version='3.11')
app=modal.App('apsp-lite2-native-collector')
volume=modal.Volume.from_name('apsp-lite-runtime-v1')
PY='/opt/apsp-runtime/bin/python'


def env():
    return {**os.environ,'PYTHONPATH':'/opt/apsp','HF_HOME':'/persist/lite2-model-addon-v1/restored/hf',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
            'HF_HUB_DISABLE_PROGRESS_BARS':'1'}


@app.function(image=image,cpu=2,memory=8192,timeout=240,volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def preflight(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    try:
        subprocess.run([PY,'/opt/apsp/env/lite2/check_cpu.py','--out',str(root/'cpu-check')],env=env(),check=True)
        subprocess.run([PY,'-m','apsp.lite2.manifests','--v1-manifests','/persist/text/lite-v1-20261019/manifests',
                        '--out',str(root/'manifests')],env=env(),check=True)
    finally:volume.commit()


@app.function(image=image,gpu='B200',cpu=6,memory=65536,timeout=1200,retries=0,
              volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def qualify(out):
    import zipfile
    root=Path(out)
    with zipfile.ZipFile(root/'measured-source.zip','w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for base in ('apsp','env/lite2','env/lite'):
            for p in (Path('/opt/apsp')/base).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.lock','.sh'):
                    z.write(p,p.relative_to('/opt/apsp').as_posix())
    print('LITE2_OUTPUT='+out,flush=True)
    try:
        child=subprocess.Popen([PY,'-m','apsp.lite2.qualify','--tasks',str(root/'manifests/qualification.jsonl'),
                                '--out',str(root/'qualification')],env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:print(line.rstrip(),flush=True)
        if child.wait()!=0:raise RuntimeError('Lite2 qualification failed; retain first exception and receipts')
        subprocess.run([PY,'/opt/apsp/env/lite2/check_pairs_worker.py','--qualified-root',out],env=env(),check=True)
    finally:volume.commit()


@app.local_entrypoint()
def run():
    out='/persist/lite2-collector-v2/'+str(time.time_ns())
    print('LITE2_OUTPUT='+out,flush=True)
    preflight.remote(out)
    qualify.remote(out)


@app.function(image=image,gpu='B200',cpu=6,memory=65536,timeout=600,retries=0,
              volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def postcheck(out):
    try:
        child=subprocess.Popen([PY,'/opt/apsp/env/lite2/check_pairs_worker.py','--qualified-root',out],
                               env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:print(line.rstrip(),flush=True)
        if child.wait()!=0:raise RuntimeError('Crossover or worker CLI regression failed')
    finally:volume.commit()


@app.local_entrypoint()
def run_postcheck(qualified_root: str):postcheck.remote(qualified_root)
