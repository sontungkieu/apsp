"""Authorized CPU-only delta model packaging using the existing qualified runtime."""
import json,os,shlex,subprocess,sys
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/apsp')
if modal.is_local():
    sys.path.insert(0,str(ROOT/'env/lite'));import modal_runtime
    image=(modal_runtime.image
        .add_local_file(str(ROOT/'env/lite/lite2_addon.py'),'/opt/tools/lite2_addon.py')
        .add_local_file(str(ROOT/'env/lite/build_lite2_addon.py'),'/opt/tools/build_lite2_addon.py')
        .add_local_file(str(ROOT/'reports/lite/COMPANION_CONTENT_MANIFEST.json'),'/opt/tools/companion-manifest.json'))
else:image=modal.Image.debian_slim(python_version='3.11')
app=modal.App('apsp-lite2-model-addon-v1')
volume=modal.Volume.from_name('apsp-lite-runtime-v1')


@app.function(image=image,cpu=2,memory=16384,timeout=1800,retries=0,max_containers=1,
              scaledown_window=2,volumes={'/persist':volume})
def package():
    env={**os.environ,'HF_HOME':'/persist/lite2-model-download-cache','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
    try:
        for extra in [[],['--publish']]:
            child=subprocess.Popen(['/opt/apsp-runtime/bin/python','/opt/tools/build_lite2_addon.py',*extra],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            for line in child.stdout:
                if line.startswith('LITE2_ADDON_JSON='):print(line.rstrip(),flush=True)
            if child.wait()!=0:raise RuntimeError('Addon stage failed; sanitized marker above')
            volume.commit()
    finally:volume.commit()


@app.local_entrypoint()
def run():
    token=None
    for line in Path('/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env').read_text().splitlines():
        k,s,v=line.removeprefix('export ').partition('=')
        if s and k.strip()=='HF_TOKEN':token=shlex.split(v)[0];break
    if not token:raise ValueError('HF_TOKEN field missing')
    package.with_options(secrets=[modal.Secret.from_dict({'HF_TOKEN':token})]).remote()
