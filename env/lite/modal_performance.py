"""Bounded CPU preflight then a single B200 benchmark on the authorized profile."""
import os,subprocess,json,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/apsp')
if modal.is_local():
    sys.path.insert(0,str(ROOT/'env/lite'));import modal_runtime
    image=modal_runtime.source_image
else:image=modal.Image.debian_slim(python_version='3.11')
app=modal.App('apsp-lite-performance-v1');volume=modal.Volume.from_name('apsp-lite-runtime-v1')


def env():return {**os.environ,'PYTHONPATH':'/opt/apsp','HF_HOME':'/persist/lite2-model-addon-v1/restored/hf',
    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'}


@app.function(image=image,cpu=2,memory=8192,timeout=180,volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def preflight():
    assert Path(env()['HF_HOME']).is_dir()
    result=subprocess.run(['/opt/apsp-runtime/bin/python','/opt/apsp/env/lite/check_performance_cpu.py'],env=env(),capture_output=True,text=True)
    print(result.stdout,flush=True)
    if result.returncode:
        print(result.stderr[-2000:]);raise RuntimeError('CPU regression failed; no GPU launch')
    code="import sys,tempfile,pathlib,zipfile;sys.path.insert(0,'/opt/apsp/env/lite');from benchmark_performance import save_source_snapshot; d=tempfile.TemporaryDirectory();p=pathlib.Path(d.name)/'source.zip';save_source_snapshot(p);assert zipfile.ZipFile(p).testzip() is None;print('SOURCE_ZIP_CPU_PASS')"
    subprocess.run(['/opt/apsp-runtime/bin/python','-c',code],env=env(),check=True)
    resume=Path('/persist/performance-qualified/1792615748521785429/summary.json')
    if resume.is_file():
        code="import sys;sys.path.insert(0,'/opt/apsp/env/lite');from qualify_throughput import cpu_resume_check;cpu_resume_check('/persist/performance-qualified/1792615748521785429')"
        subprocess.run(['/opt/apsp-runtime/bin/python','-c',code],env=env(),check=True)


@app.function(image=image,gpu='B200',cpu=6,memory=65536,timeout=650,retries=0,
              volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def benchmark():
    import time
    out='/persist/performance-v1/'+str(time.time_ns())
    print('PERFORMANCE_OUTPUT='+out,flush=True)
    child=subprocess.Popen(['/opt/apsp-runtime/bin/python','/opt/apsp/env/lite/benchmark_performance.py','--out',out,'--round','final'],
        env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in child.stdout:print(line.rstrip(),flush=True)
        if child.wait()!=0:raise RuntimeError('B200 benchmark failed; see first exception above')
    finally:volume.commit()


@app.local_entrypoint()
def run():preflight.remote();benchmark.remote()


@app.function(image=image,gpu='B200',cpu=6,memory=65536,timeout=520,retries=0,
              volumes={'/persist':volume},max_containers=1,scaledown_window=2)
def confirm():
    import time
    out='/persist/performance-qualified/1792615748521785429'
    print('THROUGHPUT_OUTPUT='+out,flush=True)
    child=subprocess.Popen(['/opt/apsp-runtime/bin/python','/opt/apsp/env/lite/qualify_throughput.py','--out',out,
        '--previous','/persist/performance-v1/1792615186911983410/summary.json','--resume'],env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in child.stdout:print(line.rstrip(),flush=True)
        if child.wait()!=0:raise RuntimeError('20-task throughput qualification failed; receipts retained')
    finally:volume.commit()


@app.local_entrypoint()
def run_confirm():preflight.remote();confirm.remote()
