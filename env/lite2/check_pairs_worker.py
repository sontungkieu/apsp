"""Crossover latency pairs plus actual worker CLI/resume on the frozen qualified collector."""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import load_file

from apsp.lite.storage import atomic_json,backup_task
from apsp.lite.prompts import read_jsonl
from apsp.lite.runtime import LiteRuntime
from apsp.lite2.contracts import PROFILES,registry,identity,policy,source_hash,file_hash
from apsp.lite2.collector import collect_task
from apsp.lite2.encoder import CoreEncoder
from apsp.lite2.validate import validate_task


def run(qualified_root):
    qualified=Path(qualified_root);out=qualified/'crossover-worker-check'
    out.mkdir(exist_ok=False)
    tasks=read_jsonl(qualified/'manifests/qualification.jsonl')
    result={'pass':True,'source_hash':source_hash(),'profiles':{}}
    for name in PROFILES:
        receipt=json.loads((qualified/'qualification'/(name+'.json')).read_text())
        assert receipt['pass'] and receipt['performance_enabled']
        assert receipt['runtime']['source_hash']==source_hash()
        selected=sorted((t for t in tasks if t['profile']==name and t['bank_id']==0),key=lambda t:t['prompt']['prompt_uid'])
        runtime=LiteRuntime(registry()['profiles'][name],registry(),execution=policy())
        runtime.load_scorers();encoder=CoreEncoder();encoder.encode(selected[0]['prompt']['text'])
        fp,_=identity(policy())
        collect_task(runtime,encoder,selected[0],out/name/'warmup',fp)
        pairs=[]
        for index,t in enumerate(selected[1:3]):
            rows={};paths={}
            labels=['canonical','fast'] if index==0 else ['fast','canonical']
            for label in labels:
                runtime.execution=policy(label=='fast');fp,_=identity(runtime.execution)
                runtime.sync();started=time.perf_counter()
                path=collect_task(runtime,encoder,t,out/name/f'pair{index}'/label,fp)
                backup_task(path,out/name/f'pair{index}'/(label+'-backup')/t['task_id'])
                runtime.sync();wall=time.perf_counter()-started
                rows[label]=wall;paths[label]=path
                assert validate_task(path,check_latents=True,runtime=runtime)['all_valid']
            a,b=[load_file(str(paths[label]/'latents.safetensors')) for label in ('canonical','fast')]
            identical=set(a)==set(b) and all(a[k].dtype==b[k].dtype and torch.equal(a[k],b[k]) for k in a)
            assert identical
            pairs.append({'task_id':t['task_id'],'order':labels,'canonical_wall_s':rows['canonical'],
                          'fast_wall_s':rows['fast'],'speedup':rows['canonical']/rows['fast'],
                          'native_identical':identical})
        gain=statistics.mean(p['speedup'] for p in pairs)
        assert gain>=1.10,(name,gain)
        result['profiles'][name]={'pairs':pairs,'mean_paired_speedup':gain}
        del runtime,encoder,a,b;gc.collect();torch.cuda.empty_cache()
        atomic_json(out/'summary.json',result)
        print('LITE2_CROSSOVER='+json.dumps({'profile':name,'speedup':gain}),flush=True)
    # Worker identity is checked against the real GPU runtime before any model loading.
    root=out/'worker-smoke';backup=out/'worker-backup';manifest=qualified/'manifests/qualification.jsonl'
    subprocess.run([sys.executable,'-m','apsp.lite2.worker','--init','--tasks',str(manifest),'--out',str(root)],check=True)
    for name in PROFILES:
        args=[sys.executable,'-m','apsp.lite2.worker','--tasks',str(manifest),'--out',str(root),
              '--backup-root',str(backup),'--profile',name,'--qualification',str(qualified/'qualification'/(name+'.json')),
              '--max-tasks','1']
        subprocess.run(args,check=True)
    before={p.name:file_hash(p/'COMPLETE.json') for p in (root/'tasks').iterdir()}
    # Resume via a second process; receipt bytes must stay unchanged.
    subprocess.run(args,check=True)
    after={p.name:file_hash(p/'COMPLETE.json') for p in (root/'tasks').iterdir()}
    assert before==after and len(after)==2
    result.update(worker_cli_profiles=2,resume_marker_unchanged=True)
    atomic_json(out/'summary.json',result)
    print('LITE2_POSTCHECK='+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualified-root',required=True);run(p.parse_args().qualified_root)
