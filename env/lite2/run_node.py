"""Company entry point: node gate, storage forecast, isolated 8-worker collection, export/pack."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from forecast import forecast
from apsp.lite.storage import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-manifests',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--backup-root',required=True)
    p.add_argument('--gpu-uuids',required=True)
    p.add_argument('--old-reservations',required=True)
    p.add_argument('--old-root',action='append',default=[])
    p.add_argument('--hyper-workers',type=int,default=2)
    p.add_argument('--no-pack',action='store_true')
    a=p.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_release.py'))],check=True)
    base=[sys.executable,'-m','apsp.lite2.node','--v1-manifests',a.v1_manifests,
          '--out',a.out,'--backup-root',a.backup_root,'--gpu-uuids',a.gpu_uuids,
          '--old-reservations',a.old_reservations,'--hyper-workers',str(a.hyper_workers)]
    for root in a.old_root:base.extend(['--old-root',root])
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    subprocess.run([*base,'--stage','qualify'],env=env,check=True)
    out=Path(a.out)
    report=forecast(out/'qualification')
    atomic_json(out/'storage-forecast.json',report)
    raw=report['native_upper_bytes']+report['core_upper_bytes']+report['qa_upper_bytes']
    exported=report['full_upper_with_reserve_bytes'] if report['recommended_mode']=='core_latent_qa' else report['core_upper_bytes']+report['qa_upper_bytes']+5_000_000_000
    needed=raw+exported+(0 if a.no_pack else exported)+10_000_000_000
    free=shutil.disk_usage(out).free
    backup=Path(a.backup_root)
    backup.mkdir(parents=True,exist_ok=True)
    same_device=os.stat(out).st_dev==os.stat(backup).st_dev
    if free<needed+(raw if same_device else 0) or shutil.disk_usage(backup).free<raw+5_000_000_000:
        raise ValueError('Insufficient working disk for raw+backup+export+parts; no bulk launch')
    atomic_json(out/'working-storage-check.json',{'pass':True,'required_out_bytes':needed,
                'free_out_bytes':free,'same_st_dev':same_device,
                'durability':'Filesystem space check only; operator must attest independent retention/backup domain'})
    subprocess.run([*base,'--stage','collect','--no-export'],env=env,check=True)
    result_paths=list((out/'manager-runs').glob('*/result.json'))
    if not result_paths:raise ValueError('No collection completion receipt')
    result=json.loads(max(result_paths,key=lambda p:p.stat().st_mtime).read_text())
    release=out/'exports'/str(time.time_ns())
    args=[sys.executable,'-m','apsp.lite2.export','--root',result['main_root'],'--root',result['qa_root'],
          '--reuse-audit',result['reuse_main'],'--reuse-audit',result['reuse_calibration'],
          '--planned-tasks',str(out/'manifests/main.jsonl'),'--planned-tasks',str(out/'manifests/calibration.jsonl'),
          '--failures-root',result['main_root'],'--failures-root',result['qa_root'],
          '--qualification-root',str(out/'qualification'),
          '--include-calibration','--expected-tasks','4800','--mode',report['recommended_mode'],'--out',str(release)]
    subprocess.run(args,env=env,check=True)
    if not a.no_pack:
        subprocess.run([sys.executable,'-m','apsp.lite2.pack','pack','--release',str(release),
                        '--out',str(release)+'-parts'],env=env,check=True)
    manifest=json.loads((release/'RELEASE_MANIFEST.json').read_text())
    final={'release':str(release),'mode':report['recommended_mode'],'main_tasks':manifest['main_tasks'],
           'calibration_tasks':manifest['calibration_tasks'],'coverage':manifest['coverage'],
           'status':('CORE_READY_LATENT_PENDING' if report['recommended_mode']=='core_qa' else 'RELEASE_VERIFIED') if manifest['coverage']=='complete' else 'PARTIAL_PENDING_RESERVATIONS_OR_FAILURES',
           'native_latent_status':manifest['native_latent_status']}
    atomic_json(out/'delivery.json',final)
    print('LITE2_DELIVERY='+json.dumps(final),flush=True)


if __name__=='__main__':main()
