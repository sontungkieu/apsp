"""20 real calibration tasks/profile; batch-one concurrent VAE, fixed scorer batch eight."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from benchmark_performance import task_for,generate,score_frames,compare,save_source_snapshot,GATES
from apsp.lite.performance import ExecutionPolicy
from apsp.lite.runtime import LiteRuntime
from apsp.lite.contracts import canonical_hash,load_registry,noise_seed
from apsp.lite.collector import collect_task,fingerprint,runtime_identity
from apsp.lite.storage import atomic_json,backup_task
from apsp.lite.validate import validate_task


def cpu_resume_check(root):
    root=Path(root)
    if not (root/'summary.json').is_file():raise ValueError('Resume summary missing')
    summary=json.loads((root/'summary.json').read_text())
    from apsp.lite.contracts import file_hash
    from apsp.lite.storage import verify_complete
    source=Path(__file__).resolve().parents[2]
    code={p.relative_to(source).as_posix():file_hash(p) for p in sorted((source/'apsp').rglob('*.py'))}
    if canonical_hash(code)!=summary['source_runtime']['source_hash']:raise ValueError('Resume collector hash changed')
    n=0
    for name,profile in summary['profiles'].items():
        for row in profile['tasks']:
            if not verify_complete(root/name/'fast/tasks'/row['task_id']):raise ValueError('Resume task checksum failed')
            n+=1
    print('RESUME_CPU_PASS '+json.dumps({'verified_completed_tasks':n,'source_hash':canonical_hash(code)}),flush=True)


def frames_at(path):
    z=load_file(str(path/'latents.safetensors'))
    return [z['x0'][:,j] for j in range(z['x0'].shape[1])]+[z['final']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--previous',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume);save_source_snapshot(out/('resumed-measured-source.zip' if a.resume else 'measured-source.zip'))
    previous=json.loads(Path(a.previous).read_text());registry=load_registry()
    calibration=[json.loads(s) for s in Path('/persist/text/lite-v1-20261019/manifests/calibration.jsonl').read_text().splitlines()]
    calibration=sorted(calibration,key=lambda r:r['prompt_uid'])[:10]
    assert len(calibration)==10 and all(r['split']=='calibration' for r in calibration)
    summary={'scope':'10 existing calibration prompts x 2 banks x 8 seeds/profile; no train/dev/test sampling.',
             'gates':GATES,'profiles':{},'source_runtime':fingerprint(),'reference':'Legacy batch-one VAE and batch-one scorers on the exact same stored trajectories.'}
    if a.resume and (out/'summary.json').is_file():
        summary=json.loads((out/'summary.json').read_text())
        if summary['source_runtime']['source_hash']!=fingerprint()['source_hash']:raise ValueError('Cannot resume changed collector source')
        summary['harness_resume']='Reuse verified task receipts; capture collection outputs instead of decoding fast frames twice.'
    scorers=None
    for name in ['sdxl_lightning8','hyper_sd15_cfg8_512']:
        spec=previous['profiles'][name]['profile'];registry['profiles'][name]=spec
        runtime=LiteRuntime(spec,registry)
        if scorers is None:runtime.load_scorers();scorers=(runtime.ir,runtime.ps,runtime.scorer_provenance)
        else:runtime.ir,runtime.ps,runtime.scorer_provenance=scorers
        from apsp.data.prompts import _clip_functions
        pe=registry['prompt_encoder'];_,runtime.embed=_clip_functions(pe['repo'],pe['revision'],device='cuda');runtime.embed([calibration[0]['text']])
        first=task_for(calibration[0]['text'],name,spec)
        trials={};policies={'stream2':ExecutionPolicy(score_batch=8,fuse_checkpoints=True,decode_streams=2,png_compress_level=1,cache_prompt_embedding=True),
                            'stream4':ExecutionPolicy(score_batch=8,fuse_checkpoints=True,decode_streams=4,png_compress_level=1,cache_prompt_embedding=True)}
        if name in summary['profiles']:
            profile=summary['profiles'][name];policy=ExecutionPolicy(**profile['execution'])
        else:
            frames,_=generate(runtime,first,False)
            baseline,_=score_frames(runtime,first['prompt']['text'],frames)
            for label,policy in policies.items():
                runtime.execution=policy;runtime.decode_chunk=1
                result,elapsed=score_frames(runtime,first['prompt']['text'],frames)
                trials[label]={'seconds':elapsed,'parity':compare(baseline,result)}
            good=[label for label,r in trials.items() if r['parity']['pass']]
            if not good:raise RuntimeError('No parallel VAE candidate passed unchanged numerical gates')
            best=min(good,key=lambda k:trials[k]['seconds']);policy=policies[best]
            profile={'profile':spec,'execution':policy.to_dict(),'selection':trials,'tasks':[],'paired_full_tasks':[]}
            del frames,baseline
        runtime.execution=policy;fp=fingerprint(policy)
        summary['profiles'][name]=profile
        done={t['task_id'] for t in profile['tasks']}
        torch.cuda.reset_peak_memory_stats()
        for row in calibration:
            for bank in (0,1):
                task=task_for(row['text'],name,spec);task['prompt']=row;task['bank_id']=bank
                task['task_id']=canonical_hash([row['prompt_uid'],name,bank])
                task['seeds']=[noise_seed(row['prompt_uid'],bank,i) for i in range(8)]
                if task['task_id'] in done:
                    from apsp.lite.storage import verify_complete
                    if not verify_complete(out/name/'fast/tasks'/task['task_id']):raise ValueError('Resume task incomplete')
                    continue
                runtime.execution=policy;torch.cuda.synchronize();start=time.perf_counter()
                captured=[];original=runtime.decode_score
                def capture(prompt,latents):
                    runtime.sync();begin=time.perf_counter();result=original(prompt,latents);runtime.sync()
                    captured.append((result,time.perf_counter()-begin));return result
                runtime.decode_score=capture
                try:dest=collect_task(runtime,task,out/name/'fast',fp)
                finally:runtime.decode_score=original
                backup_task(dest,out/name/'backup'/task['task_id'])
                torch.cuda.synchronize();fast_wall=time.perf_counter()-start
                saved=frames_at(dest)
                if captured:
                    fused,fast_score_s=captured[0]
                    fast_result=[tuple([part[i*8:(i+1)*8] for part in fused[:4]]+[fused[4]]) for i in range(len(saved))]
                else:
                    fast_result,fast_score_s=score_frames(runtime,row['text'],saved)
                parity=None;legacy_score_s=None
                # Preselected bank0 covers all ten prompts. Bank1 still gets complete
                # schema/finite/re-decode validation; avoid a second full legacy pass.
                if bank==0:
                    runtime.execution=None
                    reference,legacy_score_s=score_frames(runtime,row['text'],saved)
                    parity=compare(reference,fast_result)
                runtime.execution=policy
                validation=validate_task(dest,runtime)
                entry={'task_id':task['task_id'],'prompt_uid':row['prompt_uid'],'bank_id':bank,
                    'timing_reused_task':not bool(captured),
                    'wall_s_including_backup':fast_wall,'score_decode_legacy_s':legacy_score_s,
                    'score_decode_fast_s':fast_score_s,'score_decode_speedup':None if legacy_score_s is None else legacy_score_s/fast_score_s,
                    'parity':parity,'validation':validation}
                # Two complete legacy-vs-fast tasks/profile, independent of the authored tuning prompts.
                if bank==0 and len(profile['paired_full_tasks'])<2:
                    runtime.execution=None;torch.cuda.synchronize();start=time.perf_counter()
                    old=collect_task(runtime,task,out/name/'legacy-paired',fingerprint())
                    backup_task(old,out/name/'legacy-backup'/task['task_id'])
                    torch.cuda.synchronize();old_wall=time.perf_counter()-start
                    old_tensors=frames_at(old)
                    same=all(torch.equal(x,y) for x,y in zip(saved,old_tensors))
                    pair={'task_id':task['task_id'],'legacy_wall_s':old_wall,'fast_wall_s':fast_wall,'speedup':old_wall/fast_wall,
                          'stored_trajectories_identical':same,'legacy_png_bytes':sum(p.stat().st_size for p in (old/'images').glob('*.png')),
                          'fast_png_bytes':sum(p.stat().st_size for p in (dest/'images').glob('*.png'))}
                    profile['paired_full_tasks'].append(pair);runtime.execution=policy
                entry['pass']=(parity is None or parity['pass']) and validation['all_valid'] and validation['max_pixel_diff_redecode']==0
                profile['tasks'].append(entry);atomic_json(out/'summary.json',summary)
                print('THROUGHPUT_TASK='+json.dumps({'profile':name,'done':len(profile['tasks']),'pass':entry['pass'],
                    'score_decode_speedup':entry['score_decode_speedup']}),flush=True)
        profile['reference_checked_tasks']=sum(t['parity'] is not None for t in profile['tasks'])
        profile['pass']=len(profile['tasks'])==20 and profile['reference_checked_tasks']>=10 and all(t['pass'] for t in profile['tasks']) and all(t['stored_trajectories_identical'] for t in profile['paired_full_tasks'])
        profile['mean_paired_full_speedup']=statistics.mean(t['speedup'] for t in profile['paired_full_tasks'])
        profile['median_score_decode_speedup']=statistics.median(t['score_decode_speedup'] for t in profile['tasks'] if t['score_decode_speedup'] is not None)
        profile['max_allocated_bytes']=torch.cuda.max_memory_allocated();profile['max_reserved_bytes']=torch.cuda.max_memory_reserved()
        profile['persistent_stream_pools']={str(k):len(v) for k,v in runtime._decoder_stream_pool.items()}
        receipt={'pass':profile['pass'],'profile':name,'profile_hash':canonical_hash(spec),
                 'runtime_identity_hash':canonical_hash(runtime_identity(fp)),'runtime':fp,
                 'validation':{'tasks':20,'candidates':160,'preview_rows':1280,'reference_checked_tasks':profile['reference_checked_tasks'],
                               'all_valid':all(t['validation']['all_valid'] for t in profile['tasks'])},
                 'scope':'Throughput/native-batch qualification, legacy Lite storage dtype. Not Lite2 Core schema or full EDA/online-controller acceptance.'}
        atomic_json(out/(name+'-qualification.json'),receipt)
        if profile['pass']:atomic_json(out/(name+'-execution.json'),policy.to_dict())
        print('THROUGHPUT_PROFILE='+json.dumps({'profile':name,'pass':profile['pass'],'paired_full_speedup':profile['mean_paired_full_speedup'],
                'median_score_decode_speedup':profile['median_score_decode_speedup'],'execution':policy.to_dict()}),flush=True)
        del runtime;gc.collect();torch.cuda.empty_cache()
    summary['pass']=all(p['pass'] for p in summary['profiles'].values())
    atomic_json(out/'summary.json',summary)
    print('THROUGHPUT_SUMMARY='+json.dumps({'pass':summary['pass'],'out':str(out),'profiles':{k:{f:v[f] for f in ['pass','execution','mean_paired_full_speedup','median_score_decode_speedup','max_allocated_bytes','max_reserved_bytes']} for k,v in summary['profiles'].items()}}),flush=True)


if __name__=='__main__':main()
