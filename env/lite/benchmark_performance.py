"""Paired B200 performance and numerical gate; real fixed-eight-seed workload, no test split."""
import argparse,gc,hashlib,json,os,statistics,subprocess,threading,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from apsp.lite.runtime import LiteRuntime,CleanTap
from apsp.lite.performance import ExecutionPolicy
from apsp.lite.contracts import canonical_hash,load_registry,noise_seed,uid,checkpoints
from apsp.lite.collector import collect_task,fingerprint,runtime_identity
from apsp.lite.storage import atomic_json,backup_task
from apsp.lite.validate import validate_task

TEXTS=['A red ceramic mug on a wooden table beside a blue book, natural window light.',
       'Two yellow bicycles parked beside a brick wall, soft overcast daylight.']
GATES={'max_pixel_abs':2,'mean_pixel_abs':0.01,'ir_max_abs':0.001,'pickscore_max_abs':0.001}


def save_source_snapshot(path):
    import zipfile
    source_root=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        paths=list((source_root/'apsp').rglob('*.py'))+list((source_root/'env/lite').glob('*.py'))+[source_root/'env/lite/uv.lock']
        for item in paths:z.write(item,item.relative_to(source_root).as_posix())


def task_for(text,name,profile):
    prompt={'text':text,'text_raw':text,'prompt_uid':uid(text),'split':'calibration','train_rank':None,
            'source':'performance_authored','group_id':'performance-calibration-reserved-v1'}
    return {'prompt':prompt,'profile':name,'bank_id':0,'wave':2000,
            'task_id':canonical_hash([uid(text),name,0]),'profile_hash':canonical_hash(profile),
            'seeds':[noise_seed(uid(text),0,s) for s in range(8)]}


def generate(runtime,task,capture):
    torch.cuda.synchronize();start=time.perf_counter()
    noise=torch.cat([torch.randn((1,runtime.C,runtime.h,runtime.w),generator=torch.Generator('cpu').manual_seed(s)) for s in task['seeds']]).to(runtime.dtype)
    tap=CleanTap(runtime.pipe.scheduler,checkpoints(runtime.spec['steps'],runtime.spec['checkpoints']),capture_on_gpu=capture)
    try:
        with torch.inference_mode():
            final=runtime.pipe(prompt=task['prompt']['text'],negative_prompt='',num_images_per_prompt=8,
                num_inference_steps=runtime.spec['steps'],guidance_scale=runtime.spec['guidance'],
                height=runtime.H,width=runtime.W,latents=noise.to(runtime.device),output_type='latent').images
    finally:runtime.pipe.scheduler.step=tap.original
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    frames=[tap.x0[k].to(runtime.dtype).cpu() for k in sorted(tap.x0)]+[final.to(runtime.dtype).cpu()]
    return frames,elapsed


def score_frames(runtime,text,frames):
    torch.cuda.synchronize();start=time.perf_counter()
    if runtime.execution and runtime.execution.fuse_checkpoints:
        fused=runtime.decode_score(text,torch.cat(frames,0))
        result=[tuple([part[i*8:(i+1)*8] for part in fused[:4]]+[fused[4] if i==0 else {k:0. for k in fused[4]}]) for i in range(len(frames))]
    else:result=[runtime.decode_score(text,z) for z in frames]
    torch.cuda.synchronize()
    return result,time.perf_counter()-start


def compare(reference,trial):
    maximum=0;total=0;pixels=0;identical=0;ir_delta=0.;ps_delta=0.;valid_equal=True;winner_equal=True
    for a,b in zip(reference,trial):
        if not all(len(part)==8 for part in (a[0],b[0],a[1],b[1],a[2],b[2])):raise ValueError('Eight-seed cardinality lost')
        valid_equal &= a[1]==b[1]
        for x,y in zip(a[0],b[0]):
            if x is None or y is None:
                valid_equal &= x is None and y is None;continue
            d=np.abs(np.asarray(x,dtype=np.int16)-np.asarray(y,dtype=np.int16))
            maximum=max(maximum,int(d.max()));total+=int(d.sum());pixels+=d.size;identical+=int(not d.any())
        for field in (2,3):
            for x,y in zip(a[field],b[field]):
                if x is None or y is None:
                    valid_equal &= x is None and y is None;continue
                if field==2:ir_delta=max(ir_delta,abs(x-y))
                else:ps_delta=max(ps_delta,abs(x-y))
        if all(x is not None for x in a[2]+b[2]):winner_equal &= int(np.argmax(a[2]))==int(np.argmax(b[2]))
    result={'max_pixel_abs':maximum,'mean_pixel_abs':total/max(1,pixels),'identical_images':identical,
            'image_count':len(reference)*8,'ir_max_abs':ir_delta,'pickscore_max_abs':ps_delta,
            'validity_equal':bool(valid_equal),'ir_winner_equal':bool(winner_equal)}
    result['pass']=bool(valid_equal and winner_equal and all(result[k]<=v for k,v in GATES.items()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--round',choices=['initial','final'],default='initial');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(10,0)
    # Retain the exact measured overlay: no .git or dependency/model files.
    save_source_snapshot(out/'measured-source.zip')
    phase={'name':'startup'};stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0]
                samples.append({'time':time.time(),'phase':phase['name'],'gpu_pct_memory_pct_mib_w':[float(v.strip()) for v in s.split(',')]})
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    registry=load_registry()
    registry['profiles']['hyper_sd15_cfg8_512']={'repo':'stable-diffusion-v1-5/stable-diffusion-v1-5',
       'revision':'451f4fe16113bff5a5d2269ed5ad43b0592e9a14','pipeline':'StableDiffusionPipeline',
       'scheduler':'DDIMScheduler','timestep_spacing':'trailing','dtype':'float16','height':512,'width':512,
       'steps':8,'checkpoints':8,'guidance':7.5,'cfg_branches_available':True,'disable_safety_checker':True,
       'adapter':{'kind':'lora','repo':'ByteDance/Hyper-SD','revision':'bc08d970a87c74c71209491d64e3525845698863',
                  'filename':'Hyper-SD15-8steps-CFG-lora.safetensors','fuse_scale':1.0}}
    summary={'gates':GATES,'profiles':{},'scope':'Two authored calibration prompts/profile; batch numerical/performance gate, not EDA, full dataset or online-controller qualification.',
             'runtime':fingerprint(),'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory}
    scorers=None
    try:
        for name in ['sdxl_lightning8','hyper_sd15_cfg8_512']:
            phase['name']=name+'/load';start=time.perf_counter()
            runtime=LiteRuntime(registry['profiles'][name],registry)
            if scorers is None:runtime.load_scorers();scorers=(runtime.ir,runtime.ps,runtime.scorer_provenance)
            else:runtime.ir,runtime.ps,runtime.scorer_provenance=scorers
            row={'load_s':time.perf_counter()-start,'profile':runtime.spec,'variants':{},'paired_collection':[]}
            summary['profiles'][name]=row
            task=task_for(TEXTS[0],name,runtime.spec)
            phase['name']=name+'/generate_cpu_capture';frames,g0=generate(runtime,task,False)
            phase['name']=name+'/generate_gpu_capture';gpu_frames,g1=generate(runtime,task,True)
            row['capture']={'cpu_capture_s':g0,'gpu_capture_s':g1,'measurement':'First generation is colder; do not interpret this pair as isolated capture speedup.',
                            'all_stored_tensors_identical':all(torch.equal(x,y) for x,y in zip(frames,gpu_frames))}
            # Prime both scorers/VAE before timed scoring, outside comparison statistics.
            runtime.decode_score(TEXTS[0],frames[-1])
            policies={'legacy':None,'d1s8':ExecutionPolicy(decode_batch=1),'d2s8':ExecutionPolicy(decode_batch=2),
                      'd4s8':ExecutionPolicy(decode_batch=4),'d8s8':ExecutionPolicy(decode_batch=8)}
            if a.round=='final':
                policies={'legacy':None,'s8_png1':ExecutionPolicy(png_compress_level=1,cache_prompt_embedding=True),
                    's32_fused':ExecutionPolicy(score_batch=32,fuse_checkpoints=True,png_compress_level=1,cache_prompt_embedding=True),
                    's32_stream2':ExecutionPolicy(score_batch=32,fuse_checkpoints=True,decode_streams=2,png_compress_level=1,cache_prompt_embedding=True),
                    's32_stream4':ExecutionPolicy(score_batch=32,fuse_checkpoints=True,decode_streams=4,png_compress_level=1,cache_prompt_embedding=True)}
            reference=None
            for label,policy in policies.items():
                runtime.execution=policy;runtime.decode_chunk=1 if policy is None else policy.decode_batch
                phase['name']=name+'/'+label;torch.cuda.reset_peak_memory_stats();times=[];parity=None
                try:
                    for repeat in range(2):
                        results,elapsed=score_frames(runtime,TEXTS[0],frames);times.append(elapsed)
                        if label=='legacy' and reference is None:reference=results
                        parity=compare(reference,results)
                    row['variants'][label]={'seconds':times,'median_s':statistics.median(times),'parity':parity,
                        'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'execution':None if policy is None else policy.to_dict()}
                except torch.cuda.OutOfMemoryError:
                    row['variants'][label]={'oom':True,'execution':policy.to_dict()};torch.cuda.empty_cache()
                atomic_json(out/'summary.json',summary)
                print('PERF_PROGRESS='+json.dumps({'profile':name,'variant':label,'result':row['variants'][label]}),flush=True)
            acceptable=[k for k,v in row['variants'].items() if not v.get('oom') and v['parity']['pass']]
            best=min(acceptable,key=lambda k:row['variants'][k]['median_s'])
            if not row['capture']['all_stored_tensors_identical']:best='legacy'
            row['microbenchmark_best']=best
            # Common prompt encoder/model setup is outside both steady-state task timers.
            from apsp.data.prompts import _clip_functions
            pe=registry['prompt_encoder'];_,runtime.embed=_clip_functions(pe['repo'],pe['revision'],device=str(runtime.device))
            runtime.embed([TEXTS[0]])
            for index,text in enumerate(TEXTS):
                pair={};paths={}
                for label in (['legacy',best] if index==0 else [best,'legacy']):
                    # If best is legacy, one baseline is enough; no invented speedup.
                    if label in pair:continue
                    runtime.execution=policies[label];runtime.decode_chunk=1 if policies[label] is None else policies[label].decode_batch
                    phase['name']=name+'/collect-'+label
                    fp=fingerprint(runtime.execution);task=task_for(text,name,runtime.spec)
                    torch.cuda.synchronize();start=time.perf_counter()
                    dest=collect_task(runtime,task,out/name/str(index)/label,fp)
                    backup_task(dest,out/'backup'/name/str(index)/label/task['task_id'])
                    torch.cuda.synchronize();wall=time.perf_counter()-start
                    pair[label]={'wall_s_including_backup':wall,'timings':json.loads((dest/'timings.json').read_text()),
                                 'png_bytes':sum(p.stat().st_size for p in (dest/'images').glob('*.png')),
                                 'task_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
                                 'validation':validate_task(dest,runtime)}
                    paths[label]=dest
                if best!='legacy':
                    a_lat=load_file(str(paths['legacy']/'latents.safetensors'));b_lat=load_file(str(paths[best]/'latents.safetensors'))
                    pair['stored_latents_identical']=all(torch.equal(a_lat[k],b_lat[k]) for k in a_lat)
                    import pandas as pd
                    from PIL import Image
                    reference_final=[([Image.open(paths['legacy']/f'images/s{i}.png').copy() for i in range(8)],
                        [{'valid':True}]*8,pd.read_parquet(paths['legacy']/'seeds.parquet').sort_values('seed_idx').ir.tolist(),
                        pd.read_parquet(paths['legacy']/'seeds.parquet').sort_values('seed_idx').pickscore_raw.tolist())]
                    trial_final=[([Image.open(paths[best]/f'images/s{i}.png').copy() for i in range(8)],
                        [{'valid':True}]*8,pd.read_parquet(paths[best]/'seeds.parquet').sort_values('seed_idx').ir.tolist(),
                        pd.read_parquet(paths[best]/'seeds.parquet').sort_values('seed_idx').pickscore_raw.tolist())]
                    pair['final_parity']=compare(reference_final,trial_final)
                    pair['speedup']=pair['legacy']['wall_s_including_backup']/pair[best]['wall_s_including_backup']
                row['paired_collection'].append(pair);atomic_json(out/'summary.json',summary)
            passed=best!='legacy' and all(r['stored_latents_identical'] and r['final_parity']['pass'] and
                r[best]['validation']['all_valid'] and r['legacy']['validation']['all_valid'] for r in row['paired_collection'])
            gain=statistics.mean(r.get('speedup',1.) for r in row['paired_collection'])
            row['recommendation']={'pass':passed and gain>=1.05,'execution':None if best=='legacy' else policies[best].to_dict(),
                                   'mean_paired_speedup':gain,'variant':best}
            if row['recommendation']['pass']:atomic_json(out/(name+'-execution.json'),policies[best].to_dict())
            print('PERF_PROFILE='+json.dumps({'profile':name,**row['recommendation']}),flush=True)
            del runtime,frames,gpu_frames,reference;gc.collect();torch.cuda.empty_cache()
        summary['pass']=all(p['recommendation']['pass'] for p in summary['profiles'].values())
    finally:
        stop.set();thread.join(timeout=4)
        atomic_json(out/'telemetry.json',samples);atomic_json(out/'summary.json',summary)
    print('PERFORMANCE_SUMMARY='+json.dumps(summary),flush=True)


if __name__=='__main__':main()
