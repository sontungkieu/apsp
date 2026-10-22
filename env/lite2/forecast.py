"""Conservative storage forecast from the actual twenty-task receipts, before bulk."""
import argparse
import json
from pathlib import Path

PROFILES=('hyper_sd15_cfg8_512','sdxl_lightning8_1024')
WIDTH={'float16':2,'bfloat16':2,'float32':4,'float64':8}


def forecast(qualification):
    rows={}
    for name in PROFILES:
        receipt=json.loads((Path(qualification)/(name+'.json')).read_text())
        if not receipt['pass'] or not receipt['performance_enabled'] or len(receipt['tasks'])!=20:
            raise ValueError('Twenty passing optimized calibration tasks/profile required')
        tasks=receipt['tasks'];main_latent=[];core=[];final=[];preview=[]
        for t in tasks:
            sizes=t['bytes_by_file'];native=t['native_tensors']
            init=native.get('initial');initial_bytes=0
            if init:
                import math
                initial_bytes=math.prod(init['shape'])*WIDTH[init['dtype']]
            # Header/initial metadata overhead remains included, conservatively.
            main_latent.append(sizes['latents.safetensors']-initial_bytes)
            core.append(sum(v for k,v in sizes.items() if k!='latents.safetensors' and not k.endswith('.png')))
            final.extend(v for k,v in sizes.items() if k.startswith('final-') and k.endswith('.png'))
            preview.extend(v for k,v in sizes.items() if k.startswith('j') and k.endswith('.png'))
        if not final or not preview:raise ValueError('Missing preselected QA PNG samples')
        rows[name]={'main_native_bytes':2400*max(main_latent),
                    'calibration_native_bytes':60*max(t['bytes_by_file']['latents.safetensors'] for t in tasks),
                    'core_upper_bytes':2460*max(core),
                    'qa_png_upper_bytes':400*max(final)+240*max(preview),
                    'method':'max observed file bytes/task, no compression assumption',
                    'cost_profile':receipt['cost_profile']}
    native=sum(r['main_native_bytes']+r['calibration_native_bytes'] for r in rows.values())
    core=sum(r['core_upper_bytes'] for r in rows.values())
    qa=sum(r['qa_png_upper_bytes'] for r in rows.values())
    full=native+core+qa+5_000_000_000
    mode='core_latent_qa' if native<=65_000_000_000 and core<=5_000_000_000 and qa<=5_000_000_000 and full<=80_000_000_000 else 'core_qa'
    if core+qa+5_000_000_000>40_000_000_000:
        raise ValueError('Even Core+QA cannot fit fallback; review raw samples before bulk')
    return {'profiles':rows,'native_upper_bytes':native,'core_upper_bytes':core,'qa_upper_bytes':qa,
            'full_upper_with_reserve_bytes':full,'recommended_mode':mode,'cap_bytes':80_000_000_000,
            'limitations':'Forecast, not actual final size; native retained externally if core_qa; retention must be confirmed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualification',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report=forecast(a.qualification)
    from apsp.lite.storage import atomic_json
    atomic_json(a.out,report)
    print(json.dumps(report))
