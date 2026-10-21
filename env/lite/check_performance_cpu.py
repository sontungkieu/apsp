"""Real Torch CPU regression: batching preserves every seed, masks, and exact row order."""
import json
from types import SimpleNamespace
import torch
from apsp.lite.performance import ExecutionPolicy,decode_score_batched


class Scorer:
    def __init__(self):self.calls=[]
    def score(self,prompt,images):self.calls.append(list(images));return [float(x) for x in images]


def main():
    ir,ps=Scorer(),Scorer()
    pipe=SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0),
        decode=lambda z,**kw:(z,)),image_processor=SimpleNamespace(postprocess=lambda z,**kw:[float(v.flatten()[0]) for v in z]))
    runtime=SimpleNamespace(torch=torch,pipe=pipe,device='cpu',vae_dtype=torch.float32,
        sync=lambda:None,execution=ExecutionPolicy(decode_batch=4,score_batch=8),ir=ir,ps=ps)
    z=torch.arange(8,dtype=torch.float32).view(8,1,1,1)
    images,flags,irs,pss,_=decode_score_batched(runtime,'prompt',z)
    assert images==irs==pss==list(range(8)) and len(flags)==8 and all(f['valid'] for f in flags)
    assert len(ir.calls)==1 and len(ps.calls)==1
    runtime.execution=ExecutionPolicy(score_batch=32,fuse_checkpoints=True)
    wide=torch.arange(72,dtype=torch.float32).view(72,1,1,1)
    _,_,wide_ir,_,_=decode_score_batched(runtime,'prompt',wide)
    assert wide_ir==list(range(72)) and wide_ir[64:]==list(range(64,72))
    runtime.execution=ExecutionPolicy(decode_batch=4,score_batch=8)
    z[3]=float('nan');ir.calls=[];ps.calls=[]
    images,flags,irs,pss,_=decode_score_batched(runtime,'prompt',z)
    assert len(images)==8 and images[3] is None and flags[3]['failure_code']=='LATENT_NONFINITE'
    assert irs==[0.,1.,2.,None,4.,5.,6.,7.] and ir.calls==[[0.,1.,2.,4.,5.,6.,7.]]
    # A nonfinite decoded row must not hide valid rows in the same VAE batch.
    z=torch.arange(8,dtype=torch.float32).view(8,1,1,1)
    def decode(value,**kwargs):
        value=value.clone();value[value==5]=float('inf');return (value,)
    pipe.vae.decode=decode
    _,flags,irs,_,_=decode_score_batched(runtime,'prompt',z)
    assert flags[5]['failure_code']=='DECODE_NONFINITE' and irs[4]==4 and irs[6]==6
    # No accidental silent truncation if a scorer returns the wrong number of rows.
    runtime.ir=SimpleNamespace(score=lambda p,ims:[0.])
    try:decode_score_batched(runtime,'prompt',z)
    except ValueError as e:assert 'row count' in str(e)
    else:raise AssertionError('Scorer truncation was silently accepted')
    print('PERFORMANCE_CPU_PASS '+json.dumps({'pass':True,'rows':8,'nonfinite_row_mapping':True,'batched_scorers':True,'scorer_count_guard':True}))


if __name__=='__main__':main()
