"""Opt-in execution policy; batch shape belongs to provenance, never silent v1 tuning."""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ExecutionPolicy:
    schema: str = 'apsp.lite.execution/v1'
    decode_batch: int = 1
    score_batch: int = 8
    capture_on_gpu: bool = True
    decode_streams: int = 1
    fuse_checkpoints: bool = False
    png_compress_level: int = 6
    cache_prompt_embedding: bool = False

    def __post_init__(self):
        if self.schema != 'apsp.lite.execution/v1':raise ValueError('Unknown execution schema')
        if type(self.decode_batch) is not int or self.decode_batch not in (1,2,4,8):raise ValueError('decode_batch must be 1/2/4/8')
        if type(self.score_batch) is not int or self.score_batch not in (1,2,4,8,16,32):raise ValueError('Unsupported score_batch')
        if type(self.capture_on_gpu) is not bool:raise ValueError('capture_on_gpu must be bool')
        if type(self.decode_streams) is not int or self.decode_streams not in (1,2,4):raise ValueError('Unsupported decode_streams')
        if self.decode_streams>1 and self.decode_batch!=1:raise ValueError('Parallel streams preserve batch-one VAE shape')
        if type(self.png_compress_level) is not int or not 0<=self.png_compress_level<=9:raise ValueError('Invalid lossless PNG compression')
        if type(self.fuse_checkpoints) is not bool or type(self.cache_prompt_embedding) is not bool:raise ValueError('Execution switches must be bool')

    def to_dict(self):return asdict(self)

    @classmethod
    def load(cls,path):
        with open(path,encoding='utf-8') as f:return cls(**json.load(f))


def decode_score_batched(runtime,prompt,stored_latents):
    """Every input row survives; decode/scorer failures retain the original seed index."""
    import math,time
    torch=runtime.torch;k=stored_latents.shape[0]
    policy=runtime.execution
    images=[None]*k;img_ok=[False]*k;irs=[None]*k;pss=[None]*k
    runtime.sync();start=time.perf_counter()
    lat_ok=torch.isfinite(stored_latents).reshape(k,-1).all(1).cpu().tolist()
    valid=[i for i,ok in enumerate(lat_ok) if ok]
    vc=runtime.pipe.vae.config
    def decode(z):
        if getattr(vc,'latents_mean',None) is not None and getattr(vc,'latents_std',None) is not None:
            mean=torch.tensor(vc.latents_mean,device=runtime.device).view(1,-1,1,1)
            std=torch.tensor(vc.latents_std,device=runtime.device).view(1,-1,1,1)
            z=z*std/vc.scaling_factor+mean
        else:z=z/vc.scaling_factor+(getattr(vc,'shift_factor',None) or 0.0)
        with torch.inference_mode():return runtime.pipe.vae.decode(z.to(runtime.vae_dtype),return_dict=False)[0].float()
    def keep(idx,decoded):
        ok=torch.isfinite(decoded).reshape(len(idx),-1).all(1).cpu().tolist()
        good=[n for n,v in enumerate(ok) if v]
        if good:
            pils=runtime.pipe.image_processor.postprocess(decoded[good],output_type='pil')
            if len(pils)!=len(good):raise ValueError('Decoder image count mismatch')
            for n,pil in zip(good,pils):images[idx[n]]=pil;img_ok[idx[n]]=True
    if policy.decode_streams>1 and valid:
        if torch.device(runtime.device).type!='cuda':raise ValueError('Parallel decoder streams require CUDA')
        # The VAE still sees batch=1. Independent streams overlap kernel launches without
        # selecting a different convolution shape or throwing away any candidate rows.
        source=stored_latents.float().to(runtime.device)
        current=torch.cuda.current_stream(runtime.device)
        # Reuse stream IDs across tasks. Fresh streams each call strand allocator
        # pools and can reserve almost all VRAM despite a small live tensor set.
        key=(str(runtime.device),policy.decode_streams)
        if not hasattr(runtime,'_decoder_stream_pool'):runtime._decoder_stream_pool={}
        if key not in runtime._decoder_stream_pool:
            runtime._decoder_stream_pool[key]=[torch.cuda.Stream(device=runtime.device) for _ in range(policy.decode_streams)]
        streams=runtime._decoder_stream_pool[key]
        for stream in streams:stream.wait_stream(current);source.record_stream(stream)
        pending=[]
        for offset,i in enumerate(valid):
            stream=streams[offset%len(streams)]
            with torch.cuda.stream(stream):pending.append(([i],decode(source[i:i+1])))
        for stream in streams:current.wait_stream(stream)
        for idx,decoded in pending:keep(idx,decoded)
    else:
        for offset in range(0,len(valid),policy.decode_batch):
            idx=valid[offset:offset+policy.decode_batch]
            keep(idx,decode(stored_latents[idx].float().to(runtime.device)))
    runtime.sync();decode_s=time.perf_counter()-start
    valid=[i for i,v in enumerate(img_ok) if v]
    timings={'decode_s':decode_s}
    for name,scorer,result in [('imagereward',runtime.ir,irs),('pickscore',runtime.ps,pss)]:
        runtime.sync();start=time.perf_counter()
        for offset in range(0,len(valid),policy.score_batch):
            idx=valid[offset:offset+policy.score_batch]
            values=scorer.score(prompt,[images[i] for i in idx])
            if len(values)!=len(idx):raise ValueError('Scorer row count mismatch')
            for i,value in zip(idx,values):
                value=float(value);result[i]=value if math.isfinite(value) else None
        runtime.sync();timings[name+'_s']=time.perf_counter()-start
    flags=[]
    for i in range(k):
        failure=('LATENT_NONFINITE' if not lat_ok[i] else 'DECODE_NONFINITE' if not img_ok[i] else
                 'IR_NONFINITE' if irs[i] is None else 'PICKSCORE_NONFINITE' if pss[i] is None else None)
        flags.append({'latent_ok':bool(lat_ok[i]),'img_ok':img_ok[i],'valid':failure is None,'failure_code':failure})
    return images,flags,irs,pss,timings
