"""Real Torch CPU serialization regression using a synthetic denoiser, not a GPU receipt."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image

from apsp.lite2.contracts import ENCODER, PROFILES, policy, registry, task, uid
from apsp.lite2.collector import collect_task
from apsp.lite2.validate import validate_task
from apsp.lite2.export import export_release
from apsp.lite2.pack import pack, restore
from apsp.lite2.replay import baselines


class Scheduler:
    def __init__(self):
        self.step_index=0;self.sigmas=torch.linspace(1,0,9);self.config={'fixture':'synthetic_cpu'}
    def step(self,model_output,timestep,sample,*args,**kwargs):
        self.step_index+=1
        # Deliberately contains bits that would be destroyed by an fp16 storage cast.
        x0=sample.float()+0.000123
        return SimpleNamespace(pred_original_sample=x0,prev_sample=sample)


class UNet(torch.nn.Module):
    def forward(self,z):return SimpleNamespace(sample=z)


class FakeRuntime:
    def __init__(self,name):
        self.spec=registry()['profiles'][name];self.torch=torch;self.device='cpu';self.dtype=torch.float16
        self.C,self.h,self.w=4,self.spec['height']//8,self.spec['width']//8
        self.H,self.W=self.spec['height'],self.spec['width'];self.execution=policy()
        self.vae_dtype=torch.float32;self.denoiser_calls=0;self.adapter_provenance={'fixture':True}
        self.scorer_provenance={'imagereward':{'fixture':True},'pickscore':{'fixture':True}}
        unet=UNet()
        unet.register_forward_hook(lambda *args:setattr(self,'denoiser_calls',self.denoiser_calls+1))
        runtime=self
        class Pipe:
            def __init__(self):
                self.scheduler=Scheduler();self.unet=unet
                self.vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=.18215))
            def __call__(self,**kwargs):
                sample=kwargs['latents']
                for j in range(8):
                    value=torch.cat([sample*.1,sample*.2]) if runtime.spec['cfg_branches_available'] else sample*.1
                    result=self.unet(value)
                    sample=self.scheduler.step(result.sample,j,sample).prev_sample
                return SimpleNamespace(images=sample)
        self.pipe=Pipe()
    def sync(self):pass
    def decode_score(self,text,z):
        k=len(z);images=[Image.new('RGB',(2,2),(23,41,73)) for _ in range(k)]
        flags=[dict(latent_ok=True,img_ok=True,valid=True,failure_code=None) for _ in range(k)]
        return images,flags,[1.]*k,[2.]*k,{'decode_s':0.,'imagereward_s':0.,'pickscore_s':0.}


class Encoder:
    def encode(self,text):
        z=torch.zeros((1,384),dtype=torch.float32);z[0,0]=1
        return z,{**ENCODER,'original_tokens':4,'used_tokens':4,'truncated':False}


def main(out):
    torch.set_num_threads(4)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    paths=[];dtypes={}
    for name in PROFILES:
        prompt=dict(text='A synthetic CPU test mug.',text_raw='A synthetic CPU test mug.',
                    prompt_uid=uid('A synthetic CPU test mug.'),split='calibration',train_rank=None,
                    group_id='cpu-only',source='qualification_authored',source_revision='synthetic')
        t=task(prompt,name,0,qa_final=True,qa_preview=True)
        path=collect_task(FakeRuntime(name),Encoder(),t,out/name,{'fixture':'cpu_only'},keep_initial=True)
        assert validate_task(path,check_latents=True,runtime=FakeRuntime(name))['pass']
        from safetensors.torch import load_file
        native=load_file(str(path/'latents.safetensors'))
        assert native['x0_01'].dtype==torch.float32 and native['final'].dtype==torch.float16
        assert not torch.equal(native['x0_01'],native['x0_01'].half().float())
        paths.append(path);dtypes[name]={k:str(v.dtype) for k,v in native.items()}
    report=export_release(paths,out/'export',include_calibration=True,expected_tasks=0)
    pack(out/'export',out/'parts',part_bytes=1_000_000)
    assert restore(out/'parts',out/'restored')['pass']
    assert baselines(out/'restored/core')['episodes']==6
    result={'pass':True,'stage':'real_torch_cpu_synthetic_denoiser_not_model_qualification',
            'native_dtypes':dtypes,'export_bytes':report['total_bytes'],'core_only_replay':True}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LITE2_CPU_PASS='+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
