"""Verified delta weights for Lite2; preserves the original companion and runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

BASE_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
BASE_REV = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
HYPER_REV = 'bc08d970a87c74c71209491d64e3525845698863'
MINILM_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ARCHIVE = 'apsp-lite2-model-addon-v1.tar'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def snapshot(repo, revision):
    return f'hf/hub/models--{repo.replace("/","--")}/snapshots/{revision}'


def safe(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path')


def verify(archive, expected):
    if sha(archive)!=expected:raise ValueError('Archive SHA mismatch')
    with tarfile.open(archive,'r:') as tf:
        members=tf.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate members')
        for m in members:
            safe(m.name)
            if not m.isfile():raise ValueError('Only regular archive files accepted')
        manifest=json.load(tf.extractfile('LITE2_ADDON_MANIFEST.json'))
        if set(names)!=set(manifest['files'])|{'LITE2_ADDON_MANIFEST.json'}:
            raise ValueError('Unexpected archive members')
        for m in members:
            if m.name=='LITE2_ADDON_MANIFEST.json':continue
            h=hashlib.sha256();f=tf.extractfile(m)
            for block in iter(lambda:f.read(8<<20),b''):h.update(block)
            entry=manifest['files'][m.name]
            if m.size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
                raise ValueError('Member hash mismatch')
    return manifest


def merge_cache(old, new):
    """Build a private view; never write into the existing companion."""
    new.mkdir(parents=True,exist_ok=True)
    for p in old.iterdir():
        if p.name in ('refs','.locks'):continue
        dest=new/p.name
        if not dest.exists():dest.symlink_to(p.resolve(),target_is_directory=p.is_dir())
        elif p.is_dir() and dest.is_dir():merge_cache(p,dest)
        elif p.is_file() and dest.is_file() and sha(p)!=sha(dest):
            raise ValueError('Conflicting cache file preserved: '+p.name)


def restore(archive, target, companion, expected):
    target,companion=Path(target).absolute(),Path(companion).resolve()
    if target.exists():
        marker=target/'RESTORED_LITE2_ADDON.json'
        if not target.is_symlink() and marker.is_file() and json.loads(marker.read_text())=={'archive_sha256':expected,'companion':str(companion)}:
            return {'status':'already_restored','target':str(target)}
        raise FileExistsError('Existing target preserved')
    manifest=verify(archive,expected)
    # Validate all old files this delta depends on before creating the view.
    for rel,entry in manifest['requires_companion'].items():
        safe(rel);p=companion/rel
        if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
            raise ValueError('Required companion file missing/mismatched: '+rel)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    with tarfile.open(archive,'r:') as tf:
        for m in tf:
            p=stage/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
    merge_cache(companion/'hf/hub',stage/'hf/hub')
    (stage/'RESTORED_LITE2_ADDON.json').write_text(json.dumps({'archive_sha256':expected,'companion':str(companion)})+'\n')
    (stage/'activate_lite2_models.sh').write_text('''# Source AFTER activate_companion.sh; old companion must remain in place.
export APSP_LITE2_MODELS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$APSP_LITE2_MODELS/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export APSP_SD15_SNAPSHOT="$HF_HOME/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
export APSP_HYPER_SD15_LORA="$HF_HOME/hub/models--ByteDance--Hyper-SD/snapshots/bc08d970a87c74c71209491d64e3525845698863/Hyper-SD15-8steps-CFG-lora.safetensors"
export APSP_MINILM_SNAPSHOT="$HF_HOME/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
''')
    if target.exists():raise FileExistsError('Target appeared during restore')
    stage.rename(target)
    return {'status':'restored','target':str(target),'verified_new_files':len(manifest['files'])}


def offline_check(root):
    import socket
    def denied(*args,**kwargs):raise RuntimeError('Network blocked during offline check')
    socket.socket.connect=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer,AutoModel
    from diffusers import StableDiffusionPipeline,DDIMScheduler
    from safetensors import safe_open
    root=Path(root);torch.set_num_threads(2)
    base=root/snapshot(BASE_REPO,BASE_REV)
    pipe=StableDiffusionPipeline.from_pretrained(str(base),torch_dtype=torch.float16,
        safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True)
    lora=root/snapshot('ByteDance/Hyper-SD',HYPER_REV)/'Hyper-SD15-8steps-CFG-lora.safetensors'
    with safe_open(lora,framework='pt',device='cpu') as f:lora_keys=len(f.keys())
    pipe.load_lora_weights(str(lora.parent),weight_name=lora.name,local_files_only=True)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,timestep_spacing='trailing')
    pipe.scheduler.set_timesteps(8)
    steps=pipe.scheduler.timesteps.tolist();assert len(steps)==8
    mini=root/snapshot('sentence-transformers/all-MiniLM-L6-v2',MINILM_REV)
    tok=AutoTokenizer.from_pretrained(str(mini),local_files_only=True)
    model=AutoModel.from_pretrained(str(mini),local_files_only=True).eval()
    tokens=tok(['A red cube on a table.'],return_tensors='pt',truncation=True,max_length=256)
    with torch.no_grad():
        z=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
        emb=torch.nn.functional.normalize((z*mask).sum(1)/mask.sum(1).clamp(min=1),p=2,dim=1)
    assert emb.shape==(1,384) and torch.isfinite(emb).all() and abs(emb.norm().item()-1)<1e-5
    return {'pass':True,'stage':'offline_cpu_weight_load_lora_fuse_minilm_forward',
            'sd15_loaded':True,'lora_tensor_keys':lora_keys,'ddim_trailing_steps':steps,
            'minilm_shape':[1,384],'minilm_norm':emb.norm().item(),
            'limitation':'No B200 generation, scorer rerun or Lite2 collector qualification; v1 runtime reused.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('restore');r.add_argument('--archive',required=True);r.add_argument('--target',required=True);r.add_argument('--companion',required=True);r.add_argument('--sha256',required=True)
    r=sub.add_parser('check');r.add_argument('--root',required=True)
    a=p.parse_args()
    print(json.dumps(restore(a.archive,a.target,a.companion,a.sha256) if a.cmd=='restore' else offline_check(a.root),indent=2))


if __name__=='__main__':main()
