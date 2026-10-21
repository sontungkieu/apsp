"""Run on Modal CPU: download only missing pinned files, verify/restore, publish delta."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
import lite2_addon as addon

REPO='codemaivanngu/apsp-lite-b200-cu128'
WORK=Path('/persist/lite2-model-addon-v1')
CACHE='/persist/lite2-model-download-cache'


def emit(stage,**data):print('LITE2_ADDON_JSON='+json.dumps({'stage':stage,**data}),flush=True)


def finish_preparation(archive,digest):
    restore=addon.restore(archive,WORK/'restored','/persist/companion-restored-v1',digest)
    check=addon.offline_check(WORK/'restored');emit('offline_cpu_pass',result=check)
    manifest=json.loads((WORK/'LITE2_ADDON_MANIFEST.json').read_text())
    receipt={'pass':True,'archive_file':addon.ARCHIVE,'archive_bytes':archive.stat().st_size,'archive_sha256':digest,
             'restore':restore,'offline_cpu':check,'runtime_rebuilt':False,
             'new_files':len(manifest['files']),'required_old_files':len(manifest['requires_companion'])}
    (WORK/'PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    emit('prepared',**receipt)


def main():
    api=HfApi()
    assert api.whoami()['name']=='codemaivanngu'
    assert not api.repo_info(REPO,repo_type='dataset').private
    WORK.mkdir(exist_ok=True)
    marker=WORK/'restored/RESTORED_LITE2_ADDON.json'
    if marker.is_file():
        archive=WORK/addon.ARCHIVE;digest=json.loads(marker.read_text())['archive_sha256']
        manifest=addon.verify(archive,digest)
        assert (manifest['base_revision'],manifest['hyper_revision'],manifest['minilm_revision'])==(addon.BASE_REV,addon.HYPER_REV,addon.MINILM_REV)
        emit('resume_verified_archive',bytes=archive.stat().st_size,sha256=digest)
        finish_preparation(archive,digest)
        return
    selected={};upstream={}
    jobs=[(addon.BASE_REPO,addon.BASE_REV,['model_index.json','scheduler/scheduler_config.json',
        'unet/config.json','unet/diffusion_pytorch_model.safetensors','vae/config.json','vae/diffusion_pytorch_model.safetensors','README.md']),
        ('ByteDance/Hyper-SD',addon.HYPER_REV,['Hyper-SD15-8steps-CFG-lora.safetensors','LICENSE.md','README.md']),
        ('sentence-transformers/all-MiniLM-L6-v2',addon.MINILM_REV,['config.json','model.safetensors','tokenizer.json',
         'tokenizer_config.json','special_tokens_map.json','vocab.txt','modules.json','sentence_bert_config.json',
         'config_sentence_transformers.json','1_Pooling/config.json','README.md'])]
    for repo,rev,names in jobs:
        info=api.model_info(repo,revision=rev,files_metadata=True)
        metadata={s.rfilename:s for s in info.siblings};assert info.sha==rev
        for name in names:
            emit('download',repo=repo,file=name)
            p=Path(hf_hub_download(repo,name,revision=rev,cache_dir=CACHE))
            digest=addon.sha(p);s=metadata[name]
            assert p.stat().st_size==s.size
            if s.lfs:assert digest==s.lfs.sha256
            rel=addon.snapshot(repo,rev)+'/'+name
            selected[rel]=p;upstream[rel]={'repo':repo,'revision':rev,'filename':name,'sha256':digest,'bytes':p.stat().st_size}
    # Retain full upstream licenses; no umbrella relicensing of weights.
    license_path=Path(hf_hub_download('CompVis/stable-diffusion-license','license.txt',repo_type='space',
        revision='14d42d09bffd871b1666a084fc954a50cff72ac0',cache_dir=CACHE))
    selected['LICENSES/SD15-CreativeML-OpenRAIL-M.txt']=license_path
    apache=WORK/'Apache-2.0.txt'
    with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt',timeout=60) as r:apache.write_bytes(r.read(100000))
    selected['LICENSES/MiniLM-Apache-2.0.txt']=apache
    selected['LICENSES/Hyper-SD-LICENSE.md']=selected[addon.snapshot('ByteDance/Hyper-SD',addon.HYPER_REV)+'/LICENSE.md']
    notice=WORK/'NOTICE.txt';notice.write_text('Pinned upstream SD1.5, Hyper-SD and MiniLM weights; upstream licenses remain applicable.\nThis delta depends on the verified v1 companion text encoder/tokenizer and retains it unchanged.\n')
    selected['NOTICE.txt']=notice
    old=json.loads(Path('/opt/tools/companion-manifest.json').read_text())
    prefix=addon.snapshot(addon.BASE_REPO,addon.BASE_REV)+'/'
    required={k:v for k,v in old['files'].items() if k.startswith(prefix) and '/model.fp16.' not in k}
    expected_old={'text_encoder/config.json','text_encoder/model.safetensors','tokenizer/merges.txt',
                  'tokenizer/special_tokens_map.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json'}
    assert {k[len(prefix):] for k in required}==expected_old, 'Companion dependency inventory differs'
    files={k:{'bytes':p.stat().st_size,'sha256':addon.sha(p)} for k,p in selected.items()}
    manifest={'schema':'apsp.lite2.model-addon/v1','files':files,'upstream':upstream,'requires_companion':required,
        'base_repo':addon.BASE_REPO,'base_revision':addon.BASE_REV,'base_weight_variant':'default full-precision safetensors; no file conversion',
        'hyper_revision':addon.HYPER_REV,'minilm_revision':addon.MINILM_REV,
        'minilm':{'width':384,'pooling':'attention-mask mean then L2','max_tokens':256},
        'runtime_lock_sha256':'e404558d31cc6befaa79aa693b99650eee3d269db3d8aec00517af1845aff793',
        'qualification':'asset readiness only; B200 Lite2 generator/scorer/collector pending'}
    mp=WORK/'LITE2_ADDON_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=WORK/addon.ARCHIVE
    with archive.open('wb',buffering=8<<20) as raw,tarfile.open(fileobj=raw,mode='w:') as tf:
        for name,p in sorted(selected.items()):
            info=tf.gettarinfo(str(p),arcname=name)
            # Downloads can be HF blob symlinks; archive always materializes regular files.
            info.type=tarfile.REGTYPE;info.linkname='';info.size=p.stat().st_size
            with p.open('rb') as f:tf.addfile(info,f)
        info=tf.gettarinfo(str(mp),arcname=mp.name)
        with mp.open('rb') as f:tf.addfile(info,f)
    digest=addon.sha(archive);emit('archive_created',bytes=archive.stat().st_size,sha256=digest)
    addon.verify(archive,digest)
    restored=WORK/'restored'
    if restored.exists():
        marker=restored/'RESTORED_LITE2_ADDON.json'
        if not marker.is_file() or json.loads(marker.read_text())['archive_sha256']!=digest:
            raise ValueError('Old restore differs; preserve it and use another version')
    # Offline check blocks sockets globally: upload executes in a fresh process.
    finish_preparation(archive,digest)


def publish():
    api=HfApi();assert api.whoami()['name']=='codemaivanngu'
    prep=json.loads((WORK/'PREPARATION.json').read_text());assert prep['pass']
    archive=WORK/addon.ARCHIVE;assert addon.sha(archive)==prep['archive_sha256']
    manifest=WORK/'LITE2_ADDON_DOWNLOAD.json'
    manifest.write_text(json.dumps({'parts':[{'file':addon.ARCHIVE,'bytes':prep['archive_bytes'],'sha256':prep['archive_sha256']}]},indent=2)+'\n')
    sums=WORK/'LITE2_ADDON.SHA256SUMS'
    files=[archive,manifest,WORK/'LITE2_ADDON_MANIFEST.json',WORK/'PREPARATION.json',Path('/opt/tools/lite2_addon.py')]
    names={p:('restore_lite2_addon.py' if p.name=='lite2_addon.py' else p.name) for p in files}
    sums.write_text(''.join(f'{addon.sha(p)}  {names[p]}\n' for p in files));files.append(sums);names[sums]=sums.name
    parent=api.repo_info(REPO,repo_type='dataset').sha
    result=api.create_commit(REPO,repo_type='dataset',parent_commit=parent,
        operations=[CommitOperationAdd(path_in_repo=names[p],path_or_fileobj=str(p)) for p in files],
        commit_message='Add verified Lite2 missing SD15 Hyper-CFG8 and MiniLM assets')
    info=api.repo_info(REPO,repo_type='dataset',revision=result.oid,files_metadata=True)
    remote={s.rfilename:s for s in info.siblings}
    assert remote[addon.ARCHIVE].size==prep['archive_bytes'] and remote[addon.ARCHIVE].lfs.sha256==prep['archive_sha256']
    for p in files:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{result.oid}/{names[p]}'
        if p==archive:
            req=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
            with urllib.request.urlopen(req,timeout=60) as r:assert r.status==206 and len(r.read(2))==1
        else:
            with urllib.request.urlopen(url,timeout=60) as r:body=r.read(2<<20)
            assert hashlib.sha256(body).hexdigest()==addon.sha(p)
    receipt={**prep,'hf_repo':REPO,'hf_revision':result.oid,'anonymous_range_verified':True,
        'published_files':{names[p]:{'bytes':p.stat().st_size,'sha256':addon.sha(p)} for p in files}}
    (WORK/'PUBLICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    emit('public_verified',**receipt)


if __name__=='__main__':
    import sys
    try:publish() if '--publish' in sys.argv else main()
    except Exception as exc:
        import re,traceback
        message=re.sub(r'hf_[A-Za-z0-9]+','<redacted>',str(exc))
        if 'http' in message.lower():message='Network error details withheld'
        emit('failed',error_type=type(exc).__name__,message=message[:600],
             frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)[-4:]])
        # Field-only errors, never authenticated download URLs.
        raise SystemExit(1)
