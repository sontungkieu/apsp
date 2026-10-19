"""Publish already-verified assets directly Modal -> HF, following the TTS harness.

The user authorized a new public dataset. Stage privately, verify bytes/LFS hashes,
include upstream notices, then expose and verify anonymously at the pinned commit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.request

ROOT=Path('/opt/apsp')
PERSIST=Path('/persist')
REPO='codemaivanngu/apsp-lite-b200-cu128'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()


def stage(name, **fields):
    print('HF_PUBLIC_STAGE='+json.dumps({'stage':name,**fields}),flush=True)


def main():
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    api=HfApi()
    if api.whoami()['name']!='codemaivanngu':raise RuntimeError('HF account mismatch; refusing write')
    export=PERSIST/'export'
    parts=json.loads((export/'apsp-lite-companion-v1.tar.gz.parts.json').read_text())
    archive=json.loads((export/'apsp-lite-companion-v1.receipt.json').read_text())
    runtime=json.loads((export/'manifest.json').read_text())
    offline=json.loads((export/'companion-offline-receipt.json').read_text())
    assert parts['pass'] and archive['pass'] and offline['offline']['pass']
    assert parts['original_tar_sha256']==archive['archive_sha256']
    assert len(parts['parts'])==2 and all(p['bytes']<=30_000_000_000 for p in parts['parts'])
    assert runtime['archive_sha256']==archive['runtime_archive_sha256']
    release=PERSIST/'hf-public-release-v1';release.mkdir(exist_ok=True)
    files={p['file']:export/p['file'] for p in parts['parts']}
    files['runtime.tar.gz']=export/'runtime.tar.gz'
    # Expose small receipts/helpers separately, so neither download node needs Modal.
    small={'COMPANION_PARTS.json':export/'apsp-lite-companion-v1.tar.gz.parts.json',
        'COMPANION_PARTS.SHA256SUMS':export/'apsp-lite-companion-v1.tar.gz.SHA256SUMS',
        'runtime-manifest.json':export/'manifest.json','companion-manifest.json':export/'apsp-lite-companion-v1.manifest.json',
        'download_part_hf.sh':ROOT/'env/lite/download_part_hf.sh','restore_companion.py':ROOT/'env/lite/companion.py',
        'ensure_runtime.py':ROOT/'env/lite/ensure_runtime.py','verify_companion_offline.py':ROOT/'env/lite/verify_companion_offline.py'}
    for name,path in small.items():shutil.copyfile(path,release/name);files[name]=release/name
    # Published diagnostics use relocatable labels, not Modal volume identifiers.
    offline['offline']['cache_root']='<restored-companion>/hf'
    offline['restore']['target']='<restored-companion>'
    (release/'offline-restore-receipt.json').write_text(json.dumps(offline,indent=2)+'\n')
    files['offline-restore-receipt.json']=release/'offline-restore-receipt.json'
    assets=json.loads((PERSIST/'assets.json').read_text())
    sources=sorted({(a['repo'],a['revision']) for j in assets.values() for a in j['assets']})
    licenses=[]
    for repo,rev in sources:
        available=api.list_repo_files(repo,revision=rev,repo_type='model')
        names=[n for n in available if '/' not in n and (n.lower().startswith(('license','licence','notice')) or n=='README.md')]
        info=api.model_info(repo,revision=rev)
        declared=getattr(info.card_data,'license',None)
        for name in names:
            src=Path(hf_hub_download(repo,name,revision=rev,cache_dir=str(PERSIST/'license-cache')))
            dest=release/'LICENSES'/repo.replace('/','--')/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
            files[dest.relative_to(release).as_posix()]=dest
        licenses.append({'repo':repo,'revision':rev,'declared_license':declared,'included_files':names})
    # Mandatory FLUX distribution notice accompanies the unmodified checkpoint.
    flux_license=release/'LICENSES/black-forest-labs--FLUX.1-dev/LICENSE.md'
    assert flux_license.is_file()
    notice='''This research package redistributes byte-identical upstream weights. Original licenses continue to apply to each component; this package does not relicense them.

The FLUX.1 [dev] Model is licensed by Black Forest Labs Inc. under the FLUX.1 [dev] Non-Commercial License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

Rights to use FLUX.1 [dev] are granted directly by Black Forest Labs Inc. under that license, a copy of which is in LICENSES/black-forest-labs--FLUX.1-dev/LICENSE.md. The original checkpoint and Hyper LoRA are stored separately, unmodified; the documented runtime fuses the LoRA at execution time. No endorsement by the upstream authors is implied.

Data sources: poloclub/DiffusionDB metadata (pinned revision in companion manifest); COCO2017 caption annotations from the official images.cocodataset.org bucket; Google Research PartiPrompts TSV; djghosh13/GenEval exclusion prompts. Attribution, source revisions and hashes are retained in the companion data manifests. No source images or generated training/test image bank are claimed.

Dependency runtime packages retain the license files distributed in their installed dist-info directories. Source history is included for provenance; no credential values or company proxy configuration are included.
'''
    (release/'NOTICE').write_text(notice)
    files['NOTICE']=release/'NOTICE'
    (release/'UPSTREAM_COMPONENTS.json').write_text(json.dumps(licenses,indent=2)+'\n')
    files['UPSTREAM_COMPONENTS.json']=release/'UPSTREAM_COMPONENTS.json'
    expected_large={p['file']:{'bytes':p['bytes'],'sha256':p['sha256']} for p in parts['parts']}
    expected_large['runtime.tar.gz']={'bytes':runtime['archive_bytes'],'sha256':runtime['archive_sha256']}
    # The Hub uploader hashes the large files itself. Compare its LFS readback
    # with these already-verified receipts rather than rereading 50 GB repeatedly.
    published={name:expected_large.get(name) or {'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in files.items()}
    assert all(path.stat().st_size==published[name]['bytes'] for name,path in files.items())
    for p in parts['parts']:
        assert published[p['file']]=={'bytes':p['bytes'],'sha256':p['sha256']}
    assert published['runtime.tar.gz']['sha256']==runtime['archive_sha256']
    descriptor={'schema':'apsp.lite.public-release/v1','repo':REPO,'companion_source_commit':archive['source_commit'],
        'qualified_core_source_sha256':offline['offline']['qualified_core_source_sha256'],'files':published,
        'download_threads_per_instance':12,'instances':2,'full_train_dev_candidates':0,
        'sdxl50_cross_container_drift':'unresolved; no bitwise parity claim','company_node_qualified':False}
    (release/'APSP_RELEASE.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    files['APSP_RELEASE.json']=release/'APSP_RELEASE.json'
    sums=''.join(f'{e["sha256"]}  {name}\n' for name,e in sorted(published.items()))
    (release/'SHA256SUMS').write_text(sums);files['SHA256SUMS']=release/'SHA256SUMS'
    readme=f'''---
license: other
license_name: component-specific-upstream-licenses
license_link: NOTICE
pretty_name: APSP Lite B200 runtime and companion assets
---
# APSP Lite B200 cu128

Research/runtime distribution. Two companion transport files, **{parts['parts'][0]['bytes']/1e9:.3f} GB each**, plus a separate4.572GB dependency runtime. The two parts are one losslessly compressed archive, not two independently extractable model bundles. Weights, model/scorer/tokenizer configs, common encoder, prepared prompt/task manifests, raw prompt-source metadata, source and provenance are included. No full generated training bank is present.

Use the exact commit returned by publication, never `main`, in download URLs. Company-node transport follows DSH `PROXY_DOWNLOAD_GUIDE.md` section6: curl, explicit proxy,12 simultaneous byte ranges per part, correct last-range size, bounded temporary downloads and byte resume. No Modal SDK is required on the target nodes.

On instanceA download part1, instanceB download part2. Each needs bash/curl/python3 and its existing proxy environment. Put outputs on shared persistent storage or transfer both verified parts to the same join node. Download and verify `download_part_hf.sh` and `COMPANION_PARTS.json` using their pinned hashes in `APSP_RELEASE.json` before executing. Then invoke `bash download_part_hf.sh COMPANION_PARTS.json 1 DEST {REPO} PINNED_REV` onA, and the same command with2 onB. `APSP_DOWNLOAD_THREADS` defaults12. No GPU is used by download/join.

Check `COMPANION_PARTS.SHA256SUMS`, then explicitly concatenate part01 followed by part02 and pipe through `gzip -dc` to recover `apsp-lite-companion-v1.tar`. The recovered tar SHA256 must be `{archive['archive_sha256']}`. Use `restore_companion.py restore --archive ... --target ... --sha256 ...` and source the generated `activate_companion.sh`. Restore `runtime.tar.gz` separately with `ensure_runtime.py` and `runtime-manifest.json`. Reserve adequate persistent disk space for ranges, parts, tar and restored payload (about250GB if retained together).

Engineering checks: companion full-file SHA readback PASS; both-part lossless roundtrip PASS; separate companion restore and offline CPU loader/encoder forward PASS. Original three-profile one-prompt×8seed B200 functional qualification and Ubuntu22.04 restore PASS. These do not qualify a company node or complete the dataset. SDXL50 cross-container PNG drift remains unresolved; Lightning/FLUX matched8/8 in that bounded check. Full110400-candidate collection,50-prompt drift study, independent backup and manual prompt QA remain pending. No controller result is claimed.

See `NOTICE`, `UPSTREAM_COMPONENTS.json` and `LICENSES/` before use. FLUX.1-dev remains under its upstream non-commercial license; public hosting does not remove component restrictions.
'''
    (release/'README.md').write_text(readme);files['README.md']=release/'README.md'
    published['README.md']={'bytes':files['README.md'].stat().st_size,'sha256':sha(files['README.md'])}
    descriptor['files']=dict(published)
    (release/'APSP_RELEASE.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    published['APSP_RELEASE.json']={'bytes':files['APSP_RELEASE.json'].stat().st_size,'sha256':sha(files['APSP_RELEASE.json'])}
    (release/'SHA256SUMS').write_text(''.join(f'{e["sha256"]}  {name}\n' for name,e in sorted(published.items())))
    # Root source archive/history were separately audited locally. No whole cache,
    # credential service, partial downloads or incomplete compression are uploaded.
    if api.repo_exists(REPO,repo_type='dataset'):
        existing=api.list_repo_files(REPO,repo_type='dataset')
        if set(existing)-{'.gitattributes'}:
            raise RuntimeError('nonempty existing repository; inspect before replacing')
    else:api.create_repo(REPO,repo_type='dataset',private=True)
    stage('upload_begin',files=len(files),bytes=sum(p.stat().st_size for p in files.values()))
    commit=api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in sorted(files.items())],commit_message='Publish verified B200 runtime and two companion parts with upstream notices')
    info=api.repo_info(REPO,repo_type='dataset',revision=commit.oid,files_metadata=True)
    siblings={s.rfilename:s for s in info.siblings}
    for name,path in files.items():
        s=siblings[name]
        assert s.size==path.stat().st_size
        lfs=s.lfs
        remote_sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        if remote_sha:assert remote_sha==(published[name]['sha256'] if name in published else sha(path))
    stage('authenticated_readback_pass',revision=commit.oid)
    api.update_repo_settings(REPO,repo_type='dataset',private=False)
    public=HfApi(token=False).repo_info(REPO,repo_type='dataset',revision=commit.oid,files_metadata=True)
    assert public.private is False
    probes=[]
    for name in ['SHA256SUMS',*[p['file'] for p in parts['parts']],'runtime.tar.gz']:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{commit.oid}/{name}'
        request=urllib.request.Request(url,headers={'Range':'bytes=0-0','User-Agent':'apsp-lite-release-verifier/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            data=response.read(2)
            probes.append({'file':name,'status':response.status,'range_supported':response.status==206 and len(data)==1})
        if name!='SHA256SUMS':assert probes[-1]['range_supported']
    small_verified=0
    for name,path in files.items():
        if name in expected_large:continue
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{commit.oid}/{name}'
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'apsp-lite-release-verifier/1.0'}),timeout=60) as response:
            body=response.read(8<<20)
        assert hashlib.sha256(body).hexdigest()==sha(path)
        small_verified+=1
    receipt={'pass':True,'repo':REPO,'repo_type':'dataset','revision':commit.oid,'public':True,'verified_files':len(files),'anonymous_small_files_sha256_verified':small_verified,'anonymous_range_probes':probes,'sha256sums_sha256':sha(release/'SHA256SUMS'),'download_helper_sha256':published['download_part_hf.sh']['sha256'],'parts_manifest_sha256':published['COMPANION_PARTS.json']['sha256'],'companion_source_commit':archive['source_commit'],'parts':parts['parts'],'runtime_sha256':runtime['archive_sha256'],'verification':'Read-back byte sizes/LFS SHA256, anonymous public repo access, all small-file SHA256 and real one-byte Range downloads. Company proxy/node still not exercised.'}
    (export/'hf-publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    stage('public_verified',**receipt)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        stage('failed',error_type=type(exc).__name__)
        raise SystemExit(1)
