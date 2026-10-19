"""Small, parent-bound fixup of the published curl helper and its hash manifests."""
import hashlib,json,os,shutil,urllib.request
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path('/persist/hf-public-release-v1')
    receipt_path=Path('/persist/export/hf-publication-receipt.json')
    previous=json.loads(receipt_path.read_text())
    api=HfApi()
    assert api.whoami()['name']=='codemaivanngu' and previous['public']
    repo=previous['repo']
    shutil.copyfile('/opt/apsp/env/lite/download_part_hf.sh',root/'download_part_hf.sh')
    descriptor=json.loads((root/'APSP_RELEASE.json').read_text())
    descriptor['files']['download_part_hf.sh']={'bytes':(root/'download_part_hf.sh').stat().st_size,'sha256':sha(root/'download_part_hf.sh')}
    (root/'APSP_RELEASE.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    entries={**descriptor['files'],'APSP_RELEASE.json':{'bytes':(root/'APSP_RELEASE.json').stat().st_size,'sha256':sha(root/'APSP_RELEASE.json')}}
    (root/'SHA256SUMS').write_text(''.join(f'{e["sha256"]}  {name}\n' for name,e in sorted(entries.items())))
    names=['download_part_hf.sh','APSP_RELEASE.json','SHA256SUMS']
    commit=api.create_commit(repo,repo_type='dataset',parent_commit=previous['revision'],operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(root/n)) for n in names],commit_message='Allow HF redirect headers during bounded curl Range probe')
    for name in names:
        url=f'https://huggingface.co/datasets/{repo}/resolve/{commit.oid}/{name}'
        with urllib.request.urlopen(url,timeout=60) as response:body=response.read(2<<20)
        assert hashlib.sha256(body).hexdigest()==sha(root/name)
    updated={**previous,'revision':commit.oid,'payload_parent_revision':previous['revision'],'download_helper_sha256':sha(root/'download_part_hf.sh'),'sha256sums_sha256':sha(root/'SHA256SUMS'),'helper_update':'1 MiB bounded probe permits HF302 bodies; still requires final206 and exactly1 byte. Three changed files anonymously SHA256 verified; large payload files inherited unchanged.'}
    receipt_path.write_text(json.dumps(updated,indent=2)+'\n')
    print('HF_PUBLIC_STAGE='+json.dumps({'stage':'helper_update_verified',**updated}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('HF_PUBLIC_STAGE='+json.dumps({'stage':'helper_update_failed','error_type':type(exc).__name__}),flush=True)
        raise SystemExit(1)
