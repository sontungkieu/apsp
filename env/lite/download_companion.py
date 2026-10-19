"""Download one private Modal companion part with 12 HTTP workers and block resume.

Requires the existing Modal SDK 1.6.2 and an authorized profile on the target node.
Signed block URLs stay in memory; no download tickets or credentials are persisted.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

BLOCK=8<<20


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(BLOCK): h.update(chunk)
    return h.hexdigest()


async def resolve_blocks(path):
    # This SDK adapter is deliberately version-checked: its block API is private.
    if importlib.metadata.version('modal')!='1.6.2':
        raise RuntimeError('Modal SDK 1.6.2 required for this block adapter')
    from modal.volume import _Volume, _expected_block_lengths, BLOCK_SIZE
    from modal_proto import api_pb2
    volume=_Volume.from_name('apsp-lite-runtime-v1')
    await volume.hydrate()
    response=await volume.client._stub.VolumeGetFile2(api_pb2.VolumeGetFile2Request(volume_id=volume.object_id,path=path,client_pads_blocks=True))
    if response.start!=0 or BLOCK_SIZE!=BLOCK: raise RuntimeError('unexpected Modal block layout')
    lengths=_expected_block_lengths(response.start,response.len,len(response.get_urls))
    return response.len,list(zip(response.get_urls,lengths))


def download_blocks(blocks, size, expected_sha256, dest, workers=12):
    dest=Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if dest.stat().st_size==size and sha256(dest)==expected_sha256:
            return {'pass':True,'status':'already_verified','file':dest.name,'workers':workers}
        raise FileExistsError('destination exists with different content')
    lock_path=dest.with_name(dest.name+'.lock')
    fd=os.open(lock_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.close(fd)
    partial=dest.with_name(dest.name+'.partial')
    state_path=dest.with_name(dest.name+'.resume.json')
    try:
        state={'sha256':expected_sha256,'bytes':size,'block_size':BLOCK,'completed':{}}
        if state_path.exists():
            previous=json.loads(state_path.read_text())
            if any(previous.get(k)!=state[k] for k in ('sha256','bytes','block_size')):
                raise ValueError('resume state belongs to a different artifact')
            if not partial.exists(): raise ValueError('resume data is missing')
            state=previous
        elif partial.exists(): raise FileExistsError('partial has no matching resume state')
        with partial.open('r+b' if partial.exists() else 'x+b') as f: f.truncate(size)
        mutex=threading.Lock()
        def save_state():
            tmp=state_path.with_name(state_path.name+'.tmp')
            with tmp.open('w') as f:
                json.dump(state,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
            tmp.replace(state_path)
        # A saved completed block is skipped only after its local bytes are checked.
        with partial.open('rb') as f:
            for index,saved in list(state['completed'].items()):
                i=int(index)
                if not 0<=i<len(blocks): raise ValueError('invalid saved block index')
                f.seek(i*BLOCK)
                if hashlib.sha256(f.read(blocks[i][1])).hexdigest()!=saved:
                    del state['completed'][index]
        save_state()
        resumed=len(state['completed'])
        def fetch(i, url, length):
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=120) as response:
                        if response.status!=200: raise RuntimeError('block status must be 200')
                        data=response.read(length+1)
                    if len(data)>length: raise ValueError('oversized block')
                    # Modal's documented block protocol can omit trailing zeros.
                    data=data.ljust(length,b'\0')
                    with partial.open('r+b',buffering=BLOCK) as f:
                        f.seek(i*BLOCK); f.write(data); f.flush(); os.fsync(f.fileno())
                    digest=hashlib.sha256(data).hexdigest()
                    with mutex:
                        state['completed'][str(i)]=digest; save_state()
                        n=len(state['completed'])
                        if n%64==0 or n==len(blocks): print(f'BLOCKS={n}/{len(blocks)}',flush=True)
                    return
                except Exception:
                    if attempt==2: raise RuntimeError('block download failed; retry to resume') from None
                    time.sleep(.5*(attempt+1))
        failures=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(fetch,i,url,length) for i,(url,length) in enumerate(blocks) if str(i) not in state['completed']]
            for future in as_completed(futures):
                try: future.result()
                except Exception: failures+=1
        if failures: raise RuntimeError('incomplete download; rerun to resume verified blocks')
        if sha256(partial)!=expected_sha256:
            # Do not accept a truncated HTTP response padded with the wrong bytes.
            state['completed']={}; state['last_failure']='whole_file_sha256_mismatch'; save_state()
            raise ValueError('whole part SHA256 mismatch; next run will fetch all blocks')
        partial.replace(dest)
        state_path.unlink()
        return {'pass':True,'status':'downloaded_and_sha256_verified','file':dest.name,'bytes':size,'workers':workers,'resumed_blocks':resumed,'blocks':len(blocks)}
    finally:
        lock_path.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--part',type=int,choices=[1,2],required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--profile',default='billionaireproject8')
    ap.add_argument('--workers',type=int,choices=range(1,33),default=12)
    a=ap.parse_args()
    os.environ['MODAL_PROFILE']=a.profile
    manifest=json.loads(Path(a.manifest).read_text())
    part=manifest['parts'][a.part-1]
    if part['file']!=f'apsp-lite-companion-v1.tar.gz.part{a.part:02}':
        raise ValueError('unexpected companion part filename')
    size,blocks=asyncio.run(resolve_blocks('/export/'+part['file']))
    if size!=part['bytes'] or sum(n for _,n in blocks)!=size:
        raise ValueError('remote size differs from frozen parts manifest')
    print('DOWNLOAD_PART='+str(a.part)+' HTTP_WORKERS='+str(a.workers),flush=True)
    result=download_blocks(blocks,size,part['sha256'],Path(a.out_dir)/part['file'],a.workers)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        # HTTP/SDK exception strings can contain private signed URLs. Type only.
        print('Download failed: '+type(exc).__name__+'; existing partial/state preserved. Check profile, SDK1.6.2, free space and rerun to resume.',file=sys.stderr)
        raise SystemExit(1)
