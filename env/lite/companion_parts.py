"""Lossless transport: exactly two balanced parts, each <=30,000,000,000 bytes.

The underlying uncompressed companion remains bit-for-bit identical. No model
conversion or new qualification is performed by this transport-only operation.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

BLOCK = 8 << 20
CAP = 30_000_000_000


class HashWriter:
    def __init__(self, output): self.output, self.hash = output, hashlib.sha256()
    def write(self, data):
        n = self.output.write(data); self.hash.update(memoryview(data)[:n]); return n


class PartsReader:
    def __init__(self, paths):
        self.paths = iter(paths); self.current = None
    def read(self, n=-1):
        if n < 0: raise ValueError('bounded sequential reads only')
        chunks, needed = [], n
        while needed:
            if self.current is None:
                path = next(self.paths, None)
                if path is None: break
                self.current = Path(path).open('rb', buffering=BLOCK)
            data = self.current.read(needed)
            if data: chunks.append(data); needed -= len(data)
            else: self.current.close(); self.current = None
        return b''.join(chunks)
    def close(self):
        if self.current is not None: self.current.close()


def make_parts(archive, expected_tar_sha256, prefix, cap=CAP):
    archive, prefix = Path(archive), Path(prefix)
    final = [Path(str(prefix)+f'.part{i:02}') for i in (1,2)]
    pending = [Path(str(p)+'.partial') for p in final]
    compressed = Path(str(prefix)+'.compressing')
    if any(p.exists() for p in [*final,*pending,compressed]):
        raise FileExistsError('transport output already exists; preserve existing artifacts')
    print('COMPANION_GZIP_BEGIN', flush=True)
    with archive.open('rb',buffering=BLOCK) as src, compressed.open('xb',buffering=BLOCK) as dst:
        writer=HashWriter(dst)
        with gzip.GzipFile(filename='',mode='wb',compresslevel=1,mtime=0,fileobj=writer) as gz:
            done=0
            while chunk:=src.read(BLOCK):
                gz.write(chunk); done+=len(chunk)
                if done//(4<<30)!=(done-len(chunk))//(4<<30):
                    print('COMPANION_GZIP_INPUT_BYTES='+str(done),flush=True)
        dst.flush(); os.fsync(dst.fileno())
        gzip_sha=writer.hash.hexdigest()
    total=compressed.stat().st_size
    if total>2*cap: raise ValueError('lossless gzip exceeds two requested part caps')
    sizes=[(total+1)//2,total//2]
    parts=[]
    with compressed.open('rb',buffering=BLOCK) as src:
        for path,size in zip(pending,sizes):
            h=hashlib.sha256(); remaining=size
            with path.open('xb',buffering=BLOCK) as dst:
                while remaining:
                    chunk=src.read(min(BLOCK,remaining))
                    if not chunk: raise ValueError('truncated gzip staging file')
                    dst.write(chunk); h.update(chunk); remaining-=len(chunk)
                dst.flush(); os.fsync(dst.fileno())
            parts.append({'file':final[len(parts)].name,'bytes':size,'sha256':h.hexdigest()})
    print('COMPANION_PARTS_ROUNDTRIP_BEGIN',flush=True)
    reader=PartsReader(pending)
    decoded=hashlib.sha256(); decoded_bytes=0
    try:
        with gzip.GzipFile(fileobj=reader,mode='rb') as gz:
            while chunk:=gz.read(BLOCK): decoded.update(chunk); decoded_bytes+=len(chunk)
    finally: reader.close()
    if decoded.hexdigest()!=expected_tar_sha256 or decoded_bytes!=archive.stat().st_size:
        raise ValueError('two-part lossless roundtrip differs from qualified companion tar')
    for src,dst in zip(pending,final): src.rename(dst)
    compressed.unlink()  # only this function's verified staging gzip, not the source tar
    receipt={'pass':True,'format':'concatenate part01+part02 to recover one gzip stream','compression':'gzip level1 lossless','parts':parts,'part_count':2,'max_part_bytes':cap,'compressed_bytes':total,'gzip_sha256':gzip_sha,'original_tar_sha256':expected_tar_sha256,'original_tar_bytes':decoded_bytes,'verification':'Both parts were streamed through gzip CRC validation and decoded SHA256 equals original verified tar.','join_command':f'cat {final[0].name} {final[1].name} | gzip -dc > apsp-lite-companion-v1.tar'}
    Path(str(prefix)+'.parts.json').write_text(json.dumps(receipt,indent=2)+'\n')
    Path(str(prefix)+'.SHA256SUMS').write_text(''.join(f"{p['sha256']}  {p['file']}\n" for p in parts))
    print('COMPANION_PARTS_JSON='+json.dumps(receipt),flush=True)
    return receipt
