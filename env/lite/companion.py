"""Standard-library companion archive: pinned assets/data/source, no runtime/secrets.

Only regular files are archived. HF snapshot symlinks are resolved to regular files,
so an extracted cache never depends on the original Modal Volume's blob directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 << 20): h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('unsafe companion member')
    return p


def selected_assets(assets, registry):
    """Select only the exact loader inputs; omit alternate exports/duplicate weights."""
    selected = {}
    generators = {p['repo'] for p in registry['profiles'].values()}
    adapters = {(p['adapter']['repo'], p['adapter']['filename']) for p in registry['profiles'].values() if p.get('adapter')}
    pe = registry['prompt_encoder']['repo']
    ps = registry['scorers']['pickscore']
    prefixes = ('text_encoder/', 'text_encoder_2/', 'tokenizer/', 'tokenizer_2/', 'scheduler/', 'unet/', 'vae/', 'transformer/')
    for job in assets.values():
        if not job.get('pass'):
            raise ValueError('source asset staging did not pass')
        for asset in job['assets']:
            repo, rev = asset['repo'], asset['revision']
            files = asset['files_sha256']
            for name, sha in files.items():
                keep = False
                if repo in generators:
                    keep = name == 'model_index.json' or (name.startswith(prefixes) and name.endswith(('.json', '.safetensors', '.txt', '.model')))
                elif (repo, name) in adapters:
                    keep = True
                elif repo == pe:
                    keep = name.startswith(('text_encoder/', 'tokenizer/')) and name.endswith(('.json', '.safetensors', '.txt', '.model'))
                elif repo == 'THUDM/ImageReward':
                    keep = name in ('ImageReward.pt', 'med_config.json')
                elif repo == 'google-bert/bert-base-uncased':
                    keep = name in ('vocab.txt', 'tokenizer_config.json', 'config.json')
                elif repo == ps['model_repo']:
                    keep = name in ('config.json', 'model.safetensors') or (name == 'pytorch_model.bin' and 'model.safetensors' not in files)
                elif repo == ps['processor_repo']:
                    keep = name.endswith(('.json', '.txt', '.model'))
                if keep:
                    safe_name(name)
                    key = (repo, rev, name)
                    if key in selected and selected[key] != sha:
                        raise ValueError('conflicting pinned asset checksum')
                    selected[key] = sha
    return selected


def inventory(persist, source, bundle, registry_path):
    persist = Path(persist).resolve()
    assets = json.loads((persist/'assets.json').read_text())
    registry = json.loads(Path(registry_path).read_text())
    entries = {}

    def add(src, name, expected=None):
        safe_name(name)
        src = Path(src)
        if not src.is_file(): raise FileNotFoundError(str(src))
        entries[name] = {'source': str(src), 'bytes': src.stat().st_size, 'sha256': expected}

    for (repo, rev, name), sha in sorted(selected_assets(assets, registry).items()):
        rel = f'hf/hub/models--{repo.replace("/", "--")}/snapshots/{rev}/{name}'
        src = persist/rel
        if not src.resolve().is_relative_to(persist/'hf'):
            raise ValueError('asset symlink leaves the private HF cache')
        add(src, rel, sha)
    data_root = persist/'text/lite-v1-20261019'
    data_manifest = json.loads((data_root/'manifests/manifest.json').read_text())
    raw_hashes = data_manifest['source_hashes']
    for folder in ['manifests', 'raw']:
        for p in sorted((data_root/folder).rglob('*')):
            if p.is_file():
                expected = data_manifest['files_sha256'].get(p.name) if folder == 'manifests' else None
                if p.name == 'annotations_trainval2017.zip': expected = raw_hashes['coco2017_official_annotations']
                if p.name.startswith('parti-') and p.suffix == '.tsv': expected = raw_hashes['parti_tsv']
                if p.name == 'geneval.jsonl': expected = raw_hashes['geneval']
                add(p, f'data/lite-v1/prepared/{folder}/{p.relative_to(data_root/folder).as_posix()}', expected)
    rev = data_manifest['source_revisions']['diffusiondb']
    rel = f'hf/hub/datasets--poloclub--diffusiondb/snapshots/{rev}/metadata.parquet'
    add(persist/rel, rel, data_manifest['source_hashes']['diffusiondb'])
    add(source, 'source.tar.gz', digest(source))
    add(bundle, 'source.bundle', digest(bundle))
    add(persist/'assets.json', 'provenance/assets.json')
    add(persist/'export/manifest.json', 'provenance/runtime-manifest.json')
    for name in ['qualification/summary.json', 'restored-qualification/restore-receipt.json', 'cross_container_pixels.json']:
        add(persist/name, 'provenance/'+name.replace('/', '_'))
    if not entries or not any(k.endswith('/tasks.jsonl') for k in entries):
        raise ValueError('prepared task manifest missing')
    return entries


class HashReader:
    def __init__(self, stream): self.stream, self.hash = stream, hashlib.sha256()
    def read(self, size=-1):
        data = self.stream.read(size); self.hash.update(data); return data


class HashWriter:
    def __init__(self, stream): self.stream, self.hash = stream, hashlib.sha256()
    def write(self, data):
        n = self.stream.write(data)
        self.hash.update(memoryview(data)[:n])
        return n


def verify(archive):
    seen, meta = {}, None
    with tarfile.open(archive, 'r|') as tf:
        for m in tf:
            safe_name(m.name)
            if not m.isfile() or m.name in seen:
                raise ValueError('companion requires unique regular files')
            f = tf.extractfile(m)
            h = hashlib.sha256()
            body = bytearray() if m.name == 'COMPANION_MANIFEST.json' else None
            if body is not None and m.size > 16 << 20: raise ValueError('manifest too large')
            while chunk := f.read(8 << 20):
                h.update(chunk)
                if body is not None: body.extend(chunk)
            seen[m.name] = {'sha256': h.hexdigest(), 'bytes': m.size}
            if body is not None: meta = json.loads(body)
    if meta is None: raise ValueError('companion manifest missing')
    expected = meta['files']
    if set(seen) != set(expected) | {'COMPANION_MANIFEST.json'}:
        raise ValueError('archive file set differs from manifest')
    for name, entry in expected.items():
        if seen[name] != entry: raise ValueError('companion content checksum mismatch: '+name)
    return meta


def pack(entries, archive, metadata):
    """Fail before publication if any source changes or differs from qualified hashes."""
    archive = Path(archive)
    if archive.exists(): raise FileExistsError('refusing to overwrite companion archive')
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name+'.partial')
    if partial.exists(): raise FileExistsError('partial from earlier attempt requires inspection')
    manifest = {**metadata, 'schema': 'apsp.lite.companion/v1', 'files': {}}
    # Modal Volume is a network filesystem: batch writes rather than emitting
    # tarfile's default 10 KiB stream blocks as individual filesystem writes.
    with partial.open('xb', buffering=8 << 20) as output:
        writer = HashWriter(output)
        with tarfile.open(fileobj=writer, mode='w|', bufsize=8 << 20, copybufsize=8 << 20, format=tarfile.PAX_FORMAT) as tf:
            for i, (name, e) in enumerate(sorted(entries.items())):
                safe_name(name)
                info = tarfile.TarInfo(name); info.size = e['bytes']; info.mode = 0o644
                with Path(e['source']).open('rb') as f:
                    reader = HashReader(f)
                    tf.addfile(info, reader)
                    sha = reader.hash.hexdigest()
                if e['sha256'] and sha != e['sha256']:
                    raise ValueError('qualified source checksum mismatch: '+name)
                manifest['files'][name] = {'bytes': info.size, 'sha256': sha}
                print(json.dumps({'file': i+1, 'total': len(entries), 'path': name, 'bytes': info.size}), flush=True)
            body = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
            info = tarfile.TarInfo('COMPANION_MANIFEST.json'); info.size = len(body); info.mode = 0o644
            tf.addfile(info, io.BytesIO(body))
        output.flush(); os.fsync(output.fileno())
        sha = writer.hash.hexdigest()
    print('COMPANION_VERIFY_BEGIN', flush=True)
    verified = verify(partial)
    if verified != manifest: raise ValueError('manifest readback differs')
    partial.replace(archive)
    receipt = {'pass': True, 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
        'archive_sha256': sha, 'files': len(entries)+1, 'payload_bytes': sum(e['bytes'] for e in entries.values()),
        'source_commit': metadata['source_commit'], 'runtime_archive_sha256': metadata['runtime_archive_sha256'],
        'verification': 'full sequential tar readback and every file SHA256',
        'models_offline_load': 'pending_separate_cpu_check', 'full_bank_collected': False}
    archive.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    archive.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    archive.with_suffix('.tar.sha256').write_text(f'{sha}  {archive.name}\n')
    return receipt


def restore(archive, target, expected_sha):
    """Verify first, atomically publish into a new directory, preserve existing files."""
    archive, target = Path(archive), Path(target).absolute()
    if digest(archive) != expected_sha: raise ValueError('archive SHA256 mismatch')
    marker = target/'RESTORED_COMPANION.json'
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text()).get('archive_sha256') == expected_sha:
            return {'status': 'already_restored', 'target': str(target), 'content_rechecked': False}
        raise FileExistsError('target exists without this companion marker')
    meta = verify(archive)
    stage = target.with_name(target.name+f'.staging-{os.getpid()}-{time.time_ns()}')
    stage.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r|') as tf:
        for m in tf:
            rel = safe_name(m.name)
            if not m.isfile(): raise ValueError('regular files only')
            dest = stage/str(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f: shutil.copyfileobj(tf.extractfile(m), f, 8 << 20)
    # Source archive is generated by git archive at a recorded clean commit.
    source = stage/'source'; source.mkdir()
    with tarfile.open(stage/'source.tar.gz', 'r:gz') as tf:
        for m in tf:
            rel = safe_name(m.name)
            dest = source/str(rel)
            if m.isdir(): dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as f: shutil.copyfileobj(tf.extractfile(m), f, 8 << 20)
                dest.chmod(0o755 if m.mode & 0o111 else 0o644)
            else: raise ValueError('source archive links/devices are forbidden')
    # Absolute root is calculated by Bash at source-time, not baked into the archive.
    (stage/'activate_companion.sh').write_text('''# Source this file from bash after restoring the dependency runtime.
APSP_COMPANION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export APSP_COMPANION_ROOT
export HF_HOME="$APSP_COMPANION_ROOT/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$APSP_COMPANION_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
export APSP_SOURCE="$APSP_COMPANION_ROOT/source"
export APSP_TASKS="$APSP_COMPANION_ROOT/data/lite-v1/prepared/manifests/tasks.jsonl"
export APSP_PYTHON=/opt/apsp-runtime/bin/python
''')
    (stage/'RESTORED_COMPANION.json').write_text(json.dumps({'archive_sha256': expected_sha, 'source_commit': meta['source_commit']}, indent=2)+'\n')
    if target.exists(): raise FileExistsError('target appeared during restore')
    stage.rename(target)
    return {'status': 'restored', 'target': str(target), 'files_verified': len(meta['files']), 'source_commit': meta['source_commit']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('verify'); p.add_argument('--archive', required=True)
    p = sub.add_parser('restore'); p.add_argument('--archive', required=True); p.add_argument('--target', required=True); p.add_argument('--sha256', required=True)
    a = ap.parse_args()
    if a.command == 'restore': result = restore(a.archive, a.target, a.sha256)
    else:
        m = verify(a.archive); result = {'pass': True, 'files': len(m['files']), 'source_commit': m['source_commit']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
