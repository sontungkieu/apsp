"""CPU-only private companion packaging on the existing authorized Modal Volume."""
import json
import sys
import subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/apsp')
app = modal.App('apsp-lite-companion-v1')
volume = modal.Volume.from_name('apsp-lite-runtime-v1')
image = (modal.Image.debian_slim(python_version='3.11')
    .add_local_file(str(ROOT/'env/lite/companion.py'), '/opt/tools/companion.py')
    .add_local_file(str(ROOT/'env/lite/companion_parts.py'), '/opt/tools/companion_parts.py')
    .add_local_file(str(ROOT/'env/lite/profiles.json'), '/opt/tools/profiles.json'))


@app.function(image=image, cpu=2, memory=4096, timeout=300, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def inspect():
    sys.path.insert(0, '/opt/tools')
    import companion
    # Inventory only; source artifacts are added after a signed clean checkpoint.
    assets = json.loads(Path('/persist/assets.json').read_text())
    registry = json.loads(Path('/opt/tools/profiles.json').read_text())
    chosen = companion.selected_assets(assets, registry)
    by_repo, missing = {}, []
    for (repo, rev, name), sha in chosen.items():
        p = Path('/persist/hf/hub')/f'models--{repo.replace("/", "--")}'/'snapshots'/rev/name
        if not p.is_file(): missing.append([repo, rev, name]); continue
        row = by_repo.setdefault(repo, {'bytes': 0, 'files': 0})
        row['bytes'] += p.stat().st_size; row['files'] += 1
    data = {p.relative_to('/persist/text/lite-v1-20261019').as_posix(): p.stat().st_size for p in Path('/persist/text/lite-v1-20261019').rglob('*') if p.is_file()}
    report = {'pass': not missing, 'by_repo': by_repo, 'selected_model_bytes': sum(v['bytes'] for v in by_repo.values()), 'missing': missing, 'data_files': data}
    Path('/persist/export/companion-inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    volume.commit()
    return json.dumps(report, indent=2)


@app.function(image=image, cpu=1, memory=1024, timeout=60, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def preserve_incomplete():
    # Used only after the exact task app has stopped. Preserve evidence; no delete.
    partial = Path('/persist/export/apsp-lite-companion-v1.tar.partial')
    saved = Path('/persist/export/companion-pack-9f73118-incomplete.tar.partial')
    if partial.is_file():
        if saved.exists(): raise FileExistsError('earlier incomplete evidence already exists')
        size = partial.stat().st_size
        partial.rename(saved)
        Path('/persist/export/companion-incomplete-note.json').write_text(json.dumps({'bytes':size,'path':str(saved),'reason':'prior app stopped; rerun uses bounded 8 MiB buffered Volume writes'})+'\n')
        volume.commit()
        print('INCOMPLETE_PRESERVED='+str(size), flush=True)


@app.function(image=image, cpu=8, memory=8192, timeout=1800, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def split_two_parts():
    sys.path.insert(0,'/opt/tools')
    import companion_parts
    receipt=json.loads(Path('/persist/export/apsp-lite-companion-v1.receipt.json').read_text())
    if not receipt['pass']: raise ValueError('verified companion tar required')
    try:
        result=companion_parts.make_parts('/persist/export/apsp-lite-companion-v1.tar',receipt['archive_sha256'],'/persist/export/apsp-lite-companion-v1.tar.gz')
    finally: volume.commit()
    return json.dumps(result)


@app.function(image=image, cpu=1, memory=1024, timeout=60, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def preserve_serial_compression():
    old=Path('/persist/export/apsp-lite-companion-v1.tar.gz.compressing')
    saved=Path('/persist/export/companion-serial-compressing-incomplete.partial')
    if old.exists():
        if saved.exists(): raise FileExistsError('serial compression evidence already exists')
        old.rename(saved); volume.commit()


INPUT = ROOT/'outputs/apsp-lite-companion-input'
if modal.is_local() and INPUT.exists():
    pack_image = (image
        .add_local_file(str(INPUT/'source.tar.gz'), '/opt/input/source.tar.gz')
        .add_local_file(str(INPUT/'source.bundle'), '/opt/input/source.bundle')
        .add_local_file(str(INPUT/'metadata.json'), '/opt/input/metadata.json')
        .add_local_file(str(ROOT/'data/lite-v1/prepared/manifests/exclusion_index.jsonl'), '/opt/input/exclusion_index.jsonl'))

else:
    # Function names must also exist when Modal imports this module remotely.
    # Only the client attaches local inputs; the server uses hydrated image mounts.
    pack_image = image


@app.function(image=pack_image, cpu=2, memory=4096, timeout=3600, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def package():
    sys.path.insert(0, '/opt/tools')
    import companion
    entries = companion.inventory('/persist', '/opt/input/source.tar.gz', '/opt/input/source.bundle', '/opt/tools/profiles.json')
    extra = Path('/opt/input/exclusion_index.jsonl')
    entries['data/lite-v1/prepared/manifests/exclusion_index.jsonl'] = {'source': str(extra), 'bytes': extra.stat().st_size, 'sha256': companion.digest(extra)}
    metadata = json.loads(Path('/opt/input/metadata.json').read_text())
    for name, key in [('manifest.json', 'prompt_manifest_sha256'), ('tasks.jsonl', 'task_manifest_sha256'), ('exclusion_index.jsonl', 'exclusion_index_sha256')]:
        entries['data/lite-v1/prepared/manifests/'+name]['sha256'] = metadata[key]
    total = sum(e['bytes'] for e in entries.values())
    print('COMPANION_PACK_BEGIN='+json.dumps({'files': len(entries), 'bytes': total, 'source_commit': metadata['source_commit']}), flush=True)
    try:
        result = companion.pack(entries, '/persist/export/apsp-lite-companion-v1.tar', metadata)
    finally:
        volume.commit()
    print('COMPANION_RECEIPT='+json.dumps(result), flush=True)
    return json.dumps(result)

if modal.is_local() and INPUT.exists():
    sys.path.insert(0, str(ROOT/'env/lite'))
    import modal_runtime as qualified
    check_image = qualified.source_image
else:
    check_image = image

@app.function(image=check_image, cpu=2, memory=8192, timeout=3600, volumes={'/persist': volume}, max_containers=1, scaledown_window=2)
def check_restored():
    sys.path.insert(0, '/opt/apsp/env/lite')
    import companion
    archive = Path('/persist/export/apsp-lite-companion-v1.tar')
    receipt = json.loads(archive.with_suffix('.receipt.json').read_text())
    restored = '/persist/companion-restored-v1'
    try:
        result = companion.restore(archive, restored, receipt['archive_sha256'])
        print('COMPANION_RESTORE='+json.dumps(result), flush=True)
        env = {**__import__('os').environ, 'PYTHONPATH': restored+'/source', 'HF_HOME': restored+'/hf', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1'}
        subprocess.run(['/opt/apsp-runtime/bin/python', '/opt/apsp/env/lite/verify_companion_offline.py', '--root', restored], env=env, check=True, timeout=600)
        offline = json.loads(Path(restored+'/OFFLINE_LOADER_RECEIPT.json').read_text())
        Path('/persist/export/companion-offline-receipt.json').write_text(json.dumps({'restore': result, 'offline': offline}, indent=2)+'\n')
    finally:
        volume.commit()
