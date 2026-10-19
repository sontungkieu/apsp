"""Integrity, private-cache selection, and restore tests; no models/dependencies downloaded."""
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('companion', Path(__file__).resolve().parents[2]/'env/lite/companion.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def make_archive(tmp_path):
    code = tmp_path/'code.tar.gz'
    with tarfile.open(code, 'w:gz') as tf:
        data = b'print("offline")\n'
        item = tarfile.TarInfo('hello.py'); item.size = len(data)
        tf.addfile(item, io.BytesIO(data))
    weight = tmp_path/'weight'; weight.write_bytes(b'pinned weights')
    entries = {name: {'source': str(p), 'bytes': p.stat().st_size, 'sha256': c.digest(p)} for name,p in [('source.tar.gz',code),('hf/hub/models--a--b/snapshots/abc/model.safetensors',weight)]}
    archive = tmp_path/'companion.tar'
    receipt = c.pack(entries, archive, {'source_commit': 'f'*40, 'runtime_archive_sha256': 'a'*64})
    return archive, receipt


def test_companion_roundtrip_and_preserve_target(tmp_path):
    archive, receipt = make_archive(tmp_path)
    assert receipt['archive_sha256'] == c.digest(archive)
    assert len(c.verify(archive)['files']) == 2
    target = tmp_path/'restore'
    assert c.restore(archive, target, c.digest(archive))['status'] == 'restored'
    assert (target/'source/hello.py').read_bytes() == b'print("offline")\n'
    assert 'HF_HUB_OFFLINE=1' in (target/'activate_companion.sh').read_text()
    assert c.restore(archive, target, c.digest(archive))['status'] == 'already_restored'
    with pytest.raises(ValueError): c.restore(archive, tmp_path/'bad', 'b'*64)
    other = tmp_path/'user'; other.mkdir()
    with pytest.raises(FileExistsError): c.restore(archive, other, c.digest(archive))
    with pytest.raises(FileExistsError): c.pack({}, archive, {})


def test_companion_rejects_untrusted_members_and_corruption(tmp_path):
    archive, _ = make_archive(tmp_path)
    data = bytearray(archive.read_bytes())
    at = data.index(b'pinned weights'); data[at] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match='checksum mismatch'): c.verify(archive)
    for index, (name, kind) in enumerate([('../escape', tarfile.REGTYPE), ('/absolute',tarfile.REGTYPE), ('hf/token',tarfile.SYMTYPE)]):
        bad = tmp_path/f'bad{index}.tar'
        with tarfile.open(bad, 'w') as tf:
            info = tarfile.TarInfo(name); info.type = kind
            if kind == tarfile.SYMTYPE: info.linkname = '/home/user/.env'
            tf.addfile(info)
        with pytest.raises(ValueError): c.verify(bad)


def test_companion_selection_omits_exports_and_duplicate_weights():
    registry = {'profiles': {'a': {'repo': 'base/model', 'adapter': {'repo': 'adapter/lora', 'filename': 'selected.safetensors'}}}, 'prompt_encoder': {'repo': 'prompt/encoder'}, 'scorers': {'pickscore': {'model_repo': 'score/model', 'processor_repo': 'score/processor'}}}
    files = {'base/model': ['model_index.json', 'unet/config.json','unet/model.safetensors','unet/model.onnx_data','vae_1_0/model.safetensors','raw.safetensors'], 'score/model': ['config.json','model.safetensors','pytorch_model.bin'], 'adapter/lora': ['selected.safetensors','other.safetensors'], 'unknown/repo': ['token']}
    assets = {repo: {'pass': True, 'assets': [{'repo':repo,'revision':'a'*40,'files_sha256':{n:'f'*64 for n in names}}]} for repo,names in files.items()}
    chosen = c.selected_assets(assets, registry)
    names = {key[2] for key in chosen}
    assert names == {'model_index.json','unet/config.json','unet/model.safetensors','config.json','model.safetensors','selected.safetensors'}
    assets['bad'] = {'pass': False}
    with pytest.raises(ValueError): c.selected_assets(assets, registry)


def test_companion_source_hash_mismatch_is_not_published(tmp_path):
    weight = tmp_path/'weights'; weight.write_bytes(b'wrong')
    entries = {'model': {'source':str(weight),'bytes':5,'sha256':'0'*64}}
    target = tmp_path/'bad.tar'
    with pytest.raises(ValueError, match='qualified source checksum mismatch'):
        c.pack(entries, target, {'source_commit':'a'*40})
    assert not target.exists() and target.with_name('bad.tar.partial').exists()


def test_modal_companion_functions_exist_on_remote_import(monkeypatch):
    # Remote imports have no local outputs/input directory. Decorated functions
    # must still be defined so Modal can hydrate the requested function by name.
    import sys
    from types import SimpleNamespace
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs): return cls()
        def add_local_file(self, *args, **kwargs): return self
    class App:
        def __init__(self, name): pass
        def function(self, **kwargs): return lambda fn: fn
    modal = SimpleNamespace(is_local=lambda:False, Image=Image, App=App,
        Volume=SimpleNamespace(from_name=lambda name:object()))
    monkeypatch.setitem(sys.modules, 'modal', modal)
    path = Path(__file__).resolve().parents[2]/'env/lite/modal_companion.py'
    spec = importlib.util.spec_from_file_location('remote_companion', path)
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    assert callable(remote.package) and callable(remote.inspect) and callable(remote.check_restored) and callable(remote.split_two_parts)


def test_two_parts_preserve_original_tar_and_obey_cap(tmp_path):
    import gzip
    spec=importlib.util.spec_from_file_location('companion_parts',Path(__file__).resolve().parents[2]/'env/lite/companion_parts.py')
    parts=importlib.util.module_from_spec(spec); spec.loader.exec_module(parts)
    source=tmp_path/'source.tar'; source.write_bytes(b'weights '*1234)
    prefix=tmp_path/'companion.tar.gz'
    receipt=parts.make_parts(source,c.digest(source),prefix,cap=1000)
    assert receipt['part_count']==2 and all(p['bytes']<=1000 for p in receipt['parts'])
    data=b''.join((tmp_path/p['file']).read_bytes() for p in receipt['parts'])
    assert gzip.decompress(data)==source.read_bytes()
    assert hashlib.sha256(data).hexdigest()==receipt['gzip_sha256']
    with pytest.raises(FileExistsError): parts.make_parts(source,c.digest(source),prefix,cap=1000)
