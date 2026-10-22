import io
import json
import tarfile
import hashlib
import pytest
from apsp.lite2.node import gpu_inventory
from apsp.lite2.pack import restore
from apsp.lite2.worker import initialize
from apsp.lite.prompts import write_jsonl


def test_busy_or_duplicate_gpu_assignment_is_rejected(monkeypatch):
    cards=[f'GPU-{i}' for i in range(8)]
    calls=[]
    def query(args,**kwargs):
        calls.append(args)
        if '--query-gpu=uuid,name' in args:return '\n'.join(f'{u}, NVIDIA B200' for u in cards)
        return 'GPU-3\n'
    monkeypatch.setattr('subprocess.check_output',query)
    with pytest.raises(ValueError,match='occupied'):gpu_inventory(cards)
    assert len(calls)==2 and all(c[0]=='nvidia-smi' for c in calls)
    with pytest.raises(ValueError,match='unique'):gpu_inventory([cards[0]]*8)


def test_changed_manifest_cannot_resume_namespace(tmp_path):
    path=tmp_path/'manifest.jsonl';write_jsonl(path,[{'frozen':'first'}])
    initialize(tmp_path/'new-lite2',path)
    path.write_text('{"frozen":"changed"}\n')
    with pytest.raises(ValueError,match='contract differs'):initialize(tmp_path/'new-lite2',path)


def test_archive_traversal_never_writes_outside_restore(tmp_path):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        member=tarfile.TarInfo('../outside');member.size=3
        tar.addfile(member,io.BytesIO(b'bad'))
    data=buffer.getvalue();parts=tmp_path/'parts';parts.mkdir()
    (parts/'part').write_bytes(data)
    (parts/'PARTS.json').write_text(json.dumps({'format':'streaming-tar-parts/v1',
        'parts':[{'file':'part','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}],
        'total_archive_bytes':len(data),'release_manifest_sha256':'unused'}))
    with pytest.raises(ValueError,match='Unsafe'):restore(parts,tmp_path/'restore')
    assert not (tmp_path/'outside').exists()
