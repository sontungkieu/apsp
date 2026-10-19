"""Git bundle transport must preserve server edits and reject rollback/divergence."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('apsp_update_source', ROOT/'env/lite/update_source.py')
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def git(root, *args, input=None):
    return subprocess.check_output(['git', *args], cwd=root, input=input, text=True).strip()


def commit(root, parent=None):
    # Synthetic fixture commits, no signing configuration changes or credential use.
    tree = git(root, 'mktree', input='')
    body = f'tree {tree}\n' + (f'parent {parent}\n' if parent else '')
    body += 'author Fixture <fixture@example.invalid> 1700000000 +0000\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\n\n' + root.name + '\n'
    head = subprocess.check_output(['git','hash-object','-t','commit','-w','--stdin'],cwd=root,input=body.encode()).decode().strip()
    git(root, 'update-ref', 'refs/heads/main', head)
    return head


def package(root, path, head):
    git(root, 'bundle', 'create', str(path), 'refs/heads/main')
    return {'source_branch':'main', 'source_commit':head, 'github':updater.GITHUB,
            'bundle_file':'apsp.bundle', 'bundle_bytes':path.stat().st_size,
            'bundle_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_clone_ff_idempotence_dirty_divergence_and_hash(tmp_path):
    source=tmp_path/'source';source.mkdir();git(source,'init','--initial-branch=main')
    first=commit(source);bundle=tmp_path/'apsp.bundle';manifest=package(source,bundle,first)
    old=tmp_path/'old.bundle';shutil.copyfile(bundle,old)
    target=tmp_path/'server'
    assert updater.apply_bundle(bundle,manifest,target)==first
    assert updater.apply_bundle(bundle,manifest,target)==first
    (target/'operator.txt').write_text('preserve')
    with pytest.raises(ValueError,match='Dirty'):
        updater.apply_bundle(bundle,manifest,target)
    assert (target/'operator.txt').read_text()=='preserve'
    (target/'operator.txt').unlink()
    second=commit(source,first);bundle.unlink();manifest2=package(source,bundle,second)
    assert updater.apply_bundle(bundle,manifest2,target)==second
    with pytest.raises(ValueError,match='Diverged'):
        updater.apply_bundle(old,manifest,target)
    # A different history is rejected without moving server HEAD.
    other=tmp_path/'other';other.mkdir();git(other,'init','--initial-branch=main')
    unrelated=commit(other);third=commit(other,unrelated)
    bundle.unlink();diverged=package(other,bundle,third)
    with pytest.raises(ValueError,match='Diverged'):
        updater.apply_bundle(bundle,diverged,target)
    assert git(target,'rev-parse','HEAD')==second
    bad={**diverged,'bundle_sha256':'0'*64}
    with pytest.raises(ValueError,match='hash/size'):
        updater.apply_bundle(bundle,bad,target)
