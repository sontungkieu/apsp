"""Task-local durable commits, hash validation, exclusive claims and backup."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from .contracts import file_hash


class TaskBusy(RuntimeError):
    pass


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def files_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "COMPLETE.json"}


def verify_complete(root, contract_hash=None):
    root = Path(root)
    marker = root / "COMPLETE.json"
    if not marker.is_file():
        return False
    m = json.loads(marker.read_text(encoding="utf-8"))
    if contract_hash is not None and m.get("contract_hash") != contract_hash:
        raise ValueError("existing task has a different contract; use a new release namespace")
    actual = files_manifest(root)
    if m.get("files_sha256") != actual:
        raise ValueError(f"durable checksum mismatch: {root}")
    return True


def commit_task(stage, dest, receipt):
    stage, dest = Path(stage), Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    hashes = files_manifest(stage)
    if not hashes:
        raise ValueError("cannot commit empty task")
    for name in hashes:
        with (stage / name).open("rb+") as f:
            os.fsync(f.fileno())
    atomic_json(stage / "COMPLETE.json", {**receipt, "files_sha256": hashes})
    fsync_dir(stage)
    os.rename(stage, dest)
    fsync_dir(dest.parent)
    verify_complete(dest, receipt["contract_hash"])


@contextmanager
def claim(root, task_id):
    locks = Path(root) / "claims"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / task_id
    try:
        lock.mkdir()  # atomic; never steal an existing claim
    except FileExistsError as exc:
        raise TaskBusy(task_id) from exc
    atomic_json(lock / "owner.json", {"pid": os.getpid(), "hostname": __import__("socket").gethostname()})
    try:
        yield
    finally:
        (lock / "owner.json").unlink()
        lock.rmdir()


def backup_task(source, destination):
    source, destination = Path(source), Path(destination)
    if not verify_complete(source):
        raise ValueError("source task is not complete")
    if destination.exists():
        if not verify_complete(destination):
            raise ValueError("backup exists but has no completion marker")
        if file_hash(source / "COMPLETE.json") != file_hash(destination / "COMPLETE.json"):
            raise ValueError("backup conflicts with source")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + f".backup-{os.getpid()}")
    shutil.copytree(source, stage)
    verify_complete(stage)
    for p in stage.rglob("*"):
        if p.is_file():
            with p.open("rb+") as f:
                os.fsync(f.fileno())
    fsync_dir(stage)
    os.rename(stage, destination)
    fsync_dir(destination.parent)
    verify_complete(destination)
