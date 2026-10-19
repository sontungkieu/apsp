"""Idempotent path-preserving restore; never modifies an unmanaged /opt prefix.

python3 ensure_runtime.py --archive runtime.tar.gz --manifest manifest.json
Model/data/source remain outside the dependency archive. Run qualification on each node.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import posixpath
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PREFIXES = ("opt/apsp-runtime", "opt/apsp-python")


def under_prefix(name):
    return any(name == p or name.startswith(p + "/") for p in PREFIXES)


def validate_members(members):
    for member in members:
        name = member.name
        if name.startswith("/") or ".." in PurePosixPath(name).parts or not under_prefix(name):
            raise ValueError(f"unsafe archive member: {name}")
        if member.isdev() or member.isfifo() or not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported archive member: {name}")
        if member.issym() or member.islnk():
            target = member.linkname
            if target.startswith("/"):
                resolved = posixpath.normpath(target).lstrip("/")
            elif member.islnk():
                resolved = posixpath.normpath(target)
            else:
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            if not under_prefix(resolved):
                raise ValueError(f"link escapes runtime prefixes: {name}")
    # No file may be extracted through a symlink directory (including chained links).
    links = {m.name for m in members if m.issym() or m.islnk()}
    for m in members:
        if any(str(parent) in links for parent in PurePosixPath(m.name).parents):
            raise ValueError(f"member traverses an archive link: {m.name}")


def main(argv=None):
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cpu-only", action="store_true", help="install/import only; explicitly not GPU qualification")
    a = ap.parse_args(argv)
    archive, manifest = Path(a.archive).resolve(), json.loads(Path(a.manifest).read_text())
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""): h.update(chunk)
    expected = manifest["archive_sha256"]
    if h.hexdigest() != expected or archive.stat().st_size != manifest["archive_bytes"]:
        raise ValueError("archive checksum or size mismatch")
    with Path("/opt/.apsp-lite-restore.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prefixes = [Path("/") / p for p in PREFIXES]
        if any(p.exists() for p in prefixes):
            if not all((p / ".apsp-runtime-sha256").is_file() and (p / ".apsp-runtime-sha256").read_text().strip() == expected for p in prefixes):
                raise ValueError("existing unmanaged/incomplete/different prefixes preserved; choose another node or explicit recovery")
        else:
            stage = Path(tempfile.mkdtemp(prefix=".apsp-lite-", dir="/opt"))
            with tarfile.open(archive, "r:gz") as tf:
                members = tf.getmembers()
                validate_members(members)
                # Members and all link destinations have been constrained above.
                # Validate links ourselves because portable venv links intentionally
                # point at their final absolute /opt prefixes, not the staging root.
                if "filter" in inspect.signature(tf.extractall).parameters:
                    tf.extractall(stage, members=members, filter="fully_trusted")
                else:
                    tf.extractall(stage, members=members)
            for name, prefix in zip(PREFIXES, prefixes):
                staged = stage / name
                (staged / ".apsp-runtime-sha256").write_text(expected + "\n")
                os.rename(staged, prefix)
            # Keep any failed staging directory for diagnosis; successful stage is empty.
            (stage / "opt").rmdir()
            stage.rmdir()
        code = "import torch,diffusers,transformers,peft; print(torch.__version__,torch.version.cuda)"
        if not a.cpu_only:
            code += "; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0)==(10,0); assert bool(torch.isfinite(torch.ones(1,device='cuda')+1).all())"
        subprocess.run(["/opt/apsp-runtime/bin/python", "-c", code], check=True)
    print("APSP_RUNTIME_CPU_RESTORE_ONLY; GPU_NOT_QUALIFIED" if a.cpu_only else
          "APSP_RUNTIME_RESTORED; run node-specific model/scorer qualification next")


if __name__ == "__main__":
    main()
