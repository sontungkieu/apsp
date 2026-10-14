"""Canonical, type-tagged hashing (Implementation Guide §10).

Rules enforced here:
- types are preserved: ``1``, ``1.0``, ``"1"`` and ``True`` hash differently;
- dict keys must be ``str``; two keys that collide after NFC normalisation are rejected
  (values are never silently normalised);
- non-finite floats, sets, and unsupported types are rejected;
- lists/tuples keep their order (both are tagged as sequences);
- bodies may not contain the hash they define (no self reference) nor absolute paths,
  timestamps or free-form notes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

HASH_KINDS = ("EPH", "BMH", "GCH", "CPH", "CCH")
FORBIDDEN_BODY_KEYS = frozenset({"created_at", "timestamp", "notes", "note", "comment", "wall_time"})


class HashBodyError(ValueError):
    """Raised when a value cannot be canonically hashed."""


def _tag(obj: Any, path: str) -> Any:
    # bool must be tested before int (bool is a subclass of int)
    if obj is None:
        return ["n"]
    if isinstance(obj, (bool, np.bool_)):
        return ["b", bool(obj)]
    if isinstance(obj, (int, np.integer)):
        return ["i", str(int(obj))]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            raise HashBodyError(f"non-finite float at {path}")
        return ["f", repr(f)]
    if isinstance(obj, str):
        return ["s", obj]
    if isinstance(obj, bytes):
        return ["y", obj.hex()]
    if isinstance(obj, (list, tuple)):
        return ["l", [_tag(v, f"{path}[{i}]") for i, v in enumerate(obj)]]
    if isinstance(obj, Mapping):
        items = []
        seen_nfc: dict[str, str] = {}
        for k in obj.keys():
            if not isinstance(k, str):
                raise HashBodyError(f"non-string dict key {k!r} at {path}")
            nfc = unicodedata.normalize("NFC", k)
            if nfc in seen_nfc and seen_nfc[nfc] != k:
                raise HashBodyError(f"keys {seen_nfc[nfc]!r} and {k!r} collide under NFC at {path}")
            seen_nfc[nfc] = k
        for k in sorted(obj.keys()):
            items.append([k, _tag(obj[k], f"{path}.{k}")])
        return ["d", items]
    if isinstance(obj, (set, frozenset)):
        raise HashBodyError(f"unordered set at {path}; convert to a sorted list explicitly")
    if isinstance(obj, np.ndarray):
        raise HashBodyError(f"ndarray at {path}; hash array bytes with array_sha256() instead")
    raise HashBodyError(f"unsupported type {type(obj).__name__} at {path}")


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic byte serialisation with explicit type tags."""
    return json.dumps(_tag(obj, "$"), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """Hash of dtype, shape and C-contiguous bytes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def code_sha(paths: Iterable[str | os.PathLike], root: str | os.PathLike) -> str:
    """Hash of code files, keyed by POSIX path relative to ``root`` (no absolute paths).

    Line endings are normalised to LF so a Windows checkout and a Linux checkout of the
    same commit produce the same code hash.
    """
    root = Path(root).resolve()
    entries = []
    for p in sorted({Path(p).resolve() for p in paths}):
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes().replace(b"\r\n", b"\n")
        entries.append([rel, hashlib.sha256(data).hexdigest()])
    entries.sort()
    return sha256_hex(entries)


def package_code_sha(subpackages: Iterable[str]) -> str:
    """Hash all ``*.py`` files of the given ``apsp`` sub-packages (behaviour code hash)."""
    pkg_root = Path(__file__).resolve().parents[1]
    repo_root = pkg_root.parent
    files: list[Path] = []
    for sub in subpackages:
        d = pkg_root / sub if sub else pkg_root
        if d.is_file():
            files.append(d)
        else:
            files.extend(sorted(d.rglob("*.py")))
    return code_sha(files, repo_root)


def _check_body(kind: str, body: Mapping[str, Any]) -> None:
    def walk(o: Any, path: str) -> None:
        if isinstance(o, Mapping):
            for k, v in o.items():
                if k in FORBIDDEN_BODY_KEYS:
                    raise HashBodyError(f"forbidden key {k!r} in {kind} body at {path}")
                if k.upper() == kind:
                    raise HashBodyError(f"{kind} body must not contain its own hash ({path}.{k})")
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            if os.path.isabs(o) and (o.startswith("/") or (len(o) > 2 and o[1] == ":")):
                raise HashBodyError(f"absolute path {o!r} in {kind} body at {path}")

    walk(body, "$")


def provenance_hash(kind: str, body: Mapping[str, Any]) -> str:
    """One of the five provenance hashes. ``kind`` is prepended so bodies of different kinds never collide."""
    if kind not in HASH_KINDS:
        raise HashBodyError(f"unknown hash kind {kind}")
    _check_body(kind, body)
    return sha256_hex({"kind": kind, "body": dict(body)})
