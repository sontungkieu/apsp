"""Preregistration and data-access guards (Spec §11.5, Guide §8 CX-8).

Agents create proposals only. A signed marker is a JSON file written by research under
``docs/PREREG_SIGNED/PREREG-<X>.json`` with fields ``prereg``, ``signed_by``, ``role == "research"``,
``proposal_sha256`` (sha256 of the proposal file it approves) and ``proposal_path``.
No code in this repository writes signed markers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .hashing import file_sha256, sha256_hex

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNED_DIR_ENV = "APSP_PREREG_DIR"


class PreregGuardError(PermissionError):
    pass


def signed_dir(root: Optional[Path] = None) -> Path:
    if os.environ.get(SIGNED_DIR_ENV):
        return Path(os.environ[SIGNED_DIR_ENV])
    return (root or REPO_ROOT) / "docs" / "PREREG_SIGNED"


def is_signed(which: str, root: Optional[Path] = None) -> bool:
    try:
        _check_marker(which, root)
        return True
    except PreregGuardError:
        return False


def _check_marker(which: str, root: Optional[Path]) -> dict:
    p = signed_dir(root) / f"PREREG-{which}.json"
    if not p.exists():
        raise PreregGuardError(f"PREREG-{which} is not signed ({p} missing). Agents may not sign; research must.")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("prereg") != which or d.get("role") != "research" or not d.get("signed_by"):
        raise PreregGuardError(f"PREREG-{which} marker malformed or not signed by research")
    prop = d.get("proposal_path")
    if not prop:
        raise PreregGuardError(f"PREREG-{which} marker lacks proposal_path")
    pp = (root or REPO_ROOT) / prop
    if not pp.exists() or file_sha256(pp) != d.get("proposal_sha256"):
        raise PreregGuardError(f"PREREG-{which} proposal hash mismatch: the approved proposal changed")
    return d


def require_signed(which: str, root: Optional[Path] = None) -> dict:
    return _check_marker(which, root)


def require_marker(name: str, root: Optional[Path] = None) -> dict:
    """Generic research marker such as ``G2_EVALUATED`` (needed before real RL training)."""
    p = signed_dir(root) / f"{name}.json"
    if not p.exists():
        raise PreregGuardError(f"research marker {name} missing ({p})")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("role") != "research" or not d.get("signed_by"):
        raise PreregGuardError(f"marker {name} not signed by research")
    return d


# split -> PREREG that must be signed before *outcomes* on that split are read
_SPLIT_RULES = {
    "syn_train_fit": None, "syn_train_dev": None, "syn_eval": None,  # synthetic fixtures
    "smoke": None, "technical_smoke": None, "calibration": None,
    "train_fit": "A", "train_dev": "A",   # bank collection on research splits needs A
    "val": "B", "val_ext": "B",
    "test": "C", "geneval": "C",
}


def require_split_access(split: str, purpose: str, root: Optional[Path] = None) -> None:
    if split not in _SPLIT_RULES:
        raise PreregGuardError(f"unknown split {split}")
    need = _SPLIT_RULES[split]
    if need is not None:
        try:
            require_signed(need, root)
        except PreregGuardError as e:
            raise PreregGuardError(f"access to split '{split}' for '{purpose}' refused: {e}") from e


class ValAccessLog:
    """``val_access.jsonl``: at most two looks; a contrast's config may not change after the first look."""

    MAX_LOOKS = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def record(self, *, contrast_id: str, config_hash: str, look: int, cph: str) -> None:
        if look not in (1, 2):
            raise PreregGuardError(f"look {look} not allowed (max {self.MAX_LOOKS})")
        prev = [e for e in self.entries() if e["contrast_id"] == contrast_id]
        if prev and any(e["config_hash"] != config_hash or e["cph"] != cph for e in prev):
            raise PreregGuardError(f"config/CPH of contrast {contrast_id} changed after first look")
        looks = {e["look"] for e in prev}
        if look in looks:
            raise PreregGuardError(f"look {look} already used for {contrast_id}")
        if look == 2 and 1 not in looks:
            raise PreregGuardError("look 2 requires look 1")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"contrast_id": contrast_id, "config_hash": config_hash, "look": look, "cph": cph},
                                sort_keys=True) + "\n")


def proposal_digest(obj) -> str:
    return sha256_hex(obj)
