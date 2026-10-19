"""CPU-only identity, checkpoint and profile contracts for Lite v1."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

SEED_SCHEME = "apsp-v2-sha256-cpu"
TRAIN_PROFILES = ("sdxl_lightning8", "hyper_flux8", "sdxl_base50")
QUOTAS = {
    "train": {"diffusiondb": 1000, "coco_captions": 400, "structured": 600},
    "dev": {"diffusiondb": 150, "coco_captions": 60, "structured": 90},
    "test_id": {"diffusiondb": 150, "coco_captions": 60, "structured": 90},
    "test_shift": {"parti": 200},
    "calibration": {"diffusiondb": 25, "coco_captions": 10, "structured": 15},
}


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def uid(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def noise_seed(prompt_uid, bank_id, seed_idx):
    if bank_id not in (0, 1) or seed_idx not in range(8):
        raise ValueError("offline bank_id must be 0/1 and seed_idx 0..7")
    msg = f"apsp-v2|offline|{prompt_uid}|{bank_id}|{seed_idx}".encode()
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "little") & ((1 << 63) - 1)


def checkpoints(steps, count):
    if not 1 <= count <= steps:
        raise ValueError("require 1 <= checkpoints <= steps")
    return [(j * steps + count - 1) // count for j in range(1, count + 1)]


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "env/lite/profiles.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    for name, p in registry["profiles"].items():
        if not re.fullmatch(r"[0-9a-f]{40}", p["revision"]):
            raise ValueError(f"unpinned base: {name}")
        if p.get("adapter") and not re.fullmatch(r"[0-9a-f]{40}", p["adapter"]["revision"]):
            raise ValueError(f"unpinned adapter: {name}")
        checkpoints(p["steps"], p["checkpoints"])
    return registry
