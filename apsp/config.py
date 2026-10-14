"""Single config loader for every module (Guide §3)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from apsp.contracts.hashing import sha256_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pilot.yaml"

REQUIRED = {
    "spec_version": "apsp-v1.1.1", "psp_commit": "590f59f58384169c719e431dc01d14006fb0fc0c",
    "model_repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "scheduler": "DDIM", "eta": 0.0, "steps": 64,
    "delta": 4, "cfg": 7.5, "negative_prompt": "", "height": 512, "width": 512, "grammar": "G-842-v1", "k0": 8,
    "keep": [4, 2], "reference_cuts": [4, 8], "seed_scheme": "apsp-v1-sha256-cpu", "scoring_path": "ir_pil_v1",
}


class ConfigError(ValueError):
    pass


def load_config(path: Optional[str | Path] = None, *, allow_override: bool = False) -> dict[str, Any]:
    """Load and check the scientific config. ``allow_override`` is only for explicitly labelled smoke configs."""
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not allow_override:
        bad = {k: (cfg.get(k), v) for k, v in REQUIRED.items() if cfg.get(k) != v}
        if bad:
            raise ConfigError(f"scientific config differs from Research Spec v1.1.1: {bad}")
    if cfg["steps"] % cfg["delta"] != 0:
        raise ConfigError("steps must be a multiple of delta")
    cfg["_path"] = str(p)
    return cfg


def scientific_config_hash(cfg: dict) -> str:
    return sha256_hex({k: v for k, v in cfg.items() if not k.startswith("_") and k != "paths"})


def n_checkpoints(cfg: dict) -> int:
    return cfg["steps"] // cfg["delta"]
