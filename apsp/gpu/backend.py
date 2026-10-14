"""Backend construction shared by GPU CLIs, plus the Execution Profile Hash (EPH) body."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from apsp.config import REPO_ROOT, load_config
from apsp.contracts.hashing import file_sha256, package_code_sha, provenance_hash
from apsp.gpu.runtime import configure_determinism, require_cuda

ENV_LOCK = REPO_ROOT / "env" / "requirements-gpu-linux.txt"
GENERATION_CODE = ["gpu/engine.py", "gpu/hooks.py", "gpu/decode.py", "gpu/scoring.py", "gpu/validity.py",
                   "gpu/model.py", "gpu/collect_bank.py", "contracts/identity.py"]


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--fake-backend", action="store_true",
                    help="TEST ONLY: deterministic CPU stand-ins; outputs labelled backend=fake, never research data")
    ap.add_argument("--revision", default=None, help="model revision (defaults to config model_revision, else resolves main)")
    ap.add_argument("--preview-dtype", choices=["fp16", "fp32"], default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--rm-device", choices=["cuda", "cpu"], default="cuda",
                    help="ImageReward device; 'cpu' only for small-VRAM technical smoke (recorded in EPH)")
    ap.add_argument("--rm-revision", default=None, help="THUDM/ImageReward HF revision (defaults to config reward_revision)")
    ap.add_argument("--vae-chunk", type=int, default=None,
                    help="override vae_decode_chunk (engineering setting, recorded in EPH)")


def build_backend(args) -> tuple:
    """Returns (cfg, bundle, scorer, scoring_path, numeric_runtime)."""
    from apsp.gpu.scoring import ScoringPath
    if args.fake_backend:
        from apsp.gpu.fake_backend import FAKE_CFG_OVERRIDES, FakeBundle, FakeScorer
        cfg = load_config(args.config)
        cfg.update(FAKE_CFG_OVERRIDES)
        cfg["preview_dtype"] = cfg.get("preview_dtype") or "fp16"
        b, s = FakeBundle(), FakeScorer()
        return cfg, b, s, ScoringPath(b, s, cfg.get("vae_decode_chunk")), {"backend": "fake"}
    dev = require_cuda()
    rt = configure_determinism()
    cfg = load_config(args.config)
    if getattr(args, "vae_chunk", None):
        cfg["vae_decode_chunk"] = int(args.vae_chunk)
    from apsp.gpu.model import load_generator
    from apsp.gpu.scoring import ImageRewardScorer
    b = load_generator(cfg, dev, revision=args.revision, preview_dtype=args.preview_dtype,
                       local_files_only=args.local_files_only)
    cfg["preview_dtype"] = b.provenance["preview_dtype"]
    import torch
    rm_dev = dev if getattr(args, "rm_device", "cuda") == "cuda" else torch.device("cpu")
    s = ImageRewardScorer(rm_dev, revision=getattr(args, "rm_revision", None) or cfg.get("reward_revision"),
                          local_files_only=args.local_files_only)
    return cfg, b, s, ScoringPath(b, s, cfg.get("vae_decode_chunk")), rt


def eph_body(cfg: dict, bundle, scorer, runtime: dict) -> dict:
    prov = getattr(bundle, "provenance", {})
    return {
        "backend": getattr(bundle, "backend", "diffusers"),
        "psp_commit": cfg["psp_commit"],
        "generation_code_sha": package_code_sha(GENERATION_CODE),
        "env_lock_sha256": file_sha256(ENV_LOCK) if ENV_LOCK.exists() else None,
        "model_repo": prov.get("model_repo"), "model_revision": prov.get("model_revision"),
        "weights_digest": prov.get("weights_digest"), "weights_variant": prov.get("weights_variant"),
        "timesteps": (prov.get("scheduler") or {}).get("timesteps"),
        "sampling": {k: cfg[k] for k in ("scheduler", "eta", "steps", "delta", "cfg", "negative_prompt", "height", "width")},
        "dtypes": {"unet": cfg.get("unet_dtype"), "preview": cfg.get("preview_dtype"), "deliver": cfg.get("deliver_dtype")},
        "vae_decode_chunk": cfg.get("vae_decode_chunk"),
        "scoring_path": cfg["scoring_path"], "reward_model": getattr(scorer, "provenance", {}),
        "seed_scheme": cfg["seed_scheme"],
        "numeric_runtime": {k: v for k, v in runtime.items() if k not in ("torch_num_threads",)},
    }


def compute_eph(cfg, bundle, scorer, runtime) -> tuple[str, dict]:
    body = eph_body(cfg, bundle, scorer, runtime)
    return provenance_hash("EPH", body), body
