"""Validate an atomic Lite task, including independent tensor/row-map checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import checkpoints, noise_seed
from .storage import verify_complete


def validate_task(path, runtime=None, pixel_tol=2):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from safetensors.torch import load_file
    path = Path(path)
    if not verify_complete(path):
        raise ValueError("no durable completion marker")
    task = json.loads((path / "task.json").read_text())
    prov = json.loads((path / "provenance.json").read_text())
    marker = json.loads((path / "COMPLETE.json").read_text())
    p = prov["profile"]
    ck = checkpoints(p["steps"], p["checkpoints"])
    tensors = load_file(str(path / "latents.safetensors"))
    seeds = pd.read_parquet(path / "seeds.parquet")
    prev = pd.read_parquet(path / "previews.parquet")
    if set(seeds.seed_idx) != set(range(8)) or len(seeds) != 8 or len(prev) != 8 * len(ck):
        raise ValueError("candidate/preview row counts")
    if seeds.duplicated(["prompt_uid", "profile", "bank_id", "seed_idx"]).any():
        raise ValueError("duplicate CandidateId")
    for s in range(8):
        frame = prev[prev.seed_idx == s].sort_values("j")
        if list(frame.j) != list(range(1, len(ck) + 1)) or list(frame.step) != ck:
            raise ValueError("checkpoint row map mismatch")
        sr = seeds[seeds.seed_idx == s].iloc[0]
        if int(sr.seed) != noise_seed(task["prompt"]["prompt_uid"], task["bank_id"], s):
            raise ValueError("seed mismatch")
    for name, dims in (("x0", 5), ("final", 4), ("initial", 4)):
        t = tensors[name]
        if t.ndim != dims or t.shape[0] != 8 or str(t.dtype).removeprefix("torch.") != p["dtype"]:
            raise ValueError(f"latent shape/dtype: {name}")
    if tensors["x0"].shape[1] != len(ck) or tensors["x0"].shape[2:] != tensors["final"].shape[1:]:
        raise ValueError("latent checkpoint/shape mismatch")
    emb = np.load(path / "prompt_emb.npy", allow_pickle=False)
    if emb.shape != (1, 768) or emb.dtype != np.float16 or not np.isfinite(emb).all():
        raise ValueError("common encoder shape/dtype")
    for df in (seeds, prev):
        for r in df.itertuples():
            if r.prompt_uid != task["prompt"]["prompt_uid"] or r.profile != task["profile"] or r.bank_id != task["bank_id"]:
                raise ValueError("row identity mismatch")
            if list(r.candidate_id) != [task["bank_id"], r.seed_idx]:
                raise ValueError("CandidateId/tensor row mismatch")
            if r.valid and (not np.isfinite(r.ir) or not np.isfinite(r.pickscore_raw) or not r.img_ok or not r.latent_ok):
                raise ValueError("invalid row called valid")
            if not r.valid and not r.failure_code:
                raise ValueError("missing failure code")
    max_diff = None
    if runtime is not None:
        images, flags, irs, pss, _ = runtime.decode_score(task["prompt"]["text"], tensors["final"])
        diffs = []
        for row in seeds.sort_values("seed_idx").itertuples():
            if row.png is None:
                continue
            im = images[row.seed_idx]
            if im is None:
                raise ValueError("saved valid image decodes invalid")
            with Image.open(path / row.png) as saved:
                a = np.asarray(saved.convert("RGB"), dtype=np.int16)
                b = np.asarray(im, dtype=np.int16)
            diffs.append(int(np.abs(a - b).max()))
        max_diff = max(diffs) if diffs else None
        if max_diff is None or max_diff > pixel_tol:
            raise ValueError(f"redecode difference {max_diff} exceeds {pixel_tol}")
    valid_count = int(seeds.valid.sum())
    all_valid = valid_count == 8 and bool(prev.valid.all())
    if marker["n_candidates"] != len(seeds) or marker["n_previews"] != len(prev):
        raise ValueError("completion receipt row count mismatch")
    return {"schema_valid": True, "all_valid": all_valid, "n_candidates": len(seeds), "n_valid_final": valid_count,
            "n_previews": len(prev), "max_pixel_diff_redecode": max_diff,
            "scorers": {"imagereward": bool(prov["scorers"].get("imagereward")), "pickscore": bool(prov["scorers"].get("pickscore"))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    a = ap.parse_args(argv)
    print(json.dumps(validate_task(a.task_dir), indent=2))


if __name__ == "__main__":
    main()
