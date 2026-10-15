"""Cross-session A/A: the diag `method` k=8 collection vs smoke1's technical-smoke bank shard (read-only).

Both collect the same (prompt, uid, bank) with batch 8 and no pruning, in different sessions/processes. The script
compares initial-latent hashes, the 8x16 preview IR table, delivered IR and decoded pixels (PNG is lossless) and
reports what it measured; no tolerance. ENGINEERING ONLY.

  python scripts/diag_crosscheck.py <diag_dir>/method.json <smoke1_run>/tsmoke/TSMOKE_BANK --out reports/batch_drift/x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def pixel_sha(path: Path) -> str:
    a = np.asarray(Image.open(path).convert("RGB"))
    return hashlib.sha256(f"{a.shape}|".encode() + a.tobytes()).hexdigest()


def crosscheck(method: dict, bank: Path) -> dict:
    out = {"label": "ENGINEERING ONLY; cross-session A/A at batch 8; not G1", "bank": str(bank),
           "diag_eph": method.get("eph"), "instances": []}
    for inst in method["instances"]:
        shard = next((d for d in sorted((bank / "shards").glob("*")) if d.is_dir()
                      and json.loads((d / "shard.json").read_text(encoding="utf-8"))["uid"] == inst["uid"]
                      and json.loads((d / "shard.json").read_text(encoding="utf-8"))["bank_id"] == inst["bank_id"]), None)
        if shard is None:
            continue
        meta = json.loads((shard / "shard.json").read_text(encoding="utf-8"))
        seeds = pd.read_parquet(shard / "seeds.parquet").sort_values("seed_idx")
        pv = pd.read_parquet(shard / "previews.parquet")
        dl = pd.read_parquet(shard / "delivered.parquet").sort_values("seed_idx")
        tab = np.full((8, 16), np.nan)
        for r in pv.itertuples():
            tab[r.seed_idx, r.j - 1] = r.ir
        mine = np.asarray(inst["preview_table"], dtype=float)
        # scores are float32 in the scorer contract: compare both sides rounded to float32
        d_pv = np.abs(mine.astype(np.float32).astype(float) - tab.astype(np.float32).astype(float))
        mine_dl = np.asarray(inst["delivered_table"], dtype=float)
        d_dl = np.abs(mine_dl.astype(np.float32).astype(float) - dl["ir_deliv"].to_numpy(dtype=np.float32).astype(float))
        px = [pixel_sha(bank / p) if p else None for p in dl["image_path"]]
        out["instances"].append({
            "uid": inst["uid"], "bank_id": inst["bank_id"], "smoke1_eph": meta["eph"],
            "eph_equal": meta["eph"] == method.get("eph"),
            "init_latent_sha_equal": int(sum(a == b for a, b in zip(inst["init_latent_sha"], seeds["init_latent_sha"]))),
            "preview_n_equal_as_float32": int((d_pv == 0).sum()), "preview_n": int(d_pv.size),
            "preview_max_abs": float(np.nanmax(d_pv)),
            "delivered_n_equal_as_float32": int((d_dl == 0).sum()), "delivered_max_abs": float(np.nanmax(d_dl)),
            "pixels_equal": int(sum(a == b for a, b in zip(inst["delivered_pixel_sha"], px))), "n": 8})
    if not out["instances"]:
        out["note"] = "no smoke1 shard matches any diag (uid, bank); nothing compared"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("method_json")
    ap.add_argument("bank_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = crosscheck(json.loads(Path(a.method_json).read_text(encoding="utf-8")), Path(a.bank_dir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["instances"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
