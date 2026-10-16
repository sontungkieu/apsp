"""Bank collector (CC-6): 8 seeds per (prompt, bank), all 16 checkpoints, no pruning.

Per checkpoint: preview score + validity + collector features (drift, cfg_gap, channel stats).
At the end: delivered decode, validity, IR for every seed and a PNG for every valid image. Preview@T is
never used as the delivered image.

Shards are written to a temp dir and atomically renamed. Resume skips only shards whose ``shard.json``
matches the current EPH/config and whose checksums verify; mismatches are refused (never mixed, never
silently overwritten). Leftover temp dirs move to ``failed/`` with the reason preserved.

  python -m apsp.gpu.collect_bank --config configs/pilot.yaml --smoke-prompt "a red bicycle" --bank-name SMOKE
  python -m apsp.gpu.collect_bank --config configs/pilot.yaml --calib-throughput --prompt-pool data/prompts/prompt_pool.parquet
  python -m apsp.gpu.collect_bank --config configs/pilot.yaml --test-equivalence --prompt-pool data/prompts/prompt_pool.parquet
  python -m apsp.gpu.collect_bank --config configs/pilot.yaml --split train_fit train_dev val --resume   (needs PREREG-A)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apsp.config import scientific_config_hash
from apsp.contracts.hashing import array_sha256, file_sha256, sha256_hex
from apsp.contracts.identity import bank_rows, derive_seed
from apsp.contracts.schema import (DELIVERED_COLS, PREVIEW_COLS, SEED_COLS, atomic_write_json, validate_bank,
                                   write_bank)
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.engine import DenoiseEngine
from apsp.gpu.runtime import import_torch

COLLECTOR_VERSION = "apsp-collector-v1"


class CompatibilityError(RuntimeError):
    pass


def _key_name(uid: str, b: int) -> str:
    return f"{uid}__b{b}"


LATENT_SPLITS = ("train_fit", "train_dev")     # PREREG-A A16: x0 latents stored for these splits only


def collect_instance(engine: DenoiseEngine, prompt: str, uid: str, bank_id: int, k0: int = 8,
                     store_latents: bool = False) -> dict:
    """Run one (prompt, bank) without pruning. Returns table rows, PIL images, timings and (optionally) the x0
    preview latents as float16[k0, J, C, H, W] indexed by seed_idx and j-1."""
    torch = import_torch()
    t0 = time.perf_counter()
    engine.prepare_request(prompt, uid)
    rows = bank_rows(bank_id, k0)
    engine.start(rows)
    seeds = [(uid, bank_id, s, derive_seed(uid, bank_id, s), array_sha256(engine.init[i].cpu().numpy()))
             for i, (_, s) in enumerate(rows)]
    pv = []
    J = engine.steps // engine.delta
    x0s = [] if store_latents else None
    for j in range(1, J + 1):
        engine.advance(None, engine.delta, False)
        assert engine.hook.j == j
        sb = engine.scoring.score_preview(prompt, engine.hook.x0)
        if x0s is not None:
            x0s.append(engine.hook.x0.detach().float().cpu().numpy().astype(np.float16))
        f = engine.hook.last
        t_in = int(engine.timesteps[engine.delta * j - 1])
        for i, (_, s) in enumerate(rows):
            r = sb.records[i]
            pv.append((uid, bank_id, s, j, engine.delta * j, t_in, r.score if r.score_ok else np.nan, r.latent_ok,
                       r.img_ok, r.score_ok, float(f["drift"][i]) if r.latent_ok else np.nan,
                       float(f["cfg_gap"][i]) if r.latent_ok else np.nan,
                       *[float(v) for v in f["mean_c"][i]], *[float(v) for v in f["std_c"][i]]))
    sb = engine.scoring.score_delivered(prompt, engine.x)
    dl = [(uid, bank_id, s, sb.records[i].score if sb.records[i].score_ok else np.nan, sb.records[i].latent_ok,
           sb.records[i].img_ok, sb.records[i].score_ok) for i, (_, s) in enumerate(rows)]
    lat = None
    if x0s is not None:
        a = np.stack(x0s, axis=1)                         # [row, J, C, H, W]
        lat = np.empty_like(a)
        for i, (_, s) in enumerate(rows):
            lat[s] = a[i]
    return {"seeds": seeds, "previews": pv, "delivered": dl, "images": sb.images, "seconds": time.perf_counter() - t0,
            "unet_calls": engine.n_unet_calls, "x0_latents": lat}


def write_shard(bank_dir: Path, uid: str, b: int, out: dict, eph: str, cfg_hash: str, prompt_emb: np.ndarray) -> Path:
    name = _key_name(uid, b)
    shards = bank_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    tmp = shards / f".tmp_{name}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    img_paths, img_sha = [], []
    for (_, _, s, *_), im in zip(out["delivered"], out["images"]):
        if im is None:
            img_paths.append(None)
            img_sha.append(None)
            continue
        fn = f"img_s{s}.png"
        im.save(tmp / fn, format="PNG")
        img_paths.append(f"shards/{name}/{fn}")
        img_sha.append(file_sha256(tmp / fn))
    pd.DataFrame(out["seeds"], columns=list(SEED_COLS)).to_parquet(tmp / "seeds.parquet", index=False)
    pd.DataFrame(out["previews"], columns=list(PREVIEW_COLS)).to_parquet(tmp / "previews.parquet", index=False)
    dl = pd.DataFrame(out["delivered"], columns=list(DELIVERED_COLS)[:7])
    dl["image_path"], dl["image_sha"] = img_paths, img_sha
    dl.to_parquet(tmp / "delivered.parquet", index=False)
    np.save(tmp / "prompt_emb.npy", prompt_emb.astype(np.float16))
    if out.get("x0_latents") is not None:
        np.save(tmp / "x0.npy", out["x0_latents"].astype(np.float16))
    files = sorted(p.name for p in tmp.iterdir())
    meta = {"collector": COLLECTOR_VERSION, "eph": eph, "config_hash": cfg_hash, "uid": uid, "bank_id": b,
            "checksums": {f: file_sha256(tmp / f) for f in files}, "seconds": out["seconds"], "status": "complete"}
    (tmp / "shard.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    final = shards / name
    if final.exists():
        raise CompatibilityError(f"shard {name} already exists; refusing to overwrite")
    os.replace(tmp, final)
    return final


def shard_ok(shard: Path, eph: str, cfg_hash: str) -> bool:
    """True if a finished shard is valid for this EPH/config; raises CompatibilityError on mismatch."""
    meta_p = shard / "shard.json"
    if not meta_p.exists():
        raise CompatibilityError(f"{shard.name}: no shard.json (partial shard?)")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("eph") != eph or meta.get("config_hash") != cfg_hash:
        raise CompatibilityError(f"{shard.name}: collected under EPH {meta.get('eph', '')[:12]} / config "
                                 f"{meta.get('config_hash', '')[:12]}, current {eph[:12]} / {cfg_hash[:12]}")
    for f, h in meta["checksums"].items():
        if not (shard / f).exists() or file_sha256(shard / f) != h:
            raise CompatibilityError(f"{shard.name}: checksum mismatch for {f}")
    return True


def quarantine_partials(bank_dir: Path) -> list:
    moved = []
    shards = bank_dir / "shards"
    if not shards.exists():
        return moved
    failed = bank_dir / "failed"
    for p in shards.glob(".tmp_*"):
        failed.mkdir(exist_ok=True)
        dest = failed / f"{p.name}_{int(time.time())}"
        os.replace(p, dest)
        (dest / "REASON.txt").write_text("partial shard found at resume (process interrupted); not reused\n",
                                         encoding="utf-8")
        moved.append(dest.name)
    return moved


def init_bank_dir(bank_dir: Path, eph: str, eph_body: dict, cfg_hash: str, resume: bool) -> None:
    meta_p = bank_dir / "collector.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta["eph"] != eph or meta["config_hash"] != cfg_hash:
            raise CompatibilityError(f"bank {bank_dir.name} was collected with EPH {meta['eph'][:12]}; current EPH "
                                     f"{eph[:12]}. Refusing to mix; use a new --bank-name.")
        if not resume:
            raise CompatibilityError(f"bank {bank_dir.name} exists; pass --resume to continue it")
        return
    bank_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta_p, {"collector": COLLECTOR_VERSION, "eph": eph, "eph_body": eph_body, "config_hash": cfg_hash})


def run_collection(engine: DenoiseEngine, bundle, prompts: pd.DataFrame, bank_dir: Path, *, eph: str, eph_body: dict,
                   cfg: dict, banks_per_prompt: int, resume: bool, machine_id: str, commit: str,
                   finalize: bool = True) -> dict:
    cfg_hash = scientific_config_hash(cfg)
    init_bank_dir(bank_dir, eph, eph_body, cfg_hash, resume)
    moved = quarantine_partials(bank_dir)
    done, skipped, timings = [], [], []
    for row in prompts.itertuples(index=False):
        emb = None
        for b in range(banks_per_prompt):
            shard = bank_dir / "shards" / _key_name(row.prompt_uid, b)
            if shard.exists() and shard_ok(shard, eph, cfg_hash):
                skipped.append(shard.name)
                continue
            if emb is None:
                emb = np.asarray(bundle.pooler(row.text), np.float32)
            out = collect_instance(engine, row.text, row.prompt_uid, b, cfg["k0"],
                                   store_latents=bool(cfg.get("store_train_latents")) and row.split in LATENT_SPLITS)
            write_shard(bank_dir, row.prompt_uid, b, out, eph, cfg_hash, emb)
            done.append(shard.name)
            timings.append(out["seconds"])
    summary = {"collected": done, "skipped_valid": skipped, "quarantined": moved,
               "seconds_per_prompt_bank": timings}
    if finalize:
        summary["bank"] = finalize_bank(bank_dir, prompts, eph, eph_body, cfg, machine_id, commit, banks_per_prompt)
    return summary


def finalize_bank(bank_dir: Path, prompts: pd.DataFrame, eph: str, eph_body: dict, cfg: dict, machine_id: str,
                  commit: str, banks_per_prompt: int) -> dict:
    seeds, pv, dl, emb, lat = [], [], [], [], {}
    for row in prompts.itertuples(index=False):
        for b in range(banks_per_prompt):
            sd = bank_dir / "shards" / _key_name(row.prompt_uid, b)
            if (sd / "x0.npy").exists():
                if row.split not in LATENT_SPLITS:
                    raise CompatibilityError(f"{sd.name}: x0 latents present for split {row.split}")
                dst = bank_dir / "latents" / f"x0_{row.prompt_uid}_{b}.npy"
                dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(sd / "x0.npy", dst)
                lat[f"latents/{dst.name}"] = file_sha256(dst)
            seeds.append(pd.read_parquet(sd / "seeds.parquet"))
            pv.append(pd.read_parquet(sd / "previews.parquet"))
            dl.append(pd.read_parquet(sd / "delivered.parquet"))
            if b == 0:
                emb.append(np.load(sd / "prompt_emb.npy"))
    ptab = prompts[["prompt_uid", "text", "split", "source", "group_id", "n_clip_tokens"]].copy()
    timesteps = (eph_body.get("timesteps") or [None] * cfg["steps"])
    t_ck = [int(t) for t in pd.concat(pv).drop_duplicates("j").sort_values("j")["t_in"]]
    write_bank(bank_dir, prompts=ptab, prompt_emb=np.stack(emb), seeds=pd.concat(seeds), previews=pd.concat(pv),
               delivered=pd.concat(dl), eph=eph, eph_body=eph_body,
               collection={"collector": COLLECTOR_VERSION, "mode": "full-8-seed-no-prune", "latents_stored": bool(lat),
                           "latent_policy": {"splits": list(LATENT_SPLITS), "dtype": "float16",
                                             "layout": "[seed_idx, j-1, C, H, W]"} if lat else None,
                           "latent_checksums": lat, "banks_per_prompt": banks_per_prompt},
               machine_id=machine_id, commit=commit,
               extra_manifest={"synthetic": False, "backend": eph_body.get("backend"), "timesteps_at_checkpoints": t_ck,
                               "label": "FAKE BACKEND - test fixture" if eph_body.get("backend") == "fake" else None})
    rep = validate_bank(bank_dir)
    return {"valid": rep.ok, "errors": rep.errors[:20], "stats": rep.stats}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _prompts_from_args(a) -> pd.DataFrame:
    from apsp.contracts.prereg import require_split_access
    from apsp.data.prompts import normalize, prompt_uid
    if a.smoke_prompt:
        t = normalize(a.smoke_prompt)
        return pd.DataFrame({"prompt_uid": [prompt_uid(t)], "text": [t], "split": ["smoke"], "source": ["smoke"],
                             "group_id": [prompt_uid(t)], "n_clip_tokens": [0]})
    if not a.prompt_pool:
        raise SystemExit("--prompt-pool (or --smoke-prompt) is required")
    pool = pd.read_parquet(a.prompt_pool)
    if a.test_equivalence or a.calib_throughput:
        cal = pool[pool["split"] == "calibration"].sort_values("prompt_uid")
        if a.calib_throughput:
            return cal[cal["throughput_subset"]].head(a.limit or 20)
        return cal.head(a.limit or 10)
    for s in a.split:
        require_split_access(s, "bank collection")   # research splits need signed PREREG-A
    sel = pool[pool["split"].isin(a.split)].sort_values(["split", "prompt_uid"])
    return sel.head(a.limit) if a.limit else sel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect replay banks (8 seeds x 16 checkpoints, no pruning)")
    add_backend_args(ap)
    ap.add_argument("--bank-name", default=None)
    ap.add_argument("--out-root", default="data/banks")
    ap.add_argument("--prompt-pool", default=None)
    ap.add_argument("--split", nargs="*", default=[])
    ap.add_argument("--smoke-prompt", default=None, help="technical smoke on a prompt outside the research pool")
    ap.add_argument("--test-equivalence", action="store_true")
    ap.add_argument("--calib-throughput", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--banks-per-prompt", type=int, default=None)
    ap.add_argument("--machine-id", default="unknown")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expect-eph-file", default=None, help="frozen EPH (PREREG-A); refuse to collect under another EPH")
    a = ap.parse_args(argv)
    prompts = _prompts_from_args(a)
    if a.dry_run:
        print(json.dumps({"n_prompts": len(prompts), "splits": prompts["split"].value_counts().to_dict()}, indent=2))
        return 0
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    if a.expect_eph_file:
        want = json.loads(Path(a.expect_eph_file).read_text(encoding="utf-8"))["eph"]
        if want != eph:
            raise CompatibilityError(f"EPH {eph[:12]} differs from the frozen EPH {want[:12]} ({a.expect_eph_file})")
    engine = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    if a.test_equivalence:
        from apsp.gpu.reference_psp import equivalence_report
        rep = equivalence_report(cfg, bundle, scoring, engine, prompts, a)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("status") != "error" else 5
    name = a.bank_name or ("SMOKE" if a.smoke_prompt else "CALIB_THROUGHPUT" if a.calib_throughput else "BANK") + f"_{eph[:8]}"
    bpp = a.banks_per_prompt or (1 if a.smoke_prompt else cfg.get("banks_per_prompt", 2))
    torch = import_torch()
    if torch.cuda.is_available() and not a.fake_backend:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    summ = run_collection(engine, bundle, prompts, Path(a.out_root) / name, eph=eph, eph_body=body, cfg=cfg,
                          banks_per_prompt=bpp, resume=a.resume, machine_id=a.machine_id, commit=_git_commit())
    summ["wall_seconds"] = time.perf_counter() - t0
    summ["eph"] = eph
    summ["backend"] = body["backend"]
    if torch.cuda.is_available() and not a.fake_backend:
        summ["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summ["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    bank_dir = Path(a.out_root) / name
    summ["bank_disk_bytes"] = sum(p.stat().st_size for p in bank_dir.rglob("*") if p.is_file())
    if a.calib_throughput and summ["seconds_per_prompt_bank"]:
        s = float(np.mean(summ["seconds_per_prompt_bank"]))
        summ["M9"] = {"seconds_per_prompt_bank_mean": s,
                      "gpu_hours_small_pilot": s * 2 * (190 + 50 + 80 + 220) / 3600,
                      "gpu_hours_large_pilot": s * 2 * (320 + 80 + 200 + 100) / 3600,
                      "note": "collection only; profiling and G1 time are budgeted separately; val_ext counted at max"}
    print(json.dumps({k: v for k, v in summ.items() if k != "seconds_per_prompt_bank"}, indent=2, default=str))
    return 0 if summ.get("bank", {}).get("valid", True) else 6


if __name__ == "__main__":
    sys.exit(main())
