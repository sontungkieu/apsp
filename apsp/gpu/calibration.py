"""Calibration / profiling session parts (Guide CC-4..CC-6: M3, M8, M9, collector equivalence, M2b, VAE chunk).

Trajectory reuse: each calibration (prompt, bank 0) trajectory is computed ONCE by the production collector
(`collect_instance`, unchanged; shards written with `write_shard`). At every checkpoint a wrapper around the
scoring path also decodes the same x0 with the other preview precision (M3 compares precisions on identical
trajectories, Guide CC-4) and, for the equivalence prompts, keeps x0 at j in {4,8,16}, the final latents, the initial
latents and the prompt embeddings. The equivalence part then runs only the reference side. The wrapper's extra work is
timed with CUDA synchronisation and subtracted for M9; subset-20 bank-1 instances run the plain collector (clean M9).

Production engine / decode / scoring / collector code is unchanged (EPH unaffected). Calibration prompts are not a
research split. Resumable: an instance is skipped when both its shard and its record exist.

  python -m apsp.gpu.calibration --part m3m9 --prompt-pool data/prompts/prompt_pool.parquet --out RUN/calib --bank-root RUN/banks
  python -m apsp.gpu.calibration --part equivalence|chunk --prompt-pool ... --out RUN/calib
  python -m apsp.gpu.calibration --part m2b --costs RUN/p0/m2/generator_costs.json --prompt-pool ... --out RUN/calib
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apsp.contracts.identity import bank_rows
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.runtime import import_torch, sync

EQV_JS = (4, 8, 16)
N_EQV, N_SUBSET, N_CALIB = 10, 20, 50
LABEL = "ENGINEERING calibration (calibration split only; no research split, no val/test)"


def calibration_prompts(pool_path: str) -> pd.DataFrame:
    pool = pd.read_parquet(pool_path)
    cal = pool[pool["split"] == "calibration"].sort_values("prompt_uid").reset_index(drop=True)
    if len(cal) != N_CALIB or int(cal["throughput_subset"].sum()) != N_SUBSET:
        raise SystemExit(f"calibration split must have {N_CALIB} prompts with a fixed subset of {N_SUBSET}")
    cal["eqv"] = cal.index < N_EQV          # = collect_bank --test-equivalence selection (first 10 by uid)
    return cal


# ------------------------------------------------------------------ M3 wrapper -------------------------------------
class AugmentedScoring:
    """Delegates to the production ScoringPath; adds the alternative-precision preview and captures (timed)."""

    def __init__(self, base, alt, engine_ref, capture: bool, device):
        self.base, self.alt, self.eng, self.capture, self.device = base, alt, engine_ref, capture, device
        self.extra_s, self.by_j, self.x0 = 0.0, {}, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def score_preview(self, prompt, x0):
        sb = self.base.score_preview(prompt, x0)            # production path, unchanged
        sync(self.device)
        t0 = time.perf_counter()
        j = int(self.eng.hook.j)
        alt = self.alt.score_preview(prompt, x0) if self.alt is not None else None
        self.by_j[j] = {"prod": [(r.latent_ok, r.img_ok, r.score_ok, r.score) for r in sb.records],
                        "alt": None if alt is None else [(r.latent_ok, r.img_ok, r.score_ok, r.score) for r in alt.records]}
        if self.capture and j in EQV_JS:
            self.x0[j] = x0.detach().cpu().clone()
        sync(self.device)
        self.extra_s += time.perf_counter() - t0
        return sb


def alt_preview_scoring(bundle, scorer, cfg, prod_dtype: str):
    """ScoringPath whose preview VAE uses the other precision (fp16 <-> fp32). None on the fake backend."""
    from apsp.gpu.scoring import ScoringPath
    torch = import_torch()
    if getattr(bundle, "backend", "") == "fake":
        return None, None
    alt_dtype = "fp32" if prod_dtype == "fp16" else "fp16"
    vae = copy.deepcopy(bundle.vae).to(dtype=torch.float32 if alt_dtype == "fp32" else torch.float16)
    vae.requires_grad_(False)
    return ScoringPath(dataclasses.replace(bundle, vae_preview=vae), scorer, cfg.get("vae_decode_chunk")), alt_dtype


def _fail_counts(rows) -> dict:
    lat = sum(not r[0] for r in rows)
    img = sum(r[0] and not r[1] for r in rows)
    sc = sum(r[1] and not r[2] for r in rows)
    return {"latent": lat, "image": img, "score": sc}


def run_m3m9(cfg, bundle, scorer, scoring, eph, eph_body, cal: pd.DataFrame, out: Path, bank_dir: Path,
             machine_id: str = "unknown") -> dict:
    from apsp.config import scientific_config_hash
    from apsp.gpu.collect_bank import _key_name, collect_instance, init_bank_dir, quarantine_partials, shard_ok, write_shard
    from apsp.gpu.engine import DenoiseEngine
    torch = import_torch()
    prod_dtype = bundle.provenance.get("preview_dtype") or cfg.get("preview_dtype") or cfg.get("deliver_dtype", "fp16")
    alt, alt_dtype = alt_preview_scoring(bundle, scorer, cfg, prod_dtype)
    cfg_hash = scientific_config_hash(cfg)
    init_bank_dir(bank_dir, eph, eph_body, cfg_hash, resume=True)
    quarantine_partials(bank_dir)
    rec_dir, eqv_dir = out / "m3m9_records", out / "eqv_tensors"
    rec_dir.mkdir(parents=True, exist_ok=True)
    eqv_dir.mkdir(parents=True, exist_ok=True)
    plan = [(r, 0) for r in cal.itertuples(index=False)] + [(r, 1) for r in cal.itertuples(index=False) if r.throughput_subset]
    done = skipped = 0
    for row, b in plan:
        name = _key_name(row.prompt_uid, b)
        shard, rec_p = bank_dir / "shards" / name, rec_dir / f"{name}.json"
        if shard.exists() and rec_p.exists() and shard_ok(shard, eph, cfg_hash):
            skipped += 1
            continue
        if shard.exists():                                  # shard without record: redo the instance
            q = bank_dir / "quarantine"
            q.mkdir(exist_ok=True)
            shutil.move(str(shard), str(q / f"{name}_{int(time.time())}"))
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
        aug = None
        if b == 0:
            aug = AugmentedScoring(scoring, alt, eng, bool(row.eqv), bundle.device)
            eng.scoring = aug
        if torch.cuda.is_available() and bundle.backend != "fake":
            torch.cuda.reset_peak_memory_stats()
        emb = np.asarray(bundle.pooler(row.text), np.float32)
        col = collect_instance(eng, row.text, row.prompt_uid, b, cfg["k0"])
        t0 = time.perf_counter()
        write_shard(bank_dir, row.prompt_uid, b, col, eph, cfg_hash, emb)
        write_s = time.perf_counter() - t0
        rec = {"prompt_uid": row.prompt_uid, "bank_id": b, "subset": bool(row.throughput_subset),
               "collect_seconds": col["seconds"], "extra_seconds": aug.extra_s if aug else 0.0,
               "m9_seconds": col["seconds"] - (aug.extra_s if aug else 0.0), "write_shard_seconds": write_s,
               "clean_collector": aug is None, "unet_calls": col["unet_calls"],
               "delivered_fail": _fail_counts([(d[4], d[5], d[6]) for d in col["delivered"]])}
        if torch.cuda.is_available() and bundle.backend != "fake":
            rec["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            rec["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        if aug is not None:
            rec["m3"] = {"prod_dtype": prod_dtype, "alt_dtype": alt_dtype,
                         "by_j": {str(j): {"prod": _fail_counts(v["prod"]),
                                           "alt": None if v["alt"] is None else _fail_counts(v["alt"]),
                                           "score_abs_diff": [abs(p[3] - a[3]) for p, a in zip(v["prod"], v["alt"])
                                                              if p[3] is not None and a[3] is not None] if v["alt"] else []}
                                  for j, v in sorted(aug.by_j.items())}}
            if row.eqv:
                torch.save({"x0": aug.x0, "final": eng.x.detach().cpu().clone(), "init": eng.init.detach().cpu().clone(),
                            "cond": eng.cond.detach().cpu().clone(), "uncond": eng.uncond.detach().cpu().clone(),
                            "text": row.text}, eqv_dir / f"{row.prompt_uid}.pt")
        rec_p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        done += 1
    return summarize_m3m9(rec_dir, prod_dtype, alt_dtype) | {"instances_run": done, "instances_skipped": skipped,
                                                              "bank_dir": str(bank_dir)}


def summarize_m3m9(rec_dir: Path, prod_dtype=None, alt_dtype=None) -> dict:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rec_dir.glob("*.json"))]
    m3 = [r for r in recs if "m3" in r]
    J = 16
    out = {"n_records": len(recs), "n_m3_instances": len(m3)}
    for which in ("prod", "alt"):
        if not m3 or m3[0]["m3"]["by_j"]["1"][which] is None:
            continue
        rate = {k: [sum(r["m3"]["by_j"][str(j)][which][k] for r in m3) / (8 * len(m3)) for j in range(1, J + 1)]
                for k in ("latent", "image", "score")}
        out[f"m3_{which}"] = {"dtype": m3[0]["m3"][f"{which}_dtype"], "n_previews": 8 * len(m3) * J,
                              "failure_rate_by_j": rate, "worst_rate": max(max(v) for v in rate.values())}
    dl = {k: sum(r["delivered_fail"][k] for r in m3) for k in ("latent", "image", "score")}
    out["m3_delivered"] = {"failures": dl, "rate": max(dl.values()) / (8 * len(m3)) if m3 else None}
    diffs = [d for r in m3 for v in r["m3"]["by_j"].values() for d in v["score_abs_diff"]]
    out["m3_prod_vs_alt_score_abs_diff"] = {"max": max(diffs) if diffs else None, "mean": float(np.mean(diffs)) if diffs else None}
    worst = max([out.get("m3_prod", {}).get("worst_rate", 0), out["m3_delivered"]["rate"] or 0])
    out["m3_escalate_to_research"] = bool(worst > 1e-3)
    out["m3_complete"] = len(m3) == N_CALIB
    clean = [r["m9_seconds"] for r in recs if r["clean_collector"]]
    aug = [r["m9_seconds"] for r in recs if not r["clean_collector"] and r["subset"]]
    s = float(np.mean(clean)) if clean else None
    out["m9"] = {"n_clean_subset_bank1": len(clean), "seconds_per_prompt_bank_clean_mean": s,
                 "seconds_per_prompt_bank_subset_bank0_minus_extra_mean": float(np.mean(aug)) if aug else None,
                 "write_shard_seconds_mean": float(np.mean([r["write_shard_seconds"] for r in recs])) if recs else None,
                 "gpu_hours_small_pilot_collection": s * 2 * (190 + 50 + 80 + 220) / 3600 if s else None,
                 "gpu_hours_large_pilot_collection": s * 2 * (320 + 80 + 200 + 100) / 3600 if s else None,
                 "note": "collection only (2 banks/prompt, val_ext counted at its maximum); profiling, G1 and online "
                         "evaluation are budgeted separately"}
    peaks = [r.get("peak_vram_reserved_bytes") for r in recs if r.get("peak_vram_reserved_bytes")]
    out["m8_collector_peak_vram_reserved_bytes"] = max(peaks) if peaks else None
    return out


# ------------------------------------------------------------------ equivalence (reference side only) --------------
def run_equivalence(cfg, bundle, scoring, out: Path) -> dict:
    from apsp.contracts.hashing import array_sha256
    from apsp.gpu.reference_psp import load_reference_pipeline, run_reference
    torch = import_torch()
    files = sorted((out / "eqv_tensors").glob("*.pt"))
    if len(files) < N_EQV:
        return {"status": "incomplete", "n_stored": len(files), "needed": N_EQV}
    if getattr(bundle, "backend", "") == "fake":
        return {"status": "error", "reason": "equivalence needs the real reference pipeline"}
    ref = load_reference_pipeline(cfg, bundle)
    recs = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        dev = bundle.device
        init = (d["init"].to(dev) * bundle.scheduler.init_noise_sigma)
        latA, x0A = run_reference(ref, cfg, d["text"], init, d["cond"].to(dev), d["uncond"].to(dev), capture_js=EQV_JS)
        latB, _ = run_reference(ref, cfg, d["text"], init)
        ours = d["final"].float()
        r = {"prompt_uid": f.stem,
             "A_delivered_exact_hash": array_sha256(ours.numpy()) == array_sha256(latA.float().cpu().numpy()),
             "A_delivered_max_abs": float((ours - latA.float().cpu()).abs().max()),
             "B_delivered_exact_hash": array_sha256(ours.numpy()) == array_sha256(latB.float().cpu().numpy()),
             "B_delivered_max_abs": float((ours - latB.float().cpu()).abs().max())}
        for j in EQV_JS:
            so = [x.score for x in scoring.score_preview(d["text"], d["x0"][j].to(dev)).records]
            sr = [x.score for x in scoring.score_preview(d["text"], x0A[j]).records]
            diffs = [abs(a - b) for a, b in zip(so, sr) if a is not None and b is not None]
            r[f"C_j{j}_x0_max_abs"] = float((d["x0"][j].float() - x0A[j].float().cpu()).abs().max())
            r[f"C_j{j}_score_max_abs"] = max(diffs) if diffs else None
        recs.append(r)
    return {"status": "measured", "n_prompts": len(recs), "records": recs,
            "all_A_exact": all(r["A_delivered_exact_hash"] for r in recs),
            "source_of_ours": "stored tensors of the m3m9 collector run (no second UNet pass)",
            "note": "exact-hash comparisons are reported as measured; tolerance alone is not called bitwise parity"}


# ------------------------------------------------------------------ VAE chunk (Q7) ---------------------------------
def run_chunk(bundle, out: Path, chunks=(1, 2, 4, 8, 16), ks=(8, 16), warmup=10, n=50) -> dict:
    from apsp.gpu.decode import decode_delivered, decode_preview
    torch = import_torch()
    files = sorted((out / "eqv_tensors").glob("*.pt"))
    if len(files) < 2:
        return {"status": "incomplete", "reason": "needs stored final latents of >= 2 prompts (m3m9 first)"}
    lat = torch.cat([torch.load(f, map_location="cpu")["final"] for f in files[:2]], 0).to(bundle.device)
    res = {}
    real = torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake"
    for path, fn in (("preview", decode_preview), ("delivered", decode_delivered)):
        for k in ks:
            for c in chunks:
                if c > k:
                    continue
                if real:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                ts = []
                for i in range(warmup + n):
                    sync(bundle.device)
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        fn(bundle, lat[:k], c)
                    sync(bundle.device)
                    if i >= warmup:
                        ts.append((time.perf_counter() - t0) * 1e6)
                a = np.asarray(ts)
                res[f"{path}_k{k}_chunk{c}"] = {"mean_us": float(a.mean()), "p95_us": float(np.quantile(a, 0.95)),
                                                "cv": float(a.std() / a.mean()),
                                                "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()) if real else None,
                                                "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()) if real else None}
    return {"status": "measured", "results": res, "rule": "Guide CC-4: choose by measured throughput and memory; if "
            "latency is not distinguishable, the smaller chunk; the choice is proposed in PREREG-A, not applied here"}


# ------------------------------------------------------------------ M2b: B2 online vs sum of primitives ------------
def run_m2b(cfg, bundle, scoring, cal: pd.DataFrame, costs_path: str, n: int = N_SUBSET) -> dict:
    from apsp.contracts.ledger import load_method_costs, schedule_cost
    from apsp.gpu.engine import DenoiseEngine
    from apsp.sim.env import run_schedule
    from apsp.sim.schedules import fixed_pair_spec
    torch = import_torch()
    mc = load_method_costs(costs_path)
    spec = fixed_pair_spec(4, 8, bank_rows(0))
    pred_us = schedule_cost(spec, mc)
    rows = []
    real = torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake"
    if real:
        torch.cuda.reset_peak_memory_stats()
    for r in cal[cal["throughput_subset"]].head(n).itertuples(index=False):
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
        eng.prepare_request(r.text, r.prompt_uid)
        res = run_schedule(spec, eng, mc=mc, cap_us=10 ** 13, method_id="B2")
        sync(bundle.device)
        rows.append({"prompt_uid": r.prompt_uid, "wall_us": res.wall_clock_s * 1e6, "ledger_us": res.spend_us,
                     "residual_us": res.wall_clock_s * 1e6 - pred_us})
    w = np.asarray([x["wall_us"] for x in rows])
    return {"n_episodes": len(rows), "predicted_us": int(pred_us), "wall_mean_us": float(w.mean()),
            "wall_p50_us": float(np.median(w)), "wall_p95_us": float(np.quantile(w, 0.95)),
            "residual_mean_us": float(w.mean() - pred_us), "residual_frac": float(w.mean() / pred_us - 1),
            "episodes": rows, "costs_path": costs_path, "costs_provisional": bool(mc.provisional) if hasattr(mc, "provisional") else None,
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()) if real else None,
            "note": "prepare_request (text encoding) happens before the episode timer, as in the online runner"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calibration / profiling parts (engineering)")
    add_backend_args(ap)
    ap.add_argument("--part", choices=["m3m9", "equivalence", "chunk", "m2b", "summary"], required=True)
    ap.add_argument("--prompt-pool", default="data/prompts/prompt_pool.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-root", default=None)
    ap.add_argument("--costs", default=None)
    ap.add_argument("--machine-id", default="unknown")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "summary":
        rep = summarize_m3m9(out / "m3m9_records")
        (out / "m3m9.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1, default=str))
        return 0
    cal = calibration_prompts(a.prompt_pool)
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    from apsp.gpu.fingerprint import collect as fingerprint
    t0 = time.perf_counter()
    if a.part == "m3m9":
        bank = Path(a.bank_root or out.parent / "banks") / f"CALIB_{eph[:8]}"
        rep = run_m3m9(cfg, bundle, scorer, scoring, eph, body, cal, out, bank, a.machine_id)
    elif a.part == "equivalence":
        rep = run_equivalence(cfg, bundle, scoring, out)
    elif a.part == "chunk":
        rep = run_chunk(bundle, out)
    else:
        if not a.costs:
            raise SystemExit("--costs (merged M2 generator_costs.json) is required for m2b")
        rep = run_m2b(cfg, bundle, scoring, cal, a.costs)
    rep.update({"part": a.part, "eph": eph, "label": LABEL, "seconds": time.perf_counter() - t0,
                "runtime_fingerprint": fingerprint(bundle)})
    (out / f"{a.part}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("records", "episodes", "results", "runtime_fingerprint")},
                     indent=1, default=str))
    return 0 if rep.get("status") not in ("error",) else 5


if __name__ == "__main__":
    sys.exit(main())
