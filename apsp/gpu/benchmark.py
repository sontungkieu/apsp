"""Online quality-cost benchmark (pilot). One GPU, one worker, every method run ONLINE with its own real batch.

Timing (one definition for every method):
  model load / setup     measured once, reported separately (never in a request)
  warm-up                each engine family runs untimed warm-up episodes on a technical prompt first
  e2e request time       CUDA-synchronised perf_counter from BEFORE prepare_request (text encoding = C_fixed) to the
                         moment the delivered image is on the CPU (PIL), i.e. after the terminal decode and, for
                         selecting methods, the reward-model selection. PNG writing and evaluation are after the timer.
Reward-model selection cost is inside inference (and in the ledger: ver/fin). Evaluation-only scoring (post-hoc IR
of an F(1) delivery, HPS, ...) is recorded separately and never added to inference cost.

Per episode (episodes.jsonl): prompt_uid, split, bank_id, method, family, config_hash, candidates, winner, image
path/sha256, success, inference_score, q_eval (pilot quality = ImageReward of the delivered image) and its source,
ledger spend/cap, e2e seconds, eval seconds, forwards, candidates, VERIFY count, prune checkpoints, peak VRAM,
method EPH. manifest.json: commit, EPHs, runtime fingerprint, cost table (GCH, provisional), load/warm-up times.

  python -m apsp.gpu.benchmark --technical --methods single single32 bon:2 bon:4 bon:8 psp --costs COSTS.json --out DIR
  python -m apsp.gpu.benchmark --prompt-pool data/prompts/prompt_pool.parquet --split train_dev --methods-file FROZEN.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from apsp.contracts.hashing import file_sha256
from apsp.contracts.ledger import load_method_costs
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.engine import DenoiseEngine
from apsp.gpu.runtime import import_torch, sync
from apsp.sim.caps import compute_caps
from apsp.sim.features import PCA16, NormStats
from apsp.sim.methods import RunContext, episode_quality, parse_method, run_method

CONFIRMATORY_SPLITS = ("val", "val_ext", "test", "geneval")
LABEL = "online quality-cost benchmark (pilot instrumentation)"


def load_instances(a) -> pd.DataFrame:
    from apsp.data.prompts import normalize, prompt_uid
    if a.technical:
        sc = yaml.safe_load(Path("configs/technical_smoke.yaml").read_text(encoding="utf-8"))
        ps = [normalize(p) for p in sc["prompts"]][: a.limit or None]
        return pd.DataFrame({"prompt_uid": [prompt_uid(p) for p in ps], "text": ps, "split": "technical_smoke"})
    from apsp.contracts.prereg import require_split_access
    if getattr(a, "prompts_jsonl", None):          # GenEval: official prompts, text kept verbatim (PREREG-C)
        if a.split != "geneval":
            raise SystemExit("--prompts-jsonl is only for --split geneval")
        require_split_access("geneval", "online test generation")
        rows = [json.loads(l) for l in Path(a.prompts_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame({"prompt_uid": [r["prompt_uid"] for r in rows], "text": [r["text"] for r in rows],
                           "split": "geneval"})
        return df.head(a.limit) if a.limit else df
    if not (a.prompt_pool and a.split):
        raise SystemExit("--technical or --prompt-pool with --split is required")
    require_split_access(a.split, "online quality-cost benchmark")
    pool = pd.read_parquet(a.prompt_pool)
    sel = pool[pool["split"] == a.split].sort_values("prompt_uid")
    if sel["text"].isna().any():
        raise SystemExit(f"split {a.split} has sealed texts; unseal under the protocol first")
    return sel.head(a.limit) if a.limit else sel


def load_methods(a) -> tuple[list, Optional[dict]]:
    frozen = None
    specs = list(a.methods or [])
    if a.methods_file:
        frozen = json.loads(Path(a.methods_file).read_text(encoding="utf-8"))
        specs += frozen["methods"]
    if not specs:
        raise SystemExit("no methods given")
    if a.split in CONFIRMATORY_SPLITS:
        if not frozen or frozen.get("selected_on") != "train_dev":
            raise SystemExit("confirmatory splits run only a methods file frozen on train_dev (--methods-file)")
        if a.methods:
            raise SystemExit("confirmatory runs take the frozen methods file only (no extra --methods)")
    return [parse_method(s) for s in specs], frozen


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _peak(torch, real: bool) -> dict:
    if not real:
        return {}
    return {"peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="online quality-cost benchmark")
    add_backend_args(ap)
    ap.add_argument("--technical", action="store_true", help="technical prompts (configs/technical_smoke.yaml)")
    ap.add_argument("--prompt-pool", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--methods-file", default=None, help="frozen methods selected on train_dev (required for val/test)")
    ap.add_argument("--costs", default=None, help="measured generator cost table (merged M2)")
    ap.add_argument("--synthetic-costs", action="store_true", help="SYNTHETIC cost table (plumbing only, never evidence)")
    ap.add_argument("--norm", default=None, help="NormStats JSON fitted on train_fit (threshold/b8)")
    ap.add_argument("--pca", default=None, help="PCA16 .npz fitted on train_fit (prompt features)")
    ap.add_argument("--disp-us", type=int, default=0)
    ap.add_argument("--selector-once-us", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompts-jsonl", default=None, help="GenEval prompts (scripts/geneval_prompts.py), --split geneval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not (a.costs or a.synthetic_costs):
        raise SystemExit("--costs (measured) or --synthetic-costs (plumbing only) is required")
    inst = load_instances(a)
    methods, frozen = load_methods(a)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    torch = import_torch()
    t0 = time.perf_counter()
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    load_s = time.perf_counter() - t0
    real = torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake"
    eph64, _ = compute_eph(cfg, bundle, scorer, rt)
    cfg32 = {**cfg, "steps": 32}
    eph32, _ = compute_eph(cfg32, bundle, scorer, rt)
    from apsp.data.synthetic_bank import synthetic_costs
    base = load_method_costs(a.costs) if a.costs else synthetic_costs()
    caps = compute_caps(base)
    norm = NormStats.load(a.norm) if a.norm else NormStats.identity()
    pca = PCA16.load(a.pca) if a.pca else None
    rc = RunContext(base, caps, norm, ctrl_disp_us=a.disp_us, selector_once_us=a.selector_once_us)
    engines = {64: DenoiseEngine(bundle, scoring, cfg), 32: DenoiseEngine(bundle, scoring, cfg32)}
    # ---- warm-up (untimed, not recorded)
    tw = time.perf_counter()
    for _ in range(a.warmup):
        for m in {mm.family: mm for mm in methods}.values():
            eng = engines[m.steps]
            eng.prepare_request("a warm-up prompt for timing", "warmup-technical")
            run_method(m, eng, bank_id=0, ctx=rc)
    sync(bundle.device)
    warm_s = time.perf_counter() - tw
    ep_path = out / "episodes.jsonl"
    done = set()
    if ep_path.exists():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["prompt_uid"], r["bank_id"], r["method"]))
    n_new = 0
    with open(ep_path, "a", encoding="utf-8") as fh:
        for i, row in enumerate(inst.itertuples(index=False)):
            pf = pca.transform(np.asarray(bundle.pooler(row.text), np.float64)) if pca is not None else None
            order = methods[i % len(methods):] + methods[: i % len(methods)]          # rotate to spread drift
            for b in a.banks:
                for m in order:
                    if m.per_prompt() and b != a.banks[0]:
                        continue
                    if (row.prompt_uid, b, m.spec) in done:
                        continue
                    eng = engines[m.steps]
                    if real:
                        torch.cuda.reset_peak_memory_stats()
                    n0 = eng.n_unet_calls
                    sync(bundle.device)
                    t_req = time.perf_counter()
                    eng.prepare_request(row.text, row.prompt_uid)
                    res = run_method(m, eng, bank_id=b, ctx=rc, prompt_feat=pf)
                    sync(bundle.device)
                    e2e = time.perf_counter() - t_req
                    peak = _peak(torch, real)
                    ok, inf_score = episode_quality(res)
                    rec = {"prompt_uid": row.prompt_uid, "split": row.split, "bank_id": b, "method": m.spec,
                           "family": m.family, "config_hash": m.config_hash, "alpha": m.alpha,
                           "candidates": [list(c) for c in (res.trace[0].alive if res.trace else ())],
                           "n_candidates": m.n_candidates(), "feasible": res.feasible, "success": ok,
                           "winner": list(res.delivery.winner) if ok else None,
                           "inference_score": inf_score, "rm_used_in_inference": bool(res.delivery.rm_used) if res.delivery else None,
                           "ledger_spend_us": res.spend_us, "cap_us": res.cap_us if m.alpha is not None else None,
                           "e2e_seconds": e2e, "forwards": eng.n_unet_calls - n0,
                           "candidate_forwards": sum(4 * e.k for e in res.trace if e.action == "CONTINUE"),
                           "n_verify": res.n_verify,
                           "prune_js": list(res.prune_js), "steps": m.steps, "eph": eph32 if m.steps == 32 else eph64,
                           **peak}
                    # ---- after the timer: image I/O and evaluation-only scoring
                    if ok:
                        img = eng.last_images.get(res.delivery.winner)
                        p = out / "images" / f"{row.prompt_uid}_b{b}_{m.config_hash}.png"
                        img.save(p)
                        arr = np.asarray(img.convert("RGB"))
                        rec.update({"image": str(p.relative_to(out)), "image_png_sha256": file_sha256(p),
                                    "image_pixel_sha256": hashlib.sha256(f"{arr.shape}|".encode() + arr.tobytes()).hexdigest()})
                        if inf_score is not None:
                            rec.update({"q_eval": inf_score, "q_eval_source": "inference_rm_same_image", "eval_seconds": 0.0})
                        else:
                            te = time.perf_counter()
                            v = scorer.score(row.text, [img])[0]
                            rec.update({"q_eval": float(v) if np.isfinite(v) else None,
                                        "q_eval_source": "posthoc_ir_evaluation_only",
                                        "eval_seconds": time.perf_counter() - te})
                    else:
                        rec.update({"q_eval": None, "q_eval_source": "failed"})
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_new += 1
    from apsp.gpu.fingerprint import collect as fingerprint
    man = {"label": LABEL, "commit": _git("rev-parse", "HEAD"), "eph_64": eph64, "eph_32": eph32,
           "timesteps_32": [int(t) for t in engines[32].timesteps] if hasattr(engines[32], "timesteps") else None,
           "runtime_fingerprint": fingerprint(bundle), "backend": getattr(bundle, "backend", "diffusers"),
           "costs": {"path": a.costs, "synthetic": bool(getattr(base, "synthetic", False)),
                     "provisional": bool(getattr(base, "provisional", True)),
                     "sha256": file_sha256(Path(a.costs)) if a.costs else None},
           "caps": {str(k): v for k, v in caps.caps.items()}, "C_ref": caps.C_ref, "L0": caps.L0,
           "split": "technical_smoke" if a.technical else a.split, "n_prompts": int(len(inst)), "banks": a.banks,
           "methods": [{"spec": m.spec, "family": m.family, "config_hash": m.config_hash, "artifact": m.artifact,
                        "artifact_sha256": m.artifact_sha256} for m in methods],
           "frozen_methods_file": {"path": a.methods_file, "sha256": file_sha256(Path(a.methods_file))} if a.methods_file else None,
           "norm": a.norm, "pca": a.pca, "model_load_seconds": load_s, "warmup_episodes_per_family": a.warmup,
           "warmup_seconds": warm_s, "workers": 1, "timer": "perf_counter with CUDA synchronize; prepare_request..PIL",
           "new_episodes": n_new}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("split", "n_prompts", "new_episodes", "model_load_seconds", "warmup_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
