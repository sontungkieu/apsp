"""Tune and freeze the pilot baselines on replay (train_fit search, train_dev selection, per cap) - CPU only.

B3 best fixed pair (select_b3_dev), B5 screen-and-resume (select_sr_dev), B6 threshold (tune_threshold, <= 2000 fit
evaluations), B8 one query then commit (tune_b8). Fixed-configuration methods (single, single32, BoN-2/4/8, PSP) are
always listed. Output: a methods file `{"selected_on": "train_dev", "methods": [...]}` plus the artifacts it names
(threshold parameters, B8 selector, NormStats, PCA16) and their hashes. Research splits pass the PREREG guards of the
accessor; val/test are never read here.

  python -m apsp.sim.tune_baselines --bank DIR --costs COSTS.json --fit-split train_fit --dev-split train_dev --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from apsp.contracts.hashing import file_sha256
from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import load_method_costs
from apsp.contracts.schema import load_bank
from apsp.policies.fixed import select_b3_dev
from apsp.policies.query_commit import tune_b8
from apsp.policies.screen_resume import replay_mean, select_sr_dev
from apsp.policies.threshold import ThresholdParams, ThresholdPolicy, tune_threshold
from apsp.sim.caps import fixed_method_costs
from apsp.sim.replay import ReplayContext

FIXED_METHODS = ["single", "single32", "bon:2", "bon:4", "bon:8", "psp"]
FORBIDDEN = ("val", "val_ext", "test", "geneval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", default=None)
    ap.add_argument("--synthetic-costs", action="store_true")
    ap.add_argument("--fit-split", default="train_fit")
    ap.add_argument("--dev-split", default="train_dev")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--b6-evals", type=int, default=2000)
    ap.add_argument("--disp-us", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=["b3", "b5", "b6", "b8"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.fit_split in FORBIDDEN or a.dev_split in FORBIDDEN:
        raise SystemExit("tuning never reads val/test")
    from apsp.data.synthetic_bank import synthetic_costs
    base = load_method_costs(a.costs) if a.costs else (synthetic_costs() if a.synthetic_costs else None)
    if base is None:
        raise SystemExit("--costs or --synthetic-costs is required")
    bank = load_bank(a.bank)
    ctx = ReplayContext.build(bank, base, fit_split=a.fit_split)
    fit, dev = bank.instances_of([a.fit_split]), bank.instances_of([a.dev_split])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx.norm.save(out / "norm.json")
    if ctx.pca is not None:
        ctx.pca.save(out / "pca16.npz")
    fx = fixed_method_costs(base)
    ctrl = base.with_overrides(disp=a.disp_us, method_id="controller")
    methods, log = list(FIXED_METHODS), {}
    Yf, C = ctx.schedule_table(fit)
    Yd, _ = ctx.schedule_table(dev)
    for alpha in a.alphas:
        cap = ctx.caps.cap(alpha)
        tag = str(alpha)
        log[tag] = {"cap_us": cap}
        if "b3" not in a.skip:
            b3 = select_b3_dev(Yf, Yd, C, cap)
            log[tag]["b3"] = asdict(b3) if b3 else None
            if b3:
                methods.append(f"fixed:{b3.j1},{b3.j2}@{alpha}")
        if "b5" not in a.skip:
            keys = {a.fit_split: fit, a.dev_split: dev}
            b5 = select_sr_dev(lambda c, s: replay_mean(ctx, c, keys[s], cap), fx, cap,
                               fit_split=a.fit_split, dev_split=a.dev_split)
            log[tag]["b5"] = b5
            if b5:
                methods.append(f"{b5['name']}@{alpha}")
        if "b6" not in a.skip:
            def ev(p: ThresholdParams, split: str) -> float:
                ks = fit if split == a.fit_split else dev
                ys = [ctx.run_policy(ThresholdPolicy(p), k, cap, ctrl).Y for k in ks]
                return float(np.mean([y if y is not None else ctx.r_fail for y in ys]))
            b6 = tune_threshold(ev, n_evals=a.b6_evals, fit_split=a.fit_split, dev_split=a.dev_split)
            p6 = out / f"b6_threshold_alpha{alpha}.json"
            p6.write_text(json.dumps({k: v for k, v in b6.items() if k != "trials"}, indent=1), encoding="utf-8")
            (out / f"b6_trials_alpha{alpha}.json").write_text(json.dumps(b6["trials"]), encoding="utf-8")
            log[tag]["b6"] = {"path": str(p6), "dev_mean": b6["dev_mean"], "n_fit_evals": b6["n_fit_evals"]}
            methods.append(f"threshold:{p6.as_posix()}@{alpha}")
        if "b8" not in a.skip:
            b8 = tune_b8(ctx, fit, dev, cap)
            if b8:
                p8 = out / f"b8_selector_alpha{alpha}.json"
                p8.write_text(json.dumps(b8, indent=1), encoding="utf-8")
                log[tag]["b8"] = {"path": str(p8), "j_c": b8["j_c"], "dev_mean": b8["dev_mean"]}
                methods.append(f"b8:{p8.as_posix()}@{alpha}")
    frozen = {"selected_on": a.dev_split if a.dev_split == "train_dev" else f"{a.dev_split} (not train_dev)",
              "fit_split": a.fit_split, "methods": methods,
              "bank_manifest_sha256": file_sha256(Path(a.bank) / "manifest.json") if (Path(a.bank) / "manifest.json").exists() else None,
              "costs_sha256": file_sha256(Path(a.costs)) if a.costs else "synthetic", "r_fail": ctx.r_fail,
              "norm": str(out / "norm.json"), "pca": str(out / "pca16.npz") if ctx.pca is not None else None,
              "selection_log": log,
              "note": "replay-tuned (simulator); confirmatory evaluation is online (Amendment 1)"}
    (out / "methods_frozen.json").write_text(json.dumps(frozen, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"methods": methods, "selected_on": frozen["selected_on"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
