"""SYNTHETIC end-to-end quickstart (CPU, no CUDA, no downloads).

Builds a synthetic bank in the real schema, validates it, computes caps from a synthetic cost table and
runs B2, B3 (best-fixed), B9 (random), B6 (threshold, small tuning budget), O1/O3 and a short REINFORCE
training through the shared replay runners. Every number printed is SYNTHETIC: it checks code paths only,
not image quality and not speed.

  python -m apsp.quickstart [--out data/quickstart] [--prompts 60] [--rl-iters 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def run(out: Path, n_prompts: int = 60, rl_iters: int = 30, b6_evals: int = 40, seed: int = 0) -> dict:
    from apsp.contracts.identity import bank_rows
    from apsp.contracts.schema import load_bank, validate_bank
    from apsp.data.synthetic_bank import make_synthetic_bank, synthetic_costs
    from apsp.policies.fixed import best_fixed
    from apsp.policies.random_policy import RandomFeasiblePolicy
    from apsp.policies.threshold import ThresholdParams, ThresholdPolicy, tune_threshold
    from apsp.sim.oracles import o1, o3
    from apsp.sim.replay import ReplayContext, mean_y
    from apsp.sim.schedules import enumerate_pairs, fixed_pair_spec

    t0 = time.time()
    bank_dir = make_synthetic_bank(out / "SYN_BANK", n_prompts=n_prompts, seed=seed)
    rep = validate_bank(bank_dir)
    if not rep.ok:
        raise SystemExit(f"synthetic bank failed validation: {rep.errors[:5]}")
    bank = load_bank(bank_dir)
    base = synthetic_costs()
    ctrl = base.with_overrides(disp=150, method_id="controller")
    ctx = ReplayContext.build(bank, base, fit_split="syn_train_fit")
    fit, dev = bank.instances_of(["syn_train_fit"]), bank.instances_of(["syn_train_dev"])
    Yf, C = ctx.schedule_table(fit)
    Yd, _ = ctx.schedule_table(dev)
    pairs = enumerate_pairs()
    table = {}
    for alpha in (0.5, 1.0):
        cap = ctx.caps.cap(alpha)
        row = {"cap_us": cap}
        if C[pairs.index((4, 8))] <= cap:
            row["B2_(4,8)"] = mean_y([ctx.run_spec(fixed_pair_spec(4, 8, bank_rows(k[1])), k, cap) for k in dev])
        b3 = best_fixed(Yf, C, cap)
        if b3:
            row["B3"] = {"pair": [b3.j1, b3.j2], "dev_mean_y": float(Yd[:, pairs.index((b3.j1, b3.j2))].mean())}
        row["B9_random"] = mean_y([ctx.run_policy(RandomFeasiblePolicy(seed), k, cap, ctrl) for k in dev])

        def ev(p, split, cap=cap):
            keys = fit if split == "fit" else dev
            return mean_y([ctx.run_policy(ThresholdPolicy(p), k, cap, ctrl) for k in keys])
        tuned = tune_threshold(ev, n_evals=b6_evals, top=5, seed=seed, fit_split="fit", dev_split="dev")
        row["B6_threshold"] = {"dev_mean_y": tuned["dev_mean"], "params": tuned["params"], "n_fit_evals": tuned["n_fit_evals"]}
        row["O1_dev"] = float(np.nanmean(o1(Yd, C, cap)))
        table[str(alpha)] = row
    o3v = float(np.mean([o3(ctx.accessor.instance_arrays(k, "oracle")["delivered_ir"], ctx.r_fail) for k in dev]))
    rl = None
    if rl_iters > 0:
        try:
            from apsp.policies.reinforce import synthetic_smoke
            rl = synthetic_smoke(str(out / "rl_artifact"), iterations=rl_iters, n_prompts=min(n_prompts, 40), seed=seed)
        except ImportError:
            rl = "skipped: torch not installed (pip install -e '.[rl]')"
    return {"LABEL": "SYNTHETIC - code-path check only; not image quality, not speed",
            "bank": str(bank_dir), "bank_valid": rep.ok, "caps": {str(a): c for a, c in ctx.caps.caps.items()},
            "L0": ctx.caps.L0, "C_ref": ctx.caps.C_ref, "per_cap_dev": table, "O3_dev": o3v,
            "reinforce_smoke": rl, "seconds": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SYNTHETIC end-to-end quickstart (CPU)")
    ap.add_argument("--out", default="data/quickstart")
    ap.add_argument("--prompts", type=int, default=60)
    ap.add_argument("--rl-iters", type=int, default=30)
    ap.add_argument("--b6-evals", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(out, a.prompts, a.rl_iters, a.b6_evals, a.seed)
    (out / "quickstart_summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
