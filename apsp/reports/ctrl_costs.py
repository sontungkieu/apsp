"""M2c: controller dispatch cost (view + mask + build_obs + policy.act) measured on THIS host.

The result is ``provisional`` unless run on the measurement machine with the frozen artifact. Synthetic
banks only exercise the code path.

  python -m apsp.reports.ctrl_costs --policy random --synthetic --out reports/ctrl_costs_dev.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from apsp.contracts.hashing import package_code_sha, provenance_hash
from apsp.contracts.identity import bank_rows
from apsp.contracts.paid_state import EpisodeState
from apsp.sim.budget_dp import G842, L_of, mask, solve_L
from apsp.sim.features import build_obs


def measure_dispatch(policy, ctx, keys, cap_us, mc, warmup: int = 10, n_min: int = 50) -> dict:
    g = G842()
    L = solve_L(g, mc)
    samples = []
    for k in keys:
        res = ctx.run_policy(policy, k, cap_us, mc)  # trajectory of states to time on
        state = EpisodeState(grammar_id=g.grammar_id, rows=bank_rows(k[1]), cap_us=cap_us,
                             L_fn=lambda q: L_of(L, q, g, mc), C_ref=ctx.caps.C_ref,
                             prompt_feat=ctx.prompt_feat(k[0]))
        state.charge(mc.once)
        for ev in res.trace[:8]:
            t0 = time.perf_counter_ns()
            v = state.view()
            m = mask(state.q, state.remaining_us, L, g, mc)
            o = build_obs(v, ctx.norm)
            policy.act(o, m)
            samples.append((time.perf_counter_ns() - t0) / 1000.0)
            if ev.action == "CONTINUE":
                state.charge(ev.fee_us)
                state.arrive(state.j + 1)
            else:
                break
        if len(samples) >= warmup + n_min:
            break
    s = np.array(samples[warmup:])
    return {"n": int(s.size), "mean_us": float(s.mean()), "p50_us": float(np.median(s)), "p95_us": float(np.quantile(s, 0.95)),
            "cv": float(s.std() / s.mean()) if s.mean() else 0.0, "raw_us": s.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure controller dispatch cost (M2c)")
    ap.add_argument("--policy", choices=["random", "threshold"], default="random")
    ap.add_argument("--synthetic", action="store_true", required=True,
                    help="only synthetic exercise is wired here; frozen artifacts are profiled in P3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import tempfile
    from apsp.contracts.schema import load_bank
    from apsp.data.synthetic_bank import make_synthetic_bank, synthetic_costs
    from apsp.policies.random_policy import RandomFeasiblePolicy
    from apsp.policies.threshold import ThresholdParams, ThresholdPolicy
    from apsp.sim.replay import ReplayContext
    d = tempfile.mkdtemp()
    make_synthetic_bank(d + "/b", 30, 0)
    bank = load_bank(d + "/b")
    mc = synthetic_costs()
    ctx = ReplayContext.build(bank, mc, fit_split="syn_train_fit")
    pol = RandomFeasiblePolicy(0) if a.policy == "random" else ThresholdPolicy(ThresholdParams(0, 0, 0, 0, 1, 1))
    m = measure_dispatch(pol, ctx, bank.instances_of(["syn_train_dev"]) * 5, ctx.caps.cap(1.0), mc)
    body = {"cph": pol.cph, "disp_mean_us": m["mean_us"], "host": platform.processor() or platform.machine(),
            "code": package_code_sha(["sim", "contracts", "policies"])}
    out = {"provisional": True, "synthetic": True, "policy": pol.method_id, **{k: v for k, v in m.items() if k != "raw_us"},
           "cch_body_preview": body, "cch_preview": provenance_hash("CCH", body), "raw_us": m["raw_us"]}
    txt = json.dumps(out, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(json.dumps({k: v for k, v in out.items() if k != "raw_us"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
