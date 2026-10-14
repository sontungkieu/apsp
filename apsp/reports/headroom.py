"""G2-quality diagnostics: O1 - B3 per cap with a prompt-cluster bootstrap CI.

Train/dev splits can be inspected in P1/P2 after PREREG-B; val needs the look guard. Synthetic banks are free.

  python -m apsp.reports.headroom --bank DIR --costs COSTS.json --split train_dev --look 1
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from apsp.contracts.ledger import load_method_costs
from apsp.contracts.prereg import require_signed, require_split_access
from apsp.contracts.schema import load_bank
from apsp.policies.fixed import best_fixed
from apsp.sim.oracles import o1
from apsp.sim.replay import ReplayContext
from apsp.stats.bootstrap import cluster_bootstrap


def headroom(ctx: ReplayContext, fit_split: str, eval_split: str, alphas=(0.5, 1.0), level: float = 0.95) -> dict:
    fit = ctx.bank.instances_of([fit_split])
    ev = ctx.bank.instances_of([eval_split])
    Yf, C = ctx.schedule_table(fit)
    Ye, _ = ctx.schedule_table(ev)
    out = {}
    for a in alphas:
        cap = ctx.caps.cap(a)
        b3 = best_fixed(Yf, C, cap)
        if b3 is None:
            out[str(a)] = {"feasible": False}
            continue
        from apsp.sim.schedules import enumerate_pairs
        y_b3 = Ye[:, enumerate_pairs().index((b3.j1, b3.j2))]
        diff = o1(Ye, C, cap) - y_b3
        per = {}
        for (uid, _), d in zip(ev, diff):
            per.setdefault(uid, []).append(float(d))
        ci = cluster_bootstrap(per, level)
        out[str(a)] = {"cap_us": cap, "B3": [b3.j1, b3.j2], "mean_O1_minus_B3": float(np.mean(diff)),
                       "ci": [ci.lo, ci.hi], "level": level, "descriptive": True}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="O1 - B3 headroom per cap (G2-quality diagnostic)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--split", default="train_dev")
    ap.add_argument("--fit-split", default="train_fit")
    ap.add_argument("--look", type=int, default=1)
    a = ap.parse_args(argv)
    bank = load_bank(a.bank)
    if not bank.manifest.get("synthetic"):
        require_signed("B")
        require_split_access(a.split, "headroom")
    ctx = ReplayContext.build(bank, load_method_costs(a.costs), fit_split=a.fit_split)
    print(json.dumps({"synthetic": bool(bank.manifest.get("synthetic")), "look": a.look,
                      "headroom": headroom(ctx, a.fit_split, a.split)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
