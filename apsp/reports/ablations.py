"""E7 ablations (plan v1.3), configured on train_fit/train_dev only; dev-level evidence, not confirmatory.

  verifier cost x {0, 0.5, 2}  COUNTERFACTUAL replay accounting on train_dev: the frozen replayable methods re-run with
                               the verifier primitive scaled (MethodCosts.scaled_verifier); never called wall-clock
  online dev contrasts         RL variants (seeds pooled per prompt) and baseline families on train_dev online runs:
                               nopaid vs main (does the policy use purchased information?), traj (paid trajectory
                               features) vs main, per-cap vs budget-conditioned, B10-min vs best paid, threshold /
                               one-query vs RL*; 95 % descriptive CIs, savings on actual e2e
  also reported elsewhere      replay vs online (G1, gates_b.json)
  not run                      A1 (VERIFY forced with PRUNE: grammar variant not implemented), reward-preview terminal
                               (protocol uses the delivered image)

  python -m apsp.reports.ablations --bank BANK --costs COSTS --frozen FROZEN --runs RUN/bench_dev RUN/bench_dev_rl \
      RUN/bench_dev_ablate --out RUN/ablations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from apsp.contracts.ledger import load_method_costs
from apsp.contracts.schema import load_bank
from apsp.reports.quality_cost import contrast, quality
from apsp.reports.select_dev import load_runs
from apsp.sim.env import ReplayEngine, episode_reward
from apsp.sim.methods import ONLINE_ONLY, RunContext, parse_method, run_method
from apsp.sim.replay import ReplayContext


def group(spec: str) -> str:
    """RL seeds pooled per configuration: rl:<root>/<variant>_seed<i>/artifact.json@a -> rl-<variant>*@a."""
    m = parse_method(spec)
    if m.family != "rl":
        return spec
    d = Path(m.artifact).parent.name
    variant = d.rsplit("_seed", 1)[0] if "_seed" in d else "main"
    return f"rl-{variant}*@{m.alpha}"


def verifier_cost(bank_dir: str, costs: str, methods: list, r_fail: float, factors=(0.0, 0.5, 1.0, 2.0)) -> dict:
    bank = load_bank(bank_dir)
    base = load_method_costs(costs)
    dev = bank.instances_of(["train_dev"])
    out = {}
    for f in factors:
        mc = base.scaled_verifier(f)
        ctx = ReplayContext.build(bank, mc, fit_split="train_fit", r_fail=r_fail)
        rc = RunContext(mc, ctx.caps, ctx.norm)
        row = {}
        for spec in methods:
            m = parse_method(spec)
            if m.family in ONLINE_ONLY or m.per_prompt():
                continue
            ys, sp, feas = [], [], 0
            for uid, b in dev:
                res = run_method(m, ReplayEngine(ctx.accessor, uid), bank_id=b, ctx=rc, prompt_feat=ctx.prompt_feat(uid),
                                 is_replay=True)
                feas += bool(res.feasible)
                ys.append(episode_reward(res, ctx.accessor, uid, r_fail) if res.feasible else r_fail)
                sp.append(res.spend_us)
            row[group(spec) + ("" if group(spec) == spec else f" [{spec}]")] = {
                "mean_Y": float(np.mean(ys)), "mean_ledger_us": float(np.mean(sp)), "feasible_frac": feas / len(dev)}
        out[str(f)] = row
    return {"label": "COUNTERFACTUAL replay accounting (verifier primitive scaled); not measured wall-clock",
            "split": "train_dev", "by_factor": out}


PAIRS = [("rl-nopaid*", "rl-main*", "paid-score features removed vs RL main"),
         ("rl-traj*", "rl-main*", "paid trajectory features added vs RL main"),
         ("rl-percap*", "rl-main*", "one policy per cap vs budget-conditioned RL main"),
         ("b10", "best_paid", "B10-min (no mid-run query) vs best paid method"),
         ("threshold", "rl-main*", "threshold controller vs RL main"),
         ("b8", "rl-main*", "one query then commit vs RL main")]


def online_contrasts(df, cfg: dict, alphas=(0.5, 1.0)) -> list:
    df = df.assign(method=df["method"].map(group), Q=quality(df, cfg["r_fail"]))
    meth = set(df["method"])
    meanq = df.groupby("method")["Q"].mean()
    out = []
    for a in alphas:
        at = lambda prefix: sorted(m for m in meth if m.split(":")[0].split("@")[0] == prefix.split("@")[0]
                                   and m.endswith(f"@{a}"))
        best_paid = sorted([m for m in meth if m.endswith(f"@{a}") and (m.startswith(("threshold:", "b8:", "rl-main*")))],
                           key=lambda m: -meanq[m])
        for x, y, what in PAIRS:
            xs = at(x)
            ys = best_paid[:1] if y == "best_paid" else at(y)
            if not xs or not ys:
                out.append({"alpha": a, "ablation": what, "status": "not run"})
                continue
            c = contrast(df, xs[0], ys[0], {**cfg, "approved": False}, 0.95, cfg.get("cost_measure", "e2e_seconds"))
            out.append({"alpha": a, "ablation": what, **c, "note": "train_dev online, descriptive 95 % CI"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--config", default="configs/evaluation_locked.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    df, _ = load_runs([r for r in a.runs if Path(r, "episodes.jsonl").exists()])
    if set(df["split"]) != {"train_dev"}:
        raise SystemExit("ablations are evaluated on train_dev only")
    rep = {"verifier_cost": verifier_cost(a.bank, a.costs, sorted(set(df["method"]) | set(frozen["methods"])),
                                          float(cfg["r_fail"])),
           "online_dev": online_contrasts(df, cfg),
           "not_run": {"A1 VERIFY forced with PRUNE": "grammar variant not implemented",
                       "reward-preview terminal": "protocol fixes the delivered-image reward"},
           "see_also": {"replay_vs_online": "gates_b.json (G1)"}}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps([(r["alpha"], r["ablation"], r.get("dQ"), r.get("savings")) for r in rep["online_dev"]], default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
