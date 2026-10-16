"""G1 / G2 after the PREREG-B snapshot, on train_fit/train_dev only, then the G2_EVALUATED research marker.

G1  paired-effect discrepancy online - replay vs PSP-default on the train_dev online benchmark (Amendment 1 rev. 2):
    every replayable method of the dev run is re-run on the train_dev bank (same prompt/bank), Y = delivered IR,
    failures = r_fail (PREREG-B); status from apsp.reports.replay_vs_online.g1_status with tau_prac.
G2  quality: O1-replay - B3 per primary cap on train_dev (apsp.reports.headroom), level 1 - 0.025/M.
    GO if LCB >= theta_H. Otherwise INC: O1-replay is only a bound inside the simulator, so replay alone never gives
    STOP (plan v1.3 E3). G2-cost (primal witness) is not evaluated here and is reported as INC.
RL training (P4) is allowed after this marker unless G1 is STOP-ENG.

  python -m apsp.reports.gates_b --bank BANK --costs COSTS --dev-run RUN --out RUN/gates_b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from apsp.contracts.ledger import load_method_costs
from apsp.contracts.prereg import require_signed, signed_dir
from apsp.contracts.schema import load_bank
from apsp.reports.headroom import headroom
from apsp.reports.replay_vs_online import analyse
from apsp.sim.env import ReplayEngine, episode_reward
from apsp.sim.methods import ONLINE_ONLY, MethodError, RunContext, parse_method, run_method
from apsp.sim.replay import ReplayContext
from apsp.stats.bootstrap import contrast_level
from apsp.stats.decisions import GO, INC

REPO = Path(__file__).resolve().parents[2]


def _locked_b(root: Path = REPO) -> dict:
    m = require_signed("B", root)
    return json.loads((root / m["proposal_path"]).read_text(encoding="utf-8"))


def g1_records(ctx: ReplayContext, dev: pd.DataFrame, r_fail: float, disp_us: int = 0, once_us: int = 0) -> tuple[list, list]:
    rc = RunContext(ctx.base_costs, ctx.caps, ctx.norm, ctrl_disp_us=disp_us, selector_once_us=once_us)
    dev = dev.assign(Y=dev["q_eval"].astype(float).where(dev["success"].astype(bool) & dev["q_eval"].notna(), r_fail))
    on = dev.set_index(["method", "prompt_uid", "bank_id"])["Y"]
    rep_cache: dict = {}
    skipped = []

    def replay_y(spec, uid, b):
        k = (spec, uid, b)
        if k not in rep_cache:
            res = run_method(parse_method(spec), ReplayEngine(ctx.accessor, uid), bank_id=b, ctx=rc,
                             prompt_feat=ctx.prompt_feat(uid), is_replay=True)
            rep_cache[k] = episode_reward(res, ctx.accessor, uid, r_fail) if res.feasible else r_fail
        return rep_cache[k]

    recs = []
    for spec in sorted(set(dev["method"]) - {"psp"}):
        m = parse_method(spec)
        if m.family in ONLINE_ONLY or m.per_prompt():
            skipped.append(spec)
            continue
        for (uid, b) in sorted({(u, int(bb)) for u, bb in dev.loc[dev["method"] == spec, ["prompt_uid", "bank_id"]].itertuples(index=False)}):
            if ("psp", uid, b) not in on.index:
                continue
            try:
                recs.append({"uid": uid, "bank_id": b, "method": spec, "y_online": float(on[(spec, uid, b)]),
                             "yref_online": float(on[("psp", uid, b)]), "y_replay": replay_y(spec, uid, b),
                             "yref_replay": replay_y("psp", uid, b)})
            except MethodError:
                skipped.append(spec)
                break
    return recs, sorted(set(skipped))


def evaluate(bank_dir: str, costs: str, dev_run: str, snap: dict, disp_us: int = 0, once_us: int = 0,
             fit_split: str = "train_fit", dev_split: str = "train_dev") -> dict:
    bank = load_bank(bank_dir)
    r_fail = float(snap["B4_r_fail"])
    th = snap["B3_thresholds"]
    if (fit_split, dev_split) != ("train_fit", "train_dev") and not bank.manifest.get("synthetic"):
        raise SystemExit("only synthetic banks may use other splits")
    ctx = ReplayContext.build(bank, load_method_costs(costs), fit_split=fit_split, r_fail=r_fail)
    rows = [json.loads(l) for l in (Path(dev_run) / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = pd.DataFrame(rows)
    if set(dev["split"]) != {dev_split}:
        raise SystemExit("G1 uses the train_dev online run only")
    recs, skipped = g1_records(ctx, dev, r_fail, disp_us, once_us)
    by_m = {}
    for spec in sorted({r["method"] for r in recs}):
        by_m[spec] = analyse([r for r in recs if r["method"] == spec], th["tau_prac"])
    g1_all = analyse(recs, th["tau_prac"]) if recs else {"status": "INCONCLUSIVE_EXTEND_TRAIN_DEV", "n_records": 0}
    level = contrast_level(int(snap["B12_contrasts"]["M"]))
    hr = headroom(ctx, fit_split, dev_split, alphas=(0.5, 1.0), level=level)
    g2 = {}
    for a, h in hr.items():
        if not h.get("feasible", True):
            g2[a] = {**h, "decision": "NA (infeasible cap)"}
            continue
        g2[a] = {**h, "theta_H": th["theta_H"], "decision": GO if h["ci"][0] >= th["theta_H"] else INC,
                 "note": "O1-replay is a simulator bound; no STOP from replay alone"}
    status = GO if any(v["decision"] == GO for v in g2.values()) else INC
    return {"label": "G1/G2 on train_fit/train_dev (PREREG-B snapshot)", "tau_prac": th["tau_prac"], "r_fail": r_fail,
            "G1": {"pooled": g1_all, "by_method": by_m, "not_replayable": skipped,
                   "records_n": len(recs)}, "G2_quality": g2, "G2_quality_status": status,
            "G2_cost": {"status": INC, "reason": "primal witness not evaluated"},
            "rl_training_allowed": g1_all.get("status") != "STOP-ENG"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1/G2 gates (train_fit/train_dev) and the G2_EVALUATED marker")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--dev-run", required=True)
    ap.add_argument("--disp-us", type=int, default=0)
    ap.add_argument("--selector-once-us", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    snap = _locked_b()
    rep = evaluate(a.bank, a.costs, a.dev_run, snap, a.disp_us, a.selector_once_us)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    marker = signed_dir() / "G2_EVALUATED.json"
    if not marker.exists():
        marker.write_text(json.dumps({
            "marker": "G2_EVALUATED", "role": "research",
            "signed_by": "automatic gate evaluation under the project owner's authorization (AUTHORIZATION_20261017.md); "
                         "not a signature of anyone else",
            "g1_status": rep["G1"]["pooled"].get("status"), "g2_quality_status": rep["G2_quality_status"],
            "rl_training_allowed": rep["rl_training_allowed"], "report": str(out),
            "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"G1": rep["G1"]["pooled"].get("status"), "G2_quality": rep["G2_quality_status"],
                      "rl_training_allowed": rep["rl_training_allowed"]}, indent=1))
    return 0 if rep["rl_training_allowed"] else 6


if __name__ == "__main__":
    sys.exit(main())
