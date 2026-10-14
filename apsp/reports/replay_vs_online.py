"""M7 / G1: paired-effect replay-online discrepancy (Spec §10.2).

Delta_disc = mean_online(Y_m - Y_ref) - mean_replay(Y_m - Y_ref), cluster bootstrap by prompt.
Status: CI within +-0.25 tau -> replay usable for gates; within +-0.5 tau -> replay for training only, gates online;
wide CI -> extend train_dev checks (not parity evidence); beyond -> STOP-ENG.

  python -m apsp.reports.replay_vs_online --records data/runs/G1/records.jsonl --tau 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apsp.stats.bootstrap import cluster_bootstrap


def g1_status(lo: float, hi: float, tau: float) -> str:
    if -0.25 * tau <= lo and hi <= 0.25 * tau:
        return "REPLAY_FOR_GATES"
    if -0.5 * tau <= lo and hi <= 0.5 * tau:
        return "REPLAY_TRAIN_ONLY_GATES_ONLINE"
    if hi < -0.5 * tau or lo > 0.5 * tau:
        return "STOP-ENG"
    return "INCONCLUSIVE_EXTEND_TRAIN_DEV"


def analyse(records: list[dict], tau: float, level: float = 0.95, seed: int = 0) -> dict:
    """records: {uid, bank_id, method, y_online, yref_online, y_replay, yref_replay}."""
    per_prompt: dict = {}
    agree = 0
    for r in records:
        d = (r["y_online"] - r["yref_online"]) - (r["y_replay"] - r["yref_replay"])
        per_prompt.setdefault(r["uid"], []).append(d)
        agree += int(r.get("winner_online") == r.get("winner_replay"))
    ci = cluster_bootstrap(per_prompt, level, seed=seed)
    return {"delta_disc": ci.estimate, "ci": [ci.lo, ci.hi], "level": level, "n_prompts": ci.n_clusters,
            "n_records": len(records), "winner_agreement": agree / max(len(records), 1),
            "status": g1_status(ci.lo, ci.hi, tau), "tau_prac": tau,
            "note": "winner agreement is secondary; the gate uses the paired-effect discrepancy"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay vs online paired-effect discrepancy (G1)")
    ap.add_argument("--records", required=True)
    ap.add_argument("--tau", type=float, required=True, help="tau_prac from the signed PREREG-B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    recs = [json.loads(l) for l in Path(a.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = analyse(recs, a.tau)
    txt = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if rep["status"] != "STOP-ENG" else 4


if __name__ == "__main__":
    sys.exit(main())
