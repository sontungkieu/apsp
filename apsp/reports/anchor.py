"""M11: train anchor, thresholds, failure penalty, normalisation proposal (Guide CX-8). PROPOSAL ONLY.

Reads train_fit only (plus nothing from oracle / adaptive policy modules). Writes a PREREG-B proposal,
never a signed marker.

  python -m apsp.reports.anchor --bank data/banks/PILOT_BANK --costs data/costs/generator_costs.json --proposal-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from apsp.contracts.access import PrivilegedAccessor
from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import load_method_costs
from apsp.contracts.schema import load_bank
from apsp.sim.caps import compute_caps, fixed_method_costs
from apsp.sim.features import PCA16, NormStats
from apsp.sim.replay import failure_penalty
from apsp.sim.schedules import enumerate_pairs, pair_outcomes
from apsp.stats.decisions import thresholds
from apsp.stats.variance import nested_components


def anchor_proposal(bank_dir: str | Path, costs_path: str | Path, fit_split: str = "train_fit") -> dict:
    bank = load_bank(bank_dir)
    if fit_split != "train_fit" and not bank.manifest.get("synthetic"):
        raise SystemExit("non-train_fit anchor split is only allowed on SYNTHETIC banks")
    acc = PrivilegedAccessor(bank)
    mc = fixed_method_costs(load_method_costs(costs_path))
    keys = bank.instances_of([fit_split])
    arrs = {k: acc.instance_arrays(k, "training_label") for k in keys}
    r_fail = failure_penalty([a["delivered_ir"] for a in arrs.values()])
    ref_idx = enumerate_pairs().index((4, 8))
    y_b2, y_bon4 = {}, {}
    for k, a in arrs.items():
        outs = pair_outcomes(a["preview_ir"], a["delivered_ir"], k[1], mc)
        y_b2[k] = outs[ref_idx].y if outs[ref_idx].y is not None else r_fail
        d4 = a["delivered_ir"][:4]
        y_bon4[k] = float(np.nanmax(d4)) if np.isfinite(d4).any() else r_fail
    delta_anchor = float(np.mean(list(y_b2.values())) - np.mean(list(y_bon4.values())))
    th = thresholds(delta_anchor)
    norm = NormStats.fit([a["preview_ir"] for a in arrs.values()])
    uids = sorted({k[0] for k in keys})
    pca = PCA16.fit(np.stack([acc.prompt_embedding(u) for u in uids]))
    within_sd = float(np.sqrt(np.nanmean([np.nanvar(a["delivered_ir"]) for a in arrs.values()])))
    per_prompt = {}
    for k in keys:
        per_prompt.setdefault(k[0], []).append(y_b2[k] - y_bon4[k])
    comps = nested_components(per_prompt) if all(len(v) == 2 for v in per_prompt.values()) else None
    return {
        "status": "PROPOSAL - not signed; research signs PREREG-B",
        "bank_bmh": bank.manifest.get("bmh"), "bank_eph": bank.manifest.get("eph"),
        "synthetic": bool(bank.manifest.get("synthetic", False)), "fit_split": fit_split,
        "n_instances": len(keys), "n_prompts": len(uids),
        "caps": {str(a): c for a, c in compute_caps(mc).caps.items()},
        "delta_anchor_B2_minus_BoN4": delta_anchor, "thresholds": th.__dict__,
        "failure_penalty_r_fail": r_fail,
        "sigma_R_candidates": {"pooled_within_instance_sd_delivered_ir": within_sd,
                               "note": "sigma_R definition is an open research question (docs/QUESTIONS.md Q4)"},
        "norm_stats": norm.to_dict(), "norm_hash": norm.hash(), "pca_hash": pca.hash(),
        "variance_components_B2_minus_BoN4": comps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train anchor / thresholds proposal (PREREG-B proposal only)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", required=True, help="generator cost table JSON (MethodCosts)")
    ap.add_argument("--proposal-only", action="store_true", required=True)
    ap.add_argument("--fit-split", default="train_fit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    prop = anchor_proposal(a.bank, a.costs, a.fit_split)
    out = Path(a.out or f"docs/PREREG_PROPOSALS/PREREG-B_anchor_{(prop['bank_bmh'] or 'x')[:12]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prop, indent=2), encoding="utf-8")
    print(f"proposal written to {out} (unsigned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
