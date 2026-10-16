"""PREREG-B protocol-lock snapshot (owner authorization of 2026-10-17, docs/PREREG.md).

Every value comes from a formula fixed in docs/PREREG_PROPOSALS/PREREG_BC_DECISION_PACKAGE.md / Spec §11-§12, applied to
train_fit / train_dev data only:

  r_fail        Q_0.001(valid delivered IR | train_fit bank) - SD(same), population SD           (Spec §12, B4)
  sigma_R       population SD of valid delivered IR over every train_fit candidate, floor 1e-6     (plan v1.3, B5)
  Delta_anchor  ONLINE train_fit runs, paired by prompt: mean(Q_PSP - Q_BoN4), banks averaged    (plan v1.3 E3, B2)
                (replay train_fit value reported alongside as the simulator approximation)
  tau_prac      max(0.2 Delta_anchor, 0.02) in absolute IR units; theta_H = tau; delta_IR = theta_q = tau/2;
                theta_RL = theta_pay = tau/4                                                     (Spec §11.3, B3)
  M             3 primary contrasts (frozen operating point vs PSP-default; G3 at alpha 0.5; G3 at alpha 1.0)
  N             paired power 80 % at alpha_eff = 0.025/M (two-sided CI level 1 - alpha_eff), true difference 0, margin
                delta_IR, variance components (sigma_p^2, sigma_b^2) of Q_m - Q_PSP on the train_dev online runs, the
                largest N over the tuned adaptive candidates                                    (Spec §11.2, B13)
  looks         look 1 = val (80 prompts); look 2 only if a primary contrast is INC after look 1, adding
                min(220, max(0, N - 80)) val_ext prompts (total <= 300); N > 300 is reported as a power shortfall

Writes the snapshot JSON, the locked evaluation config and the PREREG-B marker. Refuses if any val output exists.

  python -m apsp.reports.lock_b --bank BANK --costs COSTS --anchor-online RUN --dev-run RUN --anchor-replay FILE \
      --val-guard RUNS/pilot1/bench_val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from apsp.contracts.access import PrivilegedAccessor
from apsp.contracts.schema import load_bank
from apsp.sim.replay import failure_penalty
from apsp.stats.decisions import thresholds
from apsp.stats.variance import nested_components

REPO = Path(__file__).resolve().parents[2]
AUTH = "docs/PREREG_SIGNED/AUTHORIZATION_20261017.md"
FIXED = {"single", "single32", "bon", "psp"}
M_PRIMARY = 3
VAL_N, VAL_EXT_MAX, TOTAL_MAX = 80, 220, 300


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _z(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def _episodes(run: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def _q(df: pd.DataFrame, r_fail: float) -> pd.Series:
    q = df["q_eval"].astype(float)
    return q.where(df["success"].astype(bool) & q.notna(), r_fail)


def bank_stats(bank_dir: Path, fit_split: str = "train_fit") -> dict:
    bank = load_bank(bank_dir)
    if fit_split != "train_fit" and not bank.manifest.get("synthetic"):
        raise SystemExit("only synthetic banks may use another fit split")
    acc = PrivilegedAccessor(bank)
    fit = bank.instances_of([fit_split])
    dl = [acc.instance_arrays(k, "training_label")["delivered_ir"] for k in fit]
    v = np.concatenate([np.asarray(d, float).ravel() for d in dl])
    v = v[np.isfinite(v)]
    return {"r_fail": failure_penalty(dl), "sigma_R": max(float(v.std(ddof=0)), 1e-6), "n_valid_candidates": int(v.size),
            "n_instances": len(fit), "bank_bmh": bank.manifest.get("bmh"), "bank_eph": bank.manifest.get("eph")}


def online_anchor(df: pd.DataFrame, r_fail: float, split: str = "train_fit") -> dict:
    if set(df["split"]) != {split}:
        raise SystemExit(f"anchor run must be train_fit only, got {sorted(set(df['split']))}")
    df = df.assign(Q=_q(df, r_fail))
    pp = df.groupby(["method", "prompt_uid"])["Q"].mean()
    a, b = pp.get("psp"), pp.get("bon:4")
    if a is None or b is None:
        raise SystemExit("anchor run needs methods psp and bon:4")
    keys = sorted(set(a.index) & set(b.index))
    d = np.array([a[k] - b[k] for k in keys])
    return {"delta_anchor": float(d.mean()), "n_prompts": len(keys), "sd_prompt_diff": float(d.std(ddof=1)),
            "n_episodes": int(len(df)), "failures": int((~df["success"].astype(bool)).sum())}


def power(dev: pd.DataFrame, r_fail: float, delta_ir: float, M: int, split: str = "train_dev") -> dict:
    if set(dev["split"]) != {split}:
        raise SystemExit(f"power uses train_dev only, got {sorted(set(dev['split']))}")
    dev = dev.assign(Q=_q(dev, r_fail))
    ref = dev[dev["method"] == "psp"].set_index(["prompt_uid", "bank_id"])["Q"]
    a_eff = 0.025 / M
    zsum = _z(1 - a_eff / 2) + _z(0.8)
    per = {}
    for m, g in dev.groupby("method"):
        if m == "psp":
            continue
        q = g.set_index(["prompt_uid", "bank_id"])["Q"]
        common = sorted(set(q.index) & set(ref.index))
        byp: dict = {}
        for k in common:
            byp.setdefault(k[0], []).append(float(q[k] - ref[k]))
        byp = {u: v for u, v in byp.items() if len(v) == 2}
        if len(byp) < 3:
            continue
        c = nested_components(byp)
        var1 = c["sigma_p2"] + c["sigma_b2"] / 2           # per-prompt variance of the bank-averaged difference
        n = math.ceil(zsum ** 2 * var1 / delta_ir ** 2) if delta_ir > 0 else None
        per[m] = {**c, "var_per_prompt": var1, "N_required": n, "adaptive": m.split(":")[0] not in FIXED}
    cand = {m: v for m, v in per.items() if v["adaptive"]} or per
    N = max((v["N_required"] for v in cand.values() if v["N_required"] is not None), default=None)
    return {"alpha_eff": a_eff, "z_sum": zsum, "by_method": per, "N_required": N,
            "basis": "adaptive candidates" if any(v["adaptive"] for v in per.values()) else "all methods (no adaptive)"}


def build(a) -> dict:
    for g in a.val_guard:
        p = Path(g)
        if p.exists() and any(p.iterdir() if p.is_dir() else [p]):
            raise SystemExit(f"val output exists ({p}); PREREG-B must be locked before any val outcome is read")
    bs = bank_stats(Path(a.bank))
    anc = online_anchor(_episodes(Path(a.anchor_online)), bs["r_fail"])
    th = thresholds(anc["delta_anchor"])
    pw = power(_episodes(Path(a.dev_run)), bs["r_fail"], th.delta_IR, M_PRIMARY)
    N = pw["N_required"]
    look2 = min(VAL_EXT_MAX, max(0, (N or TOTAL_MAX) - VAL_N))
    replay = json.loads(Path(a.anchor_replay).read_text(encoding="utf-8")) if a.anchor_replay else {}
    inputs = {k: {"path": str(p), "sha256": _sha(p)} for k, p in
              (("bank_manifest", Path(a.bank) / "manifest.json"), ("costs", Path(a.costs)),
               ("anchor_online_episodes", Path(a.anchor_online) / "episodes.jsonl"),
               ("dev_episodes", Path(a.dev_run) / "episodes.jsonl"),
               *((("anchor_replay", Path(a.anchor_replay)),) if a.anchor_replay else ()))}
    return {
        "prereg": "B", "kind": "protocol_lock_snapshot", "authorization_record": AUTH,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_used": ["train_fit (bank, online anchor runs)", "train_dev (online benchmark)"], "val_read": False,
        "B1_reference": "psp (PSP-default 8->4->2, cuts j=4,8)",
        "B2_anchor": {"definition": "online train_fit, paired by prompt, banks averaged: mean(Q_PSP - Q_BoN4)", **anc,
                      "replay_train_fit_approximation": replay.get("delta_anchor_B2_minus_BoN4")},
        "B3_thresholds": {"tau_prac": th.tau_prac, "theta_H": th.theta_H, "delta_IR": th.delta_IR, "theta_q": th.theta_q,
                          "theta_RL": th.theta_RL, "theta_pay": th.theta_pay, "units": "absolute ImageReward units",
                          "rule": "tau = max(0.2 Delta_anchor, 0.02)"},
        "B4_r_fail": bs["r_fail"], "B5_sigma_R": bs["sigma_R"],
        "B5_definition": "population SD of valid delivered IR over all train_fit candidates (floor 1e-6)",
        "B6_norm_pca": "NormStats / PCA-16 fitted on train_fit (tune_baselines artifacts, hashes in methods_frozen.json)",
        "B7_caps": "alpha {0.25,0.5,1.0,1.5} x C_ref of the locked cost table; primary 0.5, 1.0; infeasible -> NA",
        "B8_selection": "per cap, the candidate in {B6,B7,B8,RL*} with the best train_dev online mean Q (G3); operating "
                        "point = lowest actual mean e2e cost among methods passing NI vs PSP on train_dev",
        "B9_rl": "REINFORCE/MLP (apsp.policies.reinforce defaults), 3 training seeds {0,1,2}, checkpoint = best replay "
                 "train_dev score; every seed reported; the online claim uses all seeds; ledger dispatch fee 0 for "
                 "every closed-loop method (equal accounting), measured dispatch reported, actual e2e includes it",
        "B10_endpoints": "primary quality = delivered IR; primary efficiency = actual e2e seconds; HPS v2.1 diagnostic",
        "B11_decision": "GO: LCB(dQ) > -delta_IR and LCB(savings) >= 10%; STOP: UCB(savings) < 5% or UCB(dQ) < -delta_IR; "
                        "else INC (Spec §11.3)",
        "B12_contrasts": {"M": M_PRIMARY, "ci_level_per_look": 1 - 0.025 / M_PRIMARY, "bootstrap_resamples": 10000,
                          "primary": ["operating_point_vs_psp", "G3_alpha_0.5", "G3_alpha_1.0"],
                          "secondary": ["G3b per cap (best paid vs B10-min)", "G4 per cap (RL* vs best of B6/B8)",
                                        "every tuned and fixed method vs PSP"]},
        "B13_sample_size": {**pw, "look1_val_prompts": VAL_N, "look2_val_ext_prompts_if_inc": look2,
                            "power_shortfall": bool(N and N > TOTAL_MAX)},
        "B14_amendment_1": "rev. 2 accepted (PREREG-A A17): replay = simulator; confirmatory online; G1 = CI of the "
                           "online-replay paired-effect discrepancy vs PSP-default; wide CI -> INC",
        "bank": {k: bs[k] for k in ("bank_bmh", "bank_eph", "n_instances", "n_valid_candidates")},
        "inputs": inputs,
    }


def write(snap: dict, a) -> dict:
    prop = REPO / a.proposal
    marker = REPO / a.marker
    if marker.exists():
        raise SystemExit(f"{marker} already exists; a locked snapshot is never replaced")
    prop.parent.mkdir(parents=True, exist_ok=True)
    prop.write_text(json.dumps(snap, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    ev = yaml.safe_load((REPO / a.eval_template).read_text(encoding="utf-8"))
    ev.update({"approved": True, "approved_by": "PREREG-B protocol-lock snapshot under the owner's authorization "
                                                f"({AUTH})", "epsilon_ir": snap["B3_thresholds"]["delta_IR"],
               "r_fail": snap["B4_r_fail"], "n_primary_contrasts": M_PRIMARY, "reference_method": "psp",
               "prereg_b_snapshot": a.proposal, "prereg_b_snapshot_sha256": _sha(prop)})
    evp = REPO / a.eval_out
    evp.write_text("# GENERATED by apsp.reports.lock_b (PREREG-B snapshot); do not edit\n"
                   + yaml.safe_dump(ev, sort_keys=False), encoding="utf-8", newline="\n")
    m = {"prereg": "B", "role": "research",
         "signed_by": "protocol-lock snapshot recorded by the agent under the project owner's authorization; "
                      "not a signature of anyone else",
         "approval_kind": "protocol_lock_snapshot", "authorization_record": AUTH,
         "authorization_sha256": _sha(REPO / AUTH), "proposal_path": a.proposal, "proposal_sha256": _sha(prop),
         "evaluation_config": a.eval_out, "evaluation_config_sha256": _sha(evp),
         "recorded_at": snap["locked_at"], "recorded_by": "apsp.reports.lock_b (Claude Code agent pipeline)"}
    marker.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8", newline="\n")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PREREG-B protocol-lock snapshot (train_fit/train_dev only)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--anchor-online", required=True, help="online train_fit run dir (psp, bon:4)")
    ap.add_argument("--dev-run", required=True, help="online train_dev benchmark run dir")
    ap.add_argument("--anchor-replay", default=None, help="replay anchor proposal (apsp.reports.anchor)")
    ap.add_argument("--val-guard", nargs="*", default=[], help="paths that must be absent/empty (val outputs)")
    ap.add_argument("--proposal", default="docs/PREREG_PROPOSALS/PREREG_B_LOCKED.json")
    ap.add_argument("--marker", default="docs/PREREG_SIGNED/PREREG-B.json")
    ap.add_argument("--eval-template", default="configs/evaluation.yaml")
    ap.add_argument("--eval-out", default="configs/evaluation_locked.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    snap = build(a)
    if a.dry_run:
        print(json.dumps(snap, indent=1, default=float))
        return 0
    m = write(snap, a)
    print(json.dumps({"locked": a.proposal, "proposal_sha256": m["proposal_sha256"],
                      "delta_IR": snap["B3_thresholds"]["delta_IR"], "N": snap["B13_sample_size"]["N_required"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
