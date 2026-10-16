"""Freeze methods, operating point and contrasts on train_dev (PREREG-B snapshot rules B8/B12), before any val look.

Inputs: train_dev online runs (fixed + tuned baselines, RL seeds), the locked evaluation config and the B snapshot.
  operating point  lowest actual mean e2e among methods passing NI vs PSP on train_dev (reference if none)
  G3  (per cap)    candidate in {B6 threshold, B7, B8, RL*} with the highest train_dev mean Q   vs B3 (fixed@cap)
  G3b (per cap)    best paid in {B6, B8, RL*} by train_dev mean Q                               vs B10-min@cap
  G4  (per cap)    RL* (all seeds pooled)                                            vs the better of {B6, B8}@cap
RL* = the RL configuration; its seeds are pooled per prompt in every contrast (no best-seed claim).
Every train_dev method is carried to val (no new grid on val).

  python -m apsp.reports.select_dev --runs RUN/bench_dev RUN/bench_dev_rl --config configs/evaluation_locked.yaml \
      --snapshot docs/PREREG_PROPOSALS/PREREG_B_LOCKED.json --out STORE/frozen_operating_point.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from apsp.reports.quality_cost import build_report, quality
from apsp.sim.methods import parse_method

PAID = ("threshold", "b8", "rl")
G3_FAMILIES = ("threshold", "b7", "b8", "rl")


def rl_star(spec: str) -> str:
    m = parse_method(spec)
    return f"rl*@{m.alpha}" if m.family == "rl" else spec


def load_runs(runs) -> tuple[pd.DataFrame, dict]:
    frames, man = [], {}
    for r in runs:
        p = Path(r)
        rows = [json.loads(l) for l in (p / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        frames.append(pd.DataFrame(rows))
        if not man and (p / "manifest.json").exists():
            man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    return pd.concat(frames, ignore_index=True), man


def _fam_alpha(spec: str):
    if spec.startswith("rl*@"):
        return "rl", float(spec.split("@")[1])
    m = parse_method(spec)
    return m.family, m.alpha


def contrasts_for(df: pd.DataFrame, r_fail: float, alphas=(0.5, 1.0)) -> list[dict]:
    d = df.assign(method=df["method"].map(rl_star), Q=quality(df, r_fail))
    meanq = d.groupby("method")["Q"].mean().to_dict()
    out = []
    for a in alphas:
        at = {s: _fam_alpha(s) for s in meanq if _fam_alpha(s)[1] == a}
        fam = lambda fs: [s for s, (f, _) in at.items() if f in fs]
        b3 = fam(("fixed",))
        b10 = fam(("b10",))
        g3c = sorted(fam(G3_FAMILIES), key=lambda s: (-meanq[s], s))
        paid = sorted(fam(PAID), key=lambda s: (-meanq[s], s))
        b6b8 = sorted(fam(("threshold", "b8")), key=lambda s: (-meanq[s], s))
        rl = fam(("rl",))
        if g3c and b3:
            out.append({"id": f"G3_alpha_{a}", "method": g3c[0], "reference": b3[0], "primary": True, "theta": "theta_q",
                        "rule": "best train_dev mean Q in {B6,B7,B8,RL*} vs B3"})
        if paid and b10:
            out.append({"id": f"G3b_alpha_{a}", "method": paid[0], "reference": b10[0], "primary": False,
                        "theta": "theta_pay", "rule": "best paid in {B6,B8,RL*} vs B10-min"})
        if rl and b6b8:
            out.append({"id": f"G4_alpha_{a}", "method": rl[0], "reference": b6b8[0], "primary": False,
                        "theta": "theta_RL", "rule": "RL* (seeds pooled) vs the better of {B6,B8}"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    snap = json.loads(Path(a.snapshot).read_text(encoding="utf-8"))
    if not cfg.get("approved"):
        raise SystemExit("evaluation config not locked")
    df, man = load_runs(a.runs)
    if set(df["split"]) != {"train_dev"}:
        raise SystemExit("selection uses train_dev only")
    man = {**man, "split": "train_dev"}
    rep = build_report(df, man, cfg)
    summ = {r["method"]: r for r in rep["summary"]}
    passing = sorted([c for c in rep["contrasts_vs_reference"] if c.get("ni") == "pass"],
                     key=lambda c: (summ[c["method"]]["mean_e2e_s"], c["method"]))
    op = passing[0]["method"] if passing else "psp"
    cons = [{"id": "operating_point_vs_psp", "method": op, "reference": "psp", "primary": True,
             "theta": "delta_IR", "rule": "NI + savings (Spec §11.3 cost rule)"}] + contrasts_for(df, float(cfg["r_fail"]))
    n_primary = sum(c["primary"] for c in cons)
    if n_primary > int(cfg["n_primary_contrasts"]):
        raise SystemExit(f"{n_primary} primary contrasts > locked M={cfg['n_primary_contrasts']}")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen = {"selected_on": "train_dev", "operating_point": op, "reference": "psp",
              "methods": sorted(set(df["method"])), "contrast_manifest": cons, "M_locked": int(cfg["n_primary_contrasts"]),
              "n_primary_present": n_primary, "thresholds": snap["B3_thresholds"],
              "look_plan": {"look1": "val", "look2_val_ext_prompts_if_inc": snap["B13_sample_size"]["look2_val_ext_prompts_if_inc"]},
              "evaluation_config": a.config, "evaluation_config_sha256": sha(a.config),
              "snapshot": a.snapshot, "snapshot_sha256": sha(a.snapshot),
              "train_dev_runs": {r: sha(Path(r) / "episodes.jsonl") for r in a.runs},
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rule": "operating point: lowest mean e2e among NI-passing methods vs PSP on train_dev; reference if none"}
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; the frozen selection is never replaced")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"operating_point": op, "contrasts": [c["id"] + ": " + c["method"] + " vs " + c["reference"] for c in cons]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
