"""Phase forecast P2-P7 from MEASURED calibration outputs on the measurement machine (docs/FULL_EXPERIMENT_PLAN.md §8).

Inputs (calibration run on the target GPU):
  --m3m9  calib/m3m9.json          M9 seconds per prompt-bank (clean collector)
  --costs p0/m2/generator_costs.json  merged M2 primitives (us; provisional until Q1)
  --m2b   calib/m2b.json           B2 online wall vs ledger (residual fraction applied to every ledger estimate)
  --price USD per hour (read from the provider at forecast time; recorded, not assumed)

Per method, episode seconds = ledger full-trace cost (schedule_cost) x (1 + max(0, m2b residual)) + fixed(k0) when
measured. Capped methods (B3/B4b/B5/B6/B7/B8/B10/RL at alpha) are bounded by their cap B_alpha (an upper bound).
Every row says what was measured and what is a count assumption. Synthetic tables are refused.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from apsp.contracts.identity import bank_rows, pool_rows
from apsp.contracts.ledger import CostTableError, load_method_costs, schedule_cost
from apsp.sim.caps import compute_caps, fixed_method_costs
from apsp.sim.schedules import bon_spec, fixed_pair_spec

SPLITS = {"calibration": 50, "train_fit": 190, "train_dev": 50, "val": 80, "val_ext": 220}
FIXED_ONLINE = ["single", "single32", "bon:2", "bon:4", "bon:8", "psp"]
TUNED_FAMILIES = ["b3", "b4b", "b5", "b6", "b7", "b8", "b10"]


def method_seconds(mc, residual: float) -> dict:
    fx = fixed_method_costs(mc)
    out = {}

    def sec(spec, k0):
        us = schedule_cost(spec, fx) + int(fx.fixed.get(k0, 0))
        return us / 1e6 * (1 + max(0.0, residual))
    out["single"] = sec(bon_spec(1, bank_rows(0)), 1)
    out["single32"] = sec(bon_spec(1, bank_rows(0), J=8), 1)          # 32 forwards = 8 blocks of delta=4
    for n in (2, 4, 8):
        out[f"bon:{n}"] = sec(bon_spec(n, bank_rows(0)), n)
    try:
        out["bon:16"] = sec(bon_spec(16, pool_rows(2)), 16)
    except CostTableError:
        out["bon:16"] = None
    out["psp"] = sec(fixed_pair_spec(4, 8, bank_rows(0)), 8)
    caps = compute_caps(fx)
    for a in (0.5, 1.0):
        out[f"cap@{a}"] = (caps.cap(a) + int(fx.fixed.get(8, 0))) / 1e6 * (1 + max(0.0, residual))
    return out


def forecast(m9_s: float, ms: dict, price: float, *, alphas=(0.5, 1.0), n_tuned=len(TUNED_FAMILIES),
             rl_seeds: int = 3, safety: float = 1.25, reserve_usd: float = 0.0) -> dict:
    dev_inst = SPLITS["train_dev"] * 2
    val_inst = SPLITS["val"] * 2
    fixed_dev = sum(ms[m] for m in FIXED_ONLINE) * dev_inst + (ms["bon:16"] or 0) * SPLITS["train_dev"]
    tuned_dev = sum(ms[f"cap@{a}"] for a in alphas) * n_tuned * dev_inst
    rl_dev = sum(ms[f"cap@{a}"] for a in alphas) * rl_seeds * dev_inst
    frozen_val = (sum(ms[m] for m in FIXED_ONLINE) + sum(ms[f"cap@{a}"] for a in alphas) * 3) * val_inst
    phases = [
        ("P2 bank train_fit+train_dev", (SPLITS["train_fit"] + SPLITS["train_dev"]) * 2 * m9_s,
         f"{SPLITS['train_fit'] + SPLITS['train_dev']} prompts x 2 banks x M9 {m9_s:.1f} s (measured)", "PREREG-A"),
        ("P3 online train_dev: fixed baselines", fixed_dev,
         f"{dev_inst} instances x {len(FIXED_ONLINE)} fixed methods (+BoN-16 per prompt), ledger x (1+M2b residual)",
         "PREREG-A"),
        ("P3 online train_dev: tuned baselines", tuned_dev,
         f"{n_tuned} families x {len(alphas)} caps x {dev_inst} instances, bounded by the cap", "PREREG-A"),
        ("P4 online train_dev: RL-M1 (3 seeds)", rl_dev, f"{rl_seeds} seeds x {len(alphas)} caps x {dev_inst} instances",
         "PREREG-B + G2 marker"),
        ("P5 online val (frozen)", frozen_val,
         f"{val_inst} instances x ({len(FIXED_ONLINE)} fixed + 3 frozen adaptive per cap)", "PREREG-B"),
        ("P6 GenEval test (553 prompts x 4 images x 6 methods)", 553 * 4 * 6 * ms["psp"] / 1,
         "official protocol size; method cost ~ PSP-default per image as a placeholder until PREREG-C", "PREREG-C"),
        ("P7 ablations online confirmation (dev)", 4 * sum(ms[f"cap@{a}"] for a in alphas) * dev_inst,
         "4 ablation variants x caps x dev instances (replay ablations are CPU)", "PREREG-B"),
    ]
    rows, tot_h = [], 0.0
    for name, sec, basis, gate in phases:
        h = sec / 3600
        tot_h += h
        rows.append({"phase": name, "gpu_hours": round(h, 2), "usd_central": round(h * price, 2),
                     "usd_with_safety": round(h * safety * price, 2), "basis": basis, "gate": gate})
    return {"price_usd_per_h": price, "safety_factor": safety, "rows": rows,
            "total_gpu_hours": round(tot_h, 2), "total_usd_central": round(tot_h * price, 2),
            "total_usd_with_safety_and_reserve": round(tot_h * safety * price + reserve_usd, 2),
            "method_seconds": {k: (round(v, 2) if v is not None else None) for k, v in ms.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3m9", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--m2b", default=None)
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--reserve-usd", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mc = load_method_costs(a.costs)
    if mc.synthetic:
        raise SystemExit("refusing a synthetic cost table: forecasts need measured primitives")
    m9 = json.loads(Path(a.m3m9).read_text(encoding="utf-8"))["m9"]["seconds_per_prompt_bank_clean_mean"]
    if m9 is None or not math.isfinite(m9):
        raise SystemExit("M9 not measured")
    resid = 0.0
    if a.m2b:
        resid = float(json.loads(Path(a.m2b).read_text(encoding="utf-8"))["residual_frac"])
    rep = forecast(float(m9), method_seconds(mc, resid), a.price, reserve_usd=a.reserve_usd)
    rep.update({"inputs": {"m3m9": a.m3m9, "costs": a.costs, "m2b": a.m2b, "m9_seconds": m9, "m2b_residual_frac": resid,
                           "costs_provisional": mc.provisional},
                "label": "forecast from measured calibration on the measurement machine; counts are plan assumptions"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase_forecast.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    lines = [f"# Phase forecast (price {a.price} USD/h, safety x{rep['safety_factor']})", "",
             "| Phase | GPU h | USD central | USD x safety | Basis | Gate |", "|---|---:|---:|---:|---|---|"]
    lines += [f"| {r['phase']} | {r['gpu_hours']} | {r['usd_central']} | {r['usd_with_safety']} | {r['basis']} | {r['gate']} |"
              for r in rep["rows"]]
    lines += ["", f"Total {rep['total_gpu_hours']} GPU h, {rep['total_usd_central']} USD central, "
              f"{rep['total_usd_with_safety_and_reserve']} USD with safety and reserve.", "",
              "Per-episode seconds: " + ", ".join(f"{k} {v}" for k, v in rep["method_seconds"].items())]
    (out / "PHASE_FORECAST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
