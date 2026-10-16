"""Confirmatory val report (PREREG-B snapshot): the frozen contrast manifest, two looks, decisions fixed in advance.

Look 1 = val; look 2 (only if a primary contrast is INC after look 1) = val + the locked number of val_ext prompts.
Per-contrast CI level 1 - 0.025/M per look (apsp.stats.bootstrap.contrast_level). Decisions (Spec §11.3):
  operating_point_vs_psp   cost rule (NI at delta_IR + savings 10 % / 5 %)
  G3 / G3b / G4            quality rule against theta_q / theta_pay / theta_RL
RL* pools its training seeds per prompt. INC after look 2 is reported as "inconclusive". Every look is written to
val_access.jsonl (at most two, config/CPH unchanged).

  python -m apsp.reports.confirm --runs RUN/bench_val [RUN/bench_val_ext] --frozen FROZEN --config CFG --look 1 \
      --val-log RUN/val_access.jsonl --out RUN/confirm_look1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

from apsp.contracts.prereg import ValAccessLog, require_signed
from apsp.reports.quality_cost import contrast, quality
from apsp.reports.select_dev import load_runs, rl_star
from apsp.stats.bootstrap import contrast_level
from apsp.stats.decisions import INC, after_second_look, cost_decision, quality_decision


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--look", type=int, choices=[1, 2], required=True)
    ap.add_argument("--val-log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    require_signed("B")
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    cfg_bytes = Path(a.config).read_bytes()
    if hashlib.sha256(cfg_bytes).hexdigest() != frozen["evaluation_config_sha256"]:
        raise SystemExit("evaluation config changed after the dev freeze")
    cfg = yaml.safe_load(cfg_bytes)
    df, man = load_runs(a.runs)
    allowed = {"val"} if a.look == 1 else {"val", "val_ext"}
    if not set(df["split"]) <= allowed:
        raise SystemExit(f"look {a.look} reads only {sorted(allowed)}, got {sorted(set(df['split']))}")
    if not set(df["method"]) <= set(frozen["methods"]):
        raise SystemExit("val run contains methods that were not frozen on train_dev")
    df = df.assign(method=df["method"].map(rl_star), Q=quality(df, cfg["r_fail"]))
    level = contrast_level(int(frozen["M_locked"]))
    th = frozen["thresholds"]
    cph = hashlib.sha256(json.dumps(sorted(frozen["methods"])).encode()).hexdigest()
    log = ValAccessLog(a.val_log)
    res = []
    for c in frozen["contrast_manifest"]:
        r = contrast(df, c["method"], c["reference"], cfg, level, cfg.get("cost_measure", "e2e_seconds"))
        if "dQ_lo" not in r:
            r["decision"] = "NA (" + r.get("status", "missing") + ")"
        elif c["theta"] == "delta_IR":
            r["decision"] = cost_decision(r["dQ_lo"], r["dQ_hi"], r["savings_lo"], r["savings_hi"], th["delta_IR"])
        else:
            r["decision"] = quality_decision(r["dQ_lo"], r["dQ_hi"], th[c["theta"]])
        if a.look == 2:
            r["decision"] = after_second_look(r["decision"])
        log.record(contrast_id=c["id"], config_hash=frozen["evaluation_config_sha256"], look=a.look, cph=cph)
        res.append({**c, **r})
    need2 = a.look == 1 and any(r["primary"] and r["decision"] == INC for r in res) \
        and frozen["look_plan"]["look2_val_ext_prompts_if_inc"] > 0
    rep = {"look": a.look, "level": level, "contrasts": res, "need_look2": need2,
           "look2_val_ext_prompts": frozen["look_plan"]["look2_val_ext_prompts_if_inc"] if need2 else 0,
           "n_prompts": int(df["prompt_uid"].nunique()), "splits": sorted(set(df["split"])),
           "frozen_sha256": hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest(),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "decisions follow the locked rules; negative and inconclusive results are reported as they are"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({r["id"]: r["decision"] for r in res} | {"need_look2": need2}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
