"""Quality-cost report for the pilot: curve, quality at equal budget, cost to reach reference quality, latency,
failures, peak VRAM, and the non-inferiority / savings test of  min E[C] s.t. E[Q - Q_ref] >= -epsilon.

Statistics: paired by prompt, prompt = cluster (banks of a prompt averaged inside the cluster; pool-16 methods have
one episode per prompt), percentile cluster bootstrap (apsp.stats.bootstrap). epsilon, savings thresholds and the
failure penalty are read from configs/evaluation.yaml; if they are not approved, every NI / savings conclusion is
reported as "not_concludable" (the intervals are still printed, labelled descriptive). Nothing here picks thresholds.

Operating point (``--freeze``): only on train_dev episodes, only with an approved config: among methods passing NI
against the reference, the lowest mean cost; written with hashes to a frozen file that confirmatory runs must use.

  python -m apsp.reports.quality_cost --run DIR --config configs/evaluation.yaml --out DIR/report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from apsp.stats.bootstrap import cluster_bootstrap, contrast_level
from apsp.stats.decisions import cost_decision
from apsp.stats.frontier import Point, pareto

NC = "not_concludable"


def load_run(run: Path) -> tuple[pd.DataFrame, dict]:
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    df = pd.DataFrame(rows)
    ev = run / "eval_metrics.jsonl"
    if ev.exists():
        e = pd.DataFrame([json.loads(l) for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()])
        if len(e):
            piv = e.pivot_table(index=["prompt_uid", "bank_id", "method"], columns="metric", values="value", aggfunc="first")
            piv.columns = [f"eval_{c}" for c in piv.columns]
            df = df.merge(piv.reset_index(), on=["prompt_uid", "bank_id", "method"], how="left")
    return df, man


def status_of(cfg: dict, df: pd.DataFrame) -> dict:
    reasons = []
    if not cfg.get("approved"):
        reasons.append("evaluation config not approved by research")
    if cfg.get("epsilon_ir") is None:
        reasons.append("epsilon not set (PREREG-B)")
    if (~df["success"]).any() and cfg.get("r_fail") is None:
        reasons.append("failures present but r_fail not frozen")
    if cfg.get("n_primary_contrasts") is None:
        reasons.append("number of primary contrasts M not fixed (descriptive CI level used)")
    return {"concludable": not reasons, "reasons": reasons}


def quality(df: pd.DataFrame, r_fail: Optional[float]) -> pd.Series:
    q = df["q_eval"].astype(float)
    fail = ~df["success"].astype(bool) | q.isna()
    return q.where(~fail, np.nan if r_fail is None else float(r_fail))


def per_prompt(df: pd.DataFrame, method: str, col: str) -> dict:
    d = df[df["method"] == method]
    return d.groupby("prompt_uid")[col].mean().dropna().to_dict()


def summary_table(df: pd.DataFrame, r_fail) -> pd.DataFrame:
    df = df.assign(Q=quality(df, r_fail))
    g = df.groupby("method")
    t = pd.DataFrame({
        "family": g["family"].first(), "n_episodes": g.size(), "n_prompts": g["prompt_uid"].nunique(),
        "failure_rate": 1 - g["success"].mean(), "mean_Q": g["Q"].mean(),
        "mean_ledger_us": g["ledger_spend_us"].mean(), "mean_e2e_s": g["e2e_seconds"].mean(),
        "p50_e2e_s": g["e2e_seconds"].median(), "p95_e2e_s": g["e2e_seconds"].quantile(0.95),
        "mean_forwards": g["forwards"].mean(),
        "mean_candidate_forwards": g["candidate_forwards"].mean() if "candidate_forwards" in df else np.nan,
        "n_candidates": g["n_candidates"].first(), "mean_verify": g["n_verify"].mean(),
        "rm_in_inference": g["rm_used_in_inference"].mean(),
        "peak_vram_reserved_max": g["peak_vram_reserved_bytes"].max() if "peak_vram_reserved_bytes" in df else np.nan})
    for c in [c for c in df.columns if c.startswith("eval_")]:
        t[f"mean_{c}"] = g[c].mean()
    return t.sort_values("mean_e2e_s")


def ratio_bootstrap(cm: dict, cr: dict, level: float, seed: int = 0, B: int = 10000) -> dict:
    keys = sorted(set(cm) & set(cr))
    a, b = np.array([cm[k] for k in keys]), np.array([cr[k] for k in keys])
    if len(keys) < 2:
        return {"estimate": None, "lo": None, "hi": None, "n_clusters": len(keys)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    s = 1 - a[idx].mean(1) / b[idx].mean(1)
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float(1 - a.mean() / b.mean()), "lo": float(lo), "hi": float(hi), "n_clusters": len(keys)}


def contrast(df: pd.DataFrame, m: str, ref: str, cfg: dict, level: float, cost_col: str) -> dict:
    qm, qr = per_prompt(df, m, "Q"), per_prompt(df, ref, "Q")
    keys = sorted(set(qm) & set(qr))
    out = {"method": m, "reference": ref, "n_clusters": len(keys), "level": level}
    if len(keys) < 2:
        out["status"] = "too few paired prompts"
        return out
    ci = cluster_bootstrap({k: [qm[k] - qr[k]] for k in keys}, level, seed=0, n_resamples=10000)
    sv = ratio_bootstrap(per_prompt(df, m, cost_col), per_prompt(df, ref, cost_col), level)
    out.update({"dQ": ci.estimate, "dQ_lo": ci.lo, "dQ_hi": ci.hi, "savings": sv["estimate"], "savings_lo": sv["lo"],
                "savings_hi": sv["hi"]})
    eps = cfg.get("epsilon_ir")
    if eps is None or not cfg.get("approved"):
        out["ni"] = NC
        out["cost_decision"] = NC
    else:
        out["ni"] = "pass" if ci.lo > -float(eps) else "fail"
        out["cost_decision"] = cost_decision(ci.lo, ci.hi, sv["lo"], sv["hi"], float(eps))
    return out


def build_report(df: pd.DataFrame, man: dict, cfg: dict) -> dict:
    st = status_of(cfg, df)
    r_fail = cfg.get("r_fail")
    df = df.assign(Q=quality(df, r_fail))
    M = cfg.get("n_primary_contrasts")
    level = contrast_level(int(M)) if M else float(cfg.get("descriptive_ci_level", 0.95))
    cost_col = cfg.get("cost_measure", "e2e_seconds")
    ref = cfg["reference_method"]
    summ = summary_table(df, r_fail)
    methods = list(summ.index)
    pts = [Point(m, float(summ.loc[m, "mean_e2e_s"]), float(summ.loc[m, "mean_Q"])) for m in methods
           if np.isfinite(summ.loc[m, "mean_Q"])]
    front = {p.name for p in pareto(pts)}
    curve = [{"method": m, "mean_e2e_s": float(summ.loc[m, "mean_e2e_s"]), "mean_ledger_us": float(summ.loc[m, "mean_ledger_us"]),
              "mean_Q": float(summ.loc[m, "mean_Q"]), "pareto": m in front} for m in methods]
    contrasts = [contrast(df, m, ref, cfg, level, cost_col) for m in methods if m != ref and ref in methods]
    caps = {float(k): v for k, v in (man.get("caps") or {}).items()}
    equal_budget = []
    for alpha in cfg.get("budgets_alpha", []):
        cap = caps.get(float(alpha))
        if cap is None:
            continue
        rows = []
        for m in methods:
            d = df[df["method"] == m]
            a_ = d["alpha"].dropna()
            capped_here = len(a_) and float(a_.iloc[0]) == float(alpha)
            uncapped_fits = d["alpha"].isna().all() and d["ledger_spend_us"].max() <= cap
            if capped_here or uncapped_fits:
                rows.append({"method": m, "mean_Q": float(summ.loc[m, "mean_Q"]), "max_ledger_us": int(d["ledger_spend_us"].max()),
                             "how": "run under this cap" if capped_here else "every episode within the cap"})
        rows.sort(key=lambda r: -r["mean_Q"] if np.isfinite(r["mean_Q"]) else 1e9)
        equal_budget.append({"alpha": alpha, "cap_us": cap, "methods": rows})
    reach = []
    for lvl in cfg.get("reference_levels", []):
        if lvl not in methods:
            continue
        cands = [contrast(df, m, lvl, cfg, level, cost_col) for m in methods if m != lvl]
        if st["concludable"]:
            ok = [c for c in cands if c.get("ni") == "pass"]
            basis = "NI vs this level at the approved epsilon"
        else:
            ok = [c for c in cands if c.get("dQ") is not None and c["dQ"] >= 0]
            basis = "DESCRIPTIVE ONLY (point estimate dQ >= 0); not an NI conclusion"
        ok.sort(key=lambda c: float(summ.loc[c["method"], "mean_e2e_s"]))
        reach.append({"level": lvl, "level_mean_Q": float(summ.loc[lvl, "mean_Q"]),
                      "level_mean_e2e_s": float(summ.loc[lvl, "mean_e2e_s"]), "basis": basis,
                      "cheapest": ok[0] if ok else None})
    return {"status": st, "ci_level": level, "cost_measure": cost_col, "reference": ref,
            "split": man.get("split"), "costs": man.get("costs"), "backend": man.get("backend"),
            "summary": json.loads(summ.reset_index().to_json(orient="records")), "curve": curve,
            "contrasts_vs_reference": contrasts, "equal_budget": equal_budget, "cost_to_reach": reach,
            "label": label_of(man)}


def label_of(man: dict) -> str:
    c = man.get("costs") or {}
    if man.get("backend") == "fake" or c.get("synthetic"):
        return "SYNTHETIC / FAKE BACKEND - plumbing check only; NOT evidence of quality or savings"
    if c.get("provisional", True):
        return "real backend, PROVISIONAL cost table; ledger figures are not final"
    return "real backend, final cost table"


def freeze_selection(rep: dict, df: pd.DataFrame, cfg: dict, cfg_path: Path, episodes_path: Path, out: Path) -> dict:
    if rep.get("split") != "train_dev":
        raise SystemExit("the operating point is selected on train_dev only")
    if not rep["status"]["concludable"]:
        raise SystemExit("cannot freeze: " + "; ".join(rep["status"]["reasons"]))
    passing = [c for c in rep["contrasts_vs_reference"] if c.get("ni") == "pass"]
    summ = {r["method"]: r for r in rep["summary"]}
    passing.sort(key=lambda c: (summ[c["method"]][f"mean_{'e2e_s' if rep['cost_measure'] == 'e2e_seconds' else 'ledger_us'}"], c["method"]))
    sel = passing[0]["method"] if passing else rep["reference"]
    frozen = {"selected_on": "train_dev", "operating_point": sel, "reference": rep["reference"],
              "methods": sorted({sel, rep["reference"], *[m for m in summ]}),
              "evaluation_config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
              "train_dev_episodes_sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
              "rule": "lowest mean cost among methods passing NI vs the reference on train_dev; reference if none"}
    out.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen


CSV_COLUMNS = ["method", "family", "split", "reference", "n_episodes", "n_prompts", "mean_Q", "dQ", "dQ_lo", "dQ_hi",
               "ci_level", "ni", "mean_e2e_s", "p50_e2e_s", "p95_e2e_s", "mean_ledger_us", "savings", "savings_lo",
               "savings_hi", "cost_decision", "failure_rate", "peak_vram_reserved_max", "mean_verify",
               "mean_candidate_forwards", "mean_forwards", "rm_in_inference", "pareto", "label", "backend",
               "costs_provisional", "costs_synthetic", "commit"]


def to_csv_rows(rep: dict, man: dict) -> list[dict]:
    """One row per method (plan §10.1 `sd15_quality_cost.csv`): quality, paired effect/CI vs the reference, NI,
    e2e savings/CI, latency p50/p95, failure, VRAM, VERIFY/compute counts, n, verdict and provenance."""
    con = {c["method"]: c for c in rep["contrasts_vs_reference"]}
    par = {c["method"]: c["pareto"] for c in rep["curve"]}
    costs = rep.get("costs") or {}
    rows = []
    for r in rep["summary"]:
        m = r["method"]
        c = con.get(m, {})
        row = {**{k: r.get(k) for k in CSV_COLUMNS if k in r}, "split": rep.get("split"), "reference": rep["reference"],
               "ci_level": rep["ci_level"], "pareto": par.get(m), "label": rep["label"], "backend": rep.get("backend"),
               "costs_provisional": costs.get("provisional"), "costs_synthetic": costs.get("synthetic"),
               "commit": man.get("commit")}
        for k in ("dQ", "dQ_lo", "dQ_hi", "ni", "savings", "savings_lo", "savings_hi", "cost_decision"):
            row[k] = c.get(k) if m != rep["reference"] else ("reference" if k in ("ni", "cost_decision") else None)
        rows.append({k: row.get(k) for k in CSV_COLUMNS})
    return rows


def to_markdown(rep: dict) -> str:
    L = [f"# Quality-cost report", "", f"**{rep['label']}**", "",
         f"Split: `{rep['split']}` · reference: `{rep['reference']}` · cost measure: `{rep['cost_measure']}` · CI level {rep['ci_level']:.4f}",
         "", f"Conclusions: **{'concludable' if rep['status']['concludable'] else NC}**"
         + ("" if rep["status"]["concludable"] else " (" + "; ".join(rep["status"]["reasons"]) + ")"), "",
         "## Per method", "",
         "| method | n | fail | mean Q | mean e2e s | p50 | p95 | ledger us | fwd | cand-fwd | VERIFY | RM in inference | peak VRAM |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rep["summary"]:
        L.append(f"| {r['method']} | {r['n_episodes']} | {r['failure_rate']:.3f} | {r['mean_Q']:.4f} | {r['mean_e2e_s']:.4f} | "
                 f"{r['p50_e2e_s']:.4f} | {r['p95_e2e_s']:.4f} | {r['mean_ledger_us']:.0f} | {r['mean_forwards']:.0f} | "
                 f"{(r.get('mean_candidate_forwards') or 0):.0f} | {r['mean_verify']:.1f} | {r['rm_in_inference']:.0f} | "
                 f"{r.get('peak_vram_reserved_max') or 'n/a'} |")
    L += ["", "## Contrasts vs reference (paired by prompt, cluster bootstrap)", "",
          "| method | ΔQ | CI | savings | CI | NI | cost decision |", "|---|---:|---|---:|---|---|---|"]
    for c in rep["contrasts_vs_reference"]:
        if "dQ" not in c:
            L.append(f"| {c['method']} | — | {c.get('status')} | | | | |")
            continue
        L.append(f"| {c['method']} | {c['dQ']:+.4f} | [{c['dQ_lo']:+.4f}, {c['dQ_hi']:+.4f}] | {c['savings']:+.3f} | "
                 f"[{c['savings_lo']:+.3f}, {c['savings_hi']:+.3f}] | {c['ni']} | {c['cost_decision']} |")
    L += ["", "## Quality at equal budget (shared caps B_alpha)", ""]
    for e in rep["equal_budget"]:
        L.append(f"- alpha {e['alpha']} (cap {e['cap_us']} us): " +
                 ", ".join(f"{r['method']} {r['mean_Q']:.4f}" for r in e["methods"]) if e["methods"] else f"- alpha {e['alpha']}: none")
    L += ["", "## Cost to reach reference quality", ""]
    for r in rep["cost_to_reach"]:
        ch = r["cheapest"]
        L.append(f"- level `{r['level']}` (Q {r['level_mean_Q']:.4f}, {r['level_mean_e2e_s']:.4f} s): "
                 + (f"cheapest `{ch['method']}`, savings {ch['savings']:+.3f} [{ch['savings_lo']:+.3f}, {ch['savings_hi']:+.3f}]"
                    if ch else "no method") + f" — {r['basis']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="quality-cost report")
    ap.add_argument("--run", required=True, help="benchmark output directory (episodes.jsonl, manifest.json)")
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", default=None, help="write the frozen train_dev selection to this path")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    df, man = load_run(run)
    rep = build_report(df, man, cfg)
    (out / "quality_cost.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out / "quality_cost.md").write_text(to_markdown(rep), encoding="utf-8")
    pd.DataFrame(to_csv_rows(rep, man), columns=CSV_COLUMNS).to_csv(out / "quality_cost.csv", index=False)
    if a.freeze:
        print(json.dumps(freeze_selection(rep, df, cfg, Path(a.config), run / "episodes.jsonl", Path(a.freeze)), indent=1))
    print(json.dumps({"status": rep["status"], "label": rep["label"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
