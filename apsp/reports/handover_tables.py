"""Handover tables: online results per method with paired contrasts vs PSP-default, best-fixed (B3) and threshold (B6).

Unit of analysis = prompt. Within a prompt the banks are averaged over the banks both methods ran (two banks of one
prompt are not two independent samples); CIs are percentile bootstrap over prompts (B = 10000, seed 0).
Failures count as r_fail (PREREG-B). Savings = 1 - sum_p e2e_m / sum_p e2e_ref over the paired prompts.
RL is reported per training seed and as RL* (the three seeds averaged per prompt, as the locked G3/G4 rule says).
CI levels: 95 % descriptive for everything; for the primary contrasts of the frozen contrast manifest the
confirmatory level of PREREG-B (1 - 0.025/M per look) is also given. The official val decisions are those of
apsp.reports.confirm (job p5_confirm_look1), not this table.

  python -m apsp.reports.handover_tables --runs RUN1 RUN2 --split train_dev --frozen frozen_operating_point.json \
      --snapshot PREREG_B_LOCKED.json --out OUT_PREFIX [--replay-rl rl/seed0/artifact.json ...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

B = 10000


def load(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for line in (Path(r) / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append({k: d.get(k) for k in ("prompt_uid", "split", "bank_id", "method", "family", "alpha",
                                                   "success", "q_eval", "e2e_seconds")})
    df = pd.DataFrame(rows).drop_duplicates(["prompt_uid", "bank_id", "method"], keep="last")
    return df


def short(m: str) -> str:
    m = re.sub(r"/workspace/apsp/tuned/(b\d+)_\w+?_alpha[\d.]+\.json", r"\1", m)
    return re.sub(r"rl:/workspace/apsp/rl/(seed\d)/artifact\.json", r"rl-\1", m)


def add_rl_pooled(df: pd.DataFrame) -> pd.DataFrame:
    rl = df[df["family"] == "rl"].copy()
    if rl.empty:
        return df
    rl["cap"] = rl["method"].str.split("@").str[-1]
    g = rl.groupby(["prompt_uid", "bank_id", "cap"]).agg(Q=("Q", "mean"), e2e_seconds=("e2e_seconds", "mean"),
                                                         success=("success", "mean"), n=("method", "size")).reset_index()
    g = g[g["n"] == 3]
    g["method"] = "rl*@" + g["cap"]
    g["family"] = "rl_pooled"
    g["alpha"] = g["cap"].astype(float)
    return pd.concat([df, g.drop(columns=["cap", "n"])], ignore_index=True)


def per_prompt(df: pd.DataFrame, m: str, ref: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = df[df["method"] == m].set_index(["prompt_uid", "bank_id"])
    b = df[df["method"] == ref].set_index(["prompt_uid", "bank_id"])
    j = a[["Q", "e2e_seconds"]].join(b[["Q", "e2e_seconds"]], lsuffix="_m", rsuffix="_r", how="inner")
    p = j.groupby(level=0).mean()
    return p["Q_m"].to_numpy(), p["Q_r"].to_numpy(), p["e2e_seconds_m"].to_numpy(), p["e2e_seconds_r"].to_numpy()


def contrast(df, m, ref, levels):
    qm, qr, cm, cr = per_prompt(df, m, ref)
    n = len(qm)
    if n == 0:
        return None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))
    d = qm - qr
    dq_b = d[idx].mean(axis=1)
    sv_b = 1 - cm[idx].sum(axis=1) / cr[idx].sum(axis=1)
    out = {"n_prompts": n, "dQ": float(d.mean()), "savings": float(1 - cm.sum() / cr.sum()),
           "sd_prompt_diff": float(d.std(ddof=1)) if n > 1 else float("nan")}
    for lv in levels:
        lo, hi = (1 - lv) / 2, 1 - (1 - lv) / 2
        out[f"dQ_ci{lv:.4f}"] = [float(np.quantile(dq_b, lo)), float(np.quantile(dq_b, hi))]
        out[f"sav_ci{lv:.4f}"] = [float(np.quantile(sv_b, lo)), float(np.quantile(sv_b, hi))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--replay-rl", nargs="*", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fr = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    snap = json.loads(Path(a.snapshot).read_text(encoding="utf-8"))
    r_fail = snap["B4_r_fail"]
    conf_level = snap["B12_contrasts"]["ci_level_per_look"]
    df = load(a.runs)
    df = df[df["split"] == a.split]
    df["Q"] = df["q_eval"].astype(float).where(df["success"].astype(bool), r_fail)
    df = add_rl_pooled(df)
    b3 = {"0.5": "fixed:2,6@0.5", "1.0": "fixed:4,8@1.0"}
    b6 = {c: next(m for m in fr["methods"] if m.startswith("threshold:") and m.endswith("@" + c)) for c in b3}
    primary = {(c["method"], c["reference"]) for c in fr["contrast_manifest"] if c["primary"]}
    rows = []
    for m, g in df.groupby("method", sort=False):
        cap = m.split("@")[-1] if "@" in m else None
        seed = re.search(r"seed(\d)", m)
        r = {"method": short(m), "cap_alpha": cap or "-", "split": a.split, "n_prompts": g["prompt_uid"].nunique(),
             "n_banks": g["bank_id"].nunique(), "rl_seed": seed.group(1) if seed else ("0,1,2 pooled" if m.startswith("rl*") else "-"),
             "n_episodes": len(g), "IR_mean": g["Q"].mean(), "e2e_mean": g["e2e_seconds"].mean(),
             "e2e_p50": g["e2e_seconds"].median(), "e2e_p95": g["e2e_seconds"].quantile(0.95),
             "failure_rate": 1 - g["success"].astype(float).mean()}
        refs = {"psp": "psp"}
        if cap in b3:
            refs.update({"bestfixed": b3[cap], "threshold": b6[cap]})
        for tag, ref in refs.items():
            if ref == m:
                continue
            prim = (m, ref) in primary
            c = contrast(df, m, ref, [0.95] + ([conf_level] if prim else []))
            if c is None:
                continue
            r[f"dIR_vs_{tag}"] = c["dQ"]
            r[f"dIR_vs_{tag}_ci95"] = "[%+.4f, %+.4f]" % tuple(c["dQ_ci0.9500"])
            r[f"sav_vs_{tag}"] = c["savings"]
            r[f"sav_vs_{tag}_ci95"] = "[%+.3f, %+.3f]" % tuple(c["sav_ci0.9500"])
            r[f"pairs_vs_{tag}"] = c["n_prompts"]
            if prim:
                k = f"{conf_level:.4f}"
                r[f"PRIMARY_vs_{tag}_dIR_ci{k}"] = "[%+.4f, %+.4f]" % tuple(c[f"dQ_ci{k}"])
                r[f"PRIMARY_vs_{tag}_sav_ci{k}"] = "[%+.3f, %+.3f]" % tuple(c[f"sav_ci{k}"])
        rows.append(r)
    t = pd.DataFrame(rows)
    order = {"-": 0, "0.5": 1, "1.0": 2}
    t = t.sort_values(["cap_alpha", "e2e_mean"], key=lambda s: s.map(order) if s.name == "cap_alpha" else s)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(str(out) + ".csv", index=False)
    rep = []
    for p in a.replay_rl:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        s = d.get("summary") or {}
        rep.append({"artifact": p, "seed": re.search(r"seed(\d)", p).group(1), "best_iter": d.get("best_iter"),
                    "replay_train_dev_mean_Y_minus_B3": d.get("best_dev", s.get("best_dev_replay_minus_B3")),
                    "train_seconds": s.get("train_seconds", float("nan")), "checkpoint_rule": s.get("checkpoint_rule")})
    L = [f"# Online results, split `{a.split}` {a.label}", "",
         f"Unit = prompt (banks averaged within prompt). Percentile bootstrap over prompts, B={B}. Failures = r_fail "
         f"({r_fail:.3f}). CI95 = descriptive. Primary contrasts (frozen manifest) also at the PREREG-B confirmatory level "
         f"{conf_level:.4f}; the official decision is the confirm job's, not this table.", "",
         f"Thresholds (PREREG-B): tau={snap['B3_thresholds']['tau_prac']:.4f}, delta_IR={snap['B3_thresholds']['delta_IR']:.4f}, "
         f"theta_RL={snap['B3_thresholds']['theta_RL']:.4f}", "",
         "| method | cap | seed | prompts | banks | IR | e2e mean | p50 | p95 | fail | dIR vs PSP [CI95] | sav vs PSP [CI95] | dIR vs best-fixed [CI95] | sav vs best-fixed | dIR vs threshold [CI95] |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    f = lambda v, k="%.4f": ("-" if pd.isna(v) else k % v)
    for _, r in t.iterrows():
        L.append(f"| {r['method']} | {r['cap_alpha']} | {r['rl_seed']} | {r['n_prompts']} | {r['n_banks']} | {r['IR_mean']:.4f} | "
                 f"{r['e2e_mean']:.3f} | {r['e2e_p50']:.3f} | {r['e2e_p95']:.3f} | {r['failure_rate']:.3f} | "
                 f"{f(r.get('dIR_vs_psp'), '%+.4f')} {r.get('dIR_vs_psp_ci95', '') if isinstance(r.get('dIR_vs_psp_ci95'), str) else ''} | "
                 f"{f(r.get('sav_vs_psp'), '%+.3f')} {r.get('sav_vs_psp_ci95', '') if isinstance(r.get('sav_vs_psp_ci95'), str) else ''} | "
                 f"{f(r.get('dIR_vs_bestfixed'), '%+.4f')} {r.get('dIR_vs_bestfixed_ci95', '') if isinstance(r.get('dIR_vs_bestfixed_ci95'), str) else ''} | "
                 f"{f(r.get('sav_vs_bestfixed'), '%+.3f')} | "
                 f"{f(r.get('dIR_vs_threshold'), '%+.4f')} {r.get('dIR_vs_threshold_ci95', '') if isinstance(r.get('dIR_vs_threshold_ci95'), str) else ''} |")
    pc = [c for c in t.columns if c.startswith("PRIMARY_")]
    if pc:
        L += ["", f"## Primary contrasts at the confirmatory level {conf_level:.4f}", ""]
        for _, r in t.iterrows():
            for c in pc:
                if isinstance(r.get(c), str):
                    L.append(f"- {r['method']} @{r['cap_alpha']}: {c} = {r[c]}")
    if rep:
        L += ["", "## RL in replay (simulator, train_dev bank) - NOT online", "",
              "| seed | best iter | replay mean(Y - Y_B3) on train_dev | train s |", "|---|---:|---:|---:|"]
        for x in rep:
            L.append(f"| {x['seed']} | {x['best_iter']} | {x['replay_train_dev_mean_Y_minus_B3']:+.4f} | {x['train_seconds']:.0f} |")
    Path(str(out) + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
