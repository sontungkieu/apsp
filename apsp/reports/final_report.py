"""Final report of the suite: every method that ran, per split, with CIs, latency, failures, frontier, ablations,
confirmatory decisions and illustrative cases. Works on partial results: missing parts are listed as missing.

Reads only finished artifacts (episodes.jsonl / reports / snapshots). No threshold is chosen here: NI/savings
decisions are copied from the locked confirm reports; everything else is descriptive (95 % prompt-cluster CIs).

  python -m apsp.reports.final_report --store STORE --run-id pilot1 --out STORE/final
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from apsp.reports.quality_cost import quality, ratio_bootstrap
from apsp.stats.bootstrap import cluster_bootstrap

SPLITS = [("calibration (descriptive)", "runs/calib1/bench_calib"), ("train_dev baselines", "runs/{r}/bench_dev"),
          ("train_dev RL", "runs/{r}/bench_dev_rl"), ("train_dev ablations", "runs/{r}/bench_dev_ablate"),
          ("train_fit anchor", "runs/{r}/anchor_online"), ("val (confirmatory)", "runs/{r}/bench_val"),
          ("val_ext (look 2)", "runs/{r}/bench_val_ext"), ("GenEval test", "runs/{r}/geneval"),
          ("train_dev FK-Steering (external)", "runs/{r}/fk_dev"), ("val FK-Steering (external)", "runs/{r}/fk_val")]
REFS = ["psp", "single", "bon:4", "bon:8"]


def short(spec: str) -> str:
    if ":" in spec and "/" in spec:
        fam, rest = spec.split(":", 1)
        path, _, alpha = rest.rpartition("@")
        tag = Path(path).parent.name if path.endswith("artifact.json") else Path(path).stem
        return f"{fam}:{tag}@{alpha}"
    return spec


def load(run: Path) -> pd.DataFrame | None:
    p = run / "episodes.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows) if rows else None


def table(df: pd.DataFrame, r_fail: float) -> pd.DataFrame:
    df = df.assign(Q=quality(df, r_fail))
    out = []
    for m, g in df.groupby("method"):
        pp = g.groupby("prompt_uid")["Q"].mean()
        ci = cluster_bootstrap({k: [v] for k, v in pp.items()}, 0.95, seed=0, n_resamples=4000) if len(pp) > 1 else None
        row = {"method": short(m), "spec": m, "n_prompts": int(g["prompt_uid"].nunique()), "n_episodes": len(g),
               "mean_IR": float(g["Q"].mean()), "IR_lo": ci.lo if ci else np.nan, "IR_hi": ci.hi if ci else np.nan,
               "e2e_mean_s": float(g["e2e_seconds"].mean()), "e2e_p50_s": float(g["e2e_seconds"].median()),
               "e2e_p95_s": float(g["e2e_seconds"].quantile(0.95)), "fail_rate": float(1 - g["success"].mean())}
        if "eval_hpsv2" in g:
            row["mean_HPS"] = float(g["eval_hpsv2"].mean())
        for ref in REFS:
            if ref == m or ref not in set(df["method"]):
                continue
            r = df[df["method"] == ref]
            qm, qr = g.groupby("prompt_uid")["Q"].mean(), r.groupby("prompt_uid")["Q"].mean()
            keys = sorted(set(qm.index) & set(qr.index))
            if len(keys) < 3:
                continue
            d = cluster_bootstrap({k: [qm[k] - qr[k]] for k in keys}, 0.95, seed=0, n_resamples=4000)
            sv = ratio_bootstrap(g.groupby("prompt_uid")["e2e_seconds"].mean().to_dict(),
                                 r.groupby("prompt_uid")["e2e_seconds"].mean().to_dict(), 0.95, B=4000)
            tag = ref.replace(":", "")
            row.update({f"dIR_vs_{tag}": d.estimate, f"dIR_vs_{tag}_lo": d.lo, f"dIR_vs_{tag}_hi": d.hi,
                        f"savings_vs_{tag}": sv["estimate"], f"savings_vs_{tag}_lo": sv["lo"], f"savings_vs_{tag}_hi": sv["hi"]})
        out.append(row)
    return pd.DataFrame(out).sort_values("e2e_mean_s")


def pareto(t: pd.DataFrame) -> set:
    front, best = set(), -np.inf
    for _, r in t.sort_values(["e2e_mean_s", "mean_IR"], ascending=[True, False]).iterrows():
        if r["mean_IR"] > best:
            front.add(r["method"])
            best = r["mean_IR"]
    return front


def frontier_svg(t: pd.DataFrame, title: str) -> str:
    W, H, P = 720, 420, 60
    x, y = t["e2e_mean_s"].to_numpy(), t["mean_IR"].to_numpy()
    x0, x1 = 0.0, float(np.nanmax(x)) * 1.08
    y0, y1 = float(np.nanmin(t["IR_lo"].fillna(t["mean_IR"]))) - 0.05, float(np.nanmax(t["IR_hi"].fillna(t["mean_IR"]))) + 0.05
    X = lambda v: P + (v - x0) / (x1 - x0) * (W - 2 * P)
    Y = lambda v: H - P - (v - y0) / (y1 - y0) * (H - 2 * P)
    fr = pareto(t)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/><text x="{W/2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
         f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#333"/><line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#333"/>',
         f'<text x="{W/2}" y="{H-15}" text-anchor="middle">mean end-to-end seconds per request (measured)</text>',
         f'<text x="15" y="{H/2}" transform="rotate(-90 15 {H/2})" text-anchor="middle">mean delivered ImageReward (95% CI)</text>']
    for v in np.linspace(x0, x1, 6):
        s.append(f'<text x="{X(v):.0f}" y="{H-P+15}" text-anchor="middle">{v:.1f}</text>')
    for v in np.linspace(y0, y1, 6):
        s.append(f'<text x="{P-5}" y="{Y(v)+4:.0f}" text-anchor="end">{v:.2f}</text>')
    fp = t[t["method"].isin(fr)].sort_values("e2e_mean_s")
    s.append('<polyline fill="none" stroke="#2a6fdb" stroke-dasharray="4 3" points="' +
             " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(fp["e2e_mean_s"], fp["mean_IR"])) + '"/>')
    for _, r in t.iterrows():
        c = "#2a6fdb" if r["method"] in fr else "#888"
        if np.isfinite(r["IR_lo"]):
            s.append(f'<line x1="{X(r.e2e_mean_s):.1f}" y1="{Y(r.IR_lo):.1f}" x2="{X(r.e2e_mean_s):.1f}" y2="{Y(r.IR_hi):.1f}" stroke="{c}"/>')
        s.append(f'<circle cx="{X(r.e2e_mean_s):.1f}" cy="{Y(r.mean_IR):.1f}" r="4" fill="{c}"/>')
        s.append(f'<text x="{X(r.e2e_mean_s)+6:.1f}" y="{Y(r.mean_IR)-6:.1f}">{html.escape(r["method"][:28])}</text>')
    s.append("</svg>")
    return "\n".join(s)


def cases(df: pd.DataFrame, run: Path, out: Path, a: str, b: str, r_fail: float, n: int = 4) -> list:
    """Largest wins and losses of method a vs b (per prompt/bank, delivered IR), images copied for the report."""
    df = df.assign(Q=quality(df, r_fail))
    A = df[df["method"] == a].set_index(["prompt_uid", "bank_id"])
    B = df[df["method"] == b].set_index(["prompt_uid", "bank_id"])
    keys = sorted(set(A.index) & set(B.index))
    if not keys:
        return []
    d = sorted(((float(A.loc[k, "Q"] - B.loc[k, "Q"]), k) for k in keys))
    picked = [("loss", x) for x in d[:n]] + [("win", x) for x in d[-n:][::-1]]
    res = []
    (out / "cases").mkdir(parents=True, exist_ok=True)
    for kind, (diff, k) in picked:
        item = {"kind": kind, "prompt_uid": k[0], "bank_id": int(k[1]), "dIR": diff, "images": {}}
        for m, T in ((a, A), (b, B)):
            im = T.loc[k].get("image")
            if isinstance(im, str) and (run / im).exists():
                dst = out / "cases" / f"{kind}_{k[0][:10]}_b{k[1]}_{short(m).replace(':', '_').replace('@', '_')}.png"
                shutil.copyfile(run / im, dst)
                item["images"][short(m)] = dst.name
            item[f"IR_{short(m)}"] = float(T.loc[k, "Q"])
        res.append(item)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--run-id", default="pilot1")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    S, out, repo = Path(a.store), Path(a.out), Path(a.repo)
    out.mkdir(parents=True, exist_ok=True)
    snapB = repo / "docs/PREREG_PROPOSALS/PREREG_B_LOCKED.json"
    r_fail = json.loads(snapB.read_text())["B4_r_fail"] if snapB.exists() else None
    md = ["# Adaptive PSP: suite results", "",
          "All numbers are measured online on the measurement machine unless labelled replay/counterfactual. "
          "Quality = ImageReward of the delivered image (failures = r_fail of PREREG-B). CIs: 95 % prompt-cluster "
          "bootstrap (descriptive) unless a confirmatory decision is quoted from the locked reports.", ""]
    missing, all_rows = [], []
    for title, rel in SPLITS:
        run = S / rel.format(r=a.run_id)
        df = load(run)
        if df is None:
            missing.append(f"{title}: `{rel.format(r=a.run_id)}` not available")
            continue
        rf = r_fail if r_fail is not None else float(np.nanmin(df["q_eval"].astype(float))) - 1.0
        t = table(df, rf)
        t.insert(0, "split", title)
        all_rows.append(t)
        md += [f"## {title}", "", f"{t['n_prompts'].max()} prompts, {len(df)} episodes.", ""]
        cols = ["method", "n_episodes", "mean_IR", "IR_lo", "IR_hi", "e2e_mean_s", "e2e_p95_s", "fail_rate",
                "dIR_vs_psp", "dIR_vs_psp_lo", "dIR_vs_psp_hi", "savings_vs_psp", "savings_vs_psp_lo", "savings_vs_psp_hi"]
        cols = [c for c in cols if c in t] + (["mean_HPS"] if "mean_HPS" in t else [])
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "---|" * len(cols))
        for _, r in t.iterrows():
            md.append("| " + " | ".join((f"{r[c]:.3f}" if isinstance(r[c], float) and np.isfinite(r[c]) else
                                         ("" if isinstance(r[c], float) else str(r[c]))) for c in cols) + " |")
        svg = out / f"frontier_{title.split()[0]}_{rel.split('/')[-1]}.svg"
        svg.write_text(frontier_svg(t, f"Quality-cost: {title}"), encoding="utf-8")
        md += ["", f"![frontier]({svg.name})", ""]
    if all_rows:
        pd.concat(all_rows).to_csv(out / "quality_cost_all.csv", index=False)
    conf = S / "runs" / a.run_id / "confirm_final.json"
    if conf.exists():
        c = json.loads(conf.read_text())
        md += ["## Confirmatory decisions (locked rules, PREREG-B)", "", f"Look {c['look']}, CI level {c['level']:.4f}.", "",
               "| contrast | method | reference | dIR [CI] | savings [CI] | decision |", "|---|---|---|---|---|---|"]
        for r in c["contrasts"]:
            md.append(f"| {r['id']} | {short(r['method'])} | {short(r['reference'])} | "
                      f"{r.get('dQ', float('nan')):.3f} [{r.get('dQ_lo', float('nan')):.3f}, {r.get('dQ_hi', float('nan')):.3f}] | "
                      f"{r.get('savings', float('nan')):.1%} [{r.get('savings_lo', float('nan')):.1%}, {r.get('savings_hi', float('nan')):.1%}] | **{r['decision']}** |")
        md.append("")
    else:
        missing.append("confirmatory val decisions (confirm_final.json)")
    for name, rel in (("GenEval test report", f"runs/{a.run_id}/geneval/geneval_report.json"),
                      ("G1/G2 gates", f"runs/{a.run_id}/gates_b.json"), ("Ablations", f"runs/{a.run_id}/ablations.json")):
        p = S / rel
        if p.exists():
            md += [f"## {name}", "", "```json", json.dumps(json.loads(p.read_text()), indent=1, default=str)[:8000], "```", ""]
        else:
            missing.append(f"{name} (`{rel}`)")
    fz = S / "frozen_operating_point.json"
    val = load(S / "runs" / a.run_id / "bench_val")
    if fz.exists() and val is not None and r_fail is not None:
        op = json.loads(fz.read_text())["operating_point"]
        cs = cases(val, S / "runs" / a.run_id / "bench_val", out, op, "psp", r_fail)
        md += ["## Illustrative cases (val): operating point vs PSP-default, largest wins and losses", ""]
        for it in cs:
            imgs = " ".join(f"![{k}](cases/{v})" for k, v in it["images"].items())
            md.append(f"- **{it['kind']}** {it['prompt_uid'][:12]} bank {it['bank_id']}: dIR {it['dIR']:+.3f} {imgs}")
        md.append("")
    md += ["## Missing / not run", ""] + [f"- {m}" for m in missing] + [""]
    (out / "final_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"report": str(out / "final_report.md"), "missing": missing}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
