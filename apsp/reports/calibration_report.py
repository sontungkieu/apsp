"""calibration_report.md + calibration_quality_cost.csv from the measured P1 artifacts of one calibration run.

  python -m apsp.reports.calibration_report --cal RUNS/calib1 --out RUNS/calib1/report

Reads only files the P1 jobs wrote (fingerprint, VRAM, env check, full-pool check, M3/M9, chunk, M2 cost table,
equivalence, M2b, forecast, calibration-split benchmark report). Missing parts are listed as missing, never filled.
Each section shows its source path and sha256. The cost table keeps its own `provisional` flag.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import time
from pathlib import Path

SECTIONS = [
    ("Runtime fingerprint", "p0/runtime_fingerprint.json"),
    ("VRAM check (PyTorch)", "p0/vram_check.json"),
    ("VRAM check (independent CUDA memtest)", "p0/cuda_memtest.json"),
    ("Environment check", "p0/env_*.json"),
    ("Model / reward pins (M1)", "p0/M1_model.json"),
    ("Full-pool near-duplicate check", "calib/pool_full_check.json"),
    ("M3 preview precision + M9 collector throughput", "calib/m3m9.json"),
    ("VAE decode chunk", "calib/chunk.json"),
    ("Generator cost table (M2, merged)", "p0/m2/generator_costs.json"),
    ("Collector equivalence", "calib/equivalence.json"),
    ("M2b online residual", "calib/m2b.json"),
    ("Phase forecast (E2-E8 inputs)", "forecast/phase_forecast.json"),
    ("Calibration-split fixed baselines (descriptive, D49)", "bench_calib/report/quality_cost.json"),
]
BIG_KEYS = {"cross_split_full_pool", "records", "per_prompt", "episodes", "curve", "summary", "contrasts_vs_reference",
            "equal_budget", "cost_to_reach", "packages", "env", "primitives"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _compact(v, depth=0):
    if isinstance(v, dict):
        if depth >= 2:
            return f"{{{len(v)} keys}}"
        return {k: (_compact(x, depth + 1) if k not in BIG_KEYS else f"[{len(x) if hasattr(x, '__len__') else '...'} items]")
                for k, x in list(v.items())[:40]}
    if isinstance(v, list):
        return v if len(v) <= 8 and all(not isinstance(x, (dict, list)) for x in v) else f"[{len(v)} items]"
    return v


def build(cal: Path) -> tuple[str, dict]:
    lines = [f"# Calibration report — {cal.name}", "",
             f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from the files below (CUDA-real outputs of "
             "the P1 jobs). Missing sections are reported as missing. Values marked provisional stay provisional until "
             "PREREG-A fixes the measurement machine and cost table.", ""]
    index = {}
    for title, rel in SECTIONS:
        hits = sorted(glob.glob(str(cal / rel)))
        lines.append(f"## {title}")
        if not hits:
            lines += [f"**missing** (`{rel}`)", ""]
            index[title] = None
            continue
        for h in hits:
            p = Path(h)
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except ValueError as e:
                lines += [f"`{p}`: unreadable JSON ({e})", ""]
                continue
            index[title] = {"path": str(p), "sha256": _sha(p)}
            lines.append(f"Source `{p.relative_to(cal)}` · sha256 `{index[title]['sha256'][:16]}…`")
            if title.startswith("Generator cost table"):
                lines.append(f"- provisional: **{d.get('provisional', 'unspecified')}** · synthetic: {d.get('synthetic', False)}")
            if title.startswith("Calibration-split"):
                lines.append(f"- label: {d.get('label')}")
                lines.append("- DESCRIPTIVE ONLY: not used to select methods, hyperparameters, epsilon, thresholds or "
                             "pilot size; not a held-out test.")
                lines += ["", "| method | n | mean IR | mean e2e s | p95 e2e s | fail rate | peak VRAM GiB |",
                          "|---|---|---|---|---|---|---|"]
                for r in d.get("summary", []):
                    vr = r.get("peak_vram_reserved_max") or r.get("peak_vram_reserved_bytes")
                    lines.append(f"| {r.get('method')} | {r.get('n_episodes')} | {_f(r.get('mean_Q'))} | "
                                 f"{_f(r.get('mean_e2e_s'))} | {_f(r.get('p95_e2e_s'))} | {_f(r.get('failure_rate'))} | "
                                 f"{_f(vr / 2**30) if vr else 'n/a'} |")
            lines += ["", "```json", json.dumps(_compact(d), indent=1, default=str)[:6000], "```", ""]
    return "\n".join(lines) + "\n", index


def _f(x):
    return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cal, out = Path(a.cal), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, index = build(cal)
    (out / "calibration_report.md").write_text(md, encoding="utf-8")
    (out / "calibration_report_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    csv = cal / "bench_calib" / "report" / "quality_cost.csv"
    if csv.exists():
        shutil.copyfile(csv, out / "calibration_quality_cost.csv")
    print(json.dumps({"report": str(out / "calibration_report.md"),
                      "missing": [k for k, v in index.items() if v is None]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
