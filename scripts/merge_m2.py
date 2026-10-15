"""Merge per-k M2 generator profiles (profile --stage generator --ks k --out-dir <dir>/k<k>) into one cost table.

Checks: one EPH, the same warm-up/n and profile code in every part, and coverage of the keys that the methods need
(den/ver/fin k=1..16, prune pairs from required_prune_pairs(), feat/lfeat). The merged table stays `provisional: true`
unless every part was run with --measurement (research-designated machine, Q1) and is not a smoke run.

  python scripts/merge_m2.py <dir> --out <dir>/generator_costs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apsp.contracts.hashing import provenance_hash  # noqa: E402
from apsp.sim.schedules import required_prune_pairs  # noqa: E402

TABLES = ("den", "ver", "fin", "prune", "feat", "lfeat", "fixed")
BODY_KEYS = ("eph", "units", "timer", "warmup", "n", "profile_code", "smoke")


def merge(parts: list[Path]) -> tuple[dict, dict]:
    costs = {t: {} for t in TABLES}
    summary, mem, prov, bodies = {}, {}, [], []
    for d in parts:
        c = json.loads((d / "generator_costs.json").read_text(encoding="utf-8"))
        r = json.loads((d / "generator_costs_raw.json").read_text(encoding="utf-8"))
        for t in TABLES:
            costs[t].update(c.get(t, {}))
        summary.update(r["summary"])
        mem.update(r.get("peak_memory_by_k", {}))
        prov.append(bool(c.get("provisional", True)))
        bodies.append(r["gch_body"])
    for k in BODY_KEYS:
        if len({json.dumps(b[k]) for b in bodies}) != 1:
            raise SystemExit(f"parts disagree on {k}; refusing to merge")
    body = {k: bodies[0][k] for k in BODY_KEYS}
    body["summary"] = {k: [v["mean_us"], v["n"]] for k, v in sorted(summary.items())}
    need = {"den": [str(k) for k in range(1, 17)], "ver": [str(k) for k in range(2, 17)],
            "fin": [str(k) for k in range(1, 17)], "prune": [f"{a}->{b}" for a, b in required_prune_pairs()],
            "feat": [str(k) for k in (2, 4, 8, 16)], "lfeat": [str(k) for k in (2, 4, 8, 16)]}
    missing = {t: [k for k in ks if k not in costs[t]] for t, ks in need.items()}
    missing = {t: v for t, v in missing.items() if v}
    cv = sorted(k for k, v in summary.items() if v.get("cv_flag"))
    reasons = [r for r, f in (("a part was not run with --measurement", any(prov)), ("keys missing", bool(missing)),
                              ("smoke run", bool(body["smoke"]))) if f]
    out = {**costs, "disp": 0, "once": 0, "pred": {}, "uses_traj": False, "synthetic": False,
           "provisional": bool(reasons), "method_id": "fixed", "gch": provenance_hash("GCH", body)}
    rep = {"parts": [str(p) for p in parts], "missing_keys": missing, "cv_flags": cv, "peak_memory_by_k": mem,
           "provisional_reasons": reasons, "gch": out["gch"], "eph": body["eph"]}
    return out, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    parts = sorted(p for p in Path(a.dir).glob("k*") if (p / "generator_costs.json").exists())
    if not parts:
        raise SystemExit("no per-k parts found")
    costs, rep = merge(parts)
    Path(a.out).write_text(json.dumps(costs, indent=1), encoding="utf-8")
    Path(a.out).with_name("generator_costs_merge_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("missing_keys", "cv_flags", "provisional_reasons", "gch")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
