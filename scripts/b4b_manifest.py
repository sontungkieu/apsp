"""Export the B4b grid manifest from the pinned PSP code (read-only) to reports/b4b/.

  python scripts/b4b_manifest.py --out reports/b4b
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apsp.sim import b4b_grid as g  # noqa: E402

PSP = ROOT / "third_party" / "psp"


def psp_enumerate():
    """Run PSP's own enumerate_strategies (extracted from the pinned file, no tqdm/pandas needed)."""
    src = (PSP / g.PSP_SOURCE).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = {"ScheduledStrategy", "compute_total_particle_steps", "build_remaining_sequences", "enumerate_strategies"}
    mod = ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in keep], type_ignores=[])
    ns = {}
    exec(compile("from dataclasses import dataclass\nfrom itertools import combinations\n", "<h>", "exec"), ns)
    exec(compile(mod, str(PSP / g.PSP_SOURCE), "exec"), ns)
    a = g.SD15_ARGS
    return ns["enumerate_strategies"](total_steps=a["total_steps"], max_cutoffs=a["max_cutoffs"],
                                      cutoff_candidates=a["possible_cutoff_times"],
                                      possible_remaining=a["possible_remaining_seeds"], best_of_n=a["best_of_n"])


def sd15_args_from_scripts() -> list[dict]:
    out = []
    for s in g.PSP_SD15_SCRIPTS:
        txt = (PSP / s).read_text(encoding="utf-8")
        cmd = txt.split("python results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py", 2)[-1]

        def arg(name):
            part = cmd.split(f"--{name}", 1)[1].split("\\")[0].strip().strip('"')
            return part
        out.append({"script": s, "total_steps": int(arg("total-steps")), "best_of_n": int(arg("best-of-n")),
                    "max_cutoffs": int(arg("max-cutoffs")),
                    "possible_cutoff_times": [int(x) for x in arg("possible-cutoff-times").split(",")],
                    "possible_remaining_seeds": [int(x) for x in arg("possible-remaining-seeds").split(",")],
                    "guidance_metric": arg("guidance-metric")})
    return out


def git_blob(path: str) -> str:
    return subprocess.run(["git", "-C", str(PSP), "rev-parse", f"HEAD:{path}"], capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/b4b")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(PSP), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != g.PSP_COMMIT:
        raise SystemExit(f"PSP submodule at {head}, expected {g.PSP_COMMIT}")
    scripts = sd15_args_from_scripts()
    keys = ("total_steps", "best_of_n", "max_cutoffs", "possible_cutoff_times", "possible_remaining_seeds")
    args_match = all({k: s[k] for k in keys} == g.SD15_ARGS for s in scripts)
    mine = g.enumerate_grid()
    theirs = psp_enumerate()
    same = [(s.k_init, s.cutoff_times, s.remaining_particles, s.total_particle_steps) for s in mine] == \
           [(s.k_init, s.cutoff_times, s.remaining_particles, s.total_particle_steps) for s in theirs]
    off_bank, specs_ok = [], 0
    for s in mine:
        try:
            g.to_spec(s).validate()
            specs_ok += 1
        except g.OffBankCheckpoint as e:
            off_bank.append({"strategy": [s.k_init, s.cutoff_times, s.remaining_particles], "reason": str(e)})
    with open(out / "b4b_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "k_init", "cutoff_steps", "checkpoints_j", "keeps", "final_k", "psp_particle_steps"])
        for i, s in enumerate(mine):
            w.writerow([i, s.k_init, " ".join(map(str, s.cutoff_times)), " ".join(str(t // 4) for t in s.cutoff_times),
                        " ".join(map(str, s.remaining_particles)), s.remaining_particles[-1], s.total_particle_steps])
    man = {"schema": "apsp.b4b_grid/v1", "psp_commit": head,
           "sources": {p: {"git_blob": git_blob(p)} for p in (g.PSP_SOURCE,) + g.PSP_SD15_SCRIPTS},
           "sd15_args": g.SD15_ARGS, "sd15_scripts": scripts, "sd15_scripts_agree_with_args": args_match,
           "guidance_metrics_in_scripts": sorted({s["guidance_metric"] for s in scripts}),
           "grid_matches_psp_enumerate_strategies": same, "n_strategies": len(mine),
           "by_k_init": dict(sorted(Counter(s.k_init for s in mine).items(), reverse=True)),
           "by_n_cutoffs": dict(sorted(Counter(len(s.cutoff_times) for s in mine).items())),
           "required_primitives": g.required_primitives(mine),
           "off_bank_strategies": off_bank, "n_specs_valid_on_bank": specs_ok,
           "grid_csv_sha256": hashlib.sha256((out / "b4b_grid.csv").read_bytes()).hexdigest(),
           "mapping": {"checkpoint": "j = cutoff_step / 4 (delta 4); all SD1.5 cutoffs are multiples of 4 -> j in 1..15",
                       "candidates": "first k_init of pool 16 (PSP logical seed 0 window)",
                       "ranking": "preview IR desc, CandidateId asc (PSP: IR desc, seed asc)",
                       "final": "delivered IR among final survivors; fin(final_k) charged",
                       "costs": "project ledger incl. paid VERIFY at every cutoff; PSP particle-step budget only "
                                "defines the grid; caps decide feasibility"},
           "notes": ["PSP ranks with the guidance metric of the script (IR or HPS); this project's B4b uses IR (the "
                     "pilot reward).", "All grid checkpoints lie on the bank (j in 1..15): replay needs no auxiliary "
                     "bank; online B4b runs use batch sizes 16/12/8/4/3/2/1 (native batching, approximate replay)."]}
    (out / "b4b_grid_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("n_strategies", "grid_matches_psp_enumerate_strategies",
                                          "sd15_scripts_agree_with_args", "by_k_init", "required_primitives",
                                          "n_specs_valid_on_bank")}, indent=1))
    return 0 if same and args_match and not off_bank else 1


if __name__ == "__main__":
    sys.exit(main())
