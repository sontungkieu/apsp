"""PREREG-C protocol-lock snapshot (owner authorization 2026-10-17): frozen artifacts, test protocol, evaluator pins.

Written before any GenEval image is generated (refuses if the geneval run directory has content). Nothing here reads
val or test outcomes: the test method list comes from the train_dev freeze (select_dev), never from val results.

  C1  hashes of every frozen artifact (methods file, tuned artifacts, RL checkpoints, cost table, evaluation config,
      PREREG-B snapshot, frozen EPH)
  C2  official GenEval repo commit, prompt metadata sha256, detector checkpoint sha256s, official aggregation
  C3  4 independent episodes per prompt per method (banks 0-3, seed scheme apsp-v1 on uid ge<index>), one delivered
      image per episode, same seeds for every method
  C4  test methods = single, BoN-4, BoN-8, PSP-default, the frozen operating point, the G3 candidates per cap and RL*
      (3 seeds) at both primary caps. Primary test contrasts (GenEval overall, NI margin delta_GE = 0.01, Spec §11.3):
      operating point, G3@0.5, G3@1.0, each vs PSP-default; Bonferroni level 1 - 0.05/M_test. IR (delta_IR from B),
      e2e savings and HPS v2.1 are secondary and descriptive
  C5  a failed generation is unsuccessful on every metric (no image -> incorrect for GenEval, r_fail for IR)
  C6  test set = GenEval only (no new DiffusionDB test split: the split is locked in PREREG-A)

  python -m apsp.reports.lock_c --frozen FROZEN --tuned TUNED --rl-root RL --costs COSTS --frozen-eph E \
      --geneval-repo R --geneval-models M --geneval-run RUN/geneval --test-methods-out STORE/test_methods.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from apsp.contracts.prereg import require_signed

REPO = Path(__file__).resolve().parents[2]
AUTH = "docs/PREREG_SIGNED/AUTHORIZATION_20261017.md"
FIXED_TEST = ["single", "bon:4", "bon:8", "psp"]
DELTA_GE = 0.01


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_methods(frozen: dict) -> tuple[list, list]:
    cm = {c["id"]: c for c in frozen["contrast_manifest"]}
    sel = list(FIXED_TEST)
    op = frozen["operating_point"]
    contrasts = [{"id": "operating_point_vs_psp", "method": op, "primary": True}]
    sel.append(op)
    for a in ("0.5", "1.0"):
        g3 = cm.get(f"G3_alpha_{a}")
        if g3:
            contrasts.append({"id": f"G3_alpha_{a}_vs_psp", "method": g3["method"], "primary": True})
            if not g3["method"].startswith("rl*@"):
                sel.append(g3["method"])
    rl = sorted(m for m in frozen["methods"] if m.startswith("rl:") and m.rsplit("@", 1)[-1] in ("0.5", "1.0"))
    sel += rl
    for c in contrasts:                                  # RL* contrasts use the pooled seeds
        c["method_specs"] = [m for m in rl if m.endswith("@" + c["method"].split("@")[1])] if c["method"].startswith("rl*@") else [c["method"]]
    missing = [m for m in sel if m not in frozen["methods"]]
    if missing:
        raise SystemExit(f"test methods not frozen on train_dev: {missing}")
    return list(dict.fromkeys(sel)), contrasts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--rl-root", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--frozen-eph", required=True)
    ap.add_argument("--geneval-repo", required=True)
    ap.add_argument("--geneval-models", required=True)
    ap.add_argument("--geneval-run", required=True)
    ap.add_argument("--test-methods-out", required=True)
    ap.add_argument("--proposal", default="docs/PREREG_PROPOSALS/PREREG_C_LOCKED.json")
    ap.add_argument("--marker", default="docs/PREREG_SIGNED/PREREG-C.json")
    a = ap.parse_args(argv)
    mb = require_signed("B")
    run = Path(a.geneval_run)
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} has content; PREREG-C must be locked before any GenEval image is generated")
    marker, prop = REPO / a.marker, REPO / a.proposal
    if marker.exists():
        raise SystemExit(f"{marker} exists; a locked snapshot is never replaced")
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    methods, contrasts = test_methods(frozen)
    tuned = Path(a.tuned)
    rl = Path(a.rl_root)
    gr = Path(a.geneval_repo)
    commit = subprocess.run(["git", "-C", str(gr), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = gr / "prompts" / "evaluation_metadata.jsonl"
    snapB = json.loads((REPO / mb["proposal_path"]).read_text(encoding="utf-8"))
    snap = {
        "prereg": "C", "kind": "protocol_lock_snapshot", "authorization_record": AUTH,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "test_images_generated_before_lock": False,
        "C1_frozen_artifacts": {
            "frozen_selection": {"path": a.frozen, "sha256": _sha(a.frozen)},
            "tuned": {str(p.relative_to(tuned)): _sha(p) for p in sorted(tuned.rglob("*")) if p.is_file()},
            "rl": {str(p.relative_to(rl)): _sha(p) for p in sorted(rl.rglob("*")) if p.is_file() and p.suffix in (".json", ".pt")},
            "costs": {"path": a.costs, "sha256": _sha(a.costs)},
            "frozen_eph": json.loads(Path(a.frozen_eph).read_text(encoding="utf-8"))["eph"],
            "prereg_b_snapshot": {"path": mb["proposal_path"], "sha256": mb["proposal_sha256"]},
            "evaluation_config": {"path": "configs/evaluation_locked.yaml", "sha256": _sha(REPO / "configs/evaluation_locked.yaml")}},
        "C2_geneval": {"repo": "https://github.com/djghosh13/geneval", "commit": commit,
                       "metadata_sha256": _sha(meta), "n_prompts": sum(1 for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()),
                       "detector_checkpoints": {p.name: _sha(p) for p in sorted(Path(a.geneval_models).glob("*.pth"))},
                       "scoring": "official evaluation/evaluate_images.py per image; overall = mean over tags of per-tag "
                                  "accuracy (official summary_scores); prompt text verbatim (not normalised)"},
        "C3_test_protocol": {"images_per_prompt": 4, "banks": [0, 1, 2, 3], "seed_scheme": "apsp-v1-sha256-cpu",
                             "prompt_uid": "ge<index:05d>", "one_delivered_image_per_episode": True,
                             "same_seeds_for_all_methods": True},
        "C4_test_contrasts": {"methods": methods, "reference": "psp", "contrasts": contrasts,
                              "M_test": sum(c["primary"] for c in contrasts), "geneval_ni_margin": DELTA_GE,
                              "ci": "Bonferroni level 1 - 0.05/M_test, prompt bootstrap stratified by tag, 10,000 draws",
                              "secondary": {"ir_delta_IR": snapB["B3_thresholds"]["delta_IR"],
                                            "savings_rule": "e2e seconds, descriptive", "hps": "HPS v2.1 descriptive"},
                              "rl_star": "the 3 seeds pooled per prompt; every seed reported"},
        "C5_failures": "failed generation = unsuccessful on every metric (GenEval incorrect; IR r_fail)",
        "C6_test_sets": "GenEval only",
        "evaluator_cost": "GenEval/HPS scoring runs after delivery and is never added to inference cost",
    }
    prop.parent.mkdir(parents=True, exist_ok=True)
    prop.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8", newline="\n")
    Path(a.test_methods_out).write_text(json.dumps({"selected_on": "train_dev", "methods": methods,
                                                    "prereg_c": a.proposal, "prereg_c_sha256": _sha(prop)}, indent=1),
                                        encoding="utf-8")
    marker.write_text(json.dumps({
        "prereg": "C", "role": "research",
        "signed_by": "protocol-lock snapshot recorded by the agent under the project owner's authorization; "
                     "not a signature of anyone else",
        "approval_kind": "protocol_lock_snapshot", "authorization_record": AUTH, "authorization_sha256": _sha(REPO / AUTH),
        "proposal_path": a.proposal, "proposal_sha256": _sha(prop), "recorded_at": snap["locked_at"],
        "recorded_by": "apsp.reports.lock_c (Claude Code agent pipeline)"}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"locked": a.proposal, "methods": methods, "M_test": snap["C4_test_contrasts"]["M_test"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
