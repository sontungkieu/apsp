"""Dev freeze (operating point, G3/G3b/G4 manifest) and the two-look val report on fake episode files."""
import json

import numpy as np
import pytest
import yaml

from apsp.reports import confirm, select_dev
from apsp.sim.methods import parse_method


def _write(run, split, methods, seed=0, n=12, shift=None):
    rng = np.random.default_rng(seed)
    run.mkdir(parents=True)
    rows = []
    for i in range(n):
        base = rng.normal(0.5, 0.3)
        for b in (0, 1):
            for m in methods:
                pm = parse_method(m)
                rows.append({"prompt_uid": f"{split}{i}", "bank_id": b, "split": split, "method": m, "family": pm.family,
                             "alpha": pm.alpha, "q_eval": base + (shift or {}).get(m, 0) + rng.normal(0, 0.05),
                             "success": True, "ledger_spend_us": 1000, "e2e_seconds": {"psp": 5.0}.get(m, 3.0),
                             "forwards": 64, "n_candidates": 8, "n_verify": 2, "rm_used_in_inference": True})
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "manifest.json").write_text("{}")


def test_select_and_confirm(tmp_path, prereg_dir, monkeypatch):
    art = tmp_path / "a"
    art.mkdir()
    for f in ("t.json", "b10.json"):
        (art / f).write_text("{}")
    for s in ("seed0", "seed1"):
        (art / s).mkdir()
        (art / s / "artifact.json").write_text("{}")
    M = ["psp", "bon:4", "fixed:4,8@0.5", f"threshold:{art}/t.json@0.5", f"b10:{art}/b10.json@0.5",
         f"rl:{art}/seed0/artifact.json@0.5", f"rl:{art}/seed1/artifact.json@0.5"]
    _write(tmp_path / "dev", "train_dev", M[:5])
    _write(tmp_path / "devrl", "train_dev", M[5:])
    cfg = {"approved": True, "epsilon_ir": 0.05, "r_fail": -2.0, "n_primary_contrasts": 3, "reference_method": "psp",
           "cost_measure": "e2e_seconds", "descriptive_ci_level": 0.95, "budgets_alpha": [], "reference_levels": []}
    cp = tmp_path / "eval.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"B3_thresholds": {"delta_IR": 0.05, "theta_q": 0.05, "theta_pay": 0.025, "theta_RL": 0.025},
                                "B13_sample_size": {"look2_val_ext_prompts_if_inc": 100}}))
    fz = tmp_path / "frozen.json"
    assert select_dev.main(["--runs", str(tmp_path / "dev"), str(tmp_path / "devrl"), "--config", str(cp),
                            "--snapshot", str(snap), "--out", str(fz)]) == 0
    f = json.loads(fz.read_text())
    ids = {c["id"]: c for c in f["contrast_manifest"]}
    assert ids["G3_alpha_0.5"]["reference"] == "fixed:4,8@0.5" and ids["G4_alpha_0.5"]["method"] == "rl*@0.5"
    assert ids["G3b_alpha_0.5"]["reference"].startswith("b10:") and len(f["methods"]) == 7
    with pytest.raises(SystemExit, match="never replaced"):
        select_dev.main(["--runs", str(tmp_path / "dev"), "--config", str(cp), "--snapshot", str(snap), "--out", str(fz)])
    monkeypatch.setattr(confirm, "require_signed", lambda w: {})
    _write(tmp_path / "val", "val", M, seed=1)
    log = tmp_path / "val_access.jsonl"
    out = tmp_path / "c1.json"
    assert confirm.main(["--runs", str(tmp_path / "val"), "--frozen", str(fz), "--config", str(cp), "--look", "1",
                         "--val-log", str(log), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert {c["id"] for c in r["contrasts"]} == set(ids) and all("decision" in c for c in r["contrasts"])
    with pytest.raises(Exception, match="already used"):
        confirm.main(["--runs", str(tmp_path / "val"), "--frozen", str(fz), "--config", str(cp), "--look", "1",
                      "--val-log", str(log), "--out", str(out)])
    _write(tmp_path / "test", "test", M, seed=2)
    with pytest.raises(SystemExit, match="reads only"):
        confirm.main(["--runs", str(tmp_path / "test"), "--frozen", str(fz), "--config", str(cp), "--look", "2",
                      "--val-log", str(log), "--out", str(out)])
