"""PREREG-B snapshot: formulas on train data only, refuses after val output, never replaces a locked marker."""
import json

import numpy as np
import pandas as pd
import pytest

from apsp.data.synthetic_bank import make_synthetic_bank
from apsp.reports import lock_b
from apsp.stats.decisions import thresholds


def _eps(split, methods, n=12, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.normal(0.5, 0.3)
        for b in (0, 1):
            for m in methods:
                q = base + (shift or {}).get(m, 0.0) + rng.normal(0, 0.1)
                rows.append({"prompt_uid": f"u{i}", "bank_id": b, "split": split, "method": m, "q_eval": q,
                             "success": not (i == 0 and b == 0 and m == "bon:4")})
    return pd.DataFrame(rows)


def test_formulas_and_power(tmp_path):
    bank = make_synthetic_bank(tmp_path / "bank", n_prompts=30)
    bs = lock_b.bank_stats(bank, "syn_train_fit")
    assert bs["sigma_R"] > 0 and bs["r_fail"] < 0.5
    anc = lock_b.online_anchor(_eps("train_fit", ["psp", "bon:4"], shift={"psp": 0.2}), bs["r_fail"])
    assert 0.1 < anc["delta_anchor"] < 0.6 and anc["failures"] == 1
    th = thresholds(anc["delta_anchor"])
    assert th.tau_prac == max(0.2 * anc["delta_anchor"], 0.02) and th.delta_IR == th.tau_prac / 2
    pw = lock_b.power(_eps("train_dev", ["psp", "bon:4", "threshold:x@0.5"]), bs["r_fail"], th.delta_IR, 3)
    assert pw["basis"] == "adaptive candidates" and pw["N_required"] == pw["by_method"]["threshold:x@0.5"]["N_required"]
    with pytest.raises(SystemExit, match="train_dev only"):
        lock_b.power(_eps("val", ["psp", "bon:4"]), bs["r_fail"], th.delta_IR, 3)
    with pytest.raises(SystemExit, match="train_fit only"):
        lock_b.online_anchor(_eps("train_dev", ["psp", "bon:4"]), bs["r_fail"])


def test_refuses_after_val_and_never_replaces(tmp_path, monkeypatch):
    val = tmp_path / "bench_val"
    val.mkdir()
    (val / "episodes.jsonl").write_text("{}")
    a = type("A", (), {"val_guard": [str(val)]})()
    with pytest.raises(SystemExit, match="val output exists"):
        lock_b.build(a)
    monkeypatch.setattr(lock_b, "REPO", tmp_path)
    (tmp_path / "docs" / "PREREG_SIGNED").mkdir(parents=True)
    (tmp_path / "docs" / "PREREG_SIGNED" / "PREREG-B.json").write_text("{}")
    a = type("A", (), {"proposal": "p.json", "marker": "docs/PREREG_SIGNED/PREREG-B.json"})()
    with pytest.raises(SystemExit, match="never replaced"):
        lock_b.write({}, a)
