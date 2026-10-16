"""G1/G2 after PREREG-B on a SYNTHETIC bank: replay re-run of the dev methods, G2 never STOP from replay alone."""
import json

import numpy as np
import pandas as pd

from apsp.contracts.schema import load_bank
from apsp.data.synthetic_bank import make_synthetic_bank, synthetic_costs
from apsp.reports import gates_b


def test_g1_g2_synthetic(tmp_path):
    bank_dir = make_synthetic_bank(tmp_path / "bank", n_prompts=40)
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps(synthetic_costs().to_dict()))
    bank = load_bank(bank_dir)
    dev = bank.instances_of(["syn_train_dev"])
    rng = np.random.default_rng(0)
    rows = [{"prompt_uid": u, "bank_id": b, "split": "syn_train_dev", "method": m, "q_eval": float(rng.normal(0.5, 0.2)),
             "success": True} for (u, b) in dev for m in ("psp", "bon:4", "single32")]
    run = tmp_path / "dev"
    run.mkdir()
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    snap = {"B4_r_fail": -2.0, "B3_thresholds": {"tau_prac": 0.05, "theta_H": 0.05}, "B12_contrasts": {"M": 3}}
    rep = gates_b.evaluate(str(bank_dir), str(costs), str(run), snap, fit_split="syn_train_fit", dev_split="syn_train_dev")
    assert rep["G1"]["records_n"] == len(dev) and "single32" in rep["G1"]["not_replayable"]
    assert set(rep["G1"]["by_method"]) == {"bon:4"}
    assert rep["G2_quality_status"] in ("GO", "INC")
    assert all(v["decision"] in ("GO", "INC", "NA (infeasible cap)") for v in rep["G2_quality"].values())
    assert rep["G2_cost"]["status"] == "INC"


def test_rl_train_real_on_synthetic_bank(tmp_path):
    from apsp.policies.reinforce import load_policy, train_real
    bank_dir = make_synthetic_bank(tmp_path / "bank", n_prompts=40)
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps(synthetic_costs().to_dict()))
    snap = {"B4_r_fail": -2.0, "B5_sigma_R": 0.4}
    s = train_real(str(bank_dir), str(costs), snap, 0, str(tmp_path / "rl"), iterations=4,
                   fit_split="syn_train_fit", dev_split="syn_train_dev")
    assert s["sigma_R"] == 0.4 and s["r_fail"] == -2.0 and s["ctrl_dispatch_us_measured"] >= 1 and s["caps"]
    pol = load_policy(tmp_path / "rl")
    assert pol.cph.startswith("CPH") or len(pol.cph) > 8
