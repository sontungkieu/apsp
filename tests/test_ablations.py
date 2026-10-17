"""Ablation grouping (RL seeds pooled per variant) and the descriptive online dev contrasts."""
import numpy as np
import pandas as pd

from apsp.reports import ablations


def test_group_and_contrasts(tmp_path):
    def art(d):
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "artifact.json").write_text("{}")
        return f"rl:{p / 'artifact.json'}@0.5"
    main = [art(f"rl/seed{i}") for i in range(2)]
    nop = [art(f"rl_ablate/nopaid_seed{i}") for i in range(2)]
    pc = [art(f"rl_ablate/a05/percap_seed{i}") for i in range(2)]
    assert {ablations.group(m) for m in main} == {"rl-main*@0.5"}
    assert {ablations.group(m) for m in nop} == {"rl-nopaid*@0.5"}
    assert {ablations.group(m) for m in pc} == {"rl-percap*@0.5"}
    rng = np.random.default_rng(0)
    rows = [{"prompt_uid": f"u{i}", "bank_id": b, "split": "train_dev", "method": m, "family": "rl", "alpha": 0.5,
             "q_eval": float(rng.normal(0.5, 0.1)), "success": True, "e2e_seconds": 5.0, "ledger_spend_us": 1}
            for i in range(10) for b in (0, 1) for m in main + nop + pc]
    out = ablations.online_contrasts(pd.DataFrame(rows), {"r_fail": -2.0, "cost_measure": "e2e_seconds"}, alphas=(0.5,))
    by = {r["ablation"]: r for r in out}
    assert by["paid-score features removed vs RL main"]["method"] == "rl-nopaid*@0.5"
    assert by["one policy per cap vs budget-conditioned RL main"]["reference"] == "rl-main*@0.5"
    assert by["paid trajectory features added vs RL main"]["status"] == "not run"
