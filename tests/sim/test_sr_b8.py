"""B5 screen-and-resume and B8 query-then-commit on the synthetic bank (logic only; not evidence)."""
import numpy as np
import pytest

from apsp.contracts.identity import bank_rows, pool_rows
from apsp.contracts.ledger import schedule_cost
from apsp.policies import query_commit as qc
from apsp.policies import screen_resume as sr
from apsp.sim.caps import fixed_method_costs


def test_sr_specs_costs_and_rows(syn_costs):
    fx = fixed_method_costs(syn_costs)
    c = sr.SRConfig(16, 5, 2)
    spec = sr.sr_spec(c)
    assert spec.candidates == tuple(pool_rows(2)) and spec.queries == (5,) and spec.final_k == 2
    expect = 5 * 4 * fx.den[16] + fx.ver[16] + fx.prune[(16, 2)] + 11 * 4 * fx.den[2] + fx.fin[2]
    assert sr.sr_cost(c, fx) == expect
    assert sr.sr_spec(sr.SRConfig(8, 3, 1), bank_id=1).candidates == tuple(bank_rows(1))
    with pytest.raises(ValueError):
        sr.sr_spec(sr.SRConfig(4, 3, 1))
    assert len(sr.enumerate_sr()) == 2 * 15 * 2


def test_sr_replay_and_dev_selection(ctx):
    fit = ctx.bank.instances_of(["syn_train_fit"])
    dev = ctx.bank.instances_of(["syn_train_dev"])
    cap = ctx.caps.cap(1.0)
    keys = {"syn_train_fit": fit, "syn_train_dev": dev}
    choice = sr.select_sr_dev(lambda c, s: sr.replay_mean(ctx, c, keys[s], cap), fixed_method_costs(ctx.base_costs), cap,
                              fit_split="syn_train_fit", dev_split="syn_train_dev")
    assert choice is not None and choice["cost_us"] <= cap and len(choice["shortlist"]) <= 5
    # M=16 episodes are one per prompt
    assert len(sr.sr_instance_keys(fit, 16)) == len({u for u, _ in fit})
    # q=1 delivers without an inference RM call (F(1))
    r = ctx.run_spec(sr.sr_spec(sr.SRConfig(8, 4, 1), fit[0][1]), fit[0], 10 ** 12)
    assert r.feasible and r.delivery.rm_used is False


def test_b8_preflight_reuse_and_cost(ctx):
    fit = ctx.bank.instances_of(["syn_train_fit"])
    fx = fixed_method_costs(ctx.base_costs)
    sel = qc.train_selector(ctx, 3, fit)
    assert sel.W.shape == (qc.FEAT_DIM + 1, len(qc.future_schedules(3))) and sel.cv["folds_by"] == "prompt group"
    from apsp.sim.env import ReplayEngine
    key = fit[0]
    rows = bank_rows(key[1])
    r = qc.run_query_commit(sel, ReplayEngine(ctx.accessor, key[0]), rows=rows, mc=fx, cap_us=10 ** 12, norm=ctx.norm,
                            prompt_feat=ctx.prompt_feat(key[0]))
    commit = next(e.action for e in r.trace if e.action.startswith("COMMIT"))
    j1, j2 = (int(x) for x in commit[7:-1].split(","))
    assert j1 >= 3 and j2 > j1
    n_ver_expected = 1 + (j1 > 3) + 1                          # j_c, extra j1 only if j1 > j_c, j2
    assert r.n_verify == n_ver_expected
    assert r.spend_us == qc.b8_cost(rows, 3, j1, j2, fx)
    # preflight: a cap below every completion is infeasible and pays nothing
    lo = min(qc.b8_cost(rows, 3, a, b, fx) for a, b in qc.future_schedules(3)) - 1
    r2 = qc.run_query_commit(sel, ReplayEngine(ctx.accessor, key[0]), rows=rows, mc=fx, cap_us=lo, norm=ctx.norm)
    assert not r2.feasible and r2.spend_us == 0 and r2.trace == []


def test_b8_selection_masks_unaffordable(ctx):
    fit = ctx.bank.instances_of(["syn_train_fit"])
    fx = fixed_method_costs(ctx.base_costs)
    sel = qc.train_selector(ctx, 2, fit)
    rows = bank_rows(0)
    costs = [qc.b8_cost(rows, 2, a, b, fx) for a, b in sel.schedules]
    cap = sorted(costs)[3]
    x = np.zeros(qc.FEAT_DIM)
    choice = sel.choose(x, [c <= cap for c in costs], costs)
    assert costs[sel.schedules.index(choice)] <= cap
    assert sel.choose(x, [False] * len(costs), costs) is None
    d = qc.RidgeSelector.from_dict(sel.to_dict())
    assert d.choose(x, [c <= cap for c in costs], costs) == choice and d.cph == sel.cph


def test_b8_tuning_selects_jc_on_dev(ctx):
    fit = ctx.bank.instances_of(["syn_train_fit"])
    dev = ctx.bank.instances_of(["syn_train_dev"])
    out = qc.tune_b8(ctx, fit, dev, ctx.caps.cap(1.0))
    assert out is not None and 1 <= out["j_c"] <= 8 and set(out["dev_by_jc"]) <= {str(j) for j in range(1, 9)}


def test_tune_baselines_cli_freezes_methods(syn_bank_dir, tmp_path):
    import json
    from apsp.sim import tune_baselines
    assert tune_baselines.main(["--bank", str(syn_bank_dir), "--synthetic-costs", "--fit-split", "syn_train_fit",
                                "--dev-split", "syn_train_dev", "--alphas", "1.0", "--b6-evals", "10",
                                "--out", str(tmp_path)]) == 0
    fr = json.loads((tmp_path / "methods_frozen.json").read_text())
    fams = {m.split(":")[0].split("@")[0] for m in fr["methods"]}
    assert {"single", "single32", "bon", "psp", "fixed", "sr", "threshold", "b8", "b4b", "b7", "b10"} <= fams
    assert "b4b_coverage" in fr["selection_log"]
    assert fr["selected_on"].startswith("syn_train_dev") and (tmp_path / "norm.json").exists()
    with pytest.raises(SystemExit, match="val/test"):
        tune_baselines.main(["--bank", str(syn_bank_dir), "--synthetic-costs", "--dev-split", "val", "--out", str(tmp_path)])
