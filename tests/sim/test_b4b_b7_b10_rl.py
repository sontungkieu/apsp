"""B4b selection, B7 prompt-only selector, B10-min predictor and the RL registry path on the synthetic bank.
Logic only: synthetic data and costs are never evidence."""
import json

import numpy as np
import pytest

from apsp.contracts.identity import bank_rows, pool_rows
from apsp.sim import b4b_select as bs
from apsp.sim.b4b_grid import to_spec
from apsp.sim.caps import fixed_method_costs
from apsp.sim.env import ReplayEngine
from apsp.sim.methods import MethodError, RunContext, parse_method, rows_for, run_method


@pytest.fixture()
def full_costs(syn_costs):
    """Synthetic table with every prune pair of the B4b grid (synthetic_costs lacks 12/3 targets)."""
    from apsp.sim.b4b_grid import required_primitives
    pr = {tuple(p): 200 + 20 * p[0] for p in required_primitives(bs.sd15_grid())["prune_pairs"]}
    return syn_costs.with_overrides(prune={**syn_costs.prune, **pr})


def test_grid_outcomes_match_run_schedule(ctx, full_costs):
    uids = bs.prompts_with_pool(ctx.bank.instances_of(["syn_train_fit"]))[:6]
    grid = bs.sd15_grid()
    rng = np.random.default_rng(0)
    sample = sorted(set(rng.integers(0, len(grid), 60).tolist()) | {0, len(grid) - 1})
    arrays = [bs.pool_arrays(ctx.accessor, u) for u in uids]
    Y = bs.grid_outcomes(arrays, ctx.r_fail, [grid[i] for i in sample])
    fx = fixed_method_costs(full_costs)
    for n, i in enumerate(sample):
        for p, u in enumerate(uids):
            r = ctx.run_spec(to_spec(grid[i]), (u, 0), 10 ** 15, mc=fx)
            assert r.feasible and r.Y == pytest.approx(Y[p, n]), (i, u)


def test_strategy_costs_coverage_and_selection(ctx, syn_costs, full_costs):
    C_part, missing = bs.strategy_costs(fixed_method_costs(syn_costs))
    assert (C_part < 0).any() and missing                       # 12->8 etc. not in the synthetic table
    C, miss2 = bs.strategy_costs(fixed_method_costs(full_costs))
    assert (C > 0).all() and not miss2
    s0 = bs.sd15_grid()[0]
    assert C[0] == ctx.run_spec(to_spec(s0), (bs.prompts_with_pool(ctx.bank.instances_of(["syn_train_fit"]))[0], 0),
                                10 ** 15, mc=fixed_method_costs(full_costs)).spend_us
    fit = bs.prompts_with_pool(ctx.bank.instances_of(["syn_train_fit"]))
    dev = bs.prompts_with_pool(ctx.bank.instances_of(["syn_train_dev"]))
    Yf = bs.grid_outcomes([bs.pool_arrays(ctx.accessor, u) for u in fit], ctx.r_fail)
    Yd = bs.grid_outcomes([bs.pool_arrays(ctx.accessor, u) for u in dev], ctx.r_fail)
    cap = ctx.caps.cap(1.0)
    ch = bs.select_b4b_dev(Yf, Yd, C_part, cap)
    assert ch is not None and ch["cost_us"] <= cap and C_part[ch["grid_index"]] >= 0
    assert ch["n_uncovered"] == int((C_part < 0).sum()) and len(ch["shortlist"]) <= 5
    m = parse_method(f"{ch['name']}@1.0")
    assert m.per_prompt() and rows_for(m, 0) == tuple(pool_rows(2))[: ch["strategy"]["k_init"]]
    with pytest.raises(MethodError):
        parse_method(f"b4b:{len(bs.sd15_grid())}")


def test_b4b_registry_runs_one_episode_per_prompt(ctx, full_costs):
    m = parse_method("b4b:0")
    u = bs.prompts_with_pool(ctx.bank.instances_of(["syn_train_fit"]))[0]
    rc = RunContext(full_costs, ctx.caps, ctx.norm)
    r = run_method(m, ReplayEngine(ctx.accessor, u), bank_id=0, ctx=rc, is_replay=True)
    assert r.feasible and r.n_verify == len(bs.sd15_grid()[0].cutoff_times)


def test_b7_prompt_selector(ctx, tmp_path):
    from apsp.policies.prompt_selector import PromptSelector, tune_b7
    fit = ctx.bank.instances_of(["syn_train_fit"])
    dev = ctx.bank.instances_of(["syn_train_dev"])
    cap = ctx.caps.cap(1.0)
    out = tune_b7(ctx, fit, dev, cap)
    assert out is not None
    sel = PromptSelector.from_dict(out["selector"])
    assert sel.info["fold_preprocessing"].startswith("PCA16") and sel.pca_hash == ctx.pca.hash()
    assert all(c <= cap for c in sel.costs)
    p = tmp_path / "b7.json"
    p.write_text(json.dumps(out["selector"]))
    m = parse_method(f"b7:{p}@1.0")
    u, b = fit[0]
    rc = RunContext(ctx.base_costs, ctx.caps, ctx.norm)
    r = run_method(m, ReplayEngine(ctx.accessor, u), bank_id=b, ctx=rc, prompt_feat=ctx.prompt_feat(u), is_replay=True)
    assert r.feasible and r.n_verify == 2 and r.delivery.rm_used
    with pytest.raises(MethodError, match="cap"):
        parse_method(f"b7:{p}")


def test_b10_predictor_pred_cuts_no_verify(ctx, tmp_path):
    from apsp.policies.latent_predictor import LatentPredictor, b10_cost, tune_b10
    fit = ctx.bank.instances_of(["syn_train_fit"])
    dev = ctx.bank.instances_of(["syn_train_dev"])
    cap = ctx.caps.cap(1.0)
    fx = fixed_method_costs(ctx.base_costs)
    out = tune_b10(ctx, fit, dev, cap, fx)
    assert out is not None and out["cost_us"] <= cap
    pr = LatentPredictor.from_dict(out["predictor"])
    assert set(pr.models) == {pr.j1, pr.j2} and pr.info["selected_on"] == "train_dev" and pr.pred_us[8] >= 1
    p = tmp_path / "b10.json"
    p.write_text(json.dumps(out["predictor"]))
    u, b = fit[0]
    rc = RunContext(fx, ctx.caps, ctx.norm)
    r = run_method(parse_method(f"b10:{p}@1.0"), ReplayEngine(ctx.accessor, u), bank_id=b, ctx=rc, is_replay=True)
    assert r.feasible and r.n_verify == 0 and r.delivery.rm_used
    assert r.spend_us == b10_cost(pr.j1, pr.j2, fx.with_overrides(pred={**fx.pred, **pr.pred_us}))
    reads = {e["event"] for e in ctx.accessor.log if e["key"][0] == u and e["purpose"] == "env"}
    assert "preview" not in reads and "lfeat" in reads          # no intermediate reward is read


def test_rl_registry_loads_frozen_policy(ctx, tmp_path):
    torch = pytest.importorskip("torch")
    from apsp.policies import reinforce as rl
    cfg = rl.RLConfig(hidden=8)
    net = rl.ActorCritic(cfg.obs_dim(), cfg.hidden)
    res = rl.TrainResult(net, cfg)
    d = rl.save_artifact(tmp_path / "rl", res, "CPH-test", {})
    pol = rl.load_policy(d / "artifact.json")
    assert pol.mode == "greedy" and pol.cph == "CPH-test"
    for a, b in zip(net.state_dict().values(), pol.net.state_dict().values()):
        assert torch.equal(a, b)
    u, bk = ctx.bank.instances_of(["syn_train_fit"])[0]
    rc = RunContext(ctx.base_costs, ctx.caps, ctx.norm)
    r = run_method(parse_method(f"rl:{(d / 'artifact.json').as_posix()}@1.0"), ReplayEngine(ctx.accessor, u), bank_id=bk,
                   ctx=rc, prompt_feat=ctx.prompt_feat(u), is_replay=True)
    assert r.feasible and r.spend_us <= ctx.caps.cap(1.0)
