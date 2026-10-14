"""Replay end-to-end, closed-loop == schedule runner, leakage proxies, O1 upper bound."""
import copy
import math

import numpy as np
import pytest

from apsp.contracts.identity import bank_rows
from apsp.policies.fixed import FixedPairPolicy, best_fixed
from apsp.policies.random_policy import RandomFeasiblePolicy
from apsp.policies.threshold import ThresholdParams, ThresholdPolicy
from apsp.sim.oracles import o1
from apsp.sim.replay import ReplayContext
from apsp.sim.schedules import enumerate_pairs, fixed_pair_spec


def _policies():
    return [FixedPairPolicy(4, 8), RandomFeasiblePolicy(1),
            ThresholdPolicy(ThresholdParams(0.5, 0.5, 0.3, 0.0, 2, 2))]


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_replay_e2e_policies_respect_cap(ctx, syn_costs, alpha):
    cap = ctx.caps.cap(alpha)
    keys = ctx.bank.instances_of(["syn_train_dev"])
    mc = syn_costs.with_overrides(disp=100)
    for pol in [RandomFeasiblePolicy(2), ThresholdPolicy(ThresholdParams(0.0, 0.0, 0.0, 0.0, 1, 1))]:
        for k in keys:
            r = ctx.run_policy(pol, k, cap, mc)
            if not r.feasible:
                continue
            assert r.spend_us <= cap
            assert sum(e.fee_us for e in r.trace) + mc.fin[2] + mc.once == r.spend_us
            assert len(r.prune_js) == 2 and r.prune_js[0] < r.prune_js[1] <= 15
            assert math.isfinite(r.Y)


def test_closed_loop_fixed_equals_schedule_runner(ctx):
    cap = ctx.caps.cap(1.5)
    for k in ctx.bank.instances_of(["syn_train_fit"])[:20]:
        for j1, j2 in [(4, 8), (1, 2), (7, 15)]:
            b = ctx.run_spec(fixed_pair_spec(j1, j2, bank_rows(k[1])), k, cap)
            if not b.feasible:
                with pytest.raises(RuntimeError, match="infeasible"):
                    ctx.run_policy(FixedPairPolicy(j1, j2), k, cap, ctx.base_costs)
                continue
            a = ctx.run_policy(FixedPairPolicy(j1, j2), k, cap, ctx.base_costs)
            assert (a.Y, a.spend_us, a.delivery.winner, a.prune_js) == (b.Y, b.spend_us, b.delivery.winner, (j1, j2))


def test_accessor_reads_only_current_checkpoint_and_alive(ctx, syn_costs):
    cap = ctx.caps.cap(1.0)
    for pol in _policies():
        for k in ctx.bank.instances_of(["syn_train_dev"])[:10]:
            ctx.accessor.log.clear()
            r = ctx.run_policy(pol, k, cap, syn_costs)
            verify_events = [e for e in r.trace if e.action == "VERIFY"]
            reads = [l for l in ctx.accessor.log if l["event"] == "preview"]
            assert len(reads) == len(verify_events)
            for ev, rd in zip(verify_events, reads):
                assert rd["j"] == ev.j and set(rd["cids"]) == set(ev.alive) and rd["purpose"] == "env"
            dl = [l for l in ctx.accessor.log if l["event"] == "delivered" and l["purpose"] == "env"]
            assert len(dl) == 1 and len(dl[0]["cids"]) == 2


def test_sentinel_unpurchased_data_does_not_change_behaviour(syn_bank, syn_costs):
    """Overwrite every score the episode did not buy with sentinels; observations and actions must not change."""
    ctx = ReplayContext.build(syn_bank, syn_costs, fit_split="syn_train_fit")
    cap = ctx.caps.cap(1.0)
    for pol_factory in (lambda: RandomFeasiblePolicy(5), lambda: ThresholdPolicy(ThresholdParams(0.2, 0.4, 0.1, 0.2, 3, 1))):
        for k in syn_bank.instances_of(["syn_train_dev"])[:8]:
            base = ctx.run_policy(pol_factory(), k, cap, syn_costs, record_obs=True)
            bought = {(ev.j, c) for ev in base.trace if ev.action == "VERIFY" for c in ev.alive}
            final_alive = base.trace[-1].alive if base.trace[-1].action == "CONTINUE" else None
            bank2 = copy.deepcopy(syn_bank)
            ir = bank2.preview_ir[k].copy()
            for s in range(8):
                for j in range(1, 17):
                    if (j, (k[1], s)) not in bought:
                        ir[s, j - 1] = 1e6 if (s + j) % 2 else -1e6          # future / pruned / unbought sentinels
            bank2.preview_ir[k] = ir
            dl = bank2.delivered_ir[k].copy()
            winners_pool = set(final_alive or ())
            for s in range(8):
                if (k[1], s) not in winners_pool:
                    dl[s] = 1e6                                               # non-final candidates' delivered IR
            bank2.delivered_ir[k] = dl
            ctx2 = ReplayContext(bank2, type(ctx.accessor)(bank2), ctx.base_costs, ctx.caps, ctx.norm, ctx.pca, ctx.r_fail)
            alt = ctx2.run_policy(pol_factory(), k, cap, syn_costs, record_obs=True)
            assert [e.action for e in alt.trace] == [e.action for e in base.trace]
            assert all(np.array_equal(a, b) for a, b in zip(alt.obs_log, base.obs_log))
            assert alt.Y == base.Y


def test_o1_upper_bounds_main_grammar_policies(ctx, syn_costs):
    keys = ctx.bank.instances_of(["syn_train_dev"])
    Y, C = ctx.schedule_table(keys)
    for alpha in (0.5, 1.0):
        cap = ctx.caps.cap(alpha)
        ub = o1(Y, C, cap)
        for pol in _policies()[1:]:
            for i, k in enumerate(keys):
                r = ctx.run_policy(pol, k, cap, syn_costs.with_overrides(disp=50))
                if r.feasible:
                    assert r.Y <= ub[i] + 1e-12


def test_best_fixed_selection_rules(ctx):
    keys = ctx.bank.instances_of(["syn_train_fit"])
    Y, C = ctx.schedule_table(keys)
    b = best_fixed(Y, C, ctx.caps.cap(1.0))
    pairs = enumerate_pairs()
    ok = C <= ctx.caps.cap(1.0)
    assert b.mean_y == pytest.approx(Y[:, ok].mean(0).max())
    assert C[pairs.index((b.j1, b.j2))] <= ctx.caps.cap(1.0)
    assert best_fixed(Y, C, int(C.min()) - 1) is None
