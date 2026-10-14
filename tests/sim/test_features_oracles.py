import itertools
import math

import numpy as np
import pytest

from apsp.contracts.identity import bank_rows
from apsp.contracts.paid_state import EpisodeState
from apsp.contracts.policy_api import ScoreRecord
from apsp.sim.features import IDX, MAIN_DIM, TRAJ_DIM, NormStats, PCA16, build_obs
from apsp.sim.oracles import o2_avg_dual, o2_avg_primal, o2_inst, verify_primal


def _state(vals_by_j, prune_after=None, rows=None):
    s = EpisodeState(grammar_id="G-842-v1", rows=rows or bank_rows(0), cap_us=10_000, L_fn=lambda q: 1000.0,
                     C_ref=5000, prompt_feat=np.arange(16, dtype=np.float32) / 10)
    for j in range(1, 17):
        s.arrive(j)
        if j in vals_by_j:
            vals = vals_by_j[j]
            s.record_verify(j, {c: (ScoreRecord(True, True, True, vals[c[1]]) if vals[c[1]] is not None
                                    else ScoreRecord(True, False, False, None)) for c in s.alive})
        if j == 3:
            break
    return s


def test_obs_shapes_finite_and_no_score_zeros():
    s = EpisodeState(grammar_id="G-842-v1", rows=bank_rows(0), cap_us=100, L_fn=lambda q: math.inf, C_ref=50)
    o = build_obs(s.view(), NormStats.identity())
    assert o.shape == (MAIN_DIM,) and np.isfinite(o).all()
    assert o[IDX["has_score"]] == 0 and o[IDX["mean_z"]] == 0
    s.arrive(1)
    s.record_traj({c: (math.nan, 0.5) for c in s.alive})
    o = build_obs(s.view(), NormStats.identity(), "traj")
    assert o.shape == (TRAJ_DIM,) and np.isfinite(o).all()


def test_all_invalid_and_ties_are_finite():
    s = _state({2: [None] * 8, 3: [1.0] * 8})
    o = build_obs(s.view(), NormStats.identity())
    assert np.isfinite(o).all()
    assert o[IDX["has_score"]] == 1 and o[IDX["frac_invalid"]] == 0
    assert o[IDX["z_cut"]] == 0 and o[IDX["has_two"]] == 0      # no common valid pair across the two verifies
    s2 = _state({3: [None] * 8})
    o2 = build_obs(s2.view(), NormStats.identity())
    assert o2[IDX["frac_invalid"]] == 1 and o2[IDX["mean_z"]] == 0


def test_candidate_order_invariance():
    vals = {2: [0.1, 0.5, -0.3, 2.0, 1.0, 0.2, 0.0, 0.7], 3: [0.3, 0.4, -0.1, 1.8, 1.2, 0.1, 0.2, 0.6]}
    a = build_obs(_state(vals).view(), NormStats.identity())
    perm = [5, 2, 7, 0, 1, 6, 3, 4]
    rows = tuple((0, i) for i in perm)
    b = build_obs(_state(vals, rows=rows).view(), NormStats.identity())
    assert np.allclose(a, b)
    assert a[IDX["has_two"]] == 1 and -1 <= a[IDX["kendall"]] <= 1


def test_norm_uses_purchase_checkpoint():
    norm = NormStats(tuple(float(j) for j in range(16)), tuple([1.0] * 16))
    s = _state({2: [5.0] * 8})
    o = build_obs(s.view(), norm)   # now at j=3 but score bought at j=2 -> z = 5 - mu[j=2] = 4
    assert o[IDX["mean_z"]] == pytest.approx(4.0)


def test_pca_deterministic_sign():
    X = np.random.default_rng(0).normal(size=(40, 768))
    p1, p2 = PCA16.fit(X), PCA16.fit(X.copy())
    assert p1.hash() == p2.hash()
    assert all(c[np.argmax(np.abs(c))] > 0 for c in p1.components)


def test_dual_below_exact_primal_and_witness():
    rng = np.random.default_rng(0)
    for _ in range(5):
        N, S = 3, 5
        Y = rng.normal(size=(N, S))
        C = rng.integers(1, 100, S)
        y_ref = Y[:, 2]
        delta = 0.1
        best = math.inf
        for assign in itertools.product(range(S), repeat=N):
            a = np.array(assign)
            ok, cost, _ = verify_primal(a, Y, C, y_ref, delta)
            if ok:
                best = min(best, cost)
        d = o2_avg_dual(Y, C, y_ref, delta)
        p = o2_avg_primal(Y, C, y_ref, delta)
        assert d.dual_lower_bound_total <= best + 1e-6
        assert p.verified and p.primal_cost_total >= best - 1e-9
        assert p.primal_cost_total == pytest.approx(best)       # MILP optimum on a tiny problem


def test_o2_inst_is_not_a_mean_cost_lower_bound():
    # instance 0 can overshoot cheaply, instance 1 reaching its own target is expensive:
    Y = np.array([[0.0, 3.0], [0.0, 1.0]])
    C = np.array([1, 2])
    y_ref = np.array([1.0, 1.0])
    inst = o2_inst(Y, C, y_ref, 0.0).sum()       # needs schedule 1 on both: 4
    p = o2_avg_primal(Y, C, y_ref, 0.0)           # mean target lets instance 1 stay cheap: 3
    assert p.primal_cost_total < inst


def test_target_infeasible_detected():
    Y = np.zeros((2, 3))
    d = o2_avg_dual(Y, np.array([1, 2, 3]), np.ones(2), 0.0)
    assert not d.target_feasible
    assert o2_avg_primal(Y, np.array([1, 2, 3]), np.ones(2), 0.0).status == "target_infeasible"
