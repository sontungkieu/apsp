import numpy as np
import pytest

from apsp.reports.replay_vs_online import g1_status
from apsp.stats.bootstrap import cluster_bootstrap, contrast_level, required_resamples
from apsp.stats.decisions import GO, INC, STOP, after_second_look, cost_decision, quality_decision, thresholds
from apsp.stats.frontier import Point, operating_point, pareto
from apsp.stats.variance import nested_components


def test_thresholds_and_levels():
    th = thresholds(0.05)
    assert th.tau_prac == 0.02 and th.delta_IR == 0.01 and th.theta_RL == 0.005
    assert thresholds(0.5).tau_prac == pytest.approx(0.1)
    assert contrast_level(10) == pytest.approx(1 - 0.0025)
    assert required_resamples(contrast_level(10)) >= 100 / (0.0025 / 2)


def test_quality_and_cost_rules_including_grey_zone():
    assert quality_decision(0.03, 0.1, 0.02) == GO
    assert quality_decision(-0.1, 0.01, 0.02) == STOP
    assert quality_decision(0.0, 0.05, 0.02) == INC
    assert cost_decision(-0.005, 0.02, 0.11, 0.2, 0.01) == GO
    assert cost_decision(-0.005, 0.02, 0.06, 0.09, 0.01) == INC     # 5-10% savings stays INC
    assert cost_decision(-0.005, 0.02, 0.01, 0.04, 0.01) == STOP    # UCB savings < 5%
    assert cost_decision(-0.05, -0.02, 0.2, 0.3, 0.01) == STOP      # quality clearly worse
    assert cost_decision(-0.02, 0.01, 0.2, 0.3, 0.01) == INC        # NI not met
    assert after_second_look(INC) == "inconclusive"


def test_cluster_bootstrap_respects_clusters():
    rng = np.random.default_rng(0)
    # strong prompt effect: banks within a prompt are nearly identical -> CI width driven by #prompts
    per = {f"p{i}": [m, m + 1e-3] for i, m in enumerate(rng.normal(0, 1, 40))}
    ci = cluster_bootstrap(per, 0.95, seed=1, n_resamples=4000)
    naive_se = np.std([v for vs in per.values() for v in vs]) / np.sqrt(80)
    assert (ci.hi - ci.lo) / 3.92 > 1.2 * naive_se     # cluster CI wider than the iid-80 CI
    ci2 = cluster_bootstrap(per, 0.95, seed=1, n_resamples=4000)
    assert (ci.lo, ci.hi) == (ci2.lo, ci2.hi)


def test_coverage_at_adjusted_alpha_smoke():
    """Small Monte Carlo coverage check at an adjusted level (descriptive; not a proof)."""
    rng = np.random.default_rng(2)
    level, hits, R = 0.99, 0, 150
    for r in range(R):
        per = {f"p{i}": list(rng.normal(0.0, 1.0, 2) + rng.normal()) for i in range(60)}
        ci = cluster_bootstrap(per, level, seed=r, n_resamples=3000)
        hits += ci.lo <= 0.0 <= ci.hi
    assert hits / R > 0.95


def test_variance_components():
    rng = np.random.default_rng(1)
    per = {f"p{i}": list(rng.normal(0, 0.3, 2) + rng.normal(0, 1.0)) for i in range(400)}
    c = nested_components(per)
    assert c["sigma_p2"] == pytest.approx(1.0, rel=0.25) and c["sigma_b2"] == pytest.approx(0.09, rel=0.25)


def test_operating_point_uses_actual_cost():
    pts = [Point("a", 10, 1.0), Point("b", 8, 0.9), Point("c", 9, 1.1), Point("d", 7, 0.5, qualified=False)]
    assert operating_point(pts).name == "b"
    assert [p.name for p in pareto(pts)] == ["d", "b", "c"]


def test_g1_status():
    assert g1_status(-0.004, 0.004, 0.02) == "REPLAY_FOR_GATES"
    assert g1_status(-0.009, 0.004, 0.02) == "REPLAY_TRAIN_ONLY_GATES_ONLINE"
    assert g1_status(0.011, 0.03, 0.02) == "STOP-ENG"
    assert g1_status(-0.03, 0.03, 0.02) == "INCONCLUSIVE_EXTEND_TRAIN_DEV"
