import pytest

from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import (CostTableError, MethodCosts, cost_action, cost_gen, fixed_pair_cost, reserve_once,
                                   schedule_cost)
from apsp.contracts.paid_state import EpisodeState, LedgerError, StateError
from apsp.contracts.policy_api import CONT, PRUNE, VER, Cut, GrammarState, ScheduleSpec, ScoreRecord
from apsp.sim.schedules import bon_spec, enumerate_pairs, fixed_pair_spec


def _mc(**kw):
    base = dict(den={8: 10, 4: 6, 2: 4, 1: 3}, ver={8: 50, 4: 30, 2: 20}, fin={2: 70, 1: 15},
                prune={(8, 4): 2, (4, 2): 1, (8, 1): 2}, feat={8: 3, 4: 2, 2: 1}, lfeat={8: 5}, pred={8: 4})
    base.update(kw)
    return MethodCosts(**base)


def test_cont_feature_once_per_block_regression():
    mc = _mc(uses_traj=True, disp=7)
    q = GrammarState.make(1, 8)
    assert cost_gen(q, CONT, mc) == 4 * 10 + 3 == 43        # not 4*(10+3)=52
    assert cost_action(q, CONT, mc) == 43 + 7               # dispatch added separately
    assert cost_action(q, VER, mc) == 50 + 7
    assert cost_action(GrammarState(1, 8, True, False, True), PRUNE, mc) == 2 + 7


def test_once_prepaid_and_fee_validation():
    assert reserve_once(100, _mc(once=30)) == 70
    with pytest.raises(CostTableError):
        _mc(disp=-1)
    with pytest.raises(CostTableError):
        _mc(ver={8: 1.5})
    with pytest.raises(CostTableError):
        _mc(prune={(4, 8): 1})
    with pytest.raises(CostTableError):
        cost_gen(GrammarState.make(1, 16), CONT, _mc())     # missing primitive is an error, never 0


def test_schedule_cost_matches_closed_form_all_pairs():
    mc = _mc(once=11)
    for j1, j2 in enumerate_pairs():
        assert schedule_cost(fixed_pair_spec(j1, j2, bank_rows(0)), mc) == fixed_pair_cost(j1, j2, mc)


def test_bon_and_pred_cut_costs():
    mc = _mc()
    assert schedule_cost(bon_spec(1, bank_rows(0)), mc) == 16 * 4 * 3 + 15        # F(1): fin(1) without RM
    spec = ScheduleSpec(bank_rows(0), queries=(), cuts=(Cut(3, 1, "pred"),), final_k=1)
    assert schedule_cost(spec, mc) == 3 * 4 * 10 + 13 * 4 * 3 + 5 + 4 + 2 + 15      # lfeat+pred+prune, no ver


def test_schedule_validation():
    with pytest.raises(ValueError):
        ScheduleSpec(bank_rows(0), (4,), (Cut(4, 4), Cut(4, 2)), 2).validate()     # same checkpoint twice
    with pytest.raises(ValueError):
        ScheduleSpec(bank_rows(0), (), (Cut(4, 4, "paid"),), 4).validate()        # paid cut without query
    with pytest.raises(ValueError):
        ScheduleSpec(bank_rows(0), (4, 16), (Cut(4, 4),), 4).validate()           # query at 16
    with pytest.raises(ValueError):
        ScheduleSpec(bank_rows(0), (4,), (Cut(4, 4),), 2).validate()              # final_k mismatch


def _state(cap=1000):
    return EpisodeState(grammar_id="G-842-v1", rows=bank_rows(0), cap_us=cap, L_fn=lambda q: 0.0, C_ref=100)


def _recs(vals):
    return {(0, i): (ScoreRecord(True, True, True, v) if v is not None else ScoreRecord(True, False, False, None))
            for i, v in enumerate(vals)}


def test_charge_rules():
    s = _state(10)
    s.charge(4)
    with pytest.raises(LedgerError):
        s.charge(7)              # would exceed the cap; no free rescue
    with pytest.raises(LedgerError):
        s.charge(-1)
    with pytest.raises(LedgerError):
        s.charge(1.0)
    with pytest.raises(LedgerError):
        s.charge(True)


def test_verify_and_prune_guards():
    s = _state()
    with pytest.raises(StateError):
        s.record_verify(0, _recs([1] * 8))
    s.arrive(1)
    with pytest.raises(StateError):
        s.record_verify(2, _recs([1] * 8))                       # wrong j
    with pytest.raises(StateError):
        s.record_prune({(0, 0), (0, 1), (0, 2), (0, 3)})         # no fresh scores
    s.record_verify(1, _recs([5, 1, None, 7, 7, 0, 2, 3]))
    with pytest.raises(StateError):
        s.record_verify(1, _recs([1] * 8))                       # duplicate query
    with pytest.raises(StateError):
        s.record_prune({(0, 0), (0, 1), (0, 3), (0, 4)})         # not the deterministic top-4
    idx = s.record_prune({(0, 3), (0, 4), (0, 0), (0, 7)})
    assert idx == (0, 3, 4, 7) and s.alive == ((0, 0), (0, 3), (0, 4), (0, 7))
    with pytest.raises(StateError):
        s.record_prune({(0, 3), (0, 4)})                         # second prune at same checkpoint
    v = s.view()
    assert {p.cid for p in v.paid} == set(s.alive)               # pruned candidates' scores not exposed
    with pytest.raises(StateError):
        s.arrive(3)
