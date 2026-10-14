"""DP vs brute force on toy horizons; mask neither admits impossible completions nor excludes affordable ones."""
import math
import random

import pytest

from apsp.contracts.ledger import MethodCosts, cost_action, cost_final
from apsp.contracts.policy_api import CONT, PRUNE, VER, VP, GrammarState
from apsp.sim.budget_dp import G842, GA1, L_of, initial_feasible, mask, solve_L


def _rand_mc(rng, disp=None, traj=None):
    ks = (8, 4, 2)
    return MethodCosts(den={k: rng.randint(1, 20) for k in ks}, ver={k: rng.randint(0, 60) for k in ks},
                       fin={2: rng.randint(1, 40)}, prune={(8, 4): rng.randint(0, 5), (4, 2): rng.randint(0, 5)},
                       feat={k: rng.randint(0, 5) for k in ks}, disp=rng.randint(0, 6) if disp is None else disp,
                       uses_traj=rng.random() < 0.5 if traj is None else traj)


def _traces(g, q, mc):
    """All complete valid traces from q as (cost, first_action)."""
    if g.is_terminal(q):
        return [(cost_final(q.k, mc), None)] if g.terminal_ok(q) else []
    out = []
    for a in g.actions(q):
        c = cost_action(q, a, mc)
        for rest, _ in _traces(g, g.step(q, a), mc):
            out.append((c + rest, a))
    return out


def _reachable(g):
    seen, stack = set(), [g.initial()]
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        if not g.is_terminal(q):
            stack.extend(g.step(q, a) for a in g.actions(q))
    return seen


@pytest.mark.parametrize("gcls", [G842, GA1])
@pytest.mark.parametrize("seed", range(12))
def test_L_and_mask_match_brute_force(gcls, seed):
    rng = random.Random(seed)
    g = gcls(J=5)
    mc = _rand_mc(rng)
    L = solve_L(g, mc)
    for q in _reachable(g):
        tr = _traces(g, q, mc)
        bf = min((c for c, _ in tr), default=math.inf)
        assert L_of(L, q, g, mc) == bf, q
        if g.is_terminal(q):
            continue
        for r in sorted({c for c, _ in tr} | {0}):
            for budget in (r - 1, r, r + 1):
                allowed = set(mask(q, budget, L, g, mc))
                bf_allowed = {a for c, a in tr if c <= budget}
                assert allowed == bf_allowed, (q, budget)


def test_terminal_and_grammar_rules():
    g = G842()
    mc = _rand_mc(random.Random(0), disp=5)
    L = solve_L(g, mc)
    assert L[GrammarState.make(16, 2)] == mc.fin[2]                      # no dispatch at terminal
    assert L_of(L, GrammarState.make(16, 4), g, mc) == math.inf          # (16,k>2) invalid
    assert g.actions(GrammarState(3, 8, True, True, True)) == (CONT,)   # no second prune at same j
    assert VER not in g.actions(GrammarState.make(4, 2))                 # no verify at k=2
    assert g.actions(GrammarState.make(0, 8)) == (CONT,)                 # no preview at j=0
    assert PRUNE not in g.actions(GrammarState(16, 4, True, False, True))
    assert VP in GA1().actions(GrammarState.make(1, 8))


def test_once_prepaid_before_feasibility():
    g = G842()
    mc = _rand_mc(random.Random(1), disp=0, traj=False)
    L0 = solve_L(g, mc)[g.initial()]
    assert initial_feasible(int(L0), g, mc)
    mc2 = mc.with_overrides(once=1)
    assert not initial_feasible(int(L0), g, mc2)


@pytest.mark.parametrize("seed", range(20))
def test_budget_invariant_random_traces(seed):
    """Random feasible actions from any affordable start always complete within the cap (Mệnh đề 1)."""
    rng = random.Random(seed)
    g = G842()
    mc = _rand_mc(rng)
    L = solve_L(g, mc)
    q = g.initial()
    cap = int(L[q]) + rng.randint(0, 400)
    r = cap
    while not g.is_terminal(q):
        assert r >= L_of(L, q, g, mc)
        allowed = mask(q, r, L, g, mc)
        assert allowed, "mask became empty"
        a = rng.choice(allowed)
        r -= cost_action(q, a, mc)
        q = g.step(q, a)
    assert g.terminal_ok(q)
    r -= cost_final(q.k, mc)
    assert r >= 0


def test_forced_action_still_pays_dispatch():
    g = G842()
    mc = MethodCosts(den={8: 1, 4: 1, 2: 1}, ver={8: 1, 4: 1}, fin={2: 1}, prune={(8, 4): 0, (4, 2): 0}, disp=10)
    L = solve_L(g, mc)
    q = g.initial()
    # at j=0 only CONT exists; the mask still requires CONT + disp + L(next) <= r
    need = cost_action(q, CONT, mc) + L_of(L, g.step(q, CONT), g, mc)
    assert mask(q, need, L, g, mc) == (CONT,)
    assert mask(q, need - 1, L, g, mc) == ()
