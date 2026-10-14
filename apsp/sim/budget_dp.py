"""Grammars, minimum completion cost L^m(q) and the budget-guaranteeing action mask (Spec §7.3).

Ported/generalised from the legacy reference (``reference/budget_dp.py`` is kept read-only if present).
Mapping fixes of v1.1.1: CONT = delta*den[k] + uses_traj*feat[k] + disp; prune keyed by (k, k');
once prepaid before any feasibility check; terminal fin[k] without dispatch.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

from apsp.contracts.ledger import DELTA, MethodCosts, cost_action, cost_final
from apsp.contracts.policy_api import CONT, PRUNE, VER, VP, GrammarState

INF = math.inf


class Grammar:
    grammar_id: str = ""
    J: int = 16
    k0: int = 8
    terminal_k: int = 2

    def __init__(self, J: int = 16):
        # J != 16 only for brute-force toy tests; the scientific grammar is J=16 (Spec §5)
        self.J = int(J)

    def initial(self) -> GrammarState:
        return GrammarState.make(0, self.k0)

    def actions(self, q: GrammarState) -> tuple[int, ...]:
        raise NotImplementedError

    def step(self, q: GrammarState, a: int) -> GrammarState:
        raise NotImplementedError

    def is_terminal(self, q: GrammarState) -> bool:
        return q.j >= self.J

    def terminal_ok(self, q: GrammarState) -> bool:
        return q.j == self.J and q.k == self.terminal_k


class G842(Grammar):
    """``G-842-v1``: k0=8, exactly two prunes 8->4->2 at distinct checkpoints 1<=j1<j2<=15."""

    grammar_id = "G-842-v1"

    def actions(self, q: GrammarState) -> tuple[int, ...]:
        if q.j >= self.J:
            return ()
        acts = [CONT]
        inner = 1 <= q.j <= self.J - 1 and q.k > self.terminal_k
        if inner and q.preview and not q.fresh:
            acts.append(VER)
        if inner and q.fresh and not q.pruned:
            acts.append(PRUNE)
        return tuple(acts)

    def step(self, q: GrammarState, a: int) -> GrammarState:
        if a not in self.actions(q):
            raise ValueError(f"action {a} not structurally allowed at {q}")
        if a == CONT:
            return GrammarState.make(q.j + 1, q.k)
        if a == VER:
            return GrammarState(q.j, q.k, True, q.pruned, q.preview)
        if a == PRUNE:
            # survivors' scores stay fresh; a second prune at this checkpoint is locked
            return GrammarState(q.j, q.k // 2, True, True, q.preview)
        raise ValueError(a)


class GA1(Grammar):
    """``G-A1-v1`` (ablation A1): CONT and atomic VERIFY+PRUNE (one dispatch + ver + prune)."""

    grammar_id = "G-A1-v1"

    def actions(self, q: GrammarState) -> tuple[int, ...]:
        if q.j >= self.J:
            return ()
        acts = [CONT]
        if 1 <= q.j <= self.J - 1 and q.k > self.terminal_k and q.preview and not q.pruned:
            acts.append(VP)
        return tuple(acts)

    def step(self, q: GrammarState, a: int) -> GrammarState:
        if a not in self.actions(q):
            raise ValueError(f"action {a} not structurally allowed at {q}")
        if a == CONT:
            return GrammarState.make(q.j + 1, q.k)
        return GrammarState(q.j, q.k // 2, True, True, q.preview)


GRAMMARS: dict[str, Callable[[], Grammar]] = {"G-842-v1": G842, "G-A1-v1": GA1}


def get_grammar(grammar_id: str) -> Grammar:
    try:
        return GRAMMARS[grammar_id]()
    except KeyError as e:
        raise ValueError(f"unknown grammar {grammar_id}") from e


_L_CACHE: dict[tuple, dict] = {}


def solve_L(grammar: Grammar, mc: MethodCosts, delta: int = DELTA) -> Mapping[GrammarState, float]:
    """Minimum completion cost from every reachable state (+inf when no valid completion).

    Cache key includes grammar id, the full cost-table hash and delta, so L is never reused
    after ver/disp/feat change.
    """
    key = (grammar.grammar_id, grammar.J, mc.cost_hash(), delta)
    if key in _L_CACHE:
        return _L_CACHE[key]
    memo: dict[GrammarState, float] = {}

    def L(q: GrammarState) -> float:
        if q in memo:
            return memo[q]
        if grammar.is_terminal(q):
            v = float(cost_final(q.k, mc)) if grammar.terminal_ok(q) else INF
        else:
            v = INF
            for a in grammar.actions(q):
                v = min(v, cost_action(q, a, mc, delta) + L(grammar.step(q, a)))
        memo[q] = v
        return v

    L(grammar.initial())
    _L_CACHE[key] = memo
    return memo


def L_of(L: Mapping[GrammarState, float], q: GrammarState, grammar: Grammar, mc: MethodCosts, delta: int = DELTA) -> float:
    """Lookup that also solves states not reached from the initial state (e.g. tests)."""
    if q in L:
        return L[q]
    if grammar.is_terminal(q):
        return float(cost_final(q.k, mc)) if grammar.terminal_ok(q) else INF
    return min((cost_action(q, a, mc, delta) + L_of(L, grammar.step(q, a), grammar, mc, delta)
                for a in grammar.actions(q)), default=INF)


def mask(q: GrammarState, remaining_us: int, L: Mapping[GrammarState, float], grammar: Grammar,
         mc: MethodCosts, delta: int = DELTA) -> tuple[int, ...]:
    """Actions whose own cost (incl. this dispatch) plus min completion fits in ``remaining_us``.

    ``remaining_us`` is the budget BEFORE the current dispatch is charged (Guide §6.2).
    """
    out = []
    for a in grammar.actions(q):
        nxt = grammar.step(q, a)
        if cost_action(q, a, mc, delta) + L_of(L, nxt, grammar, mc, delta) <= remaining_us:
            out.append(a)
    return tuple(out)


def initial_feasible(cap_us: int, grammar: Grammar, mc: MethodCosts, delta: int = DELTA) -> bool:
    L = solve_L(grammar, mc, delta)
    return cap_us - mc.once >= L[grammar.initial()]
