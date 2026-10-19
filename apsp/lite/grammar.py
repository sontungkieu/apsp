"""Versioned G-842-optional-lite1, variable checkpoint segments and completion DP.

Independent of the frozen v1 grammar. Microsecond fees are supplied by profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

GRAMMAR = "G-842-optional-lite1"


@dataclass(frozen=True)
class State:
    j: int = 0
    k: int = 8
    fresh: bool = False
    pruned_here: bool = False


class Grammar:
    def __init__(self, checkpoint_steps, den, ver, fin, prune, setup=0, dispatch=0):
        self.ends = (0, *checkpoint_steps)
        if len(set(self.ends)) != len(self.ends) or list(self.ends) != sorted(self.ends):
            raise ValueError("checkpoint steps must increase from zero")
        self.J = len(checkpoint_steps)
        self.den, self.ver, self.fin, self.prune = den, ver, fin, prune
        self.setup, self.dispatch = setup, dispatch
        fees = [setup, dispatch] + [v for tab in (den, ver, fin, prune) for v in tab.values()]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in fees):
            raise ValueError("fees must be finite nonnegative integer microseconds")
        for k in (2, 4, 8):
            for tab in (den, ver, fin):
                if k not in tab:
                    raise ValueError(f"missing primitive for k={k}")
        if any(pair not in prune for pair in ((8, 4), (4, 2))):
            raise ValueError("missing prune cost")

    def transitions(self, q):
        if q.k not in (2, 4, 8) or not 0 <= q.j <= self.J:
            raise ValueError("invalid state")
        if q.j == self.J:
            return {}
        result = {"CONTINUE": (State(q.j + 1, q.k),
            (self.ends[q.j + 1] - self.ends[q.j]) * self.den[q.k] + self.dispatch)}
        if q.j > 0 and not q.fresh:
            result["VERIFY"] = (State(q.j, q.k, True, q.pruned_here), self.ver[q.k] + self.dispatch)
        if q.j > 0 and q.fresh and q.k > 2 and not q.pruned_here:
            result["PRUNE"] = (State(q.j, q.k // 2, False, True), self.prune[(q.k, q.k // 2)] + self.dispatch)
        return result

    @lru_cache(maxsize=None)
    def lower_bound(self, q=State()):
        if q.j == self.J:
            return self.fin[q.k]
        return min(fee + self.lower_bound(nxt) for nxt, fee in self.transitions(q).values())

    def mask(self, q, remaining):
        return {action: fee + self.lower_bound(nxt) <= remaining
                for action, (nxt, fee) in self.transitions(q).items()}

    def feasible(self, cap):
        return cap >= self.setup + self.lower_bound()


def fixed_schedules(J):
    return [()] + [(j,) for j in range(1, J)] + list(combinations(range(1, J), 2))
