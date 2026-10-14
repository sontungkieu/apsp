"""Shared caps B_alpha = L0 + floor(alpha (C_ref - L0)) (Spec §7.4).

L0 and C_ref come from the FIXED method primitives (no controller, no trajectory features).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import DELTA, MethodCosts, schedule_cost
from apsp.sim.budget_dp import G842, solve_L
from apsp.sim.schedules import fixed_pair_spec

ALPHAS = (0.25, 0.5, 1.0, 1.5)
PRIMARY_ALPHAS = (0.5, 1.0)


@dataclass(frozen=True)
class Caps:
    L0: int
    C_ref: int
    caps: dict  # alpha -> cap_us

    def cap(self, alpha: float) -> int:
        return self.caps[alpha]


def fixed_method_costs(mc: MethodCosts) -> MethodCosts:
    return mc.with_overrides(disp=0, once=0, uses_traj=False, method_id="fixed")


def compute_caps(mc: MethodCosts, alphas=ALPHAS, reference_cuts=(4, 8), delta: int = DELTA) -> Caps:
    fx = fixed_method_costs(mc)
    g = G842()
    L0 = solve_L(g, fx, delta)[g.initial()]
    if not math.isfinite(L0):
        raise ValueError("G-842 has no valid completion under these costs")
    C_ref = schedule_cost(fixed_pair_spec(*reference_cuts, bank_rows(0)), fx, delta)
    L0 = int(L0)
    if C_ref < L0:
        raise AssertionError("C_ref below L0: cost table inconsistent")
    return Caps(L0, C_ref, {a: L0 + math.floor(Fraction(str(a)) * (C_ref - L0)) for a in alphas})
