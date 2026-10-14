"""Fixed schedules: B0/B1 (BoN), B2 PSP-default (4,8), B3 best-fixed among the 105 pairs.

``FixedPairPolicy`` is the closed-loop rendering of a fixed pair inside G-842 (used to check that the
closed-loop runner and the schedule runner agree). Scheduled execution uses ``run_schedule``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from apsp.contracts.ledger import DELTA, MethodCosts, schedule_cost
from apsp.contracts.policy_api import CONT, PRUNE, VER
from apsp.contracts.identity import bank_rows
from apsp.sim.features import IDX
from apsp.sim.schedules import bon_spec, enumerate_pairs

REFERENCE_CUTS = (4, 8)


def obs_jk(obs: np.ndarray, J: int = 16) -> tuple[int, int, bool, bool]:
    j = int(round(float(obs[IDX["j"]]) * J))
    k = 8 if obs[IDX["k8"]] > 0.5 else 4 if obs[IDX["k4"]] > 0.5 else 2
    return j, k, bool(obs[IDX["fresh"]] > 0.5), bool(obs[IDX["pruned"]] > 0.5)


class FixedPairPolicy:
    grammar_id = "G-842-v1"
    uses_traj = False

    def __init__(self, j1: int, j2: int, method_id: Optional[str] = None):
        self.j1, self.j2 = j1, j2
        self.method_id = method_id or f"fixed({j1},{j2})"
        self.cph = f"fixed-{j1}-{j2}"

    def reset(self, prompt_feat, cap_us):
        pass

    def act(self, obs, mask):
        # no forced-action shortcut: a fixed pair must never silently deviate from (j1, j2)
        j, k, fresh, pruned = obs_jk(obs)
        want = CONT
        if (j == self.j1 and k == 8) or (j == self.j2 and k == 4):
            want = VER if not fresh else (PRUNE if not pruned else CONT)
        if want not in mask:
            raise RuntimeError(f"fixed pair ({self.j1},{self.j2}) infeasible under this cap at j={j} (mask={mask})")
        return want


@dataclass(frozen=True)
class BestFixed:
    j1: int
    j2: int
    mean_y: float
    cost_us: int
    n_affordable: int


def best_fixed(Y: np.ndarray, C: np.ndarray, cap_us: int, J: int = 16) -> Optional[BestFixed]:
    """B3: argmax mean Y over affordable pairs on the given instances (train split).

    Tie-break: lower cost, then lexicographic (j1, j2). Never selects per prompt.
    """
    pairs = enumerate_pairs(J)
    ok = np.where(C <= cap_us)[0]
    if ok.size == 0:
        return None
    means = Y[:, ok].mean(axis=0)
    order = sorted(range(ok.size), key=lambda i: (-means[i], int(C[ok[i]]), pairs[ok[i]]))
    i = ok[order[0]]
    return BestFixed(pairs[i][0], pairs[i][1], float(means[order[0]]), int(C[i]), int(ok.size))


def select_b3_dev(Y_fit: np.ndarray, Y_dev: np.ndarray, C: np.ndarray, cap_us: int, top: int = 5, J: int = 16) -> Optional[BestFixed]:
    """Tune on train_fit (top-``top`` pairs), choose on train_dev (Spec §9.2 'Tune train, chọn dev')."""
    pairs = enumerate_pairs(J)
    ok = np.where(C <= cap_us)[0]
    if ok.size == 0:
        return None
    fit_means = Y_fit[:, ok].mean(axis=0)
    short = sorted(range(ok.size), key=lambda i: (-fit_means[i], int(C[ok[i]]), pairs[ok[i]]))[:top]
    dev_means = {i: Y_dev[:, ok[i]].mean() for i in short}
    best = sorted(short, key=lambda i: (-dev_means[i], int(C[ok[i]]), pairs[ok[i]]))[0]
    return BestFixed(pairs[ok[best]][0], pairs[ok[best]][1], float(dev_means[best]), int(C[ok[best]]), int(ok.size))


def bon_feasible_n(mc: MethodCosts, cap_us: int, max_n: int = 8, delta: int = DELTA) -> Optional[int]:
    """B1: largest N <= max_n whose best-of-N trace fits the cap (nested seeds)."""
    best = None
    for n in range(1, max_n + 1):
        if n not in mc.den or n not in mc.fin:
            continue
        if schedule_cost(bon_spec(n, bank_rows(0, max(max_n, 8))), mc, delta) <= cap_us:
            best = n
    return best
