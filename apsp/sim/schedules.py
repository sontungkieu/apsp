"""Schedule enumeration and fixed-schedule outcomes (Spec Hệ quả 2a, Guide CX-2).

``schedule_table`` replays all 105 pairs on one bank instance with the fixed ranking/tie-break,
producing the winner, delivered Y and fixed-method cost C0 of each pair. Oracles and B3 read it.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional, Sequence

import numpy as np

from apsp.contracts.identity import CandidateId, bank_rows, rank_candidates
from apsp.contracts.ledger import DELTA, MethodCosts, schedule_cost
from apsp.contracts.policy_api import Cut, ScheduleSpec


def enumerate_pairs(J: int = 16) -> list[tuple[int, int]]:
    return list(combinations(range(1, J), 2))


def fixed_pair_spec(j1: int, j2: int, rows: Sequence[CandidateId], J: int = 16) -> ScheduleSpec:
    return ScheduleSpec(candidates=tuple(rows), queries=(j1, j2),
                        cuts=(Cut(j1, len(rows) // 2, "paid"), Cut(j2, len(rows) // 4, "paid")),
                        final_k=len(rows) // 4, J=J)


def bon_spec(n: int, rows: Sequence[CandidateId], J: int = 16) -> ScheduleSpec:
    """B0 (n=1, F(1): no inference RM) and B1 best-of-N with nested seeds (first n rows)."""
    return ScheduleSpec(candidates=tuple(rows[:n]), queries=(), cuts=(), final_k=n, J=J)


def required_prune_pairs() -> list[tuple[int, int]]:
    """Source->target pairs the main grammar, BoN-free baselines and S&R use; profiled by CC-4.

    Main G-842: 8->4, 4->2.  B5 screen-and-resume: M in {8,16} -> q in {1,2}.  B4b keeps are read
    from the PSP grid (see docs/QUESTIONS.md) and appended once pinned.
    """
    pairs = {(8, 4), (4, 2), (8, 1), (8, 2), (16, 1), (16, 2), (16, 8)}
    return sorted(pairs, reverse=True)


@dataclass(frozen=True)
class PairOutcome:
    j1: int
    j2: int
    winner: Optional[CandidateId]
    y: Optional[float]       # delivered IR of the winner, None if failed
    cost_us: int


def pair_outcomes(preview_ir: np.ndarray, delivered_ir: np.ndarray, bank_id: int, mc: MethodCosts,
                  delta: int = DELTA, J: int = 16) -> list[PairOutcome]:
    """Vectorised hindsight replay of the 105 fixed schedules for one (prompt, bank)."""
    rows = bank_rows(bank_id)
    out = []
    for j1, j2 in enumerate_pairs(J):
        s1 = {c: (None if not np.isfinite(preview_ir[c[1], j1 - 1]) else float(preview_ir[c[1], j1 - 1])) for c in rows}
        a4 = sorted(rank_candidates(rows, s1)[:4])
        s2 = {c: (None if not np.isfinite(preview_ir[c[1], j2 - 1]) else float(preview_ir[c[1], j2 - 1])) for c in a4}
        a2 = sorted(rank_candidates(a4, s2)[:2])
        sd = {c: (None if not np.isfinite(delivered_ir[c[1]]) else float(delivered_ir[c[1]])) for c in a2}
        w = rank_candidates(a2, sd)[0]
        y = sd[w]
        out.append(PairOutcome(j1, j2, w if y is not None else None, y,
                               schedule_cost(fixed_pair_spec(j1, j2, rows, J), mc, delta)))
    return out


def outcome_matrix(outcomes: Iterable[Sequence[PairOutcome]], r_fail: float) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-instance outcomes into Y[N,105] (failures -> r_fail) and C[105] (cost is instance-independent)."""
    outs = list(outcomes)
    Y = np.array([[o.y if o.y is not None else r_fail for o in inst] for inst in outs], dtype=np.float64)
    C = np.array([o.cost_us for o in outs[0]], dtype=np.int64)
    return Y, C
