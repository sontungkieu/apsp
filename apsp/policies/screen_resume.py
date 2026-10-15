"""B5 screen-and-resume (Spec §9.2, Guide CX-5): M in {8,16} candidates, ONE paid screen (VERIFY) at j_s, keep q in
{1,2} and resume to the end; full cost = prefix den(M) + ver(M) + prune(M->q) + suffix den(q) + fin(q).

M=8 uses the 8 seeds of one bank; M=16 uses pool 16 ((0,0)..(0,7),(1,0)..(1,7)) and is one episode per prompt.
This is this project's own same-bank breadth baseline. It is NOT Flash-BoN and must not be called Flash-BoN.
Tuning: affordable configurations ranked on train_fit (top-k) and selected on train_dev, per cap (as B3).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from apsp.contracts.identity import CandidateId, bank_rows, pool_rows
from apsp.contracts.ledger import DELTA, MethodCosts, schedule_cost
from apsp.contracts.policy_api import Cut, ScheduleSpec

M_CHOICES = (8, 16)
Q_CHOICES = (1, 2)


@dataclass(frozen=True)
class SRConfig:
    M: int
    j_s: int
    q: int

    @property
    def name(self) -> str:
        return f"sr:{self.M},{self.j_s},{self.q}"


def sr_rows(M: int, bank_id: int = 0) -> tuple[CandidateId, ...]:
    if M == 8:
        return tuple(bank_rows(bank_id))
    if M == 16:
        return tuple(pool_rows(2))
    raise ValueError(f"M must be 8 or 16, got {M}")


def sr_spec(c: SRConfig, bank_id: int = 0, J: int = 16) -> ScheduleSpec:
    if c.M not in M_CHOICES or c.q not in Q_CHOICES or not 1 <= c.j_s <= J - 1:
        raise ValueError(f"invalid S&R configuration {c}")
    return ScheduleSpec(candidates=sr_rows(c.M, bank_id), queries=(c.j_s,), cuts=(Cut(c.j_s, c.q, "paid"),),
                        final_k=c.q, J=J)


def enumerate_sr(J: int = 16) -> list[SRConfig]:
    return [SRConfig(M, j, q) for M in M_CHOICES for j in range(1, J) for q in Q_CHOICES]


def sr_cost(c: SRConfig, mc: MethodCosts, delta: int = DELTA) -> int:
    return schedule_cost(sr_spec(c), mc, delta)


def select_sr_dev(evaluate: Callable[[SRConfig, str], float], mc: MethodCosts, cap_us: int, *, top: int = 5,
                  fit_split: str = "train_fit", dev_split: str = "train_dev", delta: int = DELTA) -> Optional[dict]:
    """``evaluate(cfg, split)`` = mean delivered Y (failures at r_fail) on that split. Returns the dev choice or None."""
    ok = [c for c in enumerate_sr() if sr_cost(c, mc, delta) <= cap_us]
    if not ok:
        return None
    fit = {c: evaluate(c, fit_split) for c in ok}
    short = sorted(ok, key=lambda c: (-fit[c], sr_cost(c, mc, delta), (c.M, c.j_s, c.q)))[:top]
    dev = {c: evaluate(c, dev_split) for c in short}
    best = sorted(short, key=lambda c: (-dev[c], sr_cost(c, mc, delta), (c.M, c.j_s, c.q)))[0]
    return {"config": asdict(best), "name": best.name, "fit_mean": fit[best], "dev_mean": dev[best],
            "cost_us": sr_cost(best, mc, delta), "n_affordable": len(ok), "shortlist": [asdict(c) for c in short]}


def sr_instance_keys(keys: Sequence, M: int) -> list:
    """M=16 episodes are per prompt (both banks): keep one key per prompt (bank 0)."""
    if M == 8:
        return list(keys)
    seen, out = set(), []
    for uid, _ in keys:
        if uid not in seen:
            seen.add(uid)
            out.append((uid, 0))
    return out


def replay_mean(ctx, c: SRConfig, keys: Sequence, cap_us: int) -> float:
    ys = []
    for key in sr_instance_keys(keys, c.M):
        r = ctx.run_spec(sr_spec(c, key[1]), key, cap_us, method_id=c.name)
        ys.append(r.Y if r.feasible else ctx.r_fail)
    return float(np.mean(ys)) if ys else float("nan")
