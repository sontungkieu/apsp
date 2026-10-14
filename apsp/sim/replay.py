"""Replay context: bundles the privileged accessor, costs, caps, normalisation and prompt features.

Policies only ever see observations; this module is env/training-side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from apsp.contracts.access import PrivilegedAccessor
from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import DELTA, MethodCosts
from apsp.contracts.policy_api import Policy, ScheduleSpec
from apsp.contracts.schema import BankArrays
from apsp.sim.budget_dp import Grammar, G842
from apsp.sim.caps import Caps, compute_caps, fixed_method_costs
from apsp.sim.env import EpisodeResult, ReplayEngine, episode_reward, run_closed_loop, run_schedule
from apsp.sim.features import PCA16, NormStats
from apsp.sim.schedules import outcome_matrix, pair_outcomes


@dataclass
class ReplayContext:
    bank: BankArrays
    accessor: PrivilegedAccessor
    base_costs: MethodCosts           # generator primitives (fixed method)
    caps: Caps
    norm: NormStats
    pca: Optional[PCA16]
    r_fail: float
    delta: int = DELTA
    _prompt_feat: dict = field(default_factory=dict)
    _tables: dict = field(default_factory=dict)

    @staticmethod
    def build(bank: BankArrays, base_costs: MethodCosts, *, fit_split: str, r_fail: Optional[float] = None,
              keep_log: bool = False, guard: bool = True) -> "ReplayContext":
        acc = PrivilegedAccessor(bank, keep_log=keep_log, guard=guard)
        fit = bank.instances_of([fit_split])
        norm = NormStats.fit([acc.instance_arrays(k, "training_label")["preview_ir"] for k in fit])
        uids = sorted({k[0] for k in fit})
        pca = PCA16.fit(np.stack([acc.prompt_embedding(u) for u in uids])) if len(uids) >= 16 else None
        if r_fail is None:
            r_fail = failure_penalty([acc.instance_arrays(k, "training_label")["delivered_ir"] for k in fit])
        return ReplayContext(bank, acc, base_costs, compute_caps(base_costs), norm, pca, float(r_fail))

    def prompt_feat(self, uid: str) -> np.ndarray:
        if uid not in self._prompt_feat:
            if self.pca is None:
                self._prompt_feat[uid] = np.zeros(16, np.float32)
            else:
                self._prompt_feat[uid] = self.pca.transform(self.accessor.prompt_embedding(uid))
        return self._prompt_feat[uid]

    def schedule_table(self, keys: Sequence) -> tuple[np.ndarray, np.ndarray]:
        """Y[N,105], C[105] of fixed schedules (privileged, oracle/training-label purpose)."""
        fx = fixed_method_costs(self.base_costs)
        outs = []
        for k in keys:
            if k not in self._tables:
                arr = self.accessor.instance_arrays(k, "oracle")
                self._tables[k] = pair_outcomes(arr["preview_ir"], arr["delivered_ir"], k[1], fx, self.delta)
            outs.append(self._tables[k])
        return outcome_matrix(outs, self.r_fail)

    def run_policy(self, policy: Policy, key, cap_us: int, mc: MethodCosts, grammar: Optional[Grammar] = None,
                   variant: str = "main", record_obs: bool = False) -> EpisodeResult:
        uid, b = key
        res = run_closed_loop(policy, ReplayEngine(self.accessor, uid), rows=bank_rows(b), grammar=grammar or G842(),
                              mc=mc, cap_us=cap_us, C_ref=self.caps.C_ref, norm=self.norm, variant=variant,
                              prompt_feat=self.prompt_feat(uid), delta=self.delta, record_obs=record_obs)
        res.key = key
        res.Y = episode_reward(res, self.accessor, uid, self.r_fail) if res.feasible else None
        return res

    def run_spec(self, spec: ScheduleSpec, key, cap_us: int, mc: Optional[MethodCosts] = None,
                 method_id: str = "schedule") -> EpisodeResult:
        uid, _ = key
        res = run_schedule(spec, ReplayEngine(self.accessor, uid), mc=mc or fixed_method_costs(self.base_costs),
                           cap_us=cap_us, C_ref=self.caps.C_ref, method_id=method_id, delta=self.delta)
        res.key = key
        res.Y = episode_reward(res, self.accessor, uid, self.r_fail) if res.feasible else None
        return res


def failure_penalty(delivered: Iterable[np.ndarray]) -> float:
    """r_fail = Q_0.001(valid delivered IR | train_fit) - SD(...) (Spec §12). STOP-ENG if no valid data."""
    v = np.concatenate([np.asarray(d, np.float64).ravel() for d in delivered])
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise RuntimeError("STOP-ENG: not enough valid delivered train_fit scores to compute r_fail")
    return float(np.quantile(v, 0.001) - v.std(ddof=0))


def mean_y(results: Sequence[EpisodeResult]) -> float:
    ys = [r.Y for r in results if r.feasible]
    return float(np.mean(ys)) if ys else float("nan")
