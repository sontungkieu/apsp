"""B8: one paid query, then commit to a fixed schedule for the rest (Spec §9.2 B8, Guide CX-3/CX-5 and §6 preflight).

At checkpoint j_c (1..8) the 8 candidates are VERIFIED once (paid ver(8)). From the prompt feature and summaries of
those purchased scores, a selector chooses one affordable future schedule (j1, j2) with j1 >= j_c, j2 > j1, which is
then executed without further decisions (8->4 at j1, 4->2 at j2, terminal RM picks the winner).

- Preflight before the query: if min over (j1 >= j_c, j2 > j1) of C_B8(j_c, j1, j2) > cap, the episode is
  infeasible and recorded BEFORE anything is paid.
- C_B8 is the full trace from j=0: once (selector/feature cost, prepaid) + den blocks + ver(8) at j_c
  + ver at j1 (only if j1 > j_c; if j1 == j_c the purchased scores are reused) + prune(8->4) + ver(4) at j2
  + prune(4->2) + fin(2). The executor checks spend == schedule_cost(full spec) + once.
- Selector: multi-output ridge, one output per candidate schedule (privileged train_fit labels = delivered Y of the
  fixed pair on the same instance), regularisation chosen by group 5-fold CV (folds by prompt group, banks of a
  prompt together) over 13 log-spaced values 1e-3..1e3. Infeasible schedules are masked before the argmax.
  Future scores are never features: only the prompt feature and the scores purchased at j_c.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from apsp.contracts.hashing import sha256_hex
from apsp.contracts.identity import CandidateId, select_top
from apsp.contracts.ledger import DELTA, MethodCosts, schedule_cost
from apsp.contracts.paid_state import EpisodeState
from apsp.contracts.policy_api import Cut, ScheduleSpec
from apsp.sim.env import EpisodeResult, TraceEvent, _finalize
from apsp.sim.features import NormStats
from apsp.sim.schedules import enumerate_pairs

JC_CHOICES = tuple(range(1, 9))
RIDGE_ALPHAS = tuple(np.logspace(-3, 3, 13).tolist())
FEAT_DIM = 16 + 8 + 4


def b8_spec(rows: Sequence[CandidateId], j_c: int, j1: int, j2: int, J: int = 16) -> ScheduleSpec:
    if not (1 <= j_c <= j1 < j2 <= J - 1):
        raise ValueError(f"invalid B8 schedule j_c={j_c} j1={j1} j2={j2}")
    k = len(rows)
    return ScheduleSpec(candidates=tuple(rows), queries=tuple(sorted({j_c, j1, j2})),
                        cuts=(Cut(j1, k // 2, "paid"), Cut(j2, k // 4, "paid")), final_k=k // 4, J=J)


def b8_cost(rows, j_c: int, j1: int, j2: int, mc: MethodCosts, delta: int = DELTA) -> int:
    return schedule_cost(b8_spec(rows, j_c, j1, j2), mc, delta)          # includes mc.once


def future_schedules(j_c: int, J: int = 16) -> list[tuple[int, int]]:
    return [(a, b) for a, b in enumerate_pairs(J) if a >= j_c]


def query_features(prompt_feat: np.ndarray, scores: dict, j: int, norm: NormStats) -> np.ndarray:
    """Prompt PCA-16 + 8 sorted z-scores (invalid -> -10) + [mean z, std z, z1-z2, z4-z5] (clipped to [-10, 10])."""
    z = sorted(((norm.z(v, j) if v is not None else -10.0) for v in scores.values()), reverse=True)
    z = np.clip(np.asarray(z + [-10.0] * (8 - len(z)), np.float64)[:8], -10, 10)
    valid = z[z > -10]
    summ = [valid.mean() if valid.size else 0.0, valid.std() if valid.size else 0.0, z[0] - z[1], z[3] - z[4]]
    pf = np.zeros(16) if prompt_feat is None else np.asarray(prompt_feat, np.float64).ravel()[:16]
    return np.clip(np.concatenate([pf, z, np.asarray(summ)]), -10, 10).astype(np.float64)


@dataclass
class RidgeSelector:
    j_c: int
    schedules: list            # [(j1, j2)] candidate future schedules, in output order
    W: np.ndarray              # [FEAT_DIM + 1, n_sched]
    alpha: float
    feat_mu: np.ndarray
    feat_sd: np.ndarray
    cv: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.feat_mu) / self.feat_sd
        return np.concatenate([xs, [1.0]]) @ self.W

    def choose(self, x: np.ndarray, affordable: Sequence[bool], costs: Sequence[int]) -> Optional[tuple[int, int]]:
        p = self.predict(x)
        cand = [i for i, ok in enumerate(affordable) if ok]
        if not cand:
            return None
        best = sorted(cand, key=lambda i: (-p[i], costs[i], self.schedules[i]))[0]
        return tuple(self.schedules[best])

    def to_dict(self) -> dict:
        return {"j_c": self.j_c, "schedules": [list(s) for s in self.schedules], "W": self.W.tolist(), "alpha": self.alpha,
                "feat_mu": self.feat_mu.tolist(), "feat_sd": self.feat_sd.tolist(), "cv": self.cv}

    @staticmethod
    def from_dict(d: dict) -> "RidgeSelector":
        return RidgeSelector(int(d["j_c"]), [tuple(s) for s in d["schedules"]], np.asarray(d["W"]), float(d["alpha"]),
                             np.asarray(d["feat_mu"]), np.asarray(d["feat_sd"]), d.get("cv", {}))

    @property
    def cph(self) -> str:
        return "B8-" + sha256_hex(self.to_dict())[:16]


def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    reg = alpha * np.eye(Xa.shape[1])
    reg[-1, -1] = 0.0                                  # intercept not penalised
    return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Y)


def fit_selector(X: np.ndarray, Y: np.ndarray, groups: Sequence[str], j_c: int, schedules: list,
                 n_folds: int = 5, alphas=RIDGE_ALPHAS) -> RidgeSelector:
    """X [N, FEAT_DIM] features at j_c; Y [N, n_sched] delivered Y of each future schedule (train_fit labels)."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    Xs = (X - mu) / sd
    ug = sorted(set(groups))
    fold_of = {g: i % n_folds for i, g in enumerate(sorted(ug, key=lambda g: sha256_hex(g)))}
    folds = np.array([fold_of[g] for g in groups])
    cv = {}
    for a in alphas:
        err = []
        for f in range(min(n_folds, len(ug))):
            tr, te = folds != f, folds == f
            if not te.any() or not tr.any():
                continue
            W = _ridge(Xs[tr], Y[tr], a)
            P = np.concatenate([Xs[te], np.ones((te.sum(), 1))], 1) @ W
            err.append(float(((P - Y[te]) ** 2).mean()))
        cv[f"{a:.6g}"] = float(np.mean(err)) if err else float("inf")
    best = min(alphas, key=lambda a: (cv[f"{a:.6g}"], -a))
    return RidgeSelector(j_c, list(schedules), _ridge(Xs, Y, best), float(best), mu, sd,
                         {"mse_by_alpha": cv, "n_folds": n_folds, "n_train": int(len(X)), "folds_by": "prompt group"})


def run_query_commit(selector: RidgeSelector, engine, *, rows: Sequence[CandidateId], mc: MethodCosts, cap_us: int,
                     norm: NormStats, prompt_feat: Optional[np.ndarray] = None, C_ref: int = 1, delta: int = DELTA,
                     J: int = 16, method_id: Optional[str] = None,
                     clock: Callable[[], float] = time.perf_counter) -> EpisodeResult:
    j_c = selector.j_c
    sched = selector.schedules
    costs = [b8_cost(rows, j_c, a, b, mc, delta) for a, b in sched]
    res = EpisodeResult(method_id=method_id or f"B8(jc={j_c})", cap_us=cap_us, feasible=False)
    if min(costs) > cap_us:                       # preflight: infeasible BEFORE the query is paid
        return res
    t0 = clock()
    state = EpisodeState(grammar_id="SCHEDULE", rows=tuple(rows), cap_us=cap_us, L_fn=lambda q: 0.0, C_ref=C_ref, J=J)
    state.charge(mc.once, "once")                 # selector / query-feature work is prepaid here
    engine.start(tuple(rows))
    plan: Optional[ScheduleSpec] = None
    cuts, queries, prune_js = {}, {j_c}, []
    for j in range(J):
        fee = delta * mc.require("den", state.k)
        res.trace.append(TraceEvent(j, state.k, "CONTINUE", fee, state.remaining_us, (), state.alive))
        state.charge(fee, "CONTINUE")
        engine.advance(state, delta, False)
        state.arrive(j + 1)
        jn = j + 1
        if jn >= J:
            break
        if jn in queries and not (plan is not None and jn == j_c):
            fee = mc.require("ver", state.k)
            res.trace.append(TraceEvent(jn, state.k, "VERIFY", fee, state.remaining_us, (), state.alive))
            state.charge(fee, "VERIFY")
            state.record_verify(jn, engine.verify(state))
        if jn == j_c:                             # commit: choose the rest of the schedule from purchased info only
            x = query_features(prompt_feat, state.fresh_scores(), j_c, norm)
            choice = selector.choose(x, [c <= cap_us for c in costs], costs)
            j1, j2 = choice
            plan = b8_spec(rows, j_c, j1, j2, J)
            cuts = {c.j: c for c in plan.cuts}
            queries = set(plan.queries)
            res.trace.append(TraceEvent(jn, state.k, f"COMMIT({j1},{j2})", 0, state.remaining_us, (), state.alive))
        if jn in cuts:
            c = cuts[jn]
            fee = mc.require("prune", (state.k, c.keep))
            res.trace.append(TraceEvent(jn, state.k, f"PRUNE->{c.keep}", fee, state.remaining_us, (), state.alive))
            state.charge(fee, "PRUNE")
            keep = select_top(state.alive, state.fresh_scores(), c.keep)
            idx = state.record_prune(keep, n_keep=c.keep)
            engine.prune(state, idx)
            prune_js.append(jn)
    res.delivery = _finalize(state, engine, mc)
    res.wall_clock_s = clock() - t0
    res.delivery.wall_clock_s = res.wall_clock_s
    planned = schedule_cost(plan, mc, delta)
    if state.spent_us != planned:
        raise RuntimeError(f"B8 spend {state.spent_us} != planned full-trace cost {planned}")
    res.feasible, res.spend_us, res.n_verify, res.prune_js = True, state.spent_us, state.n_verify, tuple(prune_js)
    return res


# ------------------------------------------------------------------ training on replay ------------------------------
def train_selector(ctx, j_c: int, keys: Sequence, rows_of=lambda key: None) -> RidgeSelector:
    """Fit on train_fit instances of a ReplayContext: features from the preview scores at j_c (the information the
    runtime would have bought), labels = delivered Y of every future fixed schedule (privileged training labels)."""
    from apsp.contracts.identity import bank_rows
    sched = future_schedules(j_c)
    all_pairs = enumerate_pairs()
    cols = [all_pairs.index(s) for s in sched]
    Yall, _ = ctx.schedule_table(keys)
    X, groups = [], []
    for key in keys:
        uid, b = key
        arr = ctx.accessor.instance_arrays(key, "training_label")
        pv = arr["preview_ir"]
        scores = {c: (None if not np.isfinite(pv[c[1], j_c - 1]) else float(pv[c[1], j_c - 1])) for c in bank_rows(b)}
        X.append(query_features(ctx.prompt_feat(uid), scores, j_c, ctx.norm))
        groups.append(ctx.bank.prompts.set_index("prompt_uid").loc[uid, "group_id"])
    return fit_selector(np.stack(X), Yall[:, cols], groups, j_c, sched)


def replay_mean_b8(ctx, selector: RidgeSelector, keys: Sequence, cap_us: int, mc: Optional[MethodCosts] = None) -> float:
    from apsp.contracts.identity import bank_rows
    from apsp.sim.caps import fixed_method_costs
    from apsp.sim.env import ReplayEngine, episode_reward
    mc = mc or fixed_method_costs(ctx.base_costs)
    ys = []
    for uid, b in keys:
        r = run_query_commit(selector, ReplayEngine(ctx.accessor, uid), rows=bank_rows(b), mc=mc, cap_us=cap_us,
                             norm=ctx.norm, prompt_feat=ctx.prompt_feat(uid), C_ref=ctx.caps.C_ref)
        ys.append(episode_reward(r, ctx.accessor, uid, ctx.r_fail) if r.feasible else ctx.r_fail)
    return float(np.mean(ys)) if ys else float("nan")


def tune_b8(ctx, fit_keys: Sequence, dev_keys: Sequence, cap_us: int, mc: Optional[MethodCosts] = None) -> Optional[dict]:
    """j_c in 1..8: selector fitted on train_fit for each j_c; j_c chosen by mean Y on train_dev (per cap)."""
    from apsp.contracts.identity import bank_rows
    from apsp.sim.caps import fixed_method_costs
    mc = mc or fixed_method_costs(ctx.base_costs)
    rows = bank_rows(0)
    out = []
    for j_c in JC_CHOICES:
        if min(b8_cost(rows, j_c, a, b, mc) for a, b in future_schedules(j_c)) > cap_us:
            continue
        sel = train_selector(ctx, j_c, fit_keys)
        out.append((replay_mean_b8(ctx, sel, dev_keys, cap_us, mc), j_c, sel))
    if not out:
        return None
    out.sort(key=lambda t: (-t[0], t[1]))
    dev, j_c, sel = out[0]
    return {"j_c": j_c, "dev_mean": dev, "selector": sel.to_dict(), "cph": sel.cph,
            "dev_by_jc": {str(j): d for d, j, _ in out}}
