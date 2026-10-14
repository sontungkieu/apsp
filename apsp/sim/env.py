"""Shared episode runners and the replay engine (Guide §6.2, §7 CC-7, §8 CX-3).

``run_closed_loop`` and ``run_schedule`` are THE runners for both replay and online. They own the
ledger: an engine only executes work (forwards, decode/score, gather) and never charges fees.
Replay passes a ``ReplayEngine`` (privileged bank reads); online passes ``apsp.gpu.engine.DenoiseEngine``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from apsp.contracts.access import PrivilegedAccessor
from apsp.contracts.identity import CandidateId, rank_candidates, select_top
from apsp.contracts.ledger import DELTA, MethodCosts, cost_action, cost_final, schedule_cost
from apsp.contracts.paid_state import EpisodeState, LedgerError
from apsp.contracts.policy_api import ACTION_NAMES, CONT, PRUNE, VER, VP, Delivery, Policy, ScheduleSpec, ScoreRecord
from apsp.sim.budget_dp import Grammar, L_of, mask, solve_L
from apsp.sim.features import NormStats, build_obs


class PolicyError(RuntimeError):
    pass


class Engine(Protocol):
    """Execution backend. Methods are called by the runners only."""

    def start(self, rows: Sequence[CandidateId]) -> None: ...

    def advance(self, state: EpisodeState, n_forwards: int, want_traj: bool) -> Optional[dict]: ...

    def verify(self, state: EpisodeState) -> dict: ...

    def prune(self, state: EpisodeState, retained: tuple[int, ...]) -> None: ...

    def finalize_scores(self, state: EpisodeState) -> dict: ...

    def finalize_validity(self, state: EpisodeState) -> dict: ...


@dataclass
class TraceEvent:
    j: int
    k: int
    action: str
    fee_us: int
    remaining_before_us: int
    mask: tuple[int, ...] = ()
    alive: tuple = ()


@dataclass
class EpisodeResult:
    method_id: str
    cap_us: int
    feasible: bool
    delivery: Optional[Delivery] = None
    spend_us: int = 0
    trace: list = field(default_factory=list)
    n_verify: int = 0
    prune_js: tuple = ()
    obs_log: list = field(default_factory=list)
    wall_clock_s: Optional[float] = None
    key: object = None
    Y: Optional[float] = None

    def schedule_signature(self) -> tuple:
        return tuple(self.prune_js)


def _finalize(state: EpisodeState, engine: Engine, mc: MethodCosts) -> Delivery:
    k = state.k
    state.charge(cost_final(k, mc), "FINAL")
    if k == 1:
        (cid,) = state.alive
        lat_ok, img_ok = engine.finalize_validity(state)[cid]
        return Delivery(winner=None if not img_ok else cid, failed=not img_ok, latent_ok=lat_ok, img_ok=img_ok,
                        rm_used=False, inference_score=None, ledger_spend_us=state.spent_us)
    recs: dict = engine.finalize_scores(state)
    scores = {c: r.score for c, r in recs.items()}
    order = rank_candidates(state.alive, scores)
    best = order[0]
    r = recs[best]
    failed = not r.score_ok
    return Delivery(winner=None if failed else best, failed=failed, latent_ok=r.latent_ok, img_ok=r.img_ok,
                    rm_used=True, inference_score=r.score, ledger_spend_us=state.spent_us,
                    delivered_scores=scores)


def run_closed_loop(policy: Policy, engine: Engine, *, rows: Sequence[CandidateId], grammar: Grammar,
                    mc: MethodCosts, cap_us: int, C_ref: int, norm: NormStats, variant: str = "main",
                    prompt_feat: Optional[np.ndarray] = None, delta: int = DELTA, record_obs: bool = False,
                    clock: Callable[[], float] = time.perf_counter) -> EpisodeResult:
    L = solve_L(grammar, mc, delta)
    L_fn = lambda q: L_of(L, q, grammar, mc, delta)  # noqa: E731
    res = EpisodeResult(method_id=getattr(policy, "method_id", "policy"), cap_us=cap_us, feasible=False)
    if cap_us - mc.once < L_fn(grammar.initial()):
        return res  # infeasible: recorded before any generation or charge
    t0 = clock()
    state = EpisodeState(grammar_id=grammar.grammar_id, rows=rows, cap_us=cap_us, L_fn=L_fn, C_ref=C_ref,
                         prompt_feat=prompt_feat, J=grammar.J)
    state.charge(mc.once, "once")
    policy.reset(prompt_feat if prompt_feat is not None else np.zeros(16, np.float32), cap_us)
    engine.start(rows)
    prune_js = []
    while not grammar.is_terminal(state.q):
        q = state.q
        view = state.view()                                           # 1. budget before this dispatch
        allowed = mask(q, state.remaining_us, L, grammar, mc, delta)  # 2. mask incl. this dispatch fee
        if not allowed:
            raise LedgerError(f"empty mask at {q} remaining={state.remaining_us}: invariant r>=L(q) broken")
        obs = build_obs(view, norm, variant)                          # 3. observation, also for forced actions
        a = int(policy.act(obs, allowed))
        if a not in allowed:                                          # 4. validate
            raise PolicyError(f"policy {res.method_id} chose {a} outside mask {allowed} at {q}")
        if record_obs:
            res.obs_log.append(obs)
        fee = cost_action(q, a, mc, delta)
        res.trace.append(TraceEvent(q.j, q.k, ACTION_NAMES[a], fee, state.remaining_us, allowed, state.alive))
        state.charge(fee, ACTION_NAMES[a])                            # 5. charge exactly once
        if a == CONT:                                                 # 6. execute
            feats = engine.advance(state, delta, mc.uses_traj)
            state.arrive(q.j + 1)
            if mc.uses_traj:
                state.record_traj(feats)
        elif a == VER:
            state.record_verify(q.j, engine.verify(state))
        elif a in (PRUNE, VP):
            if a == VP:
                state.record_verify(q.j, engine.verify(state))
            keep = select_top(state.alive, state.fresh_scores(), state.k // 2)
            idx = state.record_prune(keep)
            engine.prune(state, idx)
            prune_js.append(q.j)
        else:
            raise PolicyError(a)
    if not grammar.terminal_ok(state.q):
        raise LedgerError(f"terminal state {state.q} is not a valid completion")
    res.delivery = _finalize(state, engine, mc)
    res.wall_clock_s = clock() - t0
    res.delivery.wall_clock_s = res.wall_clock_s
    res.feasible = True
    res.spend_us = state.spent_us
    res.n_verify = state.n_verify
    res.prune_js = tuple(prune_js)
    if sum(us for _, us in state.charges) != state.spent_us:
        raise LedgerError("ledger trace does not sum to spend")
    return res


def run_schedule(spec: ScheduleSpec, engine: Engine, *, mc: MethodCosts, cap_us: int, C_ref: int = 1,
                 method_id: str = "schedule", delta: int = DELTA,
                 predictor: Optional[Callable[[EpisodeState], dict]] = None,
                 clock: Callable[[], float] = time.perf_counter) -> EpisodeResult:
    """Execute a pre-planned schedule (disp=0). Full cost is validated up front."""
    total = schedule_cost(spec, mc, delta)
    res = EpisodeResult(method_id=method_id, cap_us=cap_us, feasible=False)
    if total > cap_us:
        return res
    t0 = clock()
    state = EpisodeState(grammar_id="SCHEDULE", rows=spec.candidates, cap_us=cap_us, L_fn=lambda q: 0.0,
                         C_ref=C_ref, J=spec.J)
    state.charge(mc.once, "once")
    engine.start(spec.candidates)
    cuts = {c.j: c for c in spec.cuts}
    queries = set(spec.queries)
    prune_js = []
    for j in range(spec.J):
        fee = delta * mc.require("den", state.k) + (mc.require("feat", state.k) if mc.uses_traj else 0)
        res.trace.append(TraceEvent(j, state.k, "CONTINUE", fee, state.remaining_us, (), state.alive))
        state.charge(fee, "CONTINUE")
        feats = engine.advance(state, delta, mc.uses_traj)
        state.arrive(j + 1)
        if mc.uses_traj:
            state.record_traj(feats)
        jn = j + 1
        if jn >= spec.J:
            break
        if jn in queries:
            fee = mc.require("ver", state.k)
            res.trace.append(TraceEvent(jn, state.k, "VERIFY", fee, state.remaining_us, (), state.alive))
            state.charge(fee, "VERIFY")
            state.record_verify(jn, engine.verify(state))
        if jn in cuts:
            c = cuts[jn]
            if c.source == "pred":
                if predictor is None:
                    raise PolicyError("pred cut requires a predictor")
                fee = mc.require("lfeat", state.k) + mc.require("pred", state.k) + mc.require("prune", (state.k, c.keep))
                ranking = predictor(state)
            else:
                fee = mc.require("prune", (state.k, c.keep))
                ranking = None
            res.trace.append(TraceEvent(jn, state.k, f"PRUNE->{c.keep}", fee, state.remaining_us, (), state.alive))
            state.charge(fee, "PRUNE")
            scores = ranking if ranking is not None else state.fresh_scores()
            keep = select_top(state.alive, scores, c.keep)
            idx = state.record_prune(keep, n_keep=c.keep, ranking_scores=ranking)
            engine.prune(state, idx)
            prune_js.append(jn)
    res.delivery = _finalize(state, engine, mc)
    res.wall_clock_s = clock() - t0
    res.delivery.wall_clock_s = res.wall_clock_s
    if state.spent_us != total:
        raise LedgerError(f"schedule spend {state.spent_us} != planned cost {total}")
    res.feasible = True
    res.spend_us = state.spent_us
    res.n_verify = state.n_verify
    res.prune_js = tuple(prune_js)
    return res


class ReplayEngine:
    """Replaces GPU execution by privileged bank reads for ONE prompt (possibly two banks)."""

    def __init__(self, accessor: PrivilegedAccessor, uid: str):
        self.acc = accessor
        self.uid = uid
        self.rows: list = []

    def _by_bank(self, cids):
        out: dict = {}
        for c in cids:
            out.setdefault(c[0], []).append(c)
        return out

    def start(self, rows):
        self.rows = list(rows)

    def advance(self, state, n_forwards, want_traj):
        if not want_traj:
            return None
        j = state.j + 1
        out = {}
        for b, cids in self._by_bank(state.alive).items():
            out.update(self.acc.traj((self.uid, b), j, cids, purpose="env"))
        return out

    def verify(self, state):
        assert tuple(self.rows) == state.alive
        out = {}
        for b, cids in self._by_bank(state.alive).items():
            out.update(self.acc.preview((self.uid, b), state.j, cids, purpose="env"))
        return out

    def prune(self, state, retained):
        self.rows = [self.rows[i] for i in retained]
        if tuple(self.rows) != state.alive:
            raise AssertionError("replay rows diverged from RowMap")

    def finalize_scores(self, state):
        out = {}
        for b, cids in self._by_bank(state.alive).items():
            out.update(self.acc.delivered((self.uid, b), cids, purpose="env"))
        return out

    def finalize_validity(self, state):
        out = {}
        for b, cids in self._by_bank(state.alive).items():
            out.update(self.acc.delivered_flags((self.uid, b), cids, purpose="env"))
        return out


def episode_reward(res: EpisodeResult, accessor: PrivilegedAccessor, uid: str, r_fail: float) -> float:
    """Delivered IR of the winner (privileged, reward purpose) or the frozen failure penalty.

    For k>1 deliveries this equals the inference score used for selection; for F(1) it is the
    post-inference evaluation, which never feeds back into decisions.
    """
    if not res.feasible or res.delivery is None or res.delivery.failed or res.delivery.winner is None:
        return float(r_fail)
    w = res.delivery.winner
    rec = accessor.delivered((uid, w[0]), [w], purpose="reward")[w]
    return float(rec.score) if rec.score_ok else float(r_fail)
