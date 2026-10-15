"""Method registry for the quality-cost pilot: one executor for replay and online (same runners and ledger).

Method strings (optional "@alpha" suffix = run under the shared cap B_alpha; without it, fixed-configuration methods
run uncapped and their ledger spend is recorded):
  single            one seed, full 64 steps, F(1): no RM in inference (B0)
  single32          one seed, 32-step sampler (scheduler initialised for T=32); ONLINE ONLY (no bank of a 32-step
                    sampler exists; checkpoints of the 64-step trajectory are not a 32-step result)
  bon:N             best-of-N, nested seeds of one bank (N<=8) or pool 16 (N=16); terminal RM picks the winner (B1)
  psp               PSP-default fixed pair (4,8) (B2)
  fixed:J1,J2       a fixed pair, e.g. the train-dev-selected best fixed schedule (B3)
  threshold:PATH    B6 threshold controller, parameters frozen in PATH (needs @alpha)
  sr:M,JS,Q         B5 screen-and-resume (our same-bank breadth baseline; not Flash-BoN)
  b8:PATH           B8 one query then commit, selector frozen in PATH (needs @alpha)
  random:SEED       B9 uniform feasible random (sanity; needs @alpha)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from apsp.contracts.hashing import file_sha256, sha256_hex
from apsp.contracts.identity import bank_rows, pool_rows
from apsp.contracts.ledger import MethodCosts
from apsp.sim.budget_dp import G842
from apsp.sim.caps import Caps, fixed_method_costs
from apsp.sim.env import EpisodeResult, run_closed_loop, run_schedule
from apsp.sim.features import NormStats
from apsp.sim.schedules import bon_spec, fixed_pair_spec

UNCAPPED = 10 ** 15
FAMILIES = ("single", "single32", "bon", "psp", "fixed", "threshold", "sr", "b8", "random")
NEEDS_CAP = ("threshold", "b8", "random")
ONLINE_ONLY = ("single32",)


class MethodError(ValueError):
    pass


@dataclass(frozen=True)
class Method:
    spec: str                       # the full string as given
    family: str
    params: tuple
    alpha: Optional[float]
    artifact: Optional[str] = None  # frozen parameter file (threshold / b8)
    artifact_sha256: Optional[str] = None

    @property
    def steps(self) -> int:
        return 32 if self.family == "single32" else 64

    @property
    def config_hash(self) -> str:
        return sha256_hex({"spec": self.spec.split("@")[0] if self.artifact is None else self.family,
                           "params": list(self.params), "artifact_sha256": self.artifact_sha256})[:16]

    @property
    def uses_inference_rm(self) -> bool:
        return self.family not in ("single", "single32") and not (self.family == "sr" and self.params[2] == 1) \
            and not (self.family == "bon" and self.params[0] == 1)

    def n_candidates(self) -> int:
        return len(rows_for(self, 0))

    def per_prompt(self) -> bool:
        """Pool-16 methods use both banks of a prompt in one episode."""
        return (self.family == "bon" and self.params[0] == 16) or (self.family == "sr" and self.params[0] == 16)


def parse_method(s: str) -> Method:
    base, alpha = (s.split("@", 1) + [None])[:2]
    alpha = float(alpha) if alpha is not None else None
    fam, _, arg = base.partition(":")
    art = sha = None
    if fam in ("single", "single32", "psp"):
        params = ()
    elif fam == "bon":
        n = int(arg)
        if not 1 <= n <= 16 or (8 < n < 16):
            raise MethodError("bon:N needs 1<=N<=8 (one bank) or N=16 (pool 16)")
        params = (n,)
    elif fam == "fixed":
        j1, j2 = (int(x) for x in arg.split(","))
        params = (j1, j2)
    elif fam == "sr":
        M, js, q = (int(x) for x in arg.split(","))
        params = (M, js, q)
    elif fam in ("threshold", "b8"):
        art = arg
        sha = file_sha256(Path(arg))
        params = ()
    elif fam == "random":
        params = (int(arg or 0),)
    else:
        raise MethodError(f"unknown method {s}")
    if fam in NEEDS_CAP and alpha is None:
        raise MethodError(f"{fam} runs under a cap: use {base}@ALPHA")
    return Method(s, fam, params, alpha, art, sha)


def rows_for(m: Method, bank_id: int):
    if m.family in ("single", "single32"):
        return tuple(bank_rows(bank_id)[:1])
    if m.family == "bon":
        n = m.params[0]
        return tuple(pool_rows(2)) if n == 16 else tuple(bank_rows(bank_id)[:n])
    if m.family == "sr" and m.params[0] == 16:
        return tuple(pool_rows(2))
    return tuple(bank_rows(bank_id))


def cap_of(m: Method, caps: Optional[Caps]) -> int:
    if m.alpha is None:
        return UNCAPPED
    if caps is None:
        raise MethodError("a capped method needs the cost table caps")
    return caps.cap(m.alpha)


@dataclass
class RunContext:
    base: MethodCosts                         # measured (or synthetic) generator primitives
    caps: Optional[Caps]
    norm: NormStats
    ctrl_disp_us: int = 0                     # controller dispatch fee (closed-loop methods)
    selector_once_us: int = 0                 # B8 selector / query-feature cost (prepaid once)
    loaded: dict = field(default_factory=dict)


def run_method(m: Method, engine, *, bank_id: int, ctx: RunContext, prompt_feat=None, is_replay: bool = False) -> EpisodeResult:
    """Execute one episode of ``m`` on ``engine`` (DenoiseEngine online or ReplayEngine). Ledger by the shared runners."""
    if is_replay and m.family in ONLINE_ONLY:
        raise MethodError(f"{m.family} has no replay: it is a different sampler (online only)")
    fx = fixed_method_costs(ctx.base)
    cap = cap_of(m, ctx.caps)
    C_ref = ctx.caps.C_ref if ctx.caps else 1
    rows = rows_for(m, bank_id)
    f = m.family
    if f in ("single", "single32", "bon"):
        J = m.steps // 4
        return run_schedule(bon_spec(len(rows), rows, J=J), engine, mc=fx, cap_us=cap, C_ref=C_ref, method_id=m.spec)
    if f in ("psp", "fixed"):
        j1, j2 = (4, 8) if f == "psp" else m.params
        return run_schedule(fixed_pair_spec(j1, j2, rows), engine, mc=fx, cap_us=cap, C_ref=C_ref, method_id=m.spec)
    if f == "sr":
        from apsp.policies.screen_resume import SRConfig, sr_spec
        M, js, q = m.params
        return run_schedule(sr_spec(SRConfig(M, js, q), bank_id), engine, mc=fx, cap_us=cap, C_ref=C_ref, method_id=m.spec)
    if f == "b8":
        from apsp.policies.query_commit import RidgeSelector, run_query_commit
        sel = ctx.loaded.setdefault(m.artifact, RidgeSelector.from_dict(_load_json(m.artifact).get("selector", _load_json(m.artifact))))
        return run_query_commit(sel, engine, rows=rows, mc=fx.with_overrides(once=ctx.selector_once_us), cap_us=cap,
                                norm=ctx.norm, prompt_feat=prompt_feat, C_ref=C_ref, method_id=m.spec)
    if f in ("threshold", "random"):
        if f == "threshold":
            from apsp.policies.threshold import ThresholdParams, ThresholdPolicy
            d = _load_json(m.artifact)
            pol = ThresholdPolicy(ThresholdParams(**d.get("params", d)))
        else:
            from apsp.policies.random_policy import RandomFeasiblePolicy
            pol = RandomFeasiblePolicy(m.params[0])
        ctrl = ctx.base.with_overrides(disp=ctx.ctrl_disp_us, method_id="controller")
        return run_closed_loop(pol, engine, rows=rows, grammar=G842(), mc=ctrl, cap_us=cap, C_ref=C_ref, norm=ctx.norm,
                               prompt_feat=prompt_feat)
    raise MethodError(f)


def _load_json(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def episode_quality(res: EpisodeResult) -> tuple[bool, Optional[float]]:
    """(success, inference-time score of the winner if the method computed one)."""
    if not res.feasible or res.delivery is None or res.delivery.failed or res.delivery.winner is None:
        return False, None
    return True, res.delivery.inference_score
