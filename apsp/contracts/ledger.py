"""The single source of fee arithmetic (Research Spec §7, Guide §5.4/§6).

All costs are integer microseconds. GPU runners and replay call these functions; nobody
else re-implements the formulas.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from .hashing import sha256_hex
from .policy_api import CONT, PRUNE, VER, VP, Cut, GrammarState, ScheduleSpec

LEDGER_VERSION = "apsp-ledger-v1.1.1"
DELTA = 4
MAX_US = 10**15  # overflow guard


class CostTableError(ValueError):
    pass


def _check_fee(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise CostTableError(f"{name}={v!r} must be an integer number of microseconds")
    if v < 0 or v > MAX_US:
        raise CostTableError(f"{name}={v} outside [0, {MAX_US}]")
    return v


def ceil_us(mean_us: float) -> int:
    """c = ceil(mean_us); rejects non-finite / negative measurements."""
    if not math.isfinite(mean_us) or mean_us < 0:
        raise CostTableError(f"invalid measured mean {mean_us}")
    return int(math.ceil(mean_us))


@dataclass
class MethodCosts:
    den: dict[int, int]
    ver: dict[int, int]
    fin: dict[int, int]
    prune: dict[tuple[int, int], int]
    feat: dict[int, int] = field(default_factory=dict)
    lfeat: dict[int, int] = field(default_factory=dict)
    disp: int = 0
    once: int = 0
    pred: dict[int, int] = field(default_factory=dict)
    fixed: dict[int, int] = field(default_factory=dict)
    uses_traj: bool = False
    synthetic: bool = False  # True for synthetic fixtures; never evidence of real speed
    provisional: bool = True
    method_id: str = "fixed"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for tname in ("den", "ver", "fin", "feat", "lfeat", "pred", "fixed"):
            tab = getattr(self, tname)
            for k, v in tab.items():
                if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                    raise CostTableError(f"{tname} key {k!r} must be a positive int")
                _check_fee(f"{tname}[{k}]", v)
        for (a, b), v in self.prune.items():
            if not (isinstance(a, int) and isinstance(b, int) and a > b >= 1):
                raise CostTableError(f"prune key {(a, b)} must satisfy source > target >= 1")
            _check_fee(f"prune[{a}->{b}]", v)
        _check_fee("disp", self.disp)
        _check_fee("once", self.once)

    def with_overrides(self, **kw) -> "MethodCosts":
        d = self.to_dict()
        d.update(kw)
        return MethodCosts.from_dict(d)

    def scaled_verifier(self, factor: float) -> "MethodCosts":
        """Ablation A4: multiply verifier primitive (ver, RM part not separable -> whole ver) and recompute L."""
        return self.with_overrides(ver={k: int(math.ceil(v * factor)) for k, v in self.ver.items()})

    # --- serialisation -------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["prune"] = {f"{a}->{b}": v for (a, b), v in self.prune.items()}
        for t in ("den", "ver", "fin", "feat", "lfeat", "pred", "fixed"):
            d[t] = {str(k): v for k, v in getattr(self, t).items()}
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "MethodCosts":
        d = dict(d)

        def ik(t):
            return {int(k): int(v) if isinstance(v, int) else v for k, v in (d.get(t) or {}).items()}

        prune = {}
        for k, v in (d.get("prune") or {}).items():
            if isinstance(k, tuple):
                prune[k] = v
            else:
                a, b = str(k).split("->")
                prune[(int(a), int(b))] = v
        return MethodCosts(
            den=ik("den"), ver=ik("ver"), fin=ik("fin"), prune=prune, feat=ik("feat"), lfeat=ik("lfeat"),
            disp=d.get("disp", 0), once=d.get("once", 0), pred=ik("pred"), fixed=ik("fixed"),
            uses_traj=bool(d.get("uses_traj", False)), synthetic=bool(d.get("synthetic", False)),
            provisional=bool(d.get("provisional", True)), method_id=str(d.get("method_id", "fixed")),
        )

    def cost_hash(self) -> str:
        return sha256_hex(self.to_dict())

    def require(self, table: str, key) -> int:
        tab = getattr(self, table)
        if key not in tab:
            raise CostTableError(f"missing primitive {table}[{key}] for method {self.method_id}")
        return tab[key]


def load_method_costs(path: str | Path, **overrides) -> MethodCosts:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.update(overrides)
    return MethodCosts.from_dict(d)


# --- per-action fees ------------------------------------------------------------------
def cost_gen(q: GrammarState, action: int, mc: MethodCosts, delta: int = DELTA) -> int:
    """Generation-side fee of an action, excluding the decision (dispatch) fee."""
    k = q.k
    if action == CONT:
        c = delta * mc.require("den", k)
        if mc.uses_traj:
            c += mc.require("feat", k)  # once per block, NOT multiplied by delta
        return c
    if action == VER:
        return mc.require("ver", k)
    if action == PRUNE:
        return mc.require("prune", (k, k // 2))
    if action == VP:
        return mc.require("ver", k) + mc.require("prune", (k, k // 2))
    raise ValueError(f"unknown action {action}")


def cost_decision(mc: MethodCosts) -> int:
    return mc.disp


def cost_action(q: GrammarState, action: int, mc: MethodCosts, delta: int = DELTA) -> int:
    return cost_gen(q, action, mc, delta) + cost_decision(mc)


def cost_final(k: int, mc: MethodCosts) -> int:
    """Terminal decode/selection/delivery; no dispatch fee."""
    return mc.require("fin", k)


def reserve_once(cap_us: int, mc: MethodCosts) -> int:
    """Budget left after prepaying ``once``; may be negative (episode then infeasible)."""
    _check_fee("cap", cap_us)
    return cap_us - mc.once


def schedule_cost(spec: ScheduleSpec, mc: MethodCosts, delta: int = DELTA, include_once: bool = True) -> int:
    """Full-trace cost of a pre-planned schedule (disp=0 by definition for scheduled methods).

    Paid cut: ver at the query + prune.  Pred cut: lfeat + pred + prune (no ver).
    Queries without a cut still pay ver.  Terminal fin[final_k].
    """
    spec.validate()
    cuts = {c.j: c for c in spec.cuts}
    queries = set(spec.queries)
    k = len(spec.candidates)
    total = mc.once if include_once else 0
    for j in range(0, spec.J):
        # block j -> j+1
        total += delta * mc.require("den", k)
        if mc.uses_traj:
            total += mc.require("feat", k)
        jn = j + 1
        if jn >= spec.J:
            break
        if jn in queries:
            total += mc.require("ver", k)
        if jn in cuts:
            c: Cut = cuts[jn]
            if c.source == "pred":
                total += mc.require("lfeat", k) + mc.require("pred", k)
            total += mc.require("prune", (k, c.keep))
            k = c.keep
    total += mc.require("fin", spec.final_k)
    return total


def fixed_pair_cost(j1: int, j2: int, mc: MethodCosts, J: int = 16, delta: int = DELTA) -> int:
    """Closed form of Research Spec §7.4 C(j1,j2) (+ once). Used to cross-check schedule_cost."""
    gen = delta * (j1 * mc.den[8] + (j2 - j1) * mc.den[4] + (J - j2) * mc.den[2])
    feat = (j1 * mc.feat[8] + (j2 - j1) * mc.feat[4] + (J - j2) * mc.feat[2]) if mc.uses_traj else 0
    return mc.once + gen + feat + mc.ver[8] + mc.prune[(8, 4)] + mc.ver[4] + mc.prune[(4, 2)] + mc.fin[2]


def startup_cost(k0: int, mc: MethodCosts) -> Optional[int]:
    """C_fixed(k0): outside the cap, inside the e2e endpoint."""
    return mc.fixed.get(k0)
