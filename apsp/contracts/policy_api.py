"""Policy / schedule / delivery interfaces shared by replay and online (Guide §5.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np

from .identity import CandidateId

API_VERSION = "apsp-policy-api-v1.1.1"

CONT, VER, PRUNE = 0, 1, 2
VP = 3  # atomic VERIFY+PRUNE, only in grammar G-A1-v1 (ablation A1)
ACTION_NAMES = {CONT: "CONTINUE", VER: "VERIFY", PRUNE: "PRUNE", VP: "VERIFY_PRUNE"}


class GrammarState(NamedTuple):
    """Reduced grammar state q=(j,k,f,p_r,v) of Research Spec §6.3."""

    j: int
    k: int
    fresh: bool
    pruned: bool
    preview: bool

    @staticmethod
    def make(j: int, k: int, fresh: bool = False, pruned: bool = False) -> "GrammarState":
        return GrammarState(int(j), int(k), bool(fresh), bool(pruned), bool(j >= 1))


@runtime_checkable
class Policy(Protocol):
    method_id: str
    grammar_id: str
    uses_traj: bool
    cph: str

    def reset(self, prompt_feat: np.ndarray, cap_us: int) -> None: ...

    def act(self, obs: np.ndarray, mask: tuple[int, ...]) -> int: ...


@dataclass(frozen=True)
class Cut:
    j: int
    keep: int
    source: Literal["paid", "pred"] = "paid"


@dataclass(frozen=True)
class ScheduleSpec:
    candidates: tuple[CandidateId, ...]
    queries: tuple[int, ...]
    cuts: tuple[Cut, ...]
    final_k: int
    J: int = 16

    def validate(self) -> None:
        if len(set(self.candidates)) != len(self.candidates) or not self.candidates:
            raise ValueError("schedule candidates must be unique and non-empty")
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("duplicate query checkpoint")
        if any(not 1 <= q <= self.J - 1 for q in self.queries):
            raise ValueError("queries must lie in 1..J-1")
        if tuple(sorted(self.queries)) != tuple(self.queries):
            raise ValueError("queries must be increasing")
        prev_j, prev_keep = 0, len(self.candidates)
        for c in self.cuts:
            if not 1 <= c.j <= self.J - 1:
                raise ValueError(f"cut j={c.j} outside 1..{self.J - 1}")
            if c.j <= prev_j:
                raise ValueError("cut checkpoints must be strictly increasing")
            if not 1 <= c.keep < prev_keep:
                raise ValueError("cut keep must be strictly decreasing")
            if c.source == "paid" and c.j not in self.queries:
                raise ValueError(f"paid cut at j={c.j} needs a query at the same checkpoint")
            if c.source not in ("paid", "pred"):
                raise ValueError(f"unknown cut source {c.source}")
            prev_j, prev_keep = c.j, c.keep
        if self.final_k != prev_keep:
            raise ValueError(f"final_k={self.final_k} must equal last keep {prev_keep}")


class Scheduler(Protocol):
    method_id: str
    cph: str

    def plan(self, view, mc, cap_us: int) -> ScheduleSpec: ...


@dataclass(frozen=True)
class ScoreRecord:
    """Result of one paid score for one candidate. ``score`` is None unless all three flags are True."""

    latent_ok: bool
    img_ok: bool
    score_ok: bool
    score: Optional[float]

    def __post_init__(self):
        if self.score_ok and self.score is None:
            raise ValueError("score_ok requires a finite score")
        if not self.score_ok and self.score is not None:
            raise ValueError("invalid score must be None in runtime views")
        if self.img_ok and not self.latent_ok:
            raise ValueError("img_ok implies latent_ok")
        if self.score_ok and not self.img_ok:
            raise ValueError("score_ok implies img_ok")


@dataclass
class Delivery:
    winner: Optional[CandidateId]
    failed: bool
    latent_ok: bool
    img_ok: bool
    rm_used: bool
    inference_score: Optional[float]
    ledger_spend_us: int
    wall_clock_s: Optional[float] = None
    image: object = None  # PIL image online; None in replay
    delivered_scores: dict = field(default_factory=dict)
