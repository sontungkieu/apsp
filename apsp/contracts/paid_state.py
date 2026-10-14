"""Mutable episode state (env/runner-owned) and the immutable PolicyView (Guide §5.2).

EpisodeState never holds a reference to the whole bank. Only purchased / permitted information
enters it, and ``view()`` only exposes data for candidates that are still alive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .identity import CandidateId, RowMap, select_top
from .policy_api import GrammarState, ScoreRecord


class LedgerError(RuntimeError):
    pass


class StateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaidScore:
    cid: CandidateId
    j: int
    score: Optional[float]
    latent_ok: bool
    img_ok: bool
    score_ok: bool


@dataclass(frozen=True)
class TrajFeature:
    cid: CandidateId
    j: int
    drift: float  # NaN when unavailable
    cfg_gap: float


@dataclass(frozen=True)
class PolicyView:
    grammar_id: str
    j: int
    k: int
    fresh: bool
    pruned: bool
    preview: bool
    alive: tuple[CandidateId, ...]
    remaining_us: int
    cap_us: int
    L_here: float
    C_ref: int
    n_verify: int
    verify_js: tuple[int, ...]
    paid: tuple[PaidScore, ...]           # history of alive candidates only
    traj: tuple[TrajFeature, ...] = ()    # paid-for trajectory features of alive candidates only
    prompt_feat: Optional[np.ndarray] = None
    J: int = 16

    @property
    def q(self) -> GrammarState:
        return GrammarState(self.j, self.k, self.fresh, self.pruned, self.preview)


class EpisodeState:
    def __init__(self, *, grammar_id: str, rows: Sequence[CandidateId], cap_us: int, L_fn: Callable[[GrammarState], float],
                 C_ref: int, prompt_feat: Optional[np.ndarray] = None, J: int = 16):
        if isinstance(cap_us, bool) or not isinstance(cap_us, int) or cap_us < 0:
            raise LedgerError("cap must be a non-negative int (us)")
        self.grammar_id = grammar_id
        self.rowmap = RowMap(rows)
        self.cap_us = cap_us
        self.spent_us = 0
        self.L_fn = L_fn  # minimum completion cost lookup L^m(q) of this method
        self.C_ref = int(C_ref)
        self.J = J
        self.j = 0
        self.fresh = False
        self.pruned = False
        self.n_verify = 0
        self.verify_js: list[int] = []
        self._paid: dict[CandidateId, list[PaidScore]] = {c: [] for c in self.rowmap.rows}
        self._traj: dict[CandidateId, list[TrajFeature]] = {c: [] for c in self.rowmap.rows}
        self.charges: list[tuple[str, int]] = []
        if prompt_feat is not None:
            pf = np.array(prompt_feat, dtype=np.float32, copy=True)
            pf.setflags(write=False)
            self.prompt_feat = pf
        else:
            self.prompt_feat = None

    # --- derived -----------------------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.rowmap)

    @property
    def alive(self) -> tuple[CandidateId, ...]:
        return self.rowmap.rows

    @property
    def remaining_us(self) -> int:
        return self.cap_us - self.spent_us

    @property
    def q(self) -> GrammarState:
        return GrammarState(self.j, self.k, self.fresh, self.pruned, self.j >= 1)

    # --- transitions ----------------------------------------------------------------
    def arrive(self, j: int) -> None:
        """Advance to checkpoint j (no fee here; CONT is charged via ``charge``)."""
        if j != self.j + 1 or j > self.J:
            raise StateError(f"arrive({j}) from j={self.j}")
        self.j = j
        self.fresh = False
        self.pruned = False

    def charge(self, us: int, label: str = "") -> None:
        if isinstance(us, bool) or not isinstance(us, (int, np.integer)):
            raise LedgerError(f"fee {us!r} must be an integer")
        us = int(us)
        if us < 0:
            raise LedgerError("negative fee")
        if self.spent_us + us > self.cap_us:
            raise LedgerError(f"charge {us} would exceed cap: spent={self.spent_us} cap={self.cap_us}")
        self.spent_us += us
        self.charges.append((label, us))

    def record_verify(self, j: int, scores: Mapping[CandidateId, ScoreRecord]) -> None:
        if j != self.j:
            raise StateError(f"verify for j={j} but state at j={self.j}")
        if not 1 <= j <= self.J:
            raise StateError("verify outside checkpoints")
        if self.fresh:
            raise StateError(f"duplicate query at j={j}")
        if set(scores) != set(self.alive):
            raise StateError("verify must cover exactly the alive candidates")
        for c in self.alive:
            r = scores[c]
            s = float(r.score) if (r.score_ok and r.score is not None and math.isfinite(r.score)) else None
            self._paid[c].append(PaidScore(c, j, s, bool(r.latent_ok), bool(r.img_ok), s is not None))
        self.fresh = True
        self.n_verify += 1
        self.verify_js.append(j)

    def fresh_scores(self) -> dict[CandidateId, Optional[float]]:
        if not self.fresh:
            raise StateError("no fresh scores at this checkpoint")
        out = {}
        for c in self.alive:
            last = self._paid[c][-1]
            if last.j != self.j:
                raise StateError("stale score")
            out[c] = last.score
        return out

    def record_prune(self, keep: frozenset[CandidateId] | set[CandidateId], n_keep: Optional[int] = None,
                     ranking_scores: Optional[Mapping[CandidateId, Optional[float]]] = None) -> tuple[int, ...]:
        """Validate ``keep`` against the fixed top-k rule and apply it. Returns retained row indices.

        Default: keep k/2 by the fresh paid scores. ``n_keep`` supports schedule grammars with arbitrary keep
        (B4b/B5); ``ranking_scores`` supports predictor cuts (B10), which do not require a paid query.
        """
        if self.pruned:
            raise StateError("second prune at the same checkpoint")
        if ranking_scores is None:
            if not self.fresh:
                raise StateError("prune requires fresh scores")
            ranking_scores = self.fresh_scores()
        elif set(ranking_scores) != set(self.alive):
            raise StateError("ranking scores must cover the alive set")
        keep = frozenset(keep)
        n_keep = self.k // 2 if n_keep is None else int(n_keep)
        expected = select_top(self.alive, ranking_scores, n_keep)
        if keep != expected:
            raise StateError(f"keep set {sorted(keep)} differs from deterministic top-k {sorted(expected)}")
        idx = self.rowmap.retained_indices(keep)
        self.rowmap.apply_keep(idx)
        for c in list(self._paid):
            if c not in keep:
                del self._paid[c]
                del self._traj[c]
        self.pruned = True
        return idx

    def record_traj(self, feats: Mapping[CandidateId, tuple[float, float]]) -> None:
        if set(feats) != set(self.alive):
            raise StateError("trajectory features must cover the alive set")
        for c, (d, g) in feats.items():
            self._traj[c].append(TrajFeature(c, self.j, float(d), float(g)))

    def view(self) -> PolicyView:
        paid = tuple(p for c in self.alive for p in self._paid[c])
        traj = tuple(t for c in self.alive for t in self._traj[c])
        return PolicyView(
            grammar_id=self.grammar_id, j=self.j, k=self.k, fresh=self.fresh, pruned=self.pruned,
            preview=self.j >= 1, alive=self.alive, remaining_us=self.remaining_us, cap_us=self.cap_us,
            L_here=float(self.L_fn(self.q)), C_ref=self.C_ref, n_verify=self.n_verify,
            verify_js=tuple(self.verify_js), paid=paid, traj=traj, prompt_feat=self.prompt_feat, J=self.J,
        )
