"""Candidate identity, stable row maps, deterministic ranking and seed derivation (Guide §5.1, §9)."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Optional, Sequence

CandidateId = tuple[int, int]  # (bank_id, seed_idx)

SEED_SCHEME = "apsp-v1-sha256-cpu"


def pool_rows(n_banks: int = 1, k0: int = 8) -> tuple[CandidateId, ...]:
    """Row order of a pool: (0,0)..(0,7),(1,0)..(1,7) for pool16."""
    return tuple((b, s) for b in range(n_banks) for s in range(k0))


def bank_rows(bank_id: int, k0: int = 8) -> tuple[CandidateId, ...]:
    return tuple((bank_id, s) for s in range(k0))


def _is_valid_score(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def rank_candidates(ids: Sequence[CandidateId], scores: Mapping[CandidateId, Optional[float]]) -> list[CandidateId]:
    """Decreasing finite score, ties by increasing CandidateId; invalid (None/NaN/inf) last by CandidateId."""
    valid = [c for c in ids if _is_valid_score(scores.get(c))]
    invalid = [c for c in ids if not _is_valid_score(scores.get(c))]
    valid.sort(key=lambda c: (-float(scores[c]), c))
    invalid.sort()
    return valid + invalid


def select_top(ids: Sequence[CandidateId], scores: Mapping[CandidateId, Optional[float]], keep: int) -> frozenset[CandidateId]:
    if not 1 <= keep <= len(ids):
        raise ValueError(f"keep={keep} out of range for {len(ids)} candidates")
    return frozenset(rank_candidates(ids, scores)[:keep])


class RowMap:
    """Mapping tensor row -> CandidateId. Rows always stay in the original relative order."""

    def __init__(self, rows: Iterable[CandidateId]):
        rows = tuple((int(b), int(s)) for b, s in rows)
        if len(set(rows)) != len(rows):
            raise ValueError("duplicate CandidateId in RowMap")
        self.rows: tuple[CandidateId, ...] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def index(self, cid: CandidateId) -> int:
        return self.rows.index(cid)

    def retained_indices(self, keep_ids: Iterable[CandidateId]) -> tuple[int, ...]:
        keep = set(keep_ids)
        missing = keep - set(self.rows)
        if missing:
            raise KeyError(f"keep ids not alive: {sorted(missing)}")
        return tuple(i for i, c in enumerate(self.rows) if c in keep)

    def apply_keep(self, indices: Sequence[int]) -> None:
        idx = tuple(int(i) for i in indices)
        if not idx:
            raise ValueError("cannot keep zero rows")
        if any(b <= a for a, b in zip(idx, idx[1:])):
            raise ValueError("retained indices must be strictly increasing (old row order is preserved)")
        if idx[0] < 0 or idx[-1] >= len(self.rows):
            raise IndexError("retained index out of range")
        self.rows = tuple(self.rows[i] for i in idx)

    def copy(self) -> "RowMap":
        return RowMap(self.rows)


def cfg_row_indices(indices: Sequence[int], k: int) -> list[int]:
    """Row indices into a CFG batch laid out as [uncond rows; cond rows] of size 2k."""
    idx = list(indices)
    return idx + [i + k for i in idx]


def derive_seed(prompt_uid: str, bank_id: int, seed_idx: int) -> int:
    """sha256 UTF-8 ``apsp-v1|{uid}|{bank_id}|{seed_idx}``; first 4 bytes little-endian & 0x7FFFFFFF."""
    msg = f"apsp-v1|{prompt_uid}|{int(bank_id)}|{int(seed_idx)}".encode("utf-8")
    d = hashlib.sha256(msg).digest()
    return int.from_bytes(d[:4], "little") & 0x7FFFFFFF


def seed_table(prompt_uid: str, n_banks: int = 2, k0: int = 8) -> dict[CandidateId, int]:
    return {(b, s): derive_seed(prompt_uid, b, s) for b in range(n_banks) for s in range(k0)}
