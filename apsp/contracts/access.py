"""Privileged bank access with purpose logging (Guide §8 CX-3).

``PrivilegedAccessor`` is for env / oracle / training-label / reward code only. Policies never
receive it: they get a ``PolicyView`` (or an observation built from it). ``EvalAccessor`` lives
apart and reads evaluator outputs only after the corresponding guard.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .identity import CandidateId
from .policy_api import ScoreRecord
from .prereg import require_split_access
from .schema import BankArrays

PURPOSES = frozenset({"env", "oracle", "training_label", "reward", "validation", "diagnostic"})


class AccessError(PermissionError):
    pass


class PrivilegedAccessor:
    def __init__(self, bank: BankArrays, *, keep_log: bool = False, guard: bool = True, prereg_root: Optional[Path] = None):
        self._bank = bank
        self.keep_log = keep_log
        self.log: list[dict] = []
        self.counts: Counter = Counter()
        self.guard = guard
        self._prereg_root = prereg_root
        self._checked_splits: set = set()

    def _note(self, purpose: str, key, j, cids, event: str) -> None:
        if purpose not in PURPOSES:
            raise AccessError(f"unknown purpose {purpose}")
        split = self._bank.split_of(key[0])
        if self.guard and split not in self._checked_splits:
            require_split_access(split, purpose, self._prereg_root)
            self._checked_splits.add(split)
        self.counts[(purpose, event)] += 1
        if self.keep_log:
            self.log.append({"purpose": purpose, "j": j, "cids": tuple(cids), "key": key, "split": split, "event": event})

    @staticmethod
    def _rows(cids: Sequence[CandidateId], key) -> list[int]:
        rows = []
        for b, s in cids:
            if b != key[1]:
                raise AccessError(f"candidate {(b, s)} not in bank {key}")
            rows.append(s)
        return rows

    def preview(self, key, j: int, cids: Sequence[CandidateId], purpose: str = "env") -> dict:
        self._note(purpose, key, j, cids, "preview")
        ir, fl = self._bank.preview_ir[key], self._bank.preview_flags[key]
        out = {}
        for c, s in zip(cids, self._rows(cids, key)):
            lo, io, so = (bool(x) for x in fl[s, j - 1])
            v = float(ir[s, j - 1])
            ok = so and math.isfinite(v)
            out[c] = ScoreRecord(lo, io and lo, ok, v if ok else None)
        return out

    def traj(self, key, j: int, cids: Sequence[CandidateId], purpose: str = "env") -> dict:
        self._note(purpose, key, j, cids, "traj")
        d, g = self._bank.drift[key], self._bank.cfg_gap[key]
        return {c: (float(d[s, j - 1]), float(g[s, j - 1])) for c, s in zip(cids, self._rows(cids, key))}

    def lfeat(self, key, j: int, cids: Sequence[CandidateId], purpose: str = "env") -> dict:
        self._note(purpose, key, j, cids, "lfeat")
        d, g, ch = self._bank.drift[key], self._bank.cfg_gap[key], self._bank.chan[key]
        return {c: np.concatenate([[d[s, j - 1], g[s, j - 1]], ch[s, j - 1]]).astype(np.float32)
                for c, s in zip(cids, self._rows(cids, key))}

    def delivered(self, key, cids: Sequence[CandidateId], purpose: str = "env") -> dict:
        self._note(purpose, key, 16, cids, "delivered")
        ir, fl = self._bank.delivered_ir[key], self._bank.delivered_flags[key]
        out = {}
        for c, s in zip(cids, self._rows(cids, key)):
            lo, io, so = (bool(x) for x in fl[s])
            v = float(ir[s])
            ok = so and math.isfinite(v)
            out[c] = ScoreRecord(lo, io and lo, ok, v if ok else None)
        return out

    def delivered_flags(self, key, cids: Sequence[CandidateId], purpose: str = "env") -> dict:
        """Validity of the delivered image without its RM score (F(1) path)."""
        self._note(purpose, key, 16, cids, "delivered_flags")
        fl = self._bank.delivered_flags[key]
        return {c: (bool(fl[s, 0]), bool(fl[s, 1])) for c, s in zip(cids, self._rows(cids, key))}

    def instance_arrays(self, key, purpose: str) -> dict:
        if purpose not in ("oracle", "training_label", "diagnostic"):
            raise AccessError("whole-instance reads are reserved for oracle / training labels / diagnostics")
        self._note(purpose, key, None, (), "instance")
        b = self._bank
        return {"preview_ir": b.preview_ir[key], "preview_flags": b.preview_flags[key],
                "delivered_ir": b.delivered_ir[key], "delivered_flags": b.delivered_flags[key]}

    def prompt_embedding(self, uid: str) -> np.ndarray:
        """Raw pooler_output (prompt information is permitted via approved setup / once)."""
        return self._bank.prompt_emb[self._bank.emb_row[uid]].astype(np.float32)

    def instances(self, splits: Iterable[str]) -> list:
        return self._bank.instances_of(splits)

    def split_of(self, uid: str) -> str:
        return self._bank.split_of(uid)


class EvalAccessor:
    """Reads ``eval/eval_scores.parquet``. Test-only columns need PREREG-C."""

    def __init__(self, bank_dir: str | Path, prereg_root: Optional[Path] = None):
        self.path = Path(bank_dir) / "eval" / "eval_scores.parquet"
        self._root = prereg_root

    def read(self, split: str, purpose: str = "evaluation"):
        import pandas as pd
        require_split_access("test" if split == "test" else split, purpose, self._root)
        df = pd.read_parquet(self.path)
        return df[df["split"] == split]
