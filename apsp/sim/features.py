"""Shared observation builder (Spec §9.1, Guide §8 CX-4). Frozen with obs version ``apsp-obs-v1``.

Main 38 dims:
  control (11): j/J, onehot k in (8,4,2), fresh, pruned, r/C_ref, L/C_ref, slack/C_ref, B/C_ref, n_verify/J
  paid (8):     has_score, age/J, mean z, std z, max z, z_cut, z_top, frac_invalid
  stability (3): has_two, kendall tau-b, delta z_cut
  prompt (16):  PCA-16 of raw pooler_output (fit on train_fit only)
Traj 42 dims: + mean/std drift, mean/std cfg_gap at the current checkpoint (paid feature).

Only data inside the PolicyView is read: alive candidates and purchased/permitted information.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from apsp.contracts.hashing import array_sha256, sha256_hex
from apsp.contracts.paid_state import PolicyView

OBS_VERSION = "apsp-obs-v1"
MAIN_DIM, TRAJ_DIM = 38, 42
CLIP = 10.0
EPS = 1e-6
SIGMA_FLOOR = 1e-6
K_ONEHOT = (8, 4, 2)

# feature indices (used by rule-based policies and tests)
IDX = {"j": 0, "k8": 1, "k4": 2, "k2": 3, "fresh": 4, "pruned": 5, "r": 6, "L": 7, "slack": 8, "B": 9, "nver": 10,
       "has_score": 11, "age": 12, "mean_z": 13, "std_z": 14, "max_z": 15, "z_cut": 16, "z_top": 17, "frac_invalid": 18,
       "has_two": 19, "kendall": 20, "dz_cut": 21, "prompt0": 22}


@dataclass(frozen=True)
class NormStats:
    mu: tuple[float, ...]     # per checkpoint j=1..16 (index j-1)
    sigma: tuple[float, ...]

    @staticmethod
    def fit(preview_ir: Sequence[np.ndarray]) -> "NormStats":
        """Fit on VALID train_fit preview scores only (arrays [8,16], NaN = invalid)."""
        a = np.concatenate([np.asarray(x, np.float64).reshape(-1, 16) for x in preview_ir], 0)
        mu, sd = [], []
        for j in range(16):
            col = a[:, j][np.isfinite(a[:, j])]
            if col.size == 0:
                raise ValueError(f"no valid train_fit scores at checkpoint {j + 1}; cannot fit normalisation")
            mu.append(float(col.mean()))
            sd.append(max(float(col.std(ddof=0)), SIGMA_FLOOR))
        return NormStats(tuple(mu), tuple(sd))

    @staticmethod
    def identity() -> "NormStats":
        return NormStats(tuple([0.0] * 16), tuple([1.0] * 16))

    def z(self, score: float, j: int) -> float:
        return (score - self.mu[j - 1]) / self.sigma[j - 1]

    def to_dict(self) -> dict:
        return {"obs_version": OBS_VERSION, "mu": list(self.mu), "sigma": list(self.sigma)}

    def hash(self) -> str:
        return sha256_hex(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "NormStats":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return NormStats(tuple(d["mu"]), tuple(d["sigma"]))


@dataclass(frozen=True)
class PCA16:
    """Deterministic PCA on raw pooler_output (float64 SVD; sign: largest-|loading| entry positive)."""

    mean: np.ndarray
    components: np.ndarray  # [16, 768]

    @staticmethod
    def fit(X: np.ndarray, n: int = 16) -> "PCA16":
        X = np.asarray(X, np.float64)
        if X.shape[0] < n:
            raise ValueError(f"need >= {n} train_fit prompts for PCA-{n}")
        m = X.mean(0)
        _, _, vt = np.linalg.svd(X - m, full_matrices=False)
        comp = vt[:n].copy()
        for i in range(n):
            if comp[i, np.argmax(np.abs(comp[i]))] < 0:
                comp[i] = -comp[i]
        m.setflags(write=False)
        comp.setflags(write=False)
        return PCA16(m, comp)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, np.float64) - self.mean) @ self.components.T).astype(np.float32)

    def hash(self) -> str:
        return sha256_hex({"mean": array_sha256(self.mean), "components": array_sha256(self.components)})

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, components=self.components)

    @staticmethod
    def load(path: str | Path) -> "PCA16":
        d = np.load(path)
        return PCA16(d["mean"], d["components"])


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    conc = disc = tx = ty = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx = np.sign(x[a] - x[b])
            dy = np.sign(y[a] - y[b])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return 0.0
    return (conc - disc) / denom


def _gap_features(vals: list[float], h: int) -> tuple[float, float]:
    """z_cut, z_top on sorted valid scores; enabled only if v >= h+1, else 0."""
    v = len(vals)
    if v < h + 1 or v < 2:
        return 0.0, 0.0
    s = sorted(vals, reverse=True)
    sd = float(np.std(s, ddof=0)) + EPS
    return (s[h - 1] - s[h]) / sd, (s[0] - s[1]) / sd


def _clip(x: float) -> float:
    return float(min(CLIP, max(-CLIP, x)))


def build_obs(view: PolicyView, norm: NormStats, variant: str = "main") -> np.ndarray:
    if variant not in ("main", "traj"):
        raise ValueError(variant)
    J = view.J
    Cr = float(view.C_ref)
    o: list[float] = []
    # --- control (11)
    L_here = view.L_here if math.isfinite(view.L_here) else 0.0
    slack = view.remaining_us - L_here if math.isfinite(view.L_here) else 0.0
    o += [view.j / J] + [1.0 if view.k == kk else 0.0 for kk in K_ONEHOT]
    o += [float(view.fresh), float(view.pruned)]
    o += [view.remaining_us / Cr, L_here / Cr, slack / Cr, view.cap_us / Cr, view.n_verify / J]
    # --- paid scores (8): latest verify of alive candidates
    alive = list(view.alive)
    by_c: dict = {c: [] for c in alive}
    for p in view.paid:
        by_c[p.cid].append(p)
    last_j = max(view.verify_js) if view.verify_js else None
    latest = [by_c[c][-1] for c in alive if by_c[c] and by_c[c][-1].j == last_j] if last_j else []
    h = max(view.k // 2, 1)
    if latest:
        valid = [p for p in latest if p.score_ok and p.score is not None]
        frac_inv = 1.0 - len(valid) / len(latest)
        if valid:
            z = [norm.z(p.score, p.j) for p in valid]
            mz, sz, xz = float(np.mean(z)), float(np.std(z, ddof=0)), float(np.max(z))
        else:
            mz = sz = xz = 0.0
        zc, zt = _gap_features([p.score for p in valid], h)
        o += [1.0, (view.j - last_j) / J, _clip(mz), _clip(sz), _clip(xz), _clip(zc), _clip(zt), frac_inv]
    else:
        o += [0.0] * 8
    # --- stability (3): two most recent verifies, common valid alive candidates
    has_two = tau = dz = 0.0
    if len(view.verify_js) >= 2:
        j_a, j_b = view.verify_js[-2], view.verify_js[-1]
        common = []
        for c in alive:
            sa = next((p.score for p in by_c[c] if p.j == j_a and p.score_ok), None)
            sb = next((p.score for p in by_c[c] if p.j == j_b and p.score_ok), None)
            if sa is not None and sb is not None:
                common.append((sa, sb))
        if len(common) >= 2:
            has_two = 1.0
            tau = _kendall_tau_b([a for a, _ in common], [b for _, b in common])
            za, _ = _gap_features([a for a, _ in common], h)
            zb, _ = _gap_features([b for _, b in common], h)
            # gap change only when both gaps are computable (v >= h+1); else 0 although tau may exist
            dz = _clip(zb - za) if len(common) >= h + 1 else 0.0
    o += [has_two, float(tau), dz]
    # --- prompt (16)
    pf = view.prompt_feat
    if pf is None:
        o += [0.0] * 16
    else:
        pf = np.asarray(pf, np.float64).reshape(-1)
        if pf.shape[0] != 16:
            raise ValueError("prompt feature must be PCA-16")
        o += [_clip(x) if math.isfinite(x) else 0.0 for x in pf]
    # --- trajectory (4)
    if variant == "traj":
        cur = [t for t in view.traj if t.j == view.j]
        dr = [t.drift for t in cur if math.isfinite(t.drift)]
        gp = [t.cfg_gap for t in cur if math.isfinite(t.cfg_gap)]
        o += [_clip(float(np.mean(dr))) if dr else 0.0, _clip(float(np.std(dr, ddof=0))) if dr else 0.0,
              _clip(float(np.mean(gp))) if gp else 0.0, _clip(float(np.std(gp, ddof=0))) if gp else 0.0]
    obs = np.asarray(o, dtype=np.float32)
    exp = MAIN_DIM if variant == "main" else TRAJ_DIM
    if obs.shape != (exp,):
        raise AssertionError(f"obs dim {obs.shape} != {exp}")
    if not np.isfinite(obs).all():
        raise AssertionError("non-finite observation")
    return obs
