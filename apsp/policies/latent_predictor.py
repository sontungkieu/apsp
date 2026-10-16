"""B10-min: cheap latent predictor with fixed cuts and no intermediate VERIFY (Spec §9.2 B10, Guide CX-3).

Schedule: 8 candidates, predictor cut 8->4 at j1 and 4->2 at j2 (source "pred"), terminal RM picks the winner among
2 (fin(2)). At a cut the ledger charges lfeat(k) + pred(k) + prune(k->k/2); no ver is paid and no intermediate
reward is read.

Features per candidate at checkpoint j (allowed, cheap, from the forward already computed; bank layout):
  raw  [drift, cfg_gap, mean_c0..3, std_c0..3]                (10)
  rel  raw minus the mean over the alive candidates             (10)   -> 20 features + intercept
drift is NaN at j=1 (no previous checkpoint) and is replaced by 0 after standardisation.

Training (train_fit only): one ridge regression per checkpoint j, target = delivered IR of the candidate
(privileged training label), regularisation by group 5-fold CV (folds by prompt, standardisation fitted inside each
fold). The pair (j1, j2) is shortlisted on train_fit using OUT-OF-FOLD predictions (top 5 affordable pairs), and
chosen on train_dev with the predictors refitted on the whole of train_fit. val/test are never read.

pred(k) is the measured wall time of the predictor on this machine (`measure_pred_us`, CPU, ceil of the mean).
It is recorded in the artifact as provisional and enters the ledger for every predictor cut.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from apsp.contracts.hashing import sha256_hex
from apsp.contracts.identity import CandidateId, bank_rows
from apsp.contracts.ledger import DELTA, CostTableError, MethodCosts, schedule_cost
from apsp.contracts.policy_api import Cut, ScheduleSpec
from apsp.sim.schedules import enumerate_pairs

RIDGE_ALPHAS = tuple(np.logspace(-3, 3, 13).tolist())
N_RAW = 10


def b10_spec(rows: Sequence[CandidateId], j1: int, j2: int, J: int = 16) -> ScheduleSpec:
    k = len(rows)
    return ScheduleSpec(candidates=tuple(rows), queries=(), cuts=(Cut(j1, k // 2, "pred"), Cut(j2, k // 4, "pred")),
                        final_k=k // 4, J=J)


def b10_cost(j1: int, j2: int, mc: MethodCosts, delta: int = DELTA) -> int:
    return schedule_cost(b10_spec(bank_rows(0), j1, j2), mc, delta)


def features(raw: np.ndarray) -> np.ndarray:
    """raw [k, 10] -> [k, 20] (raw + deviation from the alive-group mean); NaN kept for the caller to fill."""
    raw = np.asarray(raw, np.float64)
    with warnings.catch_warnings():                      # all-NaN columns (drift at j=1) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(np.where(np.isfinite(raw), raw, np.nan), axis=0) if len(raw) else np.zeros(N_RAW)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return np.concatenate([raw, raw - mu], axis=1)


@dataclass
class Ridge:
    W: np.ndarray            # [21]
    mu: np.ndarray           # [20]
    sd: np.ndarray           # [20]
    alpha: float
    cv: dict = field(default_factory=dict)

    def predict(self, F: np.ndarray) -> np.ndarray:
        Z = (np.asarray(F, np.float64) - self.mu) / self.sd
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.concatenate([Z, np.ones((len(Z), 1))], 1) @ self.W

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "alpha": self.alpha, "cv": self.cv}

    @staticmethod
    def from_dict(d: dict) -> "Ridge":
        return Ridge(np.asarray(d["W"]), np.asarray(d["mu"]), np.asarray(d["sd"]), float(d["alpha"]), d.get("cv", {}))


def _fit(F: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(F, 0), np.nanstd(F, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Z = np.where(np.isfinite((F - mu) / sd), (F - mu) / sd, 0.0)
    Xa = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    reg = alpha * np.eye(Xa.shape[1])
    reg[-1, -1] = 0.0
    return Ridge(np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y), mu, sd, float(alpha))


def _folds(groups: Sequence[str], n_folds: int) -> np.ndarray:
    ug = sorted(set(groups), key=lambda g: sha256_hex(g))
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def fit_checkpoint(F: np.ndarray, y: np.ndarray, groups: Sequence[str], n_folds: int = 5,
                   alphas=RIDGE_ALPHAS) -> tuple[Ridge, np.ndarray]:
    """Ridge for one checkpoint with group CV (standardisation inside folds). Returns (model on all rows, OOF preds)."""
    folds = _folds(groups, n_folds)
    nf = len(set(folds.tolist()))
    cv, oof_by = {}, {}
    for a in alphas:
        oof = np.full(len(y), np.nan)
        for f in range(nf):
            tr, te = folds != f, folds == f
            if tr.any() and te.any():
                oof[te] = _fit(F[tr], y[tr], a).predict(F[te])
        cv[f"{a:.6g}"] = float(np.nanmean((oof - y) ** 2))
        oof_by[a] = oof
    best = min(alphas, key=lambda a: (cv[f"{a:.6g}"], -a))
    m = _fit(F, y, best)
    m.cv = {"mse_by_alpha": cv, "n_folds": n_folds, "folds_by": "prompt", "n_rows": int(len(y))}
    return m, oof_by[best]


@dataclass
class LatentPredictor:
    j1: int
    j2: int
    models: dict             # j -> Ridge
    pred_us: dict            # k -> measured predictor time (us)
    info: dict = field(default_factory=dict)

    def score(self, j: int, raw: dict) -> dict:
        cids = list(raw)
        p = self.models[j].predict(features(np.array([raw[c] for c in cids])))
        return {c: float(v) for c, v in zip(cids, p)}

    def to_dict(self) -> dict:
        return {"j1": self.j1, "j2": self.j2, "models": {str(j): m.to_dict() for j, m in self.models.items()},
                "pred_us": {str(k): int(v) for k, v in self.pred_us.items()}, "info": self.info}

    @staticmethod
    def from_dict(d: dict) -> "LatentPredictor":
        return LatentPredictor(int(d["j1"]), int(d["j2"]), {int(j): Ridge.from_dict(m) for j, m in d["models"].items()},
                               {int(k): int(v) for k, v in d["pred_us"].items()}, d.get("info", {}))

    @property
    def cph(self) -> str:
        return "B10-" + sha256_hex(self.to_dict())[:16]


def run_b10(pr: LatentPredictor, engine, *, rows: Sequence[CandidateId], mc: MethodCosts, cap_us: int, C_ref: int = 1,
            method_id: str = "b10", delta: int = DELTA):
    """Execute B10-min through the shared schedule runner (pred cuts; ledger checks spend == schedule_cost)."""
    from apsp.sim.env import run_schedule
    m = mc.with_overrides(pred={**mc.pred, **pr.pred_us}, method_id="b10")
    return run_schedule(b10_spec(rows, pr.j1, pr.j2), engine, mc=m, cap_us=cap_us, C_ref=C_ref, method_id=method_id,
                        delta=delta, predictor=lambda st: pr.score(st.j, engine.lfeat(st)))


def measure_pred_us(pr_models: dict, ks=(8, 4), n: int = 200, clock: Callable[[], float] = time.perf_counter) -> dict:
    """Wall time (us, ceil of the mean over n calls) of features + predict for k candidates on this machine."""
    out = {}
    rng = np.random.default_rng(0)
    m = next(iter(pr_models.values()))
    for k in ks:
        raw = rng.normal(size=(k, N_RAW))
        m.predict(features(raw))
        t0 = clock()
        for _ in range(n):
            m.predict(features(raw))
        out[k] = int(np.ceil((clock() - t0) / n * 1e6))
    return out


# ---------------------------------------------------------------- training on the bank (replay) -------------------
def _bank_tables(ctx, keys: Sequence) -> dict:
    """Privileged training tables per instance: lfeat [8, J-1, 10], preview validity unused, delivered IR [8]."""
    b = ctx.bank
    out = {}
    for k in keys:
        arr = ctx.accessor.instance_arrays(k, "training_label")
        raw = np.concatenate([b.drift[k][..., None], b.cfg_gap[k][..., None], b.chan[k]], -1).astype(np.float64)
        D = np.asarray(arr["delivered_ir"], np.float64)
        dv = np.asarray(arr["delivered_flags"], bool)[:, 2] & np.isfinite(D)
        out[k] = (raw, np.where(dv, D, np.nan))
    return out


def _replay_pair(tab: dict, keys: Sequence, pred: Callable[[object, int, np.ndarray], np.ndarray], j1: int, j2: int,
                 r_fail: float) -> np.ndarray:
    """Y per instance for the pred schedule (j1, j2): rank by predicted score (ties by CandidateId), terminal RM."""
    ys = []
    for k in keys:
        raw, D = tab[k]
        alive = np.arange(8)
        for j, keep in ((j1, 4), (j2, 2)):
            s = pred(k, j, alive)
            order = np.lexsort((alive, -np.where(np.isfinite(s), s, -np.inf)))
            alive = np.sort(alive[order[:keep]])
        d = D[alive]
        ys.append(float(np.nanmax(d)) if np.isfinite(d).any() else r_fail)
    return np.asarray(ys)


def tune_b10(ctx, fit: Sequence, dev: Sequence, cap_us: int, mc: MethodCosts, top: int = 5, J: int = 16) -> Optional[dict]:
    """Per-checkpoint predictors on train_fit, pair shortlist by OOF replay on train_fit, choice on train_dev."""
    tf, td = _bank_tables(ctx, fit), _bank_tables(ctx, dev)
    models, oof = {}, {}
    for j in range(1, J):
        F, y, g, idx = [], [], [], []
        for k in fit:
            raw, D = tf[k]
            Fk = features(raw[:, j - 1, :])
            for s in range(8):
                if np.isfinite(D[s]):
                    F.append(Fk[s]); y.append(D[s]); g.append(k[0]); idx.append((k, s))
        F, y = np.asarray(F), np.asarray(y)
        models[j], o = fit_checkpoint(F, y, g)
        oof[j] = {ix: v for ix, v in zip(idx, o)}
    # NOTE: features use the alive-group mean, so OOF scores on the full 8 are an approximation at the second cut
    pred_us = measure_pred_us(models)
    m_ledger = mc.with_overrides(pred={**mc.pred, **pred_us}, method_id="b10")
    pairs, costs = [], []
    for j1, j2 in enumerate_pairs(J):
        try:
            c = b10_cost(j1, j2, m_ledger)
        except CostTableError:
            continue
        if c <= cap_us:
            pairs.append((j1, j2)); costs.append(c)
    if not pairs:
        return None

    def oof_pred(k, j, alive):
        return np.array([oof[j].get((k, int(s)), np.nan) for s in alive])

    def full_pred(tab):
        def f(k, j, alive):
            raw = tab[k][0][alive, j - 1, :]
            return models[j].predict(features(raw))
        return f

    fit_means = {p: float(_replay_pair(tf, fit, oof_pred, *p, ctx.r_fail).mean()) for p in pairs}
    cost_of = dict(zip(pairs, costs))
    short = sorted(pairs, key=lambda p: (-fit_means[p], cost_of[p], p))[:top]
    dev_means = {p: float(_replay_pair(td, dev, full_pred(td), *p, ctx.r_fail).mean()) for p in short}
    j1, j2 = sorted(short, key=lambda p: (-dev_means[p], cost_of[p], p))[0]
    pr = LatentPredictor(j1, j2, {j1: models[j1], j2: models[j2]}, pred_us,
                         {"fit_mean_oof": fit_means[(j1, j2)], "dev_mean": dev_means[(j1, j2)], "cost_us": cost_of[(j1, j2)],
                          "n_affordable": len(pairs), "shortlist": [{"pair": list(p), "fit_mean_oof": fit_means[p],
                                                                     "dev_mean": dev_means[p]} for p in short],
                          "selected_on": "train_dev", "pred_us_provisional": True,
                          "features": "drift, cfg_gap, mean_c0..3, std_c0..3, + deviation from alive mean"})
    return {"predictor": pr.to_dict(), "j1": j1, "j2": j2, "dev_mean": dev_means[(j1, j2)], "cost_us": cost_of[(j1, j2)]}
