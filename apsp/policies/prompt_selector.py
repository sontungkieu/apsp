"""B7: prompt-only selector of a fixed pair (Spec §9.2 B7). No generation-time information is used.

Before generation, from the prompt feature only (PCA-16 of the CLIP pooler output, the same text encoding the request
already computes), a multi-output ridge predicts the delivered Y of every affordable fixed pair (j1, j2) of G-842 and
the argmax is executed as a fixed schedule (8->4 at j1, 4->2 at j2, paid VERIFY at both, terminal RM).

Training is on train_fit only: labels are the replayed Y of each pair on each instance (privileged training labels).
Regularisation is chosen by group 5-fold CV (folds by prompt, banks of a prompt together) on the REALISED mean Y of
the chosen pair, with the PCA-16 and the standardisation refitted inside every fold. The three best alphas are then
compared on train_dev with the model refitted on all of train_fit; val/test are never read.
The final PCA is fitted on the train_fit prompts, like the shared ReplayContext PCA (checked by hash at load).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from apsp.contracts.hashing import sha256_hex
from apsp.sim.features import PCA16
from apsp.sim.schedules import enumerate_pairs

RIDGE_ALPHAS = tuple(np.logspace(-2, 4, 13).tolist())


def _ridge(X, Y, alpha):
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    reg = alpha * np.eye(Xa.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Y)


@dataclass
class PromptSelector:
    pairs: list               # affordable pairs under the cap, output order
    costs: list
    W: np.ndarray             # [17, n_pairs]
    mu: np.ndarray
    sd: np.ndarray
    alpha: float
    pca_hash: str
    info: dict = field(default_factory=dict)

    def predict(self, prompt_feat: np.ndarray) -> np.ndarray:
        z = (np.asarray(prompt_feat, np.float64).ravel()[:16] - self.mu) / self.sd
        return np.concatenate([z, [1.0]]) @ self.W

    def choose(self, prompt_feat: np.ndarray) -> tuple[int, int]:
        p = self.predict(prompt_feat)
        i = sorted(range(len(self.pairs)), key=lambda i: (-p[i], self.costs[i], tuple(self.pairs[i])))[0]
        return tuple(self.pairs[i])

    def to_dict(self) -> dict:
        return {"pairs": [list(p) for p in self.pairs], "costs": list(map(int, self.costs)), "W": self.W.tolist(),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "alpha": self.alpha, "pca_hash": self.pca_hash,
                "info": self.info}

    @staticmethod
    def from_dict(d: dict) -> "PromptSelector":
        return PromptSelector([tuple(p) for p in d["pairs"]], list(d["costs"]), np.asarray(d["W"]), np.asarray(d["mu"]),
                              np.asarray(d["sd"]), float(d["alpha"]), d["pca_hash"], d.get("info", {}))

    @property
    def cph(self) -> str:
        return "B7-" + sha256_hex(self.to_dict())[:16]


def _fit(E: np.ndarray, Y: np.ndarray, alpha: float, pca: Optional[PCA16] = None):
    pca = pca or PCA16.fit(E)
    X = np.stack([pca.transform(e) for e in E]).astype(np.float64)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    return pca, mu, sd, _ridge((X - mu) / sd, Y, alpha)


def _realised(pca, mu, sd, W, E, Y) -> float:
    X = (np.stack([pca.transform(e) for e in E]).astype(np.float64) - mu) / sd
    P = np.concatenate([X, np.ones((len(X), 1))], 1) @ W
    return float(Y[np.arange(len(Y)), P.argmax(1)].mean())


def tune_b7(ctx, fit: Sequence, dev: Sequence, cap_us: int, n_folds: int = 5, J: int = 16) -> Optional[dict]:
    """Fit on train_fit, CV per fold (PCA + standardisation inside the fold), choose alpha among the top 3 on train_dev."""
    Yf, C = ctx.schedule_table(fit)
    Yd, _ = ctx.schedule_table(dev)
    pairs_all = enumerate_pairs(J)
    ok = np.where(C <= cap_us)[0]
    if ok.size == 0:
        return None
    Yf, Yd = Yf[:, ok], Yd[:, ok]
    Ef = np.stack([ctx.accessor.prompt_embedding(u) for u, _ in fit]).astype(np.float64)
    Ed = np.stack([ctx.accessor.prompt_embedding(u) for u, _ in dev]).astype(np.float64)
    groups = [u for u, _ in fit]
    ug = sorted(set(groups), key=lambda g: sha256_hex(g))
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    folds = np.array([fold_of[g] for g in groups])
    cv = {}
    for a in RIDGE_ALPHAS:
        vals = []
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            if not te.any() or len({groups[i] for i in np.flatnonzero(tr)}) < 16:
                continue
            uniq_tr = {}
            for i in np.flatnonzero(tr):
                uniq_tr.setdefault(groups[i], Ef[i])
            pca = PCA16.fit(np.stack(list(uniq_tr.values())))                 # prompt PCA inside the fold
            _, mu, sd, W = _fit(Ef[tr], Yf[tr], a, pca)
            vals.append((_realised(pca, mu, sd, W, Ef[te], Yf[te]), int(te.sum())))
        cv[a] = float(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else float("-inf")
    top3 = sorted(RIDGE_ALPHAS, key=lambda a: (-cv[a], -a))[:3]
    final_pca = ctx.pca if ctx.pca is not None else PCA16.fit(np.stack(list({u: e for u, e in zip(groups, Ef)}.values())))
    dev_means = {}
    fitted = {}
    for a in top3:
        _, mu, sd, W = _fit(Ef, Yf, a, final_pca)
        fitted[a] = (mu, sd, W)
        dev_means[a] = _realised(final_pca, mu, sd, W, Ed, Yd)
    best = sorted(top3, key=lambda a: (-dev_means[a], -a))[0]
    mu, sd, W = fitted[best]
    sel = PromptSelector([pairs_all[i] for i in ok], [int(C[i]) for i in ok], W, mu, sd, float(best), final_pca.hash(),
                         {"cv_realised_mean_by_alpha": {f"{a:.6g}": v for a, v in cv.items()}, "n_folds": n_folds,
                          "folds_by": "prompt", "fold_preprocessing": "PCA16 + standardisation refitted per fold",
                          "dev_mean_by_alpha": {f"{a:.6g}": v for a, v in dev_means.items()}, "selected_on": "train_dev",
                          "n_affordable": int(ok.size), "cap_us": int(cap_us)})
    return {"selector": sel.to_dict(), "dev_mean": dev_means[best], "alpha": best}
