"""B6 threshold controller with six parameters (Spec §9.2, Guide CX-5).

tau_s(j) = a_s + b_s * j / 16 for stages s in {8, 4}; shared j_start in 1..8 and verify cadence m in {1,2,4}.
Action rule:
- forced (single allowed action) -> that action;
- fresh & unpruned & PRUNE allowed: PRUNE iff z_cut >= tau_s(j); else CONT if allowed, otherwise PRUNE;
- not fresh: VERIFY at cadence (j >= j_start, (j - j_start) % m == 0) when allowed; else CONT if allowed;
- if CONT would destroy completion (not in mask), follow the mask-forced VERIFY/PRUNE.
Never chooses outside the mask.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from apsp.contracts.hashing import sha256_hex
from apsp.contracts.policy_api import CONT, PRUNE, VER
from apsp.policies.fixed import obs_jk
from apsp.sim.features import IDX


@dataclass(frozen=True)
class ThresholdParams:
    a8: float
    b8: float
    a4: float
    b4: float
    j_start: int
    m: int

    def tau(self, k: int, j: int) -> float:
        a, b = (self.a8, self.b8) if k == 8 else (self.a4, self.b4)
        return a + b * j / 16.0


class ThresholdPolicy:
    grammar_id = "G-842-v1"
    uses_traj = False

    def __init__(self, params: ThresholdParams, method_id: str = "B6-threshold"):
        self.p = params
        self.method_id = method_id
        self.cph = "B6-" + sha256_hex(asdict(params))[:16]

    def reset(self, prompt_feat, cap_us):
        pass

    def act(self, obs, mask):
        if len(mask) == 1:
            return mask[0]
        j, k, fresh, pruned = obs_jk(obs)
        if fresh and not pruned and PRUNE in mask:
            z_cut = float(obs[IDX["z_cut"]])
            if z_cut >= self.p.tau(k, j):
                return PRUNE
            return CONT if CONT in mask else PRUNE
        if not fresh and VER in mask and j >= self.p.j_start and (j - self.p.j_start) % self.p.m == 0:
            return VER
        if CONT in mask:
            return CONT
        return VER if VER in mask else PRUNE


def sample_params(rng: np.random.Generator) -> ThresholdParams:
    return ThresholdParams(*rng.uniform(-3, 3, 4).tolist(), int(rng.integers(1, 9)), int(rng.choice([1, 2, 4])))


def tune_threshold(evaluate: Callable[[ThresholdParams, str], float], *, n_evals: int = 2000, top: int = 10,
                   seed: int = 0, fit_split: str = "train_fit", dev_split: str = "train_dev") -> dict:
    """Random search on ``fit_split`` (<= n_evals objective evaluations, cap <= 2000), select on ``dev_split``.

    ``evaluate(params, split)`` returns mean delivered Y under the cap of interest. All trials are logged.
    """
    if n_evals > 2000:
        raise ValueError("B6 tuning budget is <= 2000 objective evaluations per cap")
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_evals):
        p = sample_params(rng)
        trials.append((evaluate(p, fit_split), p))
    trials.sort(key=lambda t: (-t[0], json.dumps(asdict(t[1]), sort_keys=True)))
    short = trials[:top]
    dev = [(evaluate(p, dev_split), fit, p) for fit, p in short]
    dev.sort(key=lambda t: (-t[0], -t[1], json.dumps(asdict(t[2]), sort_keys=True)))
    best_dev, best_fit, best = dev[0]
    return {"params": asdict(best), "fit_mean": best_fit, "dev_mean": best_dev, "n_fit_evals": n_evals,
            "n_dev_evals": len(short), "trials": [{"fit": f, "params": asdict(p)} for f, p in trials]}
