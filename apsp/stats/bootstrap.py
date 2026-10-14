"""Cluster (prompt) bootstrap for paired contrasts (Spec §11.1).

Prompt is the cluster; banks are nested within prompt and resampled together; methods stay paired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def required_resamples(level: float, min_tail_draws: int = 100, floor: int = 10_000) -> int:
    """>= 10,000 and enough that each tail holds ~``min_tail_draws`` expected draws at the adjusted level."""
    tail = (1.0 - level) / 2.0
    return max(floor, int(math.ceil(min_tail_draws / tail)))


def contrast_level(M: int, alpha_look: float = 0.025) -> float:
    """Per-contrast per-look CI level 1 - 0.025/M (union bound over two looks)."""
    if M < 1:
        raise ValueError("M >= 1")
    return 1.0 - alpha_look / M


@dataclass(frozen=True)
class BootCI:
    estimate: float
    lo: float
    hi: float
    level: float
    n_resamples: int
    seed: int
    n_clusters: int
    mc_sensitivity: float  # |difference| between the bounds of two half-size bootstrap runs (Monte Carlo noise)


def cluster_bootstrap(per_prompt: Mapping[str, Sequence[float]], level: float, *, seed: int = 0,
                      n_resamples: int | None = None, stat=np.mean) -> BootCI:
    """Percentile CI of the mean paired difference. ``per_prompt[uid]`` lists bank-level differences of that prompt;
    the prompt mean is the cluster value (banks weighted equally within prompt)."""
    keys = sorted(per_prompt)
    vals = np.array([float(np.mean(per_prompt[k])) for k in keys])
    n = len(vals)
    if n < 2:
        raise ValueError("need >= 2 clusters")
    B = n_resamples or required_resamples(level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = vals[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    h = B // 2
    lo1, hi1 = np.quantile(boots[:h], [a, 1 - a])
    lo2, hi2 = np.quantile(boots[h:], [a, 1 - a])
    return BootCI(float(stat(vals)), float(lo), float(hi), level, B, seed, n,
                  float(max(abs(lo1 - lo2), abs(hi1 - hi2))))
