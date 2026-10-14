"""Planning approximation for paired contrasts (Spec §11.2). Variances come from train, never val."""
from __future__ import annotations

import math

from scipy.stats import norm


def se_paired(sigma_p2: float, sigma_b2: float, P: int, n_b: int = 2) -> float:
    return math.sqrt(sigma_p2 / P + sigma_b2 / (P * n_b))


def mde(sigma_p2: float, sigma_b2: float, P: int, M: int, n_b: int = 2, power: float = 0.8) -> float:
    a_eff = 0.025 / M
    return (norm.ppf(1 - a_eff / 2) + norm.ppf(power)) * se_paired(sigma_p2, sigma_b2, P, n_b)


def p_needed(sigma_p2: float, sigma_b2: float, target_mde: float, M: int, n_b: int = 2, power: float = 0.8,
             p_max: int = 100_000) -> int | None:
    for P in range(2, p_max):
        if mde(sigma_p2, sigma_b2, P, M, n_b, power) <= target_mde:
            return P
    return None
