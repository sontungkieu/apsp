"""Nested variance components (prompt / bank-within-prompt) of a paired difference, for M12 planning."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def nested_components(per_prompt: Mapping[str, Sequence[float]]) -> dict:
    """One-way random-effects ANOVA with equal banks per prompt: returns sigma_p^2, sigma_b^2 (clipped at 0)."""
    groups = [np.asarray(v, float) for _, v in sorted(per_prompt.items())]
    nb = {len(g) for g in groups}
    if len(nb) != 1:
        raise ValueError("equal number of banks per prompt required")
    n_b = nb.pop()
    P = len(groups)
    means = np.array([g.mean() for g in groups])
    grand = means.mean()
    msb = n_b * ((means - grand) ** 2).sum() / max(P - 1, 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(P * (n_b - 1), 1)
    sigma_b2 = float(msw)
    sigma_p2 = float(max((msb - msw) / n_b, 0.0))
    return {"sigma_p2": sigma_p2, "sigma_b2": sigma_b2, "P": P, "n_b": n_b}
