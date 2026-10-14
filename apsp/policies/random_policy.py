"""B9: uniform random action over the feasible mask (sanity baseline, reproducible eval seed)."""
from __future__ import annotations

import numpy as np


class RandomFeasiblePolicy:
    grammar_id = "G-842-v1"
    uses_traj = False
    method_id = "B9-random"
    cph = "B9-random-v1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, prompt_feat, cap_us):
        pass

    def act(self, obs, mask):
        return int(mask[int(self.rng.integers(len(mask)))])
