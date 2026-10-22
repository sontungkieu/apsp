"""Small Core-only baselines. Actor observations exclude unpaid, inactive and future scores."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class View:
    j: int
    alive: tuple
    embedding: object
    features: object
    paid: object


class CoreEpisode:
    def __init__(self, previews, delivered, embedding, *, active='ir', seed=0):
        if active not in ('ir', 'pickscore_raw'):
            raise ValueError('Unknown paid scorer')
        self.__rows = {(int(r['j']), int(r['seed_idx'])): dict(r) for r in previews}
        self.__final = {int(r['seed_idx']): r[active] for r in delivered}
        self.__embedding = tuple(float(v) for v in embedding)
        self.active, self.j, self.alive, self.__paid = active, 0, tuple(range(8)), {}
        self.__last_prune = -1
        self.rng = np.random.default_rng(seed)

    def advance(self):
        if self.j >= 8:
            raise ValueError('Already terminal')
        self.j += 1
        self.__paid = {}

    def view(self):
        fields = ('progress', 'drift', 'drift_available', 'cfg_gap', 'cfg_gap_available')
        features = {} if self.j == 0 else {
            s: MappingProxyType({k: self.__rows[self.j, s][k] for k in fields}) for s in self.alive}
        return View(self.j, self.alive, self.__embedding, MappingProxyType(features),
                    MappingProxyType(dict(self.__paid)))

    def verify(self):
        if not 1 <= self.j <= 7:
            raise ValueError('Preview VERIFY must be nonterminal')
        self.__paid = {s: self.__rows[self.j, s][self.active] for s in self.alive}
        return self.view()

    def prune(self, survivors):
        survivors = tuple(survivors)
        if (self.__last_prune == self.j or not self.__paid or len(self.alive) <= 2 or
                len(survivors) != len(self.alive) // 2 or len(set(survivors)) != len(survivors) or
                not set(survivors) <= set(self.alive)):
            raise ValueError('Fresh paid score and 8→4→2 mask required')
        self.alive = tuple(s for s in self.alive if s in survivors)
        self.__paid = {s: self.__paid[s] for s in self.alive}
        self.__last_prune = self.j

    def final(self):
        if self.j != 8:
            raise ValueError('Terminal scores are hidden until FINAL')
        return {s: self.__final[s] for s in self.alive}


def baselines(core):
    """Only Parquet and 384d embeddings are needed. No latency/savings claims from this replay."""
    core = Path(core)
    from safetensors import safe_open
    previews, delivered = [pd.read_parquet(core / (n + '.parquet')) for n in ('previews', 'delivered')]
    index = json.loads((core / 'prompt_features.index.json').read_text())
    with safe_open(str(core / 'prompt_features.safetensors'), framework='np') as f:
        embeddings = f.get_tensor('embedding')
    results = []
    keys = ['profile_id', 'prompt_uid', 'bank_id']
    for name in ('fixed', 'threshold', 'random_masked'):
        for key, frame in previews.groupby(keys, sort=True):
            mask = np.ones(len(delivered), dtype=bool)
            for column, value in zip(keys, key):
                mask &= delivered[column].eq(value).to_numpy()
            end = delivered[mask]
            env = CoreEpisode(frame.to_dict('records'), end.to_dict('records'), embeddings[index[key[1]]])
            for j in range(1, 9):
                env.advance()
                view = env.view()
                if j in (3, 6) and len(view.alive) > 2:
                    paid = env.verify().paid
                    valid = sorted(paid, key=lambda s: (-(paid[s] if paid[s] is not None and np.isfinite(paid[s]) else -np.inf), s))
                    if name == 'random_masked':
                        survivors = env.rng.choice(view.alive, len(view.alive) // 2, replace=False)
                    else:
                        values = [v for v in paid.values() if v is not None and np.isfinite(v)]
                        if name == 'threshold' and (len(values) < 2 or max(values) - min(values) < .2):
                            continue
                        survivors = valid[:len(view.alive) // 2]
                    env.prune(survivors)
            final = env.final()
            values = [v for v in final.values() if v is not None and np.isfinite(v)]
            results.append({'policy': name, 'profile_id': key[0], 'prompt_uid': key[1], 'bank_id': int(key[2]),
                            'survivors': len(env.alive), 'final_ir': max(values) if values else None})
    return {'pass': True, 'episodes': len(results), 'policies': ['fixed', 'threshold', 'random_masked'],
            'scope': 'Core-only approximate replay; no online equivalence or efficacy claim', 'results': results}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--core', required=True)
    a = p.parse_args()
    print(json.dumps(baselines(a.core), allow_nan=False))
