"""B4b: the PSP scheduled-strategy grid read from the pinned PSP code (Spec §B4b, Guide CX-2 "GridB4b").

Source (PSP @ 590f59f58384169c719e431dc01d14006fb0fc0c):
  results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py
      enumerate_strategies / build_remaining_sequences / compute_total_particle_steps / scheduled_select_prompt_seed
  results_scripts/best_strategies/sd15/sd15_search_best_scheduled_{ir,hps}_{benchmark_ir,geneval}.sh
      SD1.5 arguments: --total-steps 64 --best-of-n 4 --max-cutoffs 4
                       --possible-cutoff-times 4,8,...,60 --possible-remaining-seeds 16,12,8,4,3,2,1

The grid is reproduced exactly (same enumeration order and the PSP particle-step budget filter
total_particle_steps <= best_of_n * total_steps). Mapping to this project:
  - cutoff step t -> checkpoint j = t / delta (delta = 4 forwards); every SD1.5 cutoff is a multiple of 4, so every j
    is in 1..15 and on the bank. A cutoff that is not a multiple of delta raises; it is never rounded.
  - k_init candidates = the first k_init CandidateIds of pool 16 ((0,0)..(0,7),(1,0)..(1,7)), PSP's logical seed 0
    window. k_init <= 16 is within the two banks.
  - ranking at a cutoff: preview IR at j, descending, ties by CandidateId ascending (PSP: seed ascending);
    final: delivered IR among the last survivors (PSP: IR at the final step).
  - costs: this project's ledger (den/ver/prune/fin, plus a paid VERIFY at every cutoff); PSP's own budget filter
    defines the grid only, the caps decide feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from apsp.contracts.identity import CandidateId, pool_rows
from apsp.contracts.policy_api import Cut, ScheduleSpec

PSP_COMMIT = "590f59f58384169c719e431dc01d14006fb0fc0c"
PSP_SOURCE = "results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py"
PSP_SD15_SCRIPTS = tuple(f"results_scripts/best_strategies/sd15/sd15_search_best_scheduled_{m}_{d}.sh"
                         for m in ("ir", "hps") for d in ("benchmark_ir", "geneval"))
SD15_ARGS = {"total_steps": 64, "best_of_n": 4, "max_cutoffs": 4,
             "possible_cutoff_times": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
             "possible_remaining_seeds": [16, 12, 8, 4, 3, 2, 1]}


@dataclass(frozen=True)
class GridStrategy:
    k_init: int
    cutoff_times: tuple[int, ...]
    remaining_particles: tuple[int, ...]
    budget_steps: int
    total_particle_steps: int


def _total_particle_steps(k_init, cutoffs, remaining, total_steps) -> int:
    steps = k_init * cutoffs[0]
    for i in range(len(cutoffs) - 1):
        steps += remaining[i] * (cutoffs[i + 1] - cutoffs[i])
    return int(steps + remaining[-1] * (total_steps - cutoffs[-1]))


def _remaining_sequences(possible: Sequence[int], k_init: int, n_cut: int) -> list[tuple[int, ...]]:
    cand = [k for k in possible if k < k_init]
    out: list[tuple[int, ...]] = []

    def dfs(prefix, last):
        if len(prefix) == n_cut:
            out.append(tuple(prefix))
            return
        for k in cand:
            if k < last:
                prefix.append(k)
                dfs(prefix, k)
                prefix.pop()
    dfs([], k_init)
    return out


def enumerate_grid(total_steps=64, max_cutoffs=4, cutoff_candidates=None, possible_remaining=None,
                   best_of_n=4) -> list[GridStrategy]:
    """Same enumeration (order included) as PSP enumerate_strategies."""
    cutoff_candidates = SD15_ARGS["possible_cutoff_times"] if cutoff_candidates is None else cutoff_candidates
    possible_remaining = SD15_ARGS["possible_remaining_seeds"] if possible_remaining is None else possible_remaining
    budget = best_of_n * total_steps
    valid = sorted(t for t in cutoff_candidates if 0 < t < total_steps)
    out = []
    for k_init in sorted(set(possible_remaining), reverse=True):
        for n_cut in range(1, max_cutoffs + 1):
            rems = _remaining_sequences(possible_remaining, k_init, n_cut)
            if not rems:
                continue
            for ts in combinations(valid, n_cut):
                for rs in rems:
                    tp = _total_particle_steps(k_init, ts, rs, total_steps)
                    if tp <= budget:
                        out.append(GridStrategy(int(k_init), tuple(int(x) for x in ts), tuple(int(x) for x in rs),
                                                int(budget), tp))
    return out


class OffBankCheckpoint(ValueError):
    pass


def to_spec(s: GridStrategy, delta: int = 4, J: int = 16) -> ScheduleSpec:
    """Grid strategy -> ScheduleSpec on pool 16. Raises OffBankCheckpoint instead of rounding."""
    js = []
    for t in s.cutoff_times:
        if t % delta or not 1 <= t // delta <= J - 1:
            raise OffBankCheckpoint(f"cutoff step {t} is not a bank checkpoint (delta={delta}); needs an online path "
                                    "or an auxiliary bank, never rounding")
        js.append(t // delta)
    rows: list[CandidateId] = list(pool_rows(2))[: s.k_init]
    cuts = tuple(Cut(j, k, "paid") for j, k in zip(js, s.remaining_particles))
    return ScheduleSpec(candidates=tuple(rows), queries=tuple(js), cuts=cuts, final_k=s.remaining_particles[-1], J=J)


def required_primitives(grid: Sequence[GridStrategy]) -> dict:
    den, ver, fin, prune = set(), set(), set(), set()
    for s in grid:
        ks = (s.k_init,) + s.remaining_particles
        den.update(ks)
        ver.update(ks[:-1])                 # a paid VERIFY at k before each cut
        fin.add(ks[-1])
        prune.update(zip(ks[:-1], ks[1:]))
    return {"den_k": sorted(den), "ver_k": sorted(ver), "fin_k": sorted(fin),
            "prune_pairs": sorted(prune, reverse=True)}
