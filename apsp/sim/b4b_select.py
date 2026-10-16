"""B4b selection: best PSP-grid strategy per cap (train_fit shortlist, train_dev choice) - CPU, replay only.

The grid is `apsp.sim.b4b_grid.enumerate_grid()` (7,823 strategies of the pinned PSP SD1.5 search). A strategy is
a candidate under cap B only if
  - every primitive it needs (den/ver/fin at each k, prune for each consecutive keep pair) is in the cost table
    (otherwise it is "uncovered": reported, never priced by interpolation), and
  - its full-trace ledger cost (`schedule_cost` of `to_spec`, a paid VERIFY before every cut) is <= B.
Episodes are per prompt on pool 16 (both banks; k_init <= 8 uses bank 0 only, PSP's seed window): one episode per
prompt, never two.

`grid_outcomes` evaluates every strategy on every prompt with numpy and reproduces `run_schedule` on the
ReplayEngine exactly (ranking: valid preview desc, ties by CandidateId, invalid last; terminal: RM among k>1
survivors or the F(1) single survivor; failures -> r_fail). tests/sim/test_b4b_select.py checks this against
`ReplayContext.run_spec`.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from apsp.contracts.hashing import sha256_hex
from apsp.contracts.ledger import CostTableError, MethodCosts, schedule_cost
from apsp.sim.b4b_grid import GridStrategy, enumerate_grid, to_spec

NEG = -1e30


@lru_cache(maxsize=1)
def sd15_grid() -> tuple[GridStrategy, ...]:
    return tuple(enumerate_grid())


def grid_hash() -> str:
    return sha256_hex([asdict(s) for s in sd15_grid()])[:16]


def strategy_costs(mc: MethodCosts, grid: Sequence[GridStrategy] = None) -> tuple[np.ndarray, dict]:
    """C[n] full-trace cost (us) or -1 when a primitive is missing; plus the missing primitives with counts."""
    grid = sd15_grid() if grid is None else grid
    C = np.full(len(grid), -1, dtype=np.int64)
    missing: dict[str, int] = {}
    for i, s in enumerate(grid):
        try:
            C[i] = schedule_cost(to_spec(s), mc)
        except CostTableError as e:
            key = str(e).split(" for method")[0].replace("missing primitive ", "")
            missing[key] = missing.get(key, 0) + 1
    return C, missing


def pool_arrays(accessor, uid: str) -> dict:
    """Pool-16 arrays of one prompt (oracle purpose: hindsight evaluation of fixed schedules)."""
    a0 = accessor.instance_arrays((uid, 0), "oracle")
    a1 = accessor.instance_arrays((uid, 1), "oracle")
    P = np.concatenate([a0["preview_ir"], a1["preview_ir"]]).astype(np.float64)        # [16, J-1]
    pf = np.concatenate([a0["preview_flags"], a1["preview_flags"]]).astype(bool)        # [16, J-1, 3]
    D = np.concatenate([a0["delivered_ir"], a1["delivered_ir"]]).astype(np.float64)     # [16]
    df = np.concatenate([a0["delivered_flags"], a1["delivered_flags"]]).astype(bool)    # [16, 3]
    pv = pf[..., 2] & np.isfinite(P)
    dv = df[:, 2] & np.isfinite(D)
    return {"P": np.where(pv, P, NEG), "pvalid": pv, "D": np.where(dv, D, NEG), "dvalid": dv, "img_ok": df[:, 1] & df[:, 0]}


def grid_outcomes(arrays: Sequence[dict], r_fail: float, grid: Sequence[GridStrategy] = None,
                  delta: int = 4) -> np.ndarray:
    """Y[N prompts, n strategies]: delivered IR of the winner, or r_fail."""
    grid = sd15_grid() if grid is None else grid
    N = len(arrays)
    P = np.stack([a["P"] for a in arrays])          # [N,16,J-1]
    pv = np.stack([a["pvalid"] for a in arrays])
    D = np.stack([a["D"] for a in arrays])          # [N,16]
    dv = np.stack([a["dvalid"] for a in arrays])
    img = np.stack([a["img_ok"] for a in arrays])
    Y = np.empty((N, len(grid)), dtype=np.float64)
    idx = np.arange(16)
    for n, s in enumerate(grid):
        alive = np.broadcast_to(idx < s.k_init, (N, 16)).copy()
        for t, keep in zip(s.cutoff_times, s.remaining_particles):
            j = t // delta
            sc = np.where(alive, np.where(pv[:, :, j - 1], P[:, :, j - 1], NEG), -np.inf)
            order = np.argsort(-sc, axis=1, kind="stable")       # desc score, ties by CandidateId (row) asc
            new = np.zeros_like(alive)
            np.put_along_axis(new, order[:, :keep], True, axis=1)
            alive = new
        if s.remaining_particles[-1] == 1:
            ok = (alive & dv & img).any(axis=1)
            y = np.where(alive, D, NEG).max(axis=1)
            Y[:, n] = np.where(ok, y, r_fail)
        else:
            ok = (alive & dv).any(axis=1)
            Y[:, n] = np.where(ok, np.where(alive & dv, D, NEG).max(axis=1), r_fail)
    return Y


def select_b4b_dev(Y_fit: np.ndarray, Y_dev: np.ndarray, C: np.ndarray, cap_us: int, top: int = 5,
                   grid: Sequence[GridStrategy] = None) -> Optional[dict]:
    """Shortlist the top-``top`` affordable, covered strategies on train_fit; choose on train_dev (as B3)."""
    grid = sd15_grid() if grid is None else grid
    ok = np.where((C >= 0) & (C <= cap_us))[0]
    if ok.size == 0:
        return None
    fm = Y_fit[:, ok].mean(axis=0)
    short = sorted(range(ok.size), key=lambda i: (-fm[i], int(C[ok[i]]), int(ok[i])))[:top]
    dm = {i: float(Y_dev[:, ok[i]].mean()) for i in short}
    best = sorted(short, key=lambda i: (-dm[i], int(C[ok[i]]), int(ok[i])))[0]
    g = int(ok[best])
    return {"grid_index": g, "strategy": asdict(grid[g]), "fit_mean": float(fm[best]), "dev_mean": dm[best],
            "cost_us": int(C[g]), "n_affordable": int(ok.size), "n_uncovered": int((C < 0).sum()),
            "shortlist": [{"grid_index": int(ok[i]), "fit_mean": float(fm[i]), "dev_mean": dm[i],
                           "cost_us": int(C[ok[i]])} for i in short],
            "grid_hash": grid_hash(), "name": f"b4b:{g}"}


def prompts_with_pool(keys: Sequence) -> list[str]:
    """Prompts whose two banks are both present (pool 16), in first-seen order."""
    banks: dict[str, set] = {}
    for uid, b in keys:
        banks.setdefault(uid, set()).add(b)
    return [u for u, bs in banks.items() if {0, 1} <= bs]
