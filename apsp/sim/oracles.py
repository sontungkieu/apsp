"""Hindsight oracles (Spec §8 Mệnh đề 3/4, §10.4; Guide CX-6). Privileged: purpose ``oracle``.

O1   max over affordable fixed schedules (same paid-prune grammar, fixed ranking). Upper bound ONLY
     for methods in the identical grammar - not for B10, pool16, resampling or another model.
O1-free  verifier cost set to zero (optimistic diagnostic, not executable).
O3   best of the 8 delivered images (outer diagnostic, budget not implied).
O2-inst  per-instance cost-to-target (diagnostic; NOT a lower bound for the mean-quality problem).
O2-avg   two distinct outputs: Lagrangian dual lower bound (numeric) and a primal feasible witness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def o1(Y: np.ndarray, C: np.ndarray, cap_us: int) -> np.ndarray:
    """Per-instance O1 values; NaN where no schedule is affordable."""
    ok = C <= cap_us
    if not ok.any():
        return np.full(Y.shape[0], np.nan)
    return Y[:, ok].max(axis=1)


def o3(delivered_ir: np.ndarray, r_fail: float) -> float:
    v = delivered_ir[np.isfinite(delivered_ir)]
    return float(v.max()) if v.size else float(r_fail)


def o2_inst(Y: np.ndarray, C: np.ndarray, y_ref: np.ndarray, delta_ir: float) -> np.ndarray:
    """Cheapest schedule per instance reaching Y >= y_ref - delta (inf when unreachable). Diagnostic only."""
    out = np.full(Y.shape[0], np.inf)
    for i in range(Y.shape[0]):
        ok = Y[i] >= y_ref[i] - delta_ir
        if ok.any():
            out[i] = C[ok].min()
    return out


@dataclass
class DualResult:
    dual_lower_bound_total: float
    lam: float
    status: str               # "numeric_search" - never claimed exact
    evaluations: int
    target_feasible: bool


@dataclass
class PrimalResult:
    primal_feasible_assignment: Optional[np.ndarray]   # schedule index per instance
    verified: bool
    primal_cost_total: Optional[float]
    status: str
    gap_to_dual: Optional[float] = None
    details: dict = field(default_factory=dict)


def _D(Y, y_ref, delta):
    return Y - y_ref[:, None] + delta


def _lagrangian(lam: float, C: np.ndarray, D: np.ndarray, feasible_mask: np.ndarray) -> float:
    val = np.where(feasible_mask, C[None, :] - lam * D, np.inf)
    return float(val.min(axis=1).sum())


def o2_avg_dual(Y: np.ndarray, C: np.ndarray, y_ref: np.ndarray, delta_ir: float,
                feasible_mask: Optional[np.ndarray] = None, n_grid: int = 200, refine: int = 60) -> DualResult:
    """max_{lam>=0} LB(lam) by grid + golden-section on a concave piecewise-linear function.

    Target infeasibility is checked first (sum of best D < 0 -> primal infeasible, dual unbounded).
    """
    N, S = Y.shape
    fm = np.ones((N, S), bool) if feasible_mask is None else feasible_mask
    D = _D(Y, y_ref, delta_ir)
    best_D = np.where(fm, D, -np.inf).max(axis=1)
    if best_D.sum() < 0:
        return DualResult(math.inf, math.inf, "target_infeasible", 0, False)
    Cs = C.astype(np.float64)
    Dspan = float(np.abs(np.where(fm, D, 0)).max()) or 1.0
    lam_hi = 10.0 * float(Cs.max() - Cs.min() + 1.0) / max(Dspan, 1e-12) * 10
    grid = np.concatenate([[0.0], np.geomspace(1e-9 * lam_hi + 1e-12, lam_hi, n_grid)])
    vals = np.array([_lagrangian(l, Cs, D, fm) for l in grid])
    i = int(vals.argmax())
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _lagrangian(c, Cs, D, fm), _lagrangian(d, Cs, D, fm)
    for _ in range(refine):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _lagrangian(d, Cs, D, fm)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _lagrangian(c, Cs, D, fm)
    cands = [(vals[i], grid[i]), (fc, c), (fd, d)]
    best_val, best_lam = max(cands)
    return DualResult(float(best_val), float(best_lam), "numeric_search", len(grid) + refine + 2, True)


def verify_primal(assign: np.ndarray, Y: np.ndarray, C: np.ndarray, y_ref: np.ndarray, delta_ir: float,
                  feasible_mask: Optional[np.ndarray] = None) -> tuple[bool, float, float]:
    """Re-evaluate a witness from the original table: (constraint holds, total cost, sum D)."""
    idx = np.arange(Y.shape[0])
    if feasible_mask is not None and not feasible_mask[idx, assign].all():
        return False, math.inf, -math.inf
    sD = float((Y[idx, assign] - y_ref + delta_ir).sum())
    return sD >= -1e-9, float(C[assign].sum()), sD


def o2_avg_primal(Y: np.ndarray, C: np.ndarray, y_ref: np.ndarray, delta_ir: float,
                  feasible_mask: Optional[np.ndarray] = None, time_limit_s: float = 60.0) -> PrimalResult:
    """Primal witness via MILP (scipy.optimize.milp) with a greedy fallback; always re-verified."""
    N, S = Y.shape
    fm = np.ones((N, S), bool) if feasible_mask is None else feasible_mask
    D = _D(Y, y_ref, delta_ir)
    if np.where(fm, D, -np.inf).max(axis=1).sum() < 0:
        return PrimalResult(None, False, None, "target_infeasible")
    assign, status = None, "none"
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
        c = np.tile(C.astype(np.float64), N)
        A_eq = lil_matrix((N, N * S))
        for i in range(N):
            A_eq[i, i * S:(i + 1) * S] = 1
        A_q = np.where(fm, D, 0).reshape(1, -1)
        ub = fm.reshape(-1).astype(float)
        r = milp(c, integrality=np.ones(N * S), bounds=Bounds(np.zeros(N * S), ub),
                 constraints=[LinearConstraint(A_eq.tocsr(), 1, 1), LinearConstraint(A_q, 0, np.inf)],
                 options={"time_limit": time_limit_s})
        if r.x is not None:
            assign = r.x.reshape(N, S).argmax(axis=1)
            status = "milp_optimal" if r.status == 0 else f"milp_status_{r.status}"
    except Exception as e:  # noqa: BLE001
        status = f"milp_unavailable:{type(e).__name__}"
    if assign is None:
        # greedy: start from cheapest, upgrade by best quality-per-cost until the mean constraint holds
        assign = np.array([int(np.where(fm[i], C, np.iinfo(np.int64).max).argmin()) for i in range(N)])
        sD = D[np.arange(N), assign].sum()
        while sD < 0:
            best = None
            for i in range(N):
                for s in range(S):
                    if not fm[i, s]:
                        continue
                    gain = D[i, s] - D[i, assign[i]]
                    dc = C[s] - C[assign[i]]
                    if gain > 0:
                        ratio = gain / max(dc, 1e-9) if dc > 0 else math.inf
                        if best is None or ratio > best[0]:
                            best = (ratio, i, s, gain)
            if best is None:
                break
            _, i, s, gain = best
            assign[i] = s
            sD += gain
        status = "greedy_witness" if status == "none" else status + "+greedy_witness"
    ok, cost, sD = verify_primal(assign, Y, C, y_ref, delta_ir, fm)
    return PrimalResult(assign if ok else None, ok, cost if ok else None, status, details={"sum_D": sD})
