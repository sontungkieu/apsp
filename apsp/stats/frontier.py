"""Cost-quality frontier (plotting only) and the RQ2 operating-point rule (Spec §10.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Point:
    name: str
    cost_e2e: float
    quality: float
    qualified: bool = True


def pareto(points: Sequence[Point]) -> list[Point]:
    """Non-dominated points sorted by cost. Connecting lines are for drawing only (no interpolation claims)."""
    pts = sorted(points, key=lambda p: (p.cost_e2e, -p.quality, p.name))
    out, best = [], float("-inf")
    for p in pts:
        if p.quality > best:
            out.append(p)
            best = p.quality
    return out


def operating_point(points: Sequence[Point]) -> Point | None:
    """Among quality-qualified feasible points on train_dev: lowest mean ACTUAL e2e cost; tie -> higher quality, name."""
    q = [p for p in points if p.qualified]
    if not q:
        return None
    return sorted(q, key=lambda p: (p.cost_e2e, -p.quality, p.name))[0]
