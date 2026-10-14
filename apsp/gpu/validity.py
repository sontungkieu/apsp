"""Validity checks per row, BEFORE clamp/PIL (CC-3 / M3)."""
from __future__ import annotations


def rows_finite(x):
    """Bool tensor [k]: every element of the row is finite. Checked before any clamp or uint8 conversion."""
    return x.flatten(1).isfinite().all(dim=1)
