"""B4b grid: identical to PSP enumerate_strategies at the pinned commit; mapping to bank checkpoints never rounds."""
import json
import subprocess
import sys

import pytest

from apsp.sim import b4b_grid as g
from apsp.sim.schedules import required_prune_pairs


def test_grid_equals_psp_and_manifest(tmp_path):
    r = subprocess.run([sys.executable, "scripts/b4b_manifest.py", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "b4b_grid_manifest.json").read_text())
    assert m["grid_matches_psp_enumerate_strategies"] and m["sd15_scripts_agree_with_args"]
    assert m["n_strategies"] == 7823 and m["n_specs_valid_on_bank"] == 7823 and m["off_bank_strategies"] == []
    assert m["psp_commit"] == g.PSP_COMMIT


def test_budget_filter_and_mapping():
    grid = g.enumerate_grid()
    assert all(s.total_particle_steps <= 256 for s in grid) and max(len(s.cutoff_times) for s in grid) == 4
    s = next(x for x in grid if x.k_init == 12 and x.cutoff_times == (4, 8))
    spec = g.to_spec(s)
    assert spec.queries == (1, 2) and len(spec.candidates) == 12 and spec.candidates[8] == (1, 0)
    spec.validate()


def test_off_bank_cutoff_raises_instead_of_rounding():
    s = g.GridStrategy(8, (6,), (4,), 256, 0)
    with pytest.raises(g.OffBankCheckpoint):
        g.to_spec(s)


def test_required_prune_pairs_cover_b4b():
    req = set(required_prune_pairs())
    assert {tuple(p) for p in g.required_primitives(g.enumerate_grid())["prune_pairs"]} <= req
    assert {(8, 4), (4, 2), (16, 1), (16, 2)} <= req
