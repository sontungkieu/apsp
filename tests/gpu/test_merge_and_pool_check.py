"""merge_m2: per-k parts merge with coverage report; full_pool_check components (CPU)."""
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_merge_m2_from_fake_parts(tmp_path):
    from apsp.gpu import profile
    for k in (1, 2):
        assert profile.main(["--fake-backend", "--stage", "generator", "--smoke", "--ks", str(k),
                             "--out-dir", str(tmp_path / f"k{k:02d}")]) == 0
        # the smoke tag prefixes file names; the merge reads the non-smoke names
        for n in ("generator_costs.json", "generator_costs_raw.json"):
            (tmp_path / f"k{k:02d}" / n).write_bytes((tmp_path / f"k{k:02d}" / f"SMOKE_{n}").read_bytes())
    out = tmp_path / "generator_costs.json"
    r = subprocess.run([sys.executable, "scripts/merge_m2.py", str(tmp_path), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = json.loads(out.read_text())
    rep = json.loads((tmp_path / "generator_costs_merge_report.json").read_text())
    assert set(c["den"]) == {"1", "2"} and c["provisional"] is True
    assert "16" in rep["missing_keys"]["den"] and "smoke run" in rep["provisional_reasons"]


def test_full_pool_components_cpu():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("fpc", Path("scripts/full_pool_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    e = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(50, 16)), dtype=torch.float32), dim=1)
    e[1] = e[0]
    e[2] = torch.nn.functional.normalize(e[1] + 1e-3, dim=0)       # chain 0-1-2
    comp = m.components(e, 0.95, "cpu", block=7)
    assert comp[0] == comp[1] == comp[2] and len(set(comp)) == 48
