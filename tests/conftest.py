import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def syn_bank_dir(tmp_path_factory):
    from apsp.data.synthetic_bank import make_synthetic_bank
    d = tmp_path_factory.mktemp("bank") / "syn"
    make_synthetic_bank(d, n_prompts=40, seed=3)
    return d


@pytest.fixture(scope="session")
def syn_bank(syn_bank_dir):
    from apsp.contracts.schema import load_bank
    return load_bank(syn_bank_dir)


@pytest.fixture(scope="session")
def syn_costs():
    from apsp.data.synthetic_bank import synthetic_costs
    return synthetic_costs()


@pytest.fixture()
def ctx(syn_bank, syn_costs):
    from apsp.sim.replay import ReplayContext
    return ReplayContext.build(syn_bank, syn_costs, fit_split="syn_train_fit", keep_log=True)


@pytest.fixture()
def prereg_dir(tmp_path, monkeypatch):
    d = tmp_path / "PREREG_SIGNED"
    d.mkdir()
    monkeypatch.setenv("APSP_PREREG_DIR", str(d))
    return d
