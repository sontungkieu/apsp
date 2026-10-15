"""Technical-smoke path: separate from research data, guards still enforced, steps run end to end (fake backend)."""
import json
from argparse import Namespace

import pandas as pd
import pytest

pytest.importorskip("torch")

from apsp.contracts.prereg import PreregGuardError, require_split_access  # noqa: E402
from apsp.data.prompts import prompt_uid  # noqa: E402
from apsp.gpu.technical_smoke import (SmokeCheckError, check_outside_research, load_smoke_config,  # noqa: E402
                                      main as smoke_main)


def test_technical_split_does_not_unlock_research_splits(prereg_dir):
    require_split_access("technical_smoke", "unit-test")
    for split in ("train_fit", "train_dev", "val", "val_ext", "test", "geneval"):
        with pytest.raises(PreregGuardError):
            require_split_access(split, "unit-test")


def test_smoke_config_limits(tmp_path):
    sc = load_smoke_config("configs/technical_smoke.yaml")
    assert 1 <= len(sc["prompts"]) <= 3 and sc["split"] == "technical_smoke"
    bad = tmp_path / "bad.yaml"
    bad.write_text("split: train_fit\nprompts: [a]\n")
    with pytest.raises(SmokeCheckError):
        load_smoke_config(str(bad))
    many = tmp_path / "many.yaml"
    many.write_text("split: technical_smoke\nprompts: [a, b, c, d]\nmax_prompts: 4\n")
    with pytest.raises(SmokeCheckError):
        load_smoke_config(str(many))


def test_prompt_pool_and_geneval_clash_refused(tmp_path):
    sc = load_smoke_config("configs/technical_smoke.yaml")
    pool = tmp_path / "pool.parquet"
    pd.DataFrame({"prompt_uid": [prompt_uid(sc["prompts"][1])], "split": ["train_fit"]}).to_parquet(pool)
    with pytest.raises(SmokeCheckError, match="research prompt pool"):
        check_outside_research(sc, str(pool))
    with pytest.raises(SmokeCheckError, match="GenEval"):
        check_outside_research({**sc, "prompts": ["a photo of a bench"]}, None)


def test_steps_end_to_end_fake(tmp_path, capsys):
    rc = smoke_main(["--fake-backend", "--step", "all", "--out", str(tmp_path)])
    assert rc == 0
    for step in ("single", "psp", "collector", "parity"):
        d = json.loads((tmp_path / f"{step}.json").read_text())
        assert d["status"] == "PASS", (step, d["failures"])
    assert smoke_main(["--fake-backend", "--step", "collector", "--out", str(tmp_path)]) == 0   # idempotent rerun


def test_replay_parity_refuses_mixed_research_bank(tmp_path, prereg_dir):
    """A bank containing any research-split prompt still needs PREREG-B, even with a technical --split."""
    from apsp.data.synthetic_bank import synthetic_costs
    from apsp.gpu import online
    from apsp.gpu.backend import build_backend, compute_eph
    from apsp.gpu.collect_bank import run_collection
    from apsp.gpu.engine import DenoiseEngine
    args = Namespace(config="configs/pilot.yaml", fake_backend=True, revision=None, preview_dtype=None, local_files_only=False)
    cfg, bundle, scorer, scoring, rt = build_backend(args)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    pr = pd.DataFrame({"prompt_uid": ["t1", "r1"], "text": ["tech prompt", "research prompt"],
                       "split": ["technical_smoke", "train_dev"], "source": "t", "group_id": ["t1", "r1"], "n_clip_tokens": 0})
    bank = tmp_path / "MIXED"
    run_collection(DenoiseEngine(bundle, scoring, cfg, hook_mode="collect"), bundle, pr, bank, eph=eph, eph_body=body,
                   cfg=cfg, banks_per_prompt=1, resume=False, machine_id="t", commit="t")
    # make the manifest look like a real (non-fake) bank so only the split rule decides
    m = json.loads((bank / "manifest.json").read_text())
    m["backend"] = "diffusers"
    (bank / "manifest.json").write_text(json.dumps(m))
    costs = tmp_path / "c.json"
    costs.write_text(json.dumps(synthetic_costs().to_dict()))
    with pytest.raises(PreregGuardError):
        online.main(["--fake-backend", "--task", "replay-parity", "--bank", str(bank), "--costs", str(costs),
                     "--split", "technical_smoke", "--fit-split", "technical_smoke", "--out", str(tmp_path / "r")])
