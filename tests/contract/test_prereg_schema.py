import json
import shutil

import pandas as pd
import pytest

from apsp.contracts.access import AccessError, PrivilegedAccessor
from apsp.contracts.hashing import file_sha256
from apsp.contracts.prereg import PreregGuardError, ValAccessLog, require_signed, require_split_access
from apsp.contracts.schema import load_bank, validate_bank


def test_unsigned_splits_refused(prereg_dir):
    for split in ("train_fit", "val", "test", "geneval"):
        with pytest.raises(PreregGuardError):
            require_split_access(split, "unit-test")
    require_split_access("syn_eval", "unit-test")
    require_split_access("calibration", "unit-test")


def test_signed_marker_must_match_proposal(prereg_dir, tmp_path, monkeypatch):
    import apsp.contracts.prereg as pr
    monkeypatch.setattr(pr, "REPO_ROOT", tmp_path)
    prop = tmp_path / "docs" / "PREREG_PROPOSALS" / "A.json"
    prop.parent.mkdir(parents=True)
    prop.write_text("{}")
    marker = {"prereg": "A", "role": "research", "signed_by": "PI", "proposal_path": "docs/PREREG_PROPOSALS/A.json",
              "proposal_sha256": file_sha256(prop)}
    (prereg_dir / "PREREG-A.json").write_text(json.dumps(marker))
    require_signed("A")
    require_split_access("train_fit", "unit-test")
    prop.write_text('{"changed": 1}')
    with pytest.raises(PreregGuardError):
        require_signed("A")
    marker["role"] = "agent"
    (prereg_dir / "PREREG-A.json").write_text(json.dumps(marker))
    with pytest.raises(PreregGuardError):
        require_signed("A")


def test_val_access_log_two_looks_and_frozen_config(tmp_path):
    log = ValAccessLog(tmp_path / "val_access.jsonl")
    with pytest.raises(PreregGuardError):
        log.record(contrast_id="c1", config_hash="h", look=2, cph="p")
    log.record(contrast_id="c1", config_hash="h", look=1, cph="p")
    with pytest.raises(PreregGuardError):
        log.record(contrast_id="c1", config_hash="h2", look=2, cph="p")
    log.record(contrast_id="c1", config_hash="h", look=2, cph="p")
    with pytest.raises(PreregGuardError):
        log.record(contrast_id="c1", config_hash="h", look=3, cph="p")


def test_synthetic_bank_validates(syn_bank_dir):
    rep = validate_bank(syn_bank_dir)
    assert rep.ok, rep.errors
    assert rep.stats["synthetic"] and rep.stats["preview_invalid"] > 0


def _corrupt(src, tmp_path, fn, table):
    d = tmp_path / "bad"
    shutil.copytree(src, d)
    df = pd.read_parquet(d / table)
    fn(df)
    df.to_parquet(d / table, index=False)
    return validate_bank(d)


@pytest.mark.parametrize("table,fn,needle", [
    ("seeds.parquet", lambda df: df.__setitem__("seed_global", df["seed_global"] + 1), "derivation"),
    ("previews.parquet", lambda df: df.__setitem__("img_ok", ~df["latent_ok"] | df["img_ok"]), "chain"),
    ("previews.parquet", lambda df: df.drop(df.index[:3], inplace=True), "8x16"),
    ("delivered.parquet", lambda df: df.__setitem__("ir_deliv", df["ir_deliv"].fillna(0.0)), "finite iff"),
])
def test_validate_detects_corruption(syn_bank_dir, tmp_path, table, fn, needle):
    rep = _corrupt(syn_bank_dir, tmp_path, fn, table)
    assert not rep.ok
    assert any(needle in e for e in rep.errors), rep.errors
    assert any("BMH" in e for e in rep.errors)  # content change also breaks BMH


def test_accessor_purposes(syn_bank):
    acc = PrivilegedAccessor(syn_bank)
    key = syn_bank.instances[0]
    with pytest.raises(AccessError):
        acc.instance_arrays(key, "env")
    with pytest.raises(AccessError):
        acc.preview(key, 1, [(1 - key[1], 0)])
    with pytest.raises(AccessError):
        acc.preview(key, 1, [(key[1], 0)], purpose="policy")
