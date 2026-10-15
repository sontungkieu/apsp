"""Runtime fingerprint v1: collection on CPU, numeric hash, diff, CLI, harness metadata and artifact reconstruction."""
import json
import subprocess
import sys

import pytest

from apsp.gpu import fingerprint as fpm


def test_collect_cpu_schema_and_hash():
    fp = fpm.collect(backend="fake")
    assert fp["schema"] == "apsp.runtime_fingerprint/v1" and fp["python"].count(".") == 2
    assert fp["fingerprint_sha"] == fpm.numeric_sha(fp) and len(fp["fingerprint_sha"]) == 64
    fp2 = dict(fp, os="another os string")             # context field: same numeric hash
    assert fpm.numeric_sha(fp2) == fp["fingerprint_sha"]
    fp3 = dict(fp, driver="999.0")                      # numeric-relevant field: different hash
    assert fpm.numeric_sha(fp3) != fp["fingerprint_sha"]
    assert fpm.diff(fp, fp3) == {"driver": [fp["driver"], "999.0"]}


def test_cli_writes_eph_coverage(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "fp.json"
    r = subprocess.run([sys.executable, "-m", "apsp.gpu.fingerprint", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert "GPU model and compute capability" in d["eph_v1_misses"] and d["flags"]["cudnn_deterministic"] is True


def test_harness_meta_carries_fingerprint(tmp_path):
    pytest.importorskip("torch")
    from apsp.gpu import drift_harness as dh
    import yaml
    d = dh.load_diag_config("configs/drift_diagnostic.yaml")
    small = {"split": d["split"], "prompts": d["prompts"][:1], "repro": {"prompt": d["prompts"][0], "uid": "m6-technical"}}
    cfgp = tmp_path / "dc.yaml"
    cfgp.write_text(yaml.safe_dump(small))
    assert dh.main(["--fake-backend", "--diag-config", str(cfgp), "--part", "audit", "--out", str(tmp_path / "o")]) == 0
    rep = json.loads((tmp_path / "o" / "audit.json").read_text())
    assert rep["runtime_fingerprint"]["schema"] == fpm.SCHEMA and rep["runtime_fingerprint"]["backend"] == "fake"


def test_retro_reconstruction_marks_missing_fields(tmp_path):
    run = tmp_path / "run"
    (run / "p0").mkdir(parents=True)
    (run / "p0" / "env_box.json").write_text(json.dumps({
        "python": "3.10.20", "os": "Linux", "nvidia_smi": ["NVIDIA GeForce RTX 3090, 24576 MiB, 580.95.05"],
        "torch": "2.4.0+cu121", "gpu": {"name": "NVIDIA GeForce RTX 3090", "capability": "8.6"},
        "numeric_runtime": {"cuda_runtime": "12.1", "cudnn": 90100}}))
    (run / "env_freeze.txt").write_text("diffusers==0.31.0.dev0\ntransformers==4.38.2\n")
    (run / "p0" / "model.json").write_text(json.dumps({"eph": "a" * 64}))
    out = tmp_path / "fps"
    r = subprocess.run([sys.executable, "scripts/retro_fingerprint.py", f"r={run}", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    fp = json.loads((out / "r.json").read_text())
    assert fp["driver"] == "580.95.05" and fp["packages"]["diffusers"] == "0.31.0.dev0"
    assert fp["attention_processors"] == "not recorded" and fp["flags"]["deterministic_algorithms"] == "not recorded"
    assert fp["ephs_in_artifacts"] == ["a" * 64]
    assert sorted(p.name for p in run.rglob("*")) == ["env_box.json", "env_freeze.txt", "model.json", "p0"]
