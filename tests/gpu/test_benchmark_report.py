"""Online benchmark, quality-cost report, NI/savings status, evaluator path - fake backend and synthetic data only.
Nothing here is evidence of quality or savings."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from apsp.gpu import benchmark  # noqa: E402
from apsp.reports import quality_cost as qcr  # noqa: E402
from apsp.sim.methods import MethodError, parse_method, run_method  # noqa: E402

BASE = ["single", "single32", "bon:2", "bon:4", "bon:8", "psp", "sr:8,3,2", "bon:16"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    assert benchmark.main(["--fake-backend", "--technical", "--methods", *BASE, "random:1@1.0", "--synthetic-costs",
                           "--warmup", "1", "--out", str(d)]) == 0
    return d


def recs(d):
    return [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]


def test_online_batches_forwards_and_eval_sources(run_dir):
    R = recs(run_dir)
    by = {}
    for r in R:
        by.setdefault(r["method"], []).append(r)
    assert len(by["bon:16"]) == 3 and len(by["single"]) == 6           # pool-16: one episode per prompt
    for n in (2, 4, 8):
        r = by[f"bon:{n}"][0]
        assert r["n_candidates"] == n and r["candidate_forwards"] == 64 * n and r["forwards"] == 64
    s32, s64 = by["single32"][0], by["single"][0]
    assert s32["forwards"] == 32 and s64["forwards"] == 64 and s32["eph"] != s64["eph"] and s32["steps"] == 32
    assert s64["q_eval_source"] == "posthoc_ir_evaluation_only" and s64["rm_used_in_inference"] is False
    assert by["psp"][0]["q_eval_source"] == "inference_rm_same_image" and by["psp"][0]["n_verify"] == 2
    assert all(r["image_png_sha256"] and (run_dir / r["image"]).exists() for r in R if r["success"])
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["costs"]["synthetic"] and man["workers"] == 1 and len(man["timesteps_32"]) == 32
    assert man["model_load_seconds"] >= 0 and man["warmup_seconds"] >= 0


def test_resume_skips_done_episodes(run_dir):
    n = len(recs(run_dir))
    assert benchmark.main(["--fake-backend", "--technical", "--methods", *BASE, "random:1@1.0", "--synthetic-costs",
                           "--warmup", "0", "--out", str(run_dir)]) == 0
    assert len(recs(run_dir)) == n


def test_confirmatory_split_needs_frozen_methods_and_prereg(tmp_path, prereg_dir):
    pool = tmp_path / "pool.parquet"
    pd.DataFrame({"prompt_uid": ["a"], "text": ["t"], "split": ["val"]}).to_parquet(pool)
    with pytest.raises(PermissionError):           # PREREG-B not signed
        benchmark.main(["--fake-backend", "--prompt-pool", str(pool), "--split", "val", "--methods", "psp",
                        "--synthetic-costs", "--out", str(tmp_path / "o")])
    a = benchmark.argparse.Namespace(methods=["psp"], methods_file=None, split="val")
    with pytest.raises(SystemExit, match="frozen on train_dev"):
        benchmark.load_methods(a)


def test_method_parsing_rules():
    with pytest.raises(MethodError, match="cap"):
        parse_method("random:1")
    with pytest.raises(MethodError):
        parse_method("bon:12")
    m = parse_method("fixed:3,8@1.0")
    assert m.alpha == 1.0 and m.params == (3, 8) and m.uses_inference_rm
    assert not parse_method("single").uses_inference_rm and not parse_method("sr:8,4,1").uses_inference_rm
    with pytest.raises(MethodError, match="online only"):
        run_method(parse_method("single32"), None, bank_id=0, ctx=None, is_replay=True)


def test_report_not_concludable_without_approved_config(run_dir, tmp_path):
    assert qcr.main(["--run", str(run_dir), "--out", str(tmp_path / "r")]) == 0
    rep = json.loads((tmp_path / "r" / "quality_cost.json").read_text())
    assert rep["status"]["concludable"] is False and "SYNTHETIC" in rep["label"]
    assert all(c["ni"] == "not_concludable" for c in rep["contrasts_vs_reference"] if "ni" in c)
    assert any(r["basis"].startswith("DESCRIPTIVE") for r in rep["cost_to_reach"])
    assert {p["method"] for p in rep["curve"]} >= set(BASE)
    with pytest.raises(SystemExit, match="train_dev"):
        qcr.main(["--run", str(run_dir), "--out", str(tmp_path / "r2"), "--freeze", str(tmp_path / "f.json")])


def _approved(tmp_path, eps=0.05):
    cfg = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    cfg.update({"approved": True, "approved_by": "test", "epsilon_ir": eps, "r_fail": -3.0, "n_primary_contrasts": 2})
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_ni_and_freeze_with_approved_config_on_constructed_data(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        for b in (0, 1):
            q = rng.normal(0, 1)
            for m, dq, c in (("psp", 0.0, 1.0), ("cheap_ok", -0.01, 0.6), ("cheap_bad", -0.5, 0.3)):
                rows.append({"prompt_uid": f"p{i}", "bank_id": b, "method": m, "family": "x", "success": True,
                             "q_eval": q + dq + rng.normal(0, 0.005), "e2e_seconds": c * (1 + rng.normal(0, 0.01)),
                             "ledger_spend_us": int(c * 1e6), "forwards": 64, "candidate_forwards": 256, "n_candidates": 8,
                             "n_verify": 2, "rm_used_in_inference": True, "alpha": None})
    d = tmp_path / "run"
    d.mkdir()
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "manifest.json").write_text(json.dumps({"split": "train_dev", "backend": "fake", "costs": {"synthetic": True}}))
    cfgp = _approved(tmp_path)
    assert qcr.main(["--run", str(d), "--config", str(cfgp), "--out", str(d / "rep"), "--freeze", str(d / "frozen.json")]) == 0
    rep = json.loads((d / "rep" / "quality_cost.json").read_text())
    ni = {c["method"]: c["ni"] for c in rep["contrasts_vs_reference"]}
    assert rep["status"]["concludable"] and ni == {"cheap_ok": "pass", "cheap_bad": "fail"}
    fr = json.loads((d / "frozen.json").read_text())
    assert fr["selected_on"] == "train_dev" and fr["operating_point"] == "cheap_ok"
    # failures stay in the denominator at r_fail
    rows[0].update({"success": False, "q_eval": None})
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    df, man = qcr.load_run(d)
    s = qcr.summary_table(df, -3.0)
    assert s.loc[rows[0]["method"], "failure_rate"] > 0


def test_independent_evaluator_path(run_dir, tmp_path):
    from apsp.gpu.evaluators import independent as ev
    assert ev.main(["--run", str(run_dir), "--metric", "fake"]) == 0
    m = [json.loads(l) for l in (run_dir / "eval_metrics.jsonl").read_text().splitlines()]
    assert m and all(r["evaluation_only"] and r["metric"] == "fake" for r in m)
    df, _ = qcr.load_run(run_dir)
    assert "eval_fake" in df.columns
    with pytest.raises(ev.EvaluatorError, match="not pinned"):
        ev.make_evaluator("hpsv2", {"hpsv2": {"package_version": None}})
    with pytest.raises(ev.EvaluatorError, match="PREREG-C"):
        ev.make_evaluator("geneval", {})


def test_evaluators_isolated_from_env_features_policies_trainer():
    root = Path("apsp")
    for sub in ("sim", "policies", "contracts"):
        for f in (root / sub).rglob("*.py"):
            src = f.read_text(encoding="utf-8")
            assert "evaluators" not in src, f"{f} must not import evaluation-only code"
