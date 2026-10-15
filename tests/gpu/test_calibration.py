"""Calibration parts on the fake backend: one trajectory per calibration instance (M3 records + production shard),
resume, M9 summary, chunk timing, M2b and the equivalence guard."""
import json
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from apsp.data.prompts import prompt_uid  # noqa: E402
from apsp.gpu import calibration as cal  # noqa: E402


@pytest.fixture
def pool(tmp_path):
    texts = [f"calibration prompt number {i}" for i in range(50)] + ["a train prompt"]
    df = pd.DataFrame({"prompt_uid": [prompt_uid(t) for t in texts], "text": texts,
                       "split": ["calibration"] * 50 + ["train_fit"], "source": "test", "group_id": "g",
                       "n_clip_tokens": 5})
    df["group_id"] = df["prompt_uid"]
    srt = df[df["split"] == "calibration"].sort_values("prompt_uid")
    df["throughput_subset"] = df["prompt_uid"].isin(set(srt["prompt_uid"].iloc[:20]))
    p = tmp_path / "pool.parquet"
    df.to_parquet(p, index=False)
    return p


def run(args):
    return cal.main(["--fake-backend"] + args)


def test_m3m9_one_pass_resume_and_summary(tmp_path, pool, monkeypatch):
    out, banks = tmp_path / "calib", tmp_path / "banks"
    calls = {"n": 0}
    from apsp.gpu import collect_bank
    orig = collect_bank.collect_instance

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(collect_bank, "collect_instance", spy)
    assert run(["--part", "m3m9", "--prompt-pool", str(pool), "--out", str(out), "--bank-root", str(banks)]) == 0
    assert calls["n"] == 70                                  # 50 bank-0 + 20 subset bank-1, no second pass
    rep = json.loads((out / "m3m9.json").read_text())
    assert rep["n_records"] == 70 and rep["n_m3_instances"] == 50 and rep["m3_complete"]
    assert rep["m9"]["n_clean_subset_bank1"] == 20 and rep["m9"]["seconds_per_prompt_bank_clean_mean"] > 0
    assert rep["m3_prod"]["n_previews"] == 50 * 8 * 16 and rep["m3_escalate_to_research"] is False
    assert len(list((out / "eqv_tensors").glob("*.pt"))) == 10
    d = torch.load(next((out / "eqv_tensors").glob("*.pt")))
    assert set(d["x0"]) == {4, 8, 16} and d["final"].shape[0] == 8
    shards = list(banks.rglob("shards/*__b*"))
    assert len(shards) == 70
    # resume: nothing recomputed
    calls["n"] = 0
    assert run(["--part", "m3m9", "--prompt-pool", str(pool), "--out", str(out), "--bank-root", str(banks)]) == 0
    assert calls["n"] == 0 and json.loads((out / "m3m9.json").read_text())["instances_skipped"] == 70
    # a shard without its record is quarantined and redone
    rec = sorted((out / "m3m9_records").glob("*.json"))[0]
    rec.unlink()
    assert run(["--part", "m3m9", "--prompt-pool", str(pool), "--out", str(out), "--bank-root", str(banks)]) == 0
    assert calls["n"] == 1 and rec.exists()


def test_chunk_m2b_and_equivalence_guard(tmp_path, pool):
    out, banks = tmp_path / "calib", tmp_path / "banks"
    assert run(["--part", "m3m9", "--prompt-pool", str(pool), "--out", str(out), "--bank-root", str(banks)]) == 0
    assert run(["--part", "chunk", "--prompt-pool", str(pool), "--out", str(out)]) == 0
    ch = json.loads((out / "chunk.json").read_text())
    assert "delivered_k16_chunk16" in ch["results"] and "preview_k8_chunk1" in ch["results"]
    from apsp.gpu.technical_smoke import engineering_costs
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps(engineering_costs().to_dict()))
    assert run(["--part", "m2b", "--prompt-pool", str(pool), "--out", str(out), "--costs", str(costs)]) == 0
    m2b = json.loads((out / "m2b.json").read_text())
    assert m2b["n_episodes"] == 20 and m2b["predicted_us"] > 0 and m2b["costs_provisional"] is True
    assert run(["--part", "equivalence", "--prompt-pool", str(pool), "--out", str(out)]) == 5   # fake: refused


def test_calibration_split_shape_enforced(tmp_path):
    df = pd.DataFrame({"prompt_uid": ["a"], "text": ["t"], "split": ["calibration"], "throughput_subset": [True]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(SystemExit):
        cal.calibration_prompts(str(p))
