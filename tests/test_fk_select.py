"""FK-Steering configuration chosen on train_dev only, per particle count."""
import json

import pytest

from apsp.gpu import fk_baseline as fk


def test_parse_and_select(tmp_path):
    c = fk.parse_config("k8_max")
    assert c["num_particles"] == 8 and c["potential_type"] == "max" and c["resample_frequency"] == 13
    rows = [{"prompt_uid": f"u{i}", "bank_id": 0, "split": "train_dev", "method": m, "success": True,
             "q_eval": q + 0.01 * i, "e2e_seconds": 5.0}
            for i in range(5) for m, q in (("fk:k4_max", 0.5), ("fk:k4_diff", 0.6), ("fk:k8_max", 0.9), ("fk:k8_diff", 0.7))]
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    a = type("A", (), {"select": str(tmp_path / "dev"), "out": str(tmp_path / "f.json")})()
    assert fk.select(a) == 0
    assert json.loads((tmp_path / "f.json").read_text())["configs"] == ["k4_diff", "k8_max"]
    for r in rows:
        r["split"] = "val"
    (tmp_path / "dev" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(SystemExit, match="train_dev only"):
        fk.select(a)
