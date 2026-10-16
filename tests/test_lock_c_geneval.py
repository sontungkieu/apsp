"""PREREG-C test-method list from the train_dev freeze, and the GenEval aggregation (official rule, failures incorrect)."""
import importlib.util
import json
from pathlib import Path

import pytest

from apsp.reports.lock_c import test_methods as _tm

spec = importlib.util.spec_from_file_location("geneval_tools", Path(__file__).resolve().parents[1] / "scripts" / "geneval_tools.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def test_methods_from_freeze_only():
    fz = {"operating_point": "threshold:t.json@0.5",
          "methods": ["single", "bon:4", "bon:8", "psp", "threshold:t.json@0.5", "fixed:4,8@0.5",
                      "rl:r0/artifact.json@0.5", "rl:r1/artifact.json@0.5", "rl:r0/artifact.json@1.0", "rl:r1/artifact.json@1.0"],
          "contrast_manifest": [{"id": "G3_alpha_0.5", "method": "rl*@0.5"}, {"id": "G3_alpha_1.0", "method": "threshold:t.json@0.5"}]}
    m, c = _tm(fz)
    assert m[:4] == ["single", "bon:4", "bon:8", "psp"] and "fixed:4,8@0.5" not in m
    assert sum(x.startswith("rl:") for x in m) == 4
    g3 = {x["id"]: x for x in c}["G3_alpha_0.5_vs_psp"]
    assert g3["method_specs"] == ["rl:r0/artifact.json@0.5", "rl:r1/artifact.json@0.5"]
    fz["operating_point"] = "b8:x.json@1.0"
    with pytest.raises(SystemExit, match="not frozen"):
        _tm(fz)


def test_geneval_official_aggregation_and_failures():
    tags = {0: "single_object", 1: "single_object", 2: "counting"}
    per = {(0, 0): ("single_object", True), (0, 1): ("single_object", True), (1, 0): ("single_object", False),
           (2, 0): ("counting", True)}               # (1,1), (2,1) missing = failed episodes -> incorrect
    overall, per_tag, bp = gt.score(per, 2, tags)
    assert per_tag == {"single_object": 0.5, "counting": 0.5} and overall == 0.5
    d = gt.boot_diff(bp, bp, 0.95, B=200)
    assert d["estimate"] == 0 and d["lo"] == 0 and d["hi"] == 0
