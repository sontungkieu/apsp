"""apsp.reports.calibration_report: missing parts listed, sources hashed, provisional flag and the descriptive label
kept, CSV copied. Fixture JSON only (not evidence)."""
import json

from apsp.reports import calibration_report as cr


def test_report_lists_missing_and_keeps_flags(tmp_path):
    cal = tmp_path / "calib1"
    (cal / "p0" / "m2").mkdir(parents=True)
    (cal / "p0" / "m2" / "generator_costs.json").write_text(json.dumps({"provisional": True, "primitives": [1, 2]}))
    rep = cal / "bench_calib" / "report"
    rep.mkdir(parents=True)
    (rep / "quality_cost.json").write_text(json.dumps({"label": "real backend, PROVISIONAL cost table",
                                                       "summary": [{"method": "psp", "n_episodes": 100, "mean_Q": 0.5}]}))
    (rep / "quality_cost.csv").write_text("method\npsp\n")
    assert cr.main(["--cal", str(cal), "--out", str(cal / "report")]) == 0
    md = (cal / "report" / "calibration_report.md").read_text(encoding="utf-8")
    assert "provisional: **True**" in md and "DESCRIPTIVE ONLY" in md and "| psp | 100 | 0.5000" in md
    assert "**missing** (`calib/m3m9.json`)" in md
    idx = json.loads((cal / "report" / "calibration_report_index.json").read_text())
    assert idx["VAE decode chunk"] is None and len(idx["Generator cost table (M2, merged)"]["sha256"]) == 64
    assert (cal / "report" / "calibration_quality_cost.csv").read_text() == "method\npsp\n"
