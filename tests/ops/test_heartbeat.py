"""Heartbeat states: real progress only (no synthetic load), dead / waiting / upload distinguished."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("heartbeat", Path(__file__).resolve().parents[2] / "scripts" / "heartbeat.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)


def _pr(alive=True, job=None, blocked=0, failed=()):
    return {"run_project_alive": alive, "running_job": job, "blocked_n": blocked, "failed": list(failed),
            "project_state": "waiting_research"}


def test_states(monkeypatch):
    monkeypatch.setattr(hb, "rclone_active", lambda: False)
    monkeypatch.setattr(hb, "setup_active", lambda: False)
    now = 10_000.0
    p, prev = {"newest_output_mtime": now - 10}, {"newest_output_mtime": now - 100}
    g = {"util": 97, "n_procs": 1}
    assert hb.classify(g, p, prev, _pr(job={"id": "p2_bank", "gpu": True}), now)[0] == "gpu_compute"
    assert hb.classify({"util": 0, "n_procs": 0}, p, prev, _pr(job={"id": "p3_tune", "gpu": False}), now)[0] == "cpu_progress"
    old = {"newest_output_mtime": now - 4000}
    assert hb.classify({"util": 0}, old, old, _pr(job={"id": "p3_tune", "gpu": False}), now)[0] == "job_dead"
    assert hb.classify({"util": 0}, p, prev, _pr(alive=False, job={"id": "x", "gpu": True}), now)[0] == "job_dead"
    assert hb.classify({"util": 0}, old, old, _pr(alive=False, blocked=3), now)[0] == "waiting_dependency"
    assert hb.classify({"util": 0}, old, old, _pr(alive=False, failed=["p2_bank"]), now)[0] == "failed_jobs"
    monkeypatch.setattr(hb, "rclone_active", lambda: True)
    assert hb.classify({"util": 0}, old, old, _pr(alive=False), now)[0] == "upload"
