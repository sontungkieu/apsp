"""apsp.run_project: dependency order, PREREG/gate/budget/price/watchdog blocking, resume, provenance re-run,
failure classes and cumulative budget. Fake runner only; nothing is executed on a GPU."""
import json
from pathlib import Path

import pytest
import yaml

from apsp import run_project as rp


def _plan(tmp_path, jobs, **kw):
    d = {"project_run": "t", "instance_id": 1, "approved_total_usd": kw.get("total", 10.0),
         "approved_max_usd_per_h": kw.get("max_h", 1.0), "bandwidth_storage_reserve_usd": 0.5,
         "forecast_safety": 1.0, "pack_stop_margin_min": 6, "runs_root": str(tmp_path / "runs"),
         "destroy_instance": False, "env": {"OUT": str(tmp_path / "o")}, "jobs": jobs}
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def _rt(price=0.6, armed=True, deadline=None):
    return rp.Runtime(price, "test", lambda: (armed, "ok" if armed else "not armed", deadline), on_vast=False)


class FakeRunner:
    def __init__(self, tmp_path, rc=None):
        self.calls, self.tmp, self.rc = [], tmp_path, rc or {}

    def __call__(self, cmd, log, timeout_min, env):
        self.calls.append(cmd)
        log.write_text(cmd)
        name = cmd.split()[-1]
        (self.tmp / "o").mkdir(exist_ok=True)
        if self.rc.get(name, 0) == 0:
            (self.tmp / "o" / f"{name}.json").write_text("{}")
        return self.rc.get(name, 0)


def J(id, needs=(), gpu=True, est=0.5, **kw):
    return {"id": id, "phase": "P1", "gpu": gpu, "est_hours": est, "needs": list(needs),
            "cmd": f"echo {id}", "outputs": [f"${{OUT}}/{id}.json"], **kw}


def test_runs_in_order_and_resumes(tmp_path, prereg_dir):
    p = _plan(tmp_path, [J("a"), J("b", ["a"]), J("c", ["b"], gpu=False, est=None)])
    fr = FakeRunner(tmp_path)
    assert rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=_rt()) == 0
    assert fr.calls == ["echo a", "echo b", "echo c"]
    fr2 = FakeRunner(tmp_path)
    assert rp.main(["--plan", str(p), "--resume"], runner=fr2, runtime=_rt()) == 0
    assert fr2.calls == []                                     # done with valid outputs and same hash -> skipped
    (tmp_path / "o" / "b.json").write_text("")                 # empty output: not valid provenance
    fr3 = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr3, runtime=_rt())
    assert fr3.calls == ["echo b"]
    st = rp.State(tmp_path / "runs" / "t")
    assert st.get("b")["attempts"] == 2
    ev = [json.loads(l) for l in (tmp_path / "runs" / "t" / "job_manifest.jsonl").read_text().splitlines()]
    assert any(e["id"] == "b" and e["reason"] and "provenance" in e["reason"] for e in ev)


def test_blocking_reasons(tmp_path, prereg_dir):
    jobs = [J("a"), J("needs_a_prereg", prereg=["A"]), J("gated", gates=["evaluation_approved"]),
            J("never", gates=["never"]), J("cpu", gpu=False, est=None), J("off", enabled=False)]
    p = _plan(tmp_path, jobs)
    fr = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=_rt())
    st = rp.State(tmp_path / "runs" / "t")
    assert st.get("needs_a_prereg")["reason"] == "prereg:A"
    assert st.get("gated")["reason"].startswith("gate:evaluation_approved")
    assert st.get("never")["reason"].startswith("gate:never")
    assert st.get("off")["status"] == "disabled"
    assert fr.calls == ["echo a", "echo cpu"]


@pytest.mark.parametrize("kw,rt,expect", [
    ({"total": None}, _rt(), "budget:not_approved"),
    ({}, _rt(price=None), "price:unknown"),
    ({"max_h": 0.5}, _rt(price=0.778), "price:0.778>approved_max"),
    ({"total": 1.0}, _rt(price=0.6), "budget:need"),                  # 0.6*(2.0+0.1)=1.26 > 1.0-0.5
    ({}, _rt(armed=False), "watchdog:"),
    ({}, _rt(deadline=1.0), "watchdog:deadline too close"),
])
def test_gpu_preflight(tmp_path, prereg_dir, kw, rt, expect):
    p = _plan(tmp_path, [J("big", est=2.0)], **kw)
    fr = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=rt)
    assert fr.calls == []
    assert rp.State(tmp_path / "runs" / "t").get("big")["reason"].startswith(expect)


def test_failure_class_and_dependants_blocked(tmp_path, prereg_dir):
    p = _plan(tmp_path, [J("a"), J("b", ["a"]), J("indep", gpu=False, est=None)])
    fr = FakeRunner(tmp_path, rc={"a": 4})
    assert rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=_rt()) == 1
    st = rp.State(tmp_path / "runs" / "t")
    assert st.get("a")["status"] == "failed" and st.get("a")["failure_class"] == "data_error"
    assert st.get("b")["reason"] == "needs:a" and st.get("indep")["status"] == "done"
    fr2 = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr2, runtime=_rt())
    assert fr2.calls == []                                     # failed jobs are not retried silently
    rp.main(["--plan", str(p), "--resume", "--retry-failed", "a"], runner=fr2, runtime=_rt())
    assert fr2.calls == ["echo a", "echo b"]


def test_budget_is_cumulative_across_restarts(tmp_path, prereg_dir):
    p = _plan(tmp_path, [J("a", est=1.0)], total=2.0)
    t = [1000.0]
    rp.main(["--plan", str(p), "--add-spend", "0.9", "--note", "setup session"], runtime=_rt())
    fr = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=_rt(price=0.6), now_fn=lambda: t[0])
    # need 0.6*(1.0+0.1)=0.66 > left 2.0-0.5-0.9=0.6 -> blocked by the earlier, externally recorded spend
    assert fr.calls == [] and rp.State(tmp_path / "runs" / "t").get("a")["reason"].startswith("budget:need")


def test_plan_validation(tmp_path):
    with pytest.raises(rp.PlanError, match="comes later"):
        rp.load_plan(_plan(tmp_path, [J("b", ["a"]), J("a")]))
    p = _plan(tmp_path, [J("a")])
    d = yaml.safe_load(p.read_text())
    d["destroy_instance"] = True
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(rp.PlanError, match="never destroys"):
        rp.load_plan(p)


def test_repo_plan_loads_in_explicit_no_cap_mode(tmp_path, prereg_dir, monkeypatch, capsys):
    monkeypatch.setenv("APSP_STORE", str(tmp_path / "store"))
    plan, jobs = rp.load_plan(Path(rp.REPO) / "configs" / "full_experiments.yaml")
    assert plan["financial_cap_mode"] == "none" and plan["destroy_instance"] is False
    assert plan["instance_id"] == 56272061
    assert plan["stop_instance_before_project_complete"] is False and plan["stop_instance_on_deadline"] is False
    assert not plan.get("finished_marker")                    # plan v1.3: no completion marker from the runner
    assert rp.main(["--plan", str(Path(rp.REPO) / "configs" / "full_experiments.yaml"), "--status"], runtime=_rt()) == 0
    out = capsys.readouterr().out
    assert "no financial cap (explicit)" in out and "budget:" not in out
    byid = {j.id: j for j in jobs}                            # science gates unchanged
    assert byid["p2_bank"].prereg == ["A"] and byid["p5_bench_val"].prereg == ["B"] and byid["p6_geneval"].prereg == ["C"]
    assert byid["p1_bench_calib"].gates == ["pool_check_calibration"] and not byid["p1_bench_calib"].prereg


def test_no_cap_mode_skips_only_budget_and_price(tmp_path, prereg_dir):
    p = _plan(tmp_path, [J("a", est=50.0), J("b", prereg=["A"])], total=None, max_h=None)
    d = yaml.safe_load(p.read_text())
    assert rp.main(["--plan", str(p), "--resume"], runner=FakeRunner(tmp_path), runtime=_rt(price=None)) == 0
    assert rp.State(tmp_path / "runs" / "t").get("a")["reason"] == "budget:not_approved"   # null without the mode
    d["financial_cap_mode"] = "none"
    d["workload_pidfile"] = str(tmp_path / "wl.pid")
    d["finished_marker"] = str(tmp_path / "wl.finished")
    p.write_text(yaml.safe_dump(d))
    fr = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr, runtime=_rt(price=0.527))
    st = rp.State(tmp_path / "runs" / "t")
    assert fr.calls == ["echo a"] and st.get("b")["reason"] == "prereg:A"
    assert st.get("a")["cost_usd_est"] >= 0 and (tmp_path / "wl.pid").read_text().strip().isdigit()
    fin = json.loads((tmp_path / "wl.finished").read_text())
    assert fin["blocked"] == {"b": "prereg:A"} and fin["project_state"] == "waiting_research"
    fr2 = FakeRunner(tmp_path)
    rp.main(["--plan", str(p), "--resume"], runner=fr2, runtime=_rt(armed=False))   # watchdog still required
    d["jobs"].append(J("c"))
    p.write_text(yaml.safe_dump(d))
    rp.main(["--plan", str(p), "--resume"], runner=fr2, runtime=_rt(armed=False))
    assert fr2.calls == [] and st.get("c")["reason"].startswith("watchdog:")
    d["financial_cap_mode"] = "unlimited"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(rp.PlanError, match="financial_cap_mode"):
        rp.load_plan(p)


def test_pool_check_calibration_gate(tmp_path):
    env = {"CAL": str(tmp_path)}
    assert not rp.gate_status("pool_check_calibration", env=env)[0]
    (tmp_path / "calib").mkdir()
    f = tmp_path / "calib" / "pool_full_check.json"
    f.write_text(json.dumps({"cross_split_full_pool": [{"prompt_uid": "x", "split": "train_fit"},
                                                       {"prompt_uid": "y", "split": "val"}],
                             "selected_not_in_survivors": []}))
    assert rp.gate_status("pool_check_calibration", env=env)[0]       # research splits only: calibration unaffected
    f.write_text(json.dumps({"cross_split_full_pool": [{"prompt_uid": "x", "split": "calibration"}],
                             "selected_not_in_survivors": []}))
    assert not rp.gate_status("pool_check_calibration", env=env)[0]
