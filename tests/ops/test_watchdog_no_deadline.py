"""vast_watchdog.py explicit no-financial-deadline mode (D47): arming needs marker + pidfile + stall timeout; the stop
triggers are completion (idle grace), workload death and a stall; a progressing workload is never stopped for time.
FakeAPI only; nothing talks to the real provider."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vast_watchdog.py"
spec = importlib.util.spec_from_file_location("vast_watchdog_nd", SCRIPT)
wdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wdm)
IID = 56261825


@pytest.fixture(autouse=True)
def _no_container(monkeypatch):
    monkeypatch.delenv("CONTAINER_ID", raising=False)


class Clock:
    def __init__(self, t=None):
        t = float(int(__import__("time").time()) + 1000) if t is None else t
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(tmp_path, c, util=lambda: 0.0, **kw):
    kw.setdefault("finished_marker", str(tmp_path / "fin"))
    kw.setdefault("workload_pidfile", str(tmp_path / "pid"))
    kw.setdefault("stall_s", 600)
    kw.setdefault("progress_paths", [str(tmp_path / "logs")])
    w = wdm.Watchdog(IID, None, str(tmp_path / "st"), wdm.FakeAPI(str(tmp_path / "fake")), idle_grace_s=120,
                     retry_s=1, poll_s=10, gpu_util=util, clock=c, sleep=c.sleep, **kw)
    return w


def test_arm_requires_all_safety_triggers(tmp_path):
    c = Clock()
    for missing in ("finished_marker", "workload_pidfile", "stall_s"):
        w = make(tmp_path / missing, c, **{missing: None})
        with pytest.raises(wdm.WatchdogError, match="without a deadline"):
            w.arm()
    w = make(tmp_path, c)
    w.arm()
    st = wdm.read_state(str(tmp_path / "st"))
    assert st["deadline_epoch"] is None and st["financial_deadline"] is False


def test_is_armed_accepts_no_deadline_only_in_explicit_mode(tmp_path):
    c = Clock()
    make(tmp_path, c).arm()
    sd = str(tmp_path / "st")
    ok, why = wdm.is_armed(sd, IID)            # pid is this test process: alive
    assert ok, why
    st = wdm.read_state(sd)
    st["financial_deadline"] = True
    (tmp_path / "st" / "state.json").write_text(__import__("json").dumps(st))
    assert not wdm.is_armed(sd, IID)[0]


def test_progressing_workload_is_not_stopped_for_time(tmp_path):
    c = Clock()
    (tmp_path / "pid").write_text(f"{os.getpid()}\n")
    w = make(tmp_path, c, util=lambda: 95.0)   # GPU busy all along, no log changes
    w.arm()
    for _ in range(24 * 360):                  # 24 h of polls
        assert w.trigger_due() is None
        c.sleep(10)


def test_stall_triggers_when_no_file_change_and_gpu_idle(tmp_path):
    c = Clock()
    t_start = c()
    (tmp_path / "pid").write_text(f"{os.getpid()}\n")
    (tmp_path / "logs").mkdir()
    w = make(tmp_path, c)
    w.arm()
    t = None
    for i in range(200):
        if i < 30:                             # log grows for 5 min
            f = tmp_path / "logs" / "a.log"
            f.write_text("x" * i)
            os.utime(f, (c() , c()))
        t = w.trigger_due()
        if t:
            break
        c.sleep(10)
    assert t == "stalled"
    assert c() - t_start >= 600 + 290          # not before the last change + stall_s


def test_completion_then_idle_grace(tmp_path):
    c = Clock()
    (tmp_path / "pid").write_text(f"{os.getpid()}\n")
    w = make(tmp_path, c, util=lambda: 90.0)
    w.arm()
    assert w.trigger_due() is None
    (tmp_path / "fin").write_text("{}")
    assert w.trigger_due() is None
    c.sleep(121)
    assert w.trigger_due() == "idle_after_finish"


def test_workload_death_without_marker(tmp_path):
    c = Clock()
    (tmp_path / "pid").write_text(f"{os.getpid()}\n")
    w = make(tmp_path, c, util=lambda: 90.0)
    w.arm()
    assert w.trigger_due() is None
    (tmp_path / "pid").write_text("999999999\n")   # no such process
    assert w.trigger_due() is None
    c.sleep(121)
    assert w.trigger_due() == "workload_died"


def test_never_started_workload_is_stopped(tmp_path):
    c = Clock()
    w = make(tmp_path, c, util=lambda: 0.0, stall_s=10 ** 6)
    w.arm()
    trig = None
    for _ in range(100):
        trig = w.trigger_due()
        if trig:
            break
        c.sleep(10)
    assert trig == "workload_died"


def test_run_stops_once_and_never_destroys(tmp_path):
    c = Clock()
    (tmp_path / "pid").write_text("999999999\n")
    w = make(tmp_path, c)
    w.arm()
    st = w.run()
    assert st["trigger"] == "workload_died" and st["api_accepted"]
    reqs = (tmp_path / "fake" / "stop_requests.jsonl").read_text().splitlines()
    assert len(reqs) == 1 and str(IID) in reqs[0]


def test_save_failure_does_not_prevent_stop(tmp_path, monkeypatch):
    c = Clock()
    (tmp_path / "pid").write_text("999999999\n")
    w = make(tmp_path, c, save_cmds=["exit 3"])
    w.arm()
    monkeypatch.setattr(w, "save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    st = w.run()
    assert st["api_accepted"] and "OSError" in st["save_error"]


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_terminate_workload_by_process_group_not_pattern(tmp_path):
    import subprocess
    import time as _t
    # workload in its own session (as launched by setsid), with a child; the watchdog survives
    p = subprocess.Popen(["bash", "-c", "sleep 300 & sleep 300; wait"], start_new_session=True)
    (tmp_path / "pid").write_text(f"{p.pid}\n")
    w = wdm.Watchdog(IID, None, str(tmp_path / "st"), wdm.FakeAPI(str(tmp_path / "fake")), idle_grace_s=1,
                     finished_marker=str(tmp_path / "fin"), workload_pidfile=str(tmp_path / "pid"), stall_s=600,
                     gpu_util=lambda: 0.0)
    w.arm()
    pgid = os.getpgid(p.pid)
    w.terminate_workload(grace_s=10)
    p.wait(timeout=15)
    _t.sleep(0.5)
    alive = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
    assert alive == [] and w.state["workload_terminate"]["own_group"] is False


def test_monitor_only_never_stops_and_keeps_running(tmp_path, monkeypatch):
    """Plan v1.3: no stop path before project completion. No provider client exists; alerts only."""
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    c = Clock()
    (tmp_path / "pid").write_text("999999999\n")            # workload dead from the start
    w = wdm.Watchdog(IID, None, str(tmp_path / "st"), None, monitor_only=True, workload_pidfile=str(tmp_path / "pid"),
                     stall_s=600, idle_grace_s=60, poll_s=10, gpu_util=lambda: 0.0, clock=c, sleep=c.sleep)
    w.arm()
    st = w.monitor(max_polls=24 * 360)                     # 24 h: died + stalled alerts, never a stop
    assert w.api is None and st["monitor_only"] and st["stop_path"].startswith("disabled")
    assert not st["stop_requested"] and not st["api_accepted"]
    assert {a["alert"] for a in st["alerts"]} <= {"workload_died", "stalled"} and st["alerts"]
    w.stop()                                               # even a direct call is refused
    assert wdm.read_state(str(tmp_path / "st"))["stop_blocked"] is True
    assert not (tmp_path / "fake").exists()
    assert wdm.is_armed(str(tmp_path / "st"), IID)[0]       # pid of this test process is the watchdog pid


def test_monitor_only_cli_refuses_stop_related_options(tmp_path):
    for extra in (["--deadline-epoch", "1e10"], ["--save-cmd", "true"]):
        with pytest.raises(SystemExit):
            wdm.main(["--instance-id", str(IID), "--state-dir", str(tmp_path), "--monitor-only",
                      "--workload-pidfile", str(tmp_path / "p"), *extra])


def test_monitor_only_source_has_no_stop_call_on_its_path():
    src = SCRIPT.read_text(encoding="utf-8")
    body = src.split("def monitor(", 1)[1].split("\n    def ", 1)[0]
    assert "stop(" not in body and "api." not in body


def test_cli_rejects_mixed_or_missing_modes(tmp_path, capsys):
    with pytest.raises(SystemExit):
        wdm.main(["--instance-id", str(IID), "--state-dir", str(tmp_path), "--fake-api-dir", str(tmp_path / "f")])
    with pytest.raises(SystemExit):
        wdm.main(["--instance-id", str(IID), "--state-dir", str(tmp_path), "--fake-api-dir", str(tmp_path / "f"),
                  "--deadline-epoch", "1e10", "--no-financial-deadline"])
