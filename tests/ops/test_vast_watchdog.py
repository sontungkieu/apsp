"""scripts/vast_watchdog.py: arming guards, deadline / idle triggers, bounded save, stop levels, detachment from the
launching session, hanging downloads, and that the API key never reaches argv, logs or the state file.

Everything runs against FakeAPI or a local HTTP mock. Nothing here talks to the real provider.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vast_watchdog.py"
spec = importlib.util.spec_from_file_location("vast_watchdog", SCRIPT)
wdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wdm)

IID = 56070930
PY = f'"{sys.executable}"'


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(tmp_path, clock, mode="accept", **kw):
    api = wdm.FakeAPI(str(tmp_path / "fake"))
    (tmp_path / "fake" / "mode").write_text(mode)
    kw.setdefault("save_margin_s", 300)
    kw.setdefault("retry_s", 1)
    kw.setdefault("poll_s", 10)
    w = wdm.Watchdog(IID, clock() + 3600, str(tmp_path / "st"), api, clock=clock, sleep=clock.sleep, **kw)
    return w, api


def stops(tmp_path):
    p = tmp_path / "fake" / "stop_requests.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------ arming --------------------------------------
def test_arm_refuses_wrong_instance_and_bad_deadlines(tmp_path):
    c = Clock()
    w, _ = make(tmp_path, c)
    with pytest.raises(wdm.WatchdogError, match="instance 123"):
        w.arm(own_instance_id="123")
    w2 = wdm.Watchdog(IID, c() + 30, str(tmp_path / "s2"), wdm.FakeAPI(str(tmp_path / "f2")), clock=c)
    with pytest.raises(wdm.WatchdogError, match="less than"):
        w2.arm()
    w3 = wdm.Watchdog(IID, c() + 7 * 3600, str(tmp_path / "s3"), wdm.FakeAPI(str(tmp_path / "f3")), clock=c)
    with pytest.raises(wdm.WatchdogError, match="more than"):
        w3.arm(max_hours=6)
    w4 = wdm.Watchdog(IID, c() + 200, str(tmp_path / "s4"), wdm.FakeAPI(str(tmp_path / "f4")), clock=c, save_margin_s=300)
    with pytest.raises(wdm.WatchdogError, match="save margin"):
        w4.arm()
    w.arm(own_instance_id=str(IID))
    assert wdm.read_state(str(tmp_path / "st"))["phase"] == "armed"


# ------------------------------------------------------------------ triggers and stop levels -------------------
def test_deadline_stop_accepted_but_not_confirmed(tmp_path):
    c = Clock()
    w, _ = make(tmp_path, c)
    w.arm()
    st = w.run()
    assert st["trigger"] == "deadline"
    assert st["triggered_at"] >= w.deadline - w.save_margin_s
    assert st["stop_requested"] and st["api_accepted"] and st["api_result"] == "accepted"
    assert st["confirmed_stopped"] is False and st["phase"] == "api_accepted"
    assert [o["status"] for o in st["observed_status"]] == ["running"] * 3
    assert st["stop_requested_at"] <= w.deadline
    assert len(stops(tmp_path)) == 1 and stops(tmp_path)[0]["instance_id"] == IID


def test_confirmed_only_when_status_reads_stopped(tmp_path):
    c = Clock()
    w, _ = make(tmp_path, c, mode="accept_then_stopped")
    w.arm()
    st = w.run()
    assert st["api_accepted"] and st["confirmed_stopped"] and st["phase"] == "confirmed_stopped"


@pytest.mark.parametrize("mode,result", [("reject", "rejected"), ("error", "error")])
def test_rejected_or_failing_api_is_retried_then_reported(tmp_path, mode, result):
    c = Clock()
    w, _ = make(tmp_path, c, mode=mode, max_stop_attempts=3)
    w.arm()
    st = w.run()
    assert st["stop_requested"] and not st["api_accepted"] and st["api_result"] == result
    assert st["phase"] == "stop_failed" and len(st["stop_attempts"]) == 3 and len(stops(tmp_path)) == 3
    assert not st["confirmed_stopped"]


def test_hanging_api_call_times_out(tmp_path):
    c = Clock()
    w, _ = make(tmp_path, c, mode="hang", max_stop_attempts=2, api_timeout_s=0.2)
    w.arm()
    st = w.run()
    assert [a["error"] for a in st["stop_attempts"]] == ["timeout", "timeout"] and st["phase"] == "stop_failed"


def test_idle_after_finish_triggers_before_deadline(tmp_path):
    c = Clock()
    marker = tmp_path / "run.finished"
    w, _ = make(tmp_path, c, finished_marker=str(marker), idle_grace_s=600)
    w.arm()
    assert w.trigger_due() is None
    marker.write_text("")
    assert w.trigger_due() is None          # grace starts when the marker is first seen
    c.t += 599
    assert w.trigger_due() is None
    c.t += 1
    assert w.trigger_due() == "idle_after_finish"
    st = w.run()
    assert st["trigger"] == "idle_after_finish" and st["api_accepted"] and st["triggered_at"] < w.deadline - 300


def test_hanging_save_command_is_killed_and_stop_still_precedes_deadline(tmp_path):
    """Real clock: the save command would run 60 s; the deadline is ~3 s away; stop must be sent by the deadline."""
    api = wdm.FakeAPI(str(tmp_path / "fake"))
    t0 = time.time()
    w = wdm.Watchdog(IID, t0 + 3.0, str(tmp_path / "st"), api, save_margin_s=2.0, save_cmd_timeout_s=30, poll_s=0.1,
                     retry_s=0.1, status_checks=1,
                     save_cmds=[f'{PY} -c "import time; time.sleep(60)"', f'{PY} -c "pass"'])
    w.arm(min_lead_s=1)
    st = w.run()
    req = stops(tmp_path)[0]["t"]
    assert req <= t0 + 3.0 + 1.0, "stop request must not wait for the save command"
    assert st["save"][0].get("killed") is True
    assert st["save"][1].get("skipped") == "deadline reached" or "rc" in st["save"][1]
    assert st["api_accepted"]


# ------------------------------------------------------------------ detached process: session loss, download hang
def _launch_detached(args, log):
    kw = {"stdin": subprocess.DEVNULL, "stdout": open(log, "w"), "stderr": subprocess.STDOUT}
    if os.name == "posix":
        kw["start_new_session"] = True
    else:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008   # DETACHED_PROCESS
    return subprocess.Popen(args, **kw)


def _wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_watchdog_survives_loss_of_the_launching_session(tmp_path):
    """A 'session' process (stand-in for Claude's SSH shell) launches the watchdog detached and is then killed with
    its whole process group. The watchdog must still stop the instance at its deadline."""
    fake, st = tmp_path / "fake", tmp_path / "st"
    deadline = time.time() + 4
    wd_args = [sys.executable, str(SCRIPT), "--instance-id", str(IID), "--deadline-epoch", str(deadline),
               "--state-dir", str(st), "--fake-api-dir", str(fake), "--save-margin-s", "1", "--poll-s", "0.2",
               "--retry-s", "0.2", "--status-checks", "1", "--min-lead-s", "1"]
    session_code = (
        "import subprocess, sys, os, time\n"
        f"args = {wd_args!r}\n"
        "kw = {'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}\n"
        "kw.update({'start_new_session': True} if os.name == 'posix' else {'creationflags': 0x200 | 0x8})\n"
        "subprocess.Popen(args, **kw)\n"
        "time.sleep(600)\n")
    sess = subprocess.Popen([sys.executable, "-c", session_code], start_new_session=(os.name == "posix"))
    assert _wait(lambda: wdm.read_state(str(st)).get("phase") == "armed", 15)
    if os.name == "posix":
        os.killpg(sess.pid, signal.SIGKILL)          # the SSH session and everything in its group dies
    else:
        sess.kill()
    sess.wait(timeout=10)
    assert _wait(lambda: wdm.read_state(str(st)).get("api_accepted") is True, 20)
    s = wdm.read_state(str(st))
    assert s["trigger"] == "deadline" and s["stop_requested_at"] <= deadline + 1
    assert len(stops(tmp_path)) == 1


def test_hanging_download_does_not_delay_stop(tmp_path):
    """The workload finished; a 'download' (stand-in for scp from the laptop) keeps reading the tarball and never
    ends. The watchdog stops after the idle grace while the download is still running."""
    fake, st = tmp_path / "fake", tmp_path / "st"
    tarball = tmp_path / "artifacts.tgz"
    tarball.write_bytes(os.urandom(1024))
    marker = tmp_path / "run.finished"
    marker.write_text("")
    dl = subprocess.Popen([sys.executable, "-c",
                           f"import time\nf = open(r'{tarball}', 'rb')\nwhile True:\n    f.seek(0); f.read(); time.sleep(0.05)"])
    try:
        wd = _launch_detached([sys.executable, str(SCRIPT), "--instance-id", str(IID), "--deadline-epoch",
                               str(time.time() + 120), "--state-dir", str(st), "--fake-api-dir", str(fake),
                               "--finished-marker", str(marker), "--idle-grace-s", "1.5", "--save-margin-s", "10",
                               "--poll-s", "0.2", "--retry-s", "0.2", "--status-checks", "1", "--min-lead-s", "1"],
                              tmp_path / "wd.log")
        assert _wait(lambda: wdm.read_state(str(st)).get("api_accepted") is True, 20), (tmp_path / "wd.log").read_text()
        assert dl.poll() is None, "download still running when the stop was accepted"
        s = wdm.read_state(str(st))
        assert s["trigger"] == "idle_after_finish"
        assert s["stop_requested_at"] - s["finished_at"] < 10
        wd.wait(timeout=10)
    finally:
        dl.kill()
        dl.wait(timeout=10)


# ------------------------------------------------------------------ runner gate ---------------------------------
def test_is_armed_gate(tmp_path):
    c = Clock(time.time())
    w, _ = make(tmp_path, c)
    w.arm()
    assert wdm.is_armed(str(tmp_path / "st"), IID) == (True, "ok")
    assert wdm.is_armed(str(tmp_path / "st"), 1)[0] is False
    assert wdm.is_armed(str(tmp_path / "st"), IID, now=c() + 7200)[0] is False
    s = wdm.read_state(str(tmp_path / "st"))
    s["pid"] = 2 ** 22 + 12345
    (tmp_path / "st" / "state.json").write_text(json.dumps(s))
    assert wdm.is_armed(str(tmp_path / "st"), IID) == (False, "watchdog process not running")
    assert wdm.is_armed(str(tmp_path / "none"))[0] is False


# ------------------------------------------------------------------ real client vs local mock: secrets ----------
class _Mock(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def _reply(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        _Mock.seen.append(("PUT", self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n))))
        self._reply(200, {"success": True, "api_key_echo": "SHOULD-NOT-BE-STORED"})

    def do_GET(self):
        _Mock.seen.append(("GET", self.path, self.headers.get("Authorization"), None))
        self._reply(200, {"instances": {"id": IID, "actual_status": "running"}})

    def log_message(self, *a):
        pass


def test_vast_client_sends_key_only_in_header_and_never_stores_it(tmp_path, monkeypatch):
    secret = "sk-test-NOT-REAL-0123456789"
    monkeypatch.setenv("TEST_VAST_KEY", secret)
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Mock)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        api = wdm.VastAPI("TEST_VAST_KEY", base=f"http://127.0.0.1:{srv.server_port}/api/v0", timeout_s=5)
        c = Clock(time.time())
        w = wdm.Watchdog(IID, c() + 3600, str(tmp_path / "st"), api, clock=c, sleep=c.sleep, retry_s=1)
        w.arm()
        st = w.run()
    finally:
        srv.shutdown()
    put = [s for s in _Mock.seen if s[0] == "PUT"][-1]
    assert put[1] == f"/api/v0/instances/{IID}/" and put[3] == {"state": "stopped"} and put[2] == f"Bearer {secret}"
    assert st["api_accepted"] and not st["confirmed_stopped"]
    assert st["observed_status"][0]["status"] == "running"
    blob = (tmp_path / "st" / "state.json").read_text() + (tmp_path / "st" / "events.jsonl").read_text()
    assert secret not in blob and "SHOULD-NOT-BE-STORED" not in blob
    # no secret on any command line: the CLI never accepts the key as an argument
    assert "--api-key " not in SCRIPT.read_text().replace("--api-key-env", "")


def test_missing_key_refuses(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY_VAR", raising=False)
    with pytest.raises(wdm.WatchdogError, match="not set"):
        wdm.VastAPI("NO_SUCH_KEY_VAR")


@pytest.mark.skipif(os.name != "posix", reason="runner gate is exercised on Linux (python3 + flock)")
def test_runner_refuses_without_armed_watchdog_on_vast(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "APSP_STORE": str(tmp_path / "store"), "RUN_ID": "gate", "CONTAINER_ID": str(IID)}
    env.pop("REQUIRE_WATCHDOG", None)
    r = subprocess.run(["bash", "scripts/gpu_smoke.sh", "02_env_check"], cwd=root, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 3 and "cost watchdog not armed" in r.stderr
    assert not (tmp_path / "store" / "runs" / "gate" / "summary.tsv").exists()   # nothing ran
