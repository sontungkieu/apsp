#!/usr/bin/env python3
"""Instance cost watchdog. Runs ON the rented GPU instance, detached from any Claude session or SSH connection.

Armed BEFORE the workload with a fixed wall-clock deadline. It stops exactly one instance (the one it runs on) when
  - the deadline is reached ("deadline"), or
  - the workload has finished and an idle grace period has passed ("idle_after_finish").
Before stopping it runs bounded "save" commands (end the runner, write a tarball/state on the instance disk). It never
waits for artifacts to be downloaded to a laptop: the stop request is sent at the deadline at the latest, whether or
not a save command or a download is still running. Stop keeps the disk; this tool has no destroy path.

Stdlib only, so it runs with the system python3 before any venv exists. The API key is read from the environment
(default CONTAINER_API_KEY, which Vast injects into the instance) and is sent only in an HTTP header: it is never
put on a command line, logged or written to the state file.

Stop outcome is recorded in three separate levels:
  stop_requested     the watchdog started a stop call (timestamped before the call)
  api_accepted       the provider answered 2xx without an error flag (or api_rejected / api_error)
  confirmed_stopped  a status read returned a stopped state. From inside the instance this usually cannot be
                     observed (the container is halted), so it normally stays false here and must be confirmed
                     externally (console, or `--check-status` from a machine with API access).

  python3 scripts/vast_watchdog.py --instance-id 56070930 --deadline-epoch 1792060869 \
      --state-dir /workspace/apsp/watchdog --finished-marker /workspace/apsp/diag1.finished --idle-grace-s 900 \
      --save-cmd "pkill -TERM -f '[s]cripts/gpu_smoke.sh'" --save-cmd 'tar --exclude="*.lock" -czf ... ' &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

SCHEMA = "apsp.vast_watchdog/v1"
VAST_BASE = "https://console.vast.ai/api/v0"
STOPPED_STATES = {"stopped", "exited", "offline"}


class WatchdogError(RuntimeError):
    pass


# ------------------------------------------------------------------ providers ----------------------------------
class VastAPI:
    """Minimal Vast REST client. Endpoints follow the vastai CLI (PUT instances/{id}/ {"state": "stopped"}; GET
    instances/{id}/). NOT verified against the live provider by this repo's tests (see docs/COST_WATCHDOG.md)."""

    def __init__(self, key_env: str = "CONTAINER_API_KEY", base: str = VAST_BASE, timeout_s: float = 30.0):
        self._key = os.environ.get(key_env)
        if not self._key:
            raise WatchdogError(f"environment variable {key_env} is not set")
        self.base, self.timeout_s = base.rstrip("/"), timeout_s

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base}/{path}", data=data, method=method,
                                     headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                                              "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                raw = r.read().decode("utf-8", "replace")
                return r.status, _safe_json(raw)
        except urllib.error.HTTPError as e:
            return e.code, _safe_json(e.read().decode("utf-8", "replace"))

    def stop(self, instance_id: int) -> tuple[int, dict]:
        return self._call("PUT", f"instances/{instance_id}/", {"state": "stopped"})

    def status(self, instance_id: int) -> Optional[str]:
        code, body = self._call("GET", f"instances/{instance_id}/")
        if code // 100 != 2:
            return None
        inst = body.get("instances", body)
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
        return (inst or {}).get("actual_status")


class FakeAPI:
    """Test provider. Behaviour from <dir>/mode: accept | reject | error | hang | accept_then_stopped."""

    def __init__(self, d: str):
        self.d = Path(d)
        self.d.mkdir(parents=True, exist_ok=True)

    def _mode(self) -> str:
        p = self.d / "mode"
        return p.read_text().strip() if p.exists() else "accept"

    def stop(self, instance_id: int) -> tuple[int, dict]:
        with open(self.d / "stop_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"instance_id": instance_id, "t": time.time()}) + "\n")
        m = self._mode()
        if m == "hang":
            time.sleep(3600)
        if m == "error":
            raise OSError("simulated network error")
        if m == "reject":
            return 403, {"success": False, "msg": "simulated rejection"}
        if m == "accept_then_stopped":
            (self.d / "status").write_text("stopped")
        return 200, {"success": True}

    def status(self, instance_id: int) -> Optional[str]:
        p = self.d / "status"
        return p.read_text().strip() if p.exists() else "running"


def self_matching_kill_patterns(save_cmds, own_cmdline: str) -> list[str]:
    """Patterns of `pkill -f` / `pgrep -f` / `killall -r` in the save commands that also match the watchdog's OWN
    command line. The save commands are part of that command line, so `pkill -f scripts/gpu_smoke.sh` kills the
    watchdog itself (seen live on 2026-10-16). Bracket one character instead: `pkill -f '[s]cripts/gpu_smoke.sh'`."""
    bad = []
    for cmd in save_cmds:
        for part in re.split(r"[;&|]+", cmd):
            try:
                toks = shlex.split(part)
            except ValueError:
                continue
            if not toks or os.path.basename(toks[0]) not in ("pkill", "pgrep", "killall"):
                continue
            if "-f" not in toks and toks[0].endswith(("pkill", "pgrep")):
                continue
            pats = [t for t in toks[1:] if not t.startswith("-")]
            for pat in pats:
                try:
                    if re.search(pat, own_cmdline):
                        bad.append(pat)
                except re.error:
                    if pat in own_cmdline:
                        bad.append(pat)
    return bad


def _safe_json(raw: str) -> dict:
    try:
        v = json.loads(raw) if raw else {}
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"raw_prefix": raw[:200]}


def _sanitize(body: dict) -> dict:
    """Keep only short scalar fields of a provider answer (no tokens or nested blobs in the state file)."""
    out = {}
    for k, v in list(body.items())[:20]:
        if any(s in k.lower() for s in ("key", "token", "secret", "password", "auth")):
            continue
        if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200):
            out[k] = v
    return out


# ------------------------------------------------------------------ watchdog -----------------------------------
class Watchdog:
    def __init__(self, instance_id: int, deadline_epoch: float, state_dir: str, api, *,
                 finished_marker: Optional[str] = None, idle_grace_s: float = 900.0, save_cmds=(),
                 save_margin_s: float = 300.0, save_cmd_timeout_s: float = 240.0, poll_s: float = 10.0,
                 api_timeout_s: float = 30.0, max_stop_attempts: int = 5, retry_s: float = 20.0,
                 status_checks: int = 3, clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.iid, self.deadline, self.api = int(instance_id), float(deadline_epoch), api
        self.dir = Path(state_dir)
        self.finished_marker = Path(finished_marker) if finished_marker else None
        self.idle_grace_s, self.save_cmds = float(idle_grace_s), list(save_cmds)
        self.save_margin_s, self.save_cmd_timeout_s = float(save_margin_s), float(save_cmd_timeout_s)
        self.poll_s, self.api_timeout_s = float(poll_s), float(api_timeout_s)
        self.max_stop_attempts, self.retry_s, self.status_checks = int(max_stop_attempts), float(retry_s), int(status_checks)
        self.clock, self.sleep = clock, sleep
        self.state: dict = {}

    # -- state ------------------------------------------------------------------------------------------------
    def _write(self, **upd) -> None:
        self.state.update(upd)
        self.state["updated_at"] = self.clock()
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "state.json")

    def _event(self, kind: str, **kw) -> None:
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": self.clock(), "event": kind, **kw}, default=str) + "\n")

    # -- arming -----------------------------------------------------------------------------------------------
    def arm(self, max_hours: float = 6.0, own_instance_id: Optional[str] = None, min_lead_s: float = 60.0,
            own_cmdline: Optional[str] = None) -> None:
        now = self.clock()
        own = own_cmdline if own_cmdline is not None else " ".join(sys.argv)
        bad = self_matching_kill_patterns(self.save_cmds, own)
        if bad:
            raise WatchdogError(f"refusing to arm: save-cmd kill pattern(s) {bad} also match the watchdog's own command "
                                "line; bracket one character, e.g. '[s]cripts/gpu_smoke.sh'")
        if own_instance_id is not None and str(own_instance_id) != str(self.iid):
            raise WatchdogError(f"refusing to arm: this machine is instance {own_instance_id}, not {self.iid}")
        if self.deadline <= now + min_lead_s:
            raise WatchdogError(f"refusing to arm: deadline is less than {min_lead_s:g} s away (or past)")
        if self.deadline > now + max_hours * 3600:
            raise WatchdogError(f"refusing to arm: deadline more than {max_hours} h away")
        if self.save_margin_s >= self.deadline - now:
            raise WatchdogError("refusing to arm: save margin longer than the time to the deadline")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write(schema=SCHEMA, instance_id=self.iid, pid=os.getpid(), armed_at=now, deadline_epoch=self.deadline,
                    idle_grace_s=self.idle_grace_s, save_margin_s=self.save_margin_s, phase="armed", trigger=None,
                    finished_at=None, save=[], stop_requested=False, stop_requested_at=None, api_accepted=False,
                    api_result=None, stop_attempts=[], confirmed_stopped=False, observed_status=[],
                    note="confirmed_stopped is only set when a status read returns a stopped state; from inside the "
                         "instance that is usually impossible, so confirm externally")
        self._event("armed", deadline_epoch=self.deadline)

    # -- main loop --------------------------------------------------------------------------------------------
    def trigger_due(self) -> Optional[str]:
        now = self.clock()
        if self.finished_marker is not None and self.finished_marker.exists():
            if self.state.get("finished_at") is None:
                self._write(finished_at=now)
                self._event("workload_finished_seen")
            if now >= self.state["finished_at"] + self.idle_grace_s:
                return "idle_after_finish"
        if now >= self.deadline - self.save_margin_s:
            return "deadline"
        return None

    def run(self) -> dict:
        while True:
            t = self.trigger_due()
            if t:
                break
            self._write(heartbeat=self.clock())
            self.sleep(self.poll_s)
        self._write(phase="saving", trigger=t, triggered_at=self.clock())
        self._event("triggered", trigger=t)
        self.save()
        self.stop()
        return self.state

    def save(self) -> None:
        """Bounded: every command ends before the deadline (killed with its process group if needed)."""
        results = []
        for cmd in self.save_cmds:
            left = self.deadline - self.clock()
            if left <= 1:
                results.append({"cmd_index": len(results), "skipped": "deadline reached"})
                continue
            tmo = min(self.save_cmd_timeout_s, left)
            t0 = self.clock()
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=(os.name == "posix"))
            try:
                rc = p.wait(timeout=tmo)
                results.append({"cmd_index": len(results), "rc": rc, "seconds": round(self.clock() - t0, 2)})
            except subprocess.TimeoutExpired:
                _kill_tree(p)
                results.append({"cmd_index": len(results), "timeout_s": round(tmo, 2), "killed": True})
            self._write(save=results)
        self._write(phase="saved", save=results, saved_at=self.clock())
        self._event("saved", n=len(results))

    def stop(self) -> None:
        attempts = []
        for i in range(self.max_stop_attempts):
            now = self.clock()
            if not self.state.get("stop_requested"):
                self._write(stop_requested=True, stop_requested_at=now, phase="stop_requested")
            self._event("stop_request", attempt=i + 1)
            res = _call_with_timeout(lambda: self.api.stop(self.iid), self.api_timeout_s)
            if res[0] == "ok":
                code, body = res[1]
                ok = code // 100 == 2 and body.get("success", True) is not False
                a = {"attempt": i + 1, "http_status": code, "body": _sanitize(body), "accepted": ok}
            else:
                a = {"attempt": i + 1, "error": res[0], "detail": str(res[1])[:200], "accepted": False}
            attempts.append(a)
            self._write(stop_attempts=attempts)
            if a["accepted"]:
                self._write(api_accepted=True, api_result="accepted", phase="api_accepted")
                self._event("api_accepted")
                break
            self._write(api_result="rejected" if "http_status" in a else "error")
            self.sleep(self.retry_s)
        else:
            self._write(phase="stop_failed")
            self._event("stop_failed")
            return
        for _ in range(self.status_checks):
            res = _call_with_timeout(lambda: self.api.status(self.iid), self.api_timeout_s)
            st = res[1] if res[0] == "ok" else f"unreadable:{res[0]}"
            self._write(observed_status=self.state["observed_status"] + [{"t": self.clock(), "status": st}])
            if st in STOPPED_STATES:
                self._write(confirmed_stopped=True, phase="confirmed_stopped")
                self._event("confirmed_stopped")
                return
            self.sleep(self.retry_s)


def _call_with_timeout(fn, timeout_s: float):
    """Run fn in a daemon thread; ('ok', value) | ('timeout', None) | ('exception', exc)."""
    import threading
    box = {}

    def target():
        try:
            box["v"] = ("ok", fn())
        except Exception as e:  # noqa: BLE001
            box["v"] = ("exception", e)
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)
    return box.get("v", ("timeout", None))


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def read_state(state_dir: str) -> dict:
    p = Path(state_dir) / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def is_armed(state_dir: str, instance_id: Optional[int] = None, now: Optional[float] = None) -> tuple[bool, str]:
    """For the runner: armed, the process alive, deadline in the future, and (optionally) the right instance."""
    st = read_state(state_dir)
    now = time.time() if now is None else now
    if st.get("schema") != SCHEMA or st.get("phase") != "armed":
        return False, f"watchdog not armed (phase={st.get('phase')})"
    if instance_id is not None and int(st.get("instance_id", -1)) != int(instance_id):
        return False, "watchdog armed for another instance"
    if float(st.get("deadline_epoch", 0)) <= now:
        return False, "watchdog deadline already passed"
    try:
        alive = _pid_alive(int(st["pid"]))
    except (KeyError, ValueError):
        alive = False
    if not alive:
        return False, "watchdog process not running"
    return True, "ok"


def _pid_alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)          # signal 0: existence check only (POSIX)
            return True
        except PermissionError:
            return True
        except OSError:
            return False
    # Windows: os.kill(pid, 0) would send CTRL_C_EVENT, so query the process handle instead
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259          # STILL_ACTIVE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="instance cost watchdog (stop, never destroy)")
    ap.add_argument("--instance-id", type=int, required=True)
    ap.add_argument("--deadline-epoch", type=float)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--finished-marker")
    ap.add_argument("--idle-grace-s", type=float, default=900)
    ap.add_argument("--save-cmd", action="append", default=[])
    ap.add_argument("--save-margin-s", type=float, default=300)
    ap.add_argument("--save-cmd-timeout-s", type=float, default=240)
    ap.add_argument("--poll-s", type=float, default=10)
    ap.add_argument("--retry-s", type=float, default=20)
    ap.add_argument("--max-hours", type=float, default=6)
    ap.add_argument("--min-lead-s", type=float, default=60, help="refuse to arm closer than this to the deadline")
    ap.add_argument("--stop-attempts", type=int, default=5)
    ap.add_argument("--status-checks", type=int, default=3)
    ap.add_argument("--api-timeout-s", type=float, default=30)
    ap.add_argument("--api-key-env", default="CONTAINER_API_KEY")
    ap.add_argument("--own-id-env", default="CONTAINER_ID",
                    help="env var holding this machine's instance id; arming refuses a mismatch")
    ap.add_argument("--fake-api-dir", help="TEST ONLY: file-based fake provider")
    ap.add_argument("--check-armed", action="store_true", help="exit 0 if armed and alive, else 1 (for the runner)")
    ap.add_argument("--check-status", action="store_true", help="print the provider status of the instance and exit")
    a = ap.parse_args(argv)
    if a.check_armed:
        ok, why = is_armed(a.state_dir, a.instance_id)
        print(why)
        return 0 if ok else 1
    api = FakeAPI(a.fake_api_dir) if a.fake_api_dir else VastAPI(a.api_key_env)
    if a.check_status:
        print(json.dumps({"instance_id": a.instance_id, "actual_status": api.status(a.instance_id)}))
        return 0
    if a.deadline_epoch is None:
        ap.error("--deadline-epoch is required")
    wd = Watchdog(a.instance_id, a.deadline_epoch, a.state_dir, api, finished_marker=a.finished_marker,
                  idle_grace_s=a.idle_grace_s, save_cmds=a.save_cmd, save_margin_s=a.save_margin_s,
                  save_cmd_timeout_s=a.save_cmd_timeout_s, poll_s=a.poll_s, retry_s=a.retry_s,
                  max_stop_attempts=a.stop_attempts, status_checks=a.status_checks, api_timeout_s=a.api_timeout_s)
    wd.arm(max_hours=a.max_hours, own_instance_id=os.environ.get(a.own_id_env), min_lead_s=a.min_lead_s)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)     # survive the SSH session that launched it
    st = wd.run()
    return 0 if st.get("api_accepted") else 2


if __name__ == "__main__":
    sys.exit(main())
