"""Heartbeat with real progress (stdlib only; runs with the system python3, independent of the laptop).

Every POLL seconds writes $APSP_STORE/heartbeat.json and appends heartbeat.jsonl with one state:
  gpu_compute        a GPU process of the workload is running and GPU utilisation > 0 in the last sample window
  cpu_progress       a CPU job of run_project is running and its outputs/logs grew since the previous sample
  upload             the backup daemon is transferring (rclone running) and nothing else is computing
  cpu_setup          environment / evaluator setup scripts running (installs, pinned downloads)
  waiting_dependency run_project is not running or has no runnable job: the project state lists blocked jobs only
  job_dead           run_project/job pid gone while the job manifest says running, or no progress for STALL_S
  idle_unknown       none of the above (reported, never hidden)
Progress counters: bank shards, episode lines per run, newest output mtime, failed jobs. Errors are written to
$APSP_STORE/ALERTS.jsonl as soon as they appear (failed job, dead job, stall); the backup daemon ships both files.
This process creates no artificial load and never touches any platform idle mechanism.

  APSP_STORE=/workspace/apsp setsid nohup python3 scripts/heartbeat.py > $APSP_STORE/heartbeat.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

STORE = Path(os.environ.get("APSP_STORE", "/workspace/apsp"))
POLL = int(os.environ.get("HEARTBEAT_POLL_S", "60"))
STALL_S = int(os.environ.get("HEARTBEAT_STALL_S", "1800"))
PROJECT_RUN = os.environ.get("PROJECT_RUN", "apsp_full_sd15_v1")


def _alive(pidfile: Path) -> bool:
    try:
        os.kill(int(pidfile.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def gpu_sample() -> dict:
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0].split(",")
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.split()
        return {"util": int(u[0]), "mem_mib": int(u[1]), "n_procs": len(apps)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def progress() -> dict:
    shards = STORE / "banks" / "PILOT" / "shards"
    runs = STORE / "runs"
    eps = {}
    newest = 0.0
    if runs.exists():
        for p in runs.glob("*/*/episodes.jsonl"):
            try:
                with open(p, "rb") as fh:
                    eps[str(p.relative_to(runs))] = sum(1 for _ in fh)
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    for d in (STORE / "project_runs" / PROJECT_RUN / "logs", runs, shards):
        if d.exists():
            for p in d.rglob("*"):
                try:
                    if p.is_file():
                        newest = max(newest, p.stat().st_mtime)
                except OSError:
                    pass
    n_sh = sum(1 for p in shards.iterdir() if p.is_dir() and not p.name.startswith(".tmp")) if shards.exists() else 0
    return {"bank_shards": n_sh, "episodes": eps, "newest_output_mtime": newest}


def project() -> dict:
    st = {}
    try:
        st = json.loads((STORE / "project_state.json").read_text())
    except (OSError, ValueError):
        pass
    running = None
    man = STORE / "project_runs" / PROJECT_RUN / "job_manifest.jsonl"
    if man.exists():
        last = {}
        for l in man.read_text().splitlines()[-400:]:
            try:
                e = json.loads(l)
                last[e["id"]] = e
            except (ValueError, KeyError):
                pass
        rs = [e for e in last.values() if e.get("status") == "running"]
        running = rs[-1] if rs else None
    return {"run_project_alive": _alive(STORE / "run_project.pid"), "supervisor_alive": _alive(STORE / "supervisor.pid"),
            "backup_alive": _alive(STORE / "backup.pid"), "running_job": running and {k: running.get(k) for k in ("id", "phase", "gpu")},
            "failed": st.get("failed", []), "blocked_n": len(st.get("blocked", {})), "done_n": len(st.get("done", [])),
            "project_state": st.get("project_state")}


def rclone_active() -> bool:
    """Our backup daemon is transferring (its own rclone child), not any rclone process of the platform."""
    try:
        pid = (STORE / "backup.pid").read_text().strip()
        return subprocess.run(["pgrep", "-P", pid, "-x", "rclone"], capture_output=True).returncode == 0
    except (OSError, ValueError):
        return False


def setup_active() -> bool:
    try:
        return subprocess.run(["pgrep", "-f", "setup_gpu_linux.sh|setup_evaluators.sh"], capture_output=True).returncode == 0
    except OSError:
        return False


def classify(g: dict, p: dict, prev: dict | None, pr: dict, now: float) -> tuple[str, str]:
    job = pr["running_job"]
    grew = prev is not None and p["newest_output_mtime"] > prev.get("newest_output_mtime", 0)
    if job and not pr["run_project_alive"]:
        return "job_dead", f"job {job['id']} marked running but run_project is gone"
    if pr["run_project_alive"] and job and job.get("gpu") and g.get("util", 0) > 0 and g.get("n_procs", 0) > 0:
        return "gpu_compute", f"{job['id']} util {g['util']}%"
    if pr["run_project_alive"] and job and (grew or now - p["newest_output_mtime"] < STALL_S):
        return ("cpu_progress" if not job.get("gpu") else "gpu_job_low_util"), f"{job['id']} outputs advancing"
    if pr["run_project_alive"] and job:
        return "job_dead", f"{job['id']}: no output progress for > {STALL_S}s"
    if rclone_active():
        return "upload", "backup transfer in progress"
    if setup_active():
        return "cpu_setup", "environment / evaluator setup (installs, pinned downloads) in progress"
    if not pr["run_project_alive"] and pr["blocked_n"] and not pr["failed"]:
        return "waiting_dependency", f"{pr['blocked_n']} jobs blocked ({pr['project_state']})"
    if pr["failed"]:
        return "failed_jobs", f"failed: {pr['failed']}"
    return "idle_unknown", "no running job, no upload, no blocked work recorded"


def main() -> None:
    (STORE / "heartbeat.pid").write_text(str(os.getpid()))
    prev, last_state = None, None
    while True:
        now = time.time()
        g, p, pr = gpu_sample(), progress(), project()
        state, why = classify(g, p, prev, pr, now)
        hb = {"t": now, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "state": state, "why": why,
              "gpu": g, "progress": p, "project": pr, "instance": os.environ.get("CONTAINER_ID")}
        tmp = STORE / ".heartbeat.tmp"
        tmp.write_text(json.dumps(hb, indent=1))
        os.replace(tmp, STORE / "heartbeat.json")
        with open(STORE / "heartbeat.jsonl", "a") as fh:
            fh.write(json.dumps({k: hb[k] for k in ("utc", "state", "why")} | {"shards": p["bank_shards"],
                                "util": g.get("util")}) + "\n")
        if state in ("job_dead", "failed_jobs", "idle_unknown") and state != last_state:
            with open(STORE / "ALERTS.jsonl", "a") as fh:
                fh.write(json.dumps({"utc": hb["utc"], "state": state, "why": why}) + "\n")
        prev, last_state = p, state
        time.sleep(POLL)


if __name__ == "__main__":
    main()
