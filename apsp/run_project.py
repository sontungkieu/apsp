"""Sequential project orchestrator with resume, budget/price preflight, PREREG gates, watchdog gate and a duplicate-run
lock (docs/FULL_EXPERIMENT_PLAN.md §9).

  python -m apsp.run_project --plan configs/full_experiments.yaml --status            # table, no execution
  python -m apsp.run_project --plan configs/full_experiments.yaml --resume --dry-run  # what would run next and why
  python -m apsp.run_project --plan configs/full_experiments.yaml --resume            # run every eligible job in order
  python -m apsp.run_project --plan ... --resume --only p1_m3m9                       # one job (dependencies still checked)

State lives in <runs_root>/<project_run>/: job_manifest.jsonl (append-only events), jobs/<id>.json (current record),
budget.json (cumulative spend across restarts), logs/<id>.log, .lock.

A job runs only if
  - it is enabled and not already done with valid provenance (outputs exist, are non-empty, and the command/config
    hash matches; a done job whose hash changed is re-run under a NEW attempt, never silently reused),
  - every dependency is done,
  - the PREREG markers it lists are really signed (apsp.contracts.prereg.is_signed; never written here),
  - extra gates hold (e.g. configs/evaluation.yaml approved),
  - for GPU jobs: an approved budget exists, the hourly price read at start is <= the approved maximum, the remaining
    budget covers the job's forecast + the pack/stop margin, and (on a Vast instance) a watchdog is armed for this
    instance with enough time left before its deadline.
    `financial_cap_mode: none` (explicit, D47) drops the budget and price checks only; null budget fields without it
    still block. Spend is recorded either way, and the instance/watchdog checks stay.
With `workload_pidfile` / `finished_marker` in the plan, a --resume run writes its pid at start and the marker at exit
(whatever the outcome), so the watchdog stops the instance once no eligible work is left.
Otherwise the job is `blocked` with a machine-readable reason. A failing job is `failed` with a class
(engineering / data / gate / budget / timeout); dependants stay blocked, independent jobs continue.
Spend is wall time x quoted hourly price (estimate; the provider bill is authoritative).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from apsp.contracts.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[1]
STATUSES = ("pending", "running", "done", "failed", "blocked", "disabled")
EXIT_CLASS = {0: "ok", 2: "budget_stop", 3: "gate_refused", 4: "data_error", 124: "timeout", 137: "killed"}


class PlanError(ValueError):
    pass


@dataclass
class Job:
    id: str
    phase: str
    cmd: str
    needs: list = field(default_factory=list)
    prereg: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    gpu: bool = False
    est_hours: Optional[float] = None
    outputs: list = field(default_factory=list)
    timeout_min: int = 60
    enabled: bool = True
    notes: str = ""

    def config_hash(self, env: dict) -> str:
        return sha256_hex({"cmd": expand(self.cmd, env), "outputs": [expand(o, env) for o in self.outputs],
                           "prereg": self.prereg, "gates": self.gates})[:16]


def expand(s: str, env: dict) -> str:
    out = s
    for _ in range(3):
        for k, v in env.items():
            out = out.replace("${" + k + "}", str(v))
    return out


def load_plan(path: str | Path) -> tuple[dict, list[Job]]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    jobs = [Job(**j) for j in d.pop("jobs")]
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate job id")
    pos = {j: i for i, j in enumerate(ids)}
    for j in jobs:
        for n in j.needs:
            if n not in pos:
                raise PlanError(f"{j.id} needs unknown job {n}")
            if pos[n] >= pos[j.id]:
                raise PlanError(f"{j.id} needs {n}, which comes later: the plan must be in dependency order")
        bad = set(j.prereg) - {"A", "B", "C"}
        if bad:
            raise PlanError(f"{j.id}: unknown PREREG {bad}")
    if d.get("destroy_instance"):
        raise PlanError("destroy_instance must be false: this orchestrator never destroys")
    if d.get("financial_cap_mode", "approved_budget") not in FINANCIAL_MODES:
        raise PlanError(f"financial_cap_mode must be one of {FINANCIAL_MODES}")
    return d, jobs


FINANCIAL_MODES = ("approved_budget", "none")


def no_financial_cap(plan: dict) -> bool:
    return plan.get("financial_cap_mode") == "none"


# ------------------------------------------------------------------------- state -----------------------------------
class State:
    def __init__(self, root: Path):
        self.root = root
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)

    def get(self, jid: str) -> dict:
        p = self.root / "jobs" / f"{jid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"id": jid, "status": "pending", "attempts": 0}

    def put(self, rec: dict) -> None:
        p = self.root / "jobs" / f"{rec['id']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, p)
        with open(self.root / "job_manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **{k: rec.get(k) for k in
                    ("id", "status", "reason", "attempts", "exit_code", "failure_class", "config_hash", "seconds",
                     "cost_usd_est")}}, default=str) + "\n")

    # budget: cumulative over restarts; sessions are appended, never reset by a new RUN_ID
    def budget(self) -> dict:
        p = self.root / "budget.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"sessions": [], "adjustments": []}

    def save_budget(self, b: dict) -> None:
        p = self.root / "budget.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(b, indent=1), encoding="utf-8")
        os.replace(tmp, p)


def spent_usd(b: dict, now: Optional[float] = None) -> float:
    now = time.time() if now is None else now
    s = 0.0
    for x in b["sessions"]:
        end = x.get("end") or now
        s += max(0.0, end - x["start"]) / 3600.0 * float(x["usd_per_h"])
    return s + sum(float(a["usd"]) for a in b.get("adjustments", []))


def outputs_ok(job: Job, env: dict) -> tuple[bool, dict]:
    sums = {}
    for o in job.outputs:
        p = Path(expand(o, env))
        if not p.exists():
            return False, {"missing": str(p)}
        if p.is_file():
            if p.stat().st_size == 0:
                return False, {"empty": str(p)}
            sums[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.is_dir() and not any(p.iterdir()):
            return False, {"empty_dir": str(p)}
        else:
            sums[str(p)] = "dir"
    return True, sums


# ------------------------------------------------------------------------- gates -----------------------------------
def gate_status(name: str, root: Path = REPO, env: Optional[dict] = None) -> tuple[bool, str]:
    if name == "pool_check_calibration":
        # the GPU full-pool check found no near-duplicate component joining a calibration prompt to another split
        p = Path(expand("${CAL}/calib/pool_full_check.json", env or {}))
        if not p.exists():
            return False, f"{p} missing"
        rep = json.loads(p.read_text(encoding="utf-8"))
        cal = [r for r in rep.get("cross_split_full_pool", []) if r.get("split") == "calibration"]
        bad = cal or rep.get("selected_not_in_survivors")
        return not bad, "full-pool check: no calibration prompt in a cross-split component"
    if name == "evaluation_approved":
        cfg = yaml.safe_load((root / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
        ok = bool(cfg.get("approved")) and all(cfg.get(k) is not None for k in ("epsilon_ir", "r_fail", "n_primary_contrasts"))
        return ok, "configs/evaluation.yaml approved with epsilon_ir, r_fail, n_primary_contrasts"
    if name == "pool_frozen":
        p = root / "reports" / "prompt_pool" / "POOL_FROZEN.json"
        return p.exists(), "research marker reports/prompt_pool/POOL_FROZEN.json (Q15/Q16 approval)"
    if name == "hpsv2_pinned":
        cfg = yaml.safe_load((root / "configs" / "evaluators.yaml").read_text(encoding="utf-8"))["hpsv2"]
        ok = all(cfg.get(k) for k in ("package_version", "checkpoint_revision", "checkpoint_sha256"))
        return ok, "configs/evaluators.yaml hpsv2 fully pinned"
    if name.startswith("marker:"):
        from apsp.contracts.prereg import signed_dir
        m = name.split(":", 1)[1]
        return (signed_dir() / f"{m}.json").exists(), f"research marker {m}"
    if name == "never":
        return False, "deferred by plan (not part of this budget)"
    raise PlanError(f"unknown gate {name}")


@dataclass
class Runtime:
    price_usd_h: Optional[float]
    price_source: str
    watchdog: Callable[[], tuple[bool, str, Optional[float]]]   # -> (armed, why, deadline_epoch)
    on_vast: bool


def read_vast_price(instance_id: str) -> Optional[float]:
    """dph_total of this instance from the provider (key from the environment, never printed)."""
    key = os.environ.get("CONTAINER_API_KEY")
    if not key:
        return None
    import urllib.request
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{instance_id}/",
                                 headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None
    inst = d.get("instances", d)
    inst = inst[0] if isinstance(inst, list) and inst else inst
    v = (inst or {}).get("dph_total")
    return float(v) if v is not None else None


def default_runtime(plan: dict, price_override: Optional[float]) -> Runtime:
    iid = os.environ.get("CONTAINER_ID")
    on_vast = bool(iid)
    if price_override is not None:
        price, src = float(price_override), "--price-usd-h"
    elif on_vast:
        price, src = read_vast_price(iid), "vast api dph_total"
    else:
        price, src = None, "none (not on an instance)"

    def wd():
        if not on_vast:
            return True, "not on a Vast instance", None
        sys.path.insert(0, str(REPO / "scripts"))
        import vast_watchdog as vw
        sd = os.path.expandvars(plan.get("watchdog_state_dir", "/workspace/apsp/watchdog"))
        ok, why = vw.is_armed(sd, int(plan["instance_id"]))
        st = vw.read_state(sd)
        return ok, why, float(st["deadline_epoch"]) if st.get("deadline_epoch") else None
    return Runtime(price, src, wd, on_vast)


def blocked_reason(job: Job, plan: dict, state: State, env: dict, rt: Runtime, now: float) -> Optional[str]:
    from apsp.contracts.prereg import is_signed
    if not job.enabled:
        return "disabled"
    for n in job.needs:
        if state.get(n)["status"] != "done":
            return f"needs:{n}"
    for x in job.prereg:
        if not is_signed(x):
            return f"prereg:{x}"
    for g in job.gates:
        ok, what = gate_status(g, env=env)
        if not ok:
            return f"gate:{g} ({what})"
    if not job.gpu:
        return None
    if rt.on_vast and str(os.environ.get("CONTAINER_ID")) != str(plan.get("instance_id")):
        return f"instance:{os.environ.get('CONTAINER_ID')}!={plan.get('instance_id')}"
    est = job.est_hours
    if no_financial_cap(plan):
        ok, why, deadline = rt.watchdog()
        if not ok:
            return f"watchdog:{why}"
        if deadline is not None and est is not None and now + est * 3600 > deadline:
            return f"watchdog:deadline too close for {est:.2f} h"
        return None
    total = plan.get("approved_total_usd")
    if total is None:
        return "budget:not_approved"
    if rt.price_usd_h is None:
        return f"price:unknown ({rt.price_source})"
    cap_h = plan.get("approved_max_usd_per_h")
    if cap_h is None or rt.price_usd_h > float(cap_h) + 1e-9:
        return f"price:{rt.price_usd_h:.3f}>approved_max {cap_h}"
    if est is None:
        return "forecast:missing"
    margin_h = float(plan.get("pack_stop_margin_min", 20)) / 60.0
    need = (est * float(plan.get("forecast_safety", 1.25)) + margin_h) * rt.price_usd_h
    left = float(total) - float(plan.get("bandwidth_storage_reserve_usd") or 0.0) - spent_usd(state.budget(), now)
    if need > left:
        return f"budget:need {need:.2f} > left {left:.2f} USD"
    ok, why, deadline = rt.watchdog()
    if not ok:
        return f"watchdog:{why}"
    if deadline is not None and now + (est * float(plan.get("forecast_safety", 1.25)) + margin_h) * 3600 > deadline:
        return f"watchdog:deadline too close for {est:.2f} h"
    return None


# ------------------------------------------------------------------------- run -------------------------------------
def acquire_lock(root: Path):
    f = open(root / ".lock", "a+")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("another run_project is already running for this project run (lock held)")
    return f


def run_job(job: Job, env: dict, state: State, rt: Runtime, runner: Callable[[str, Path, int, dict], int]) -> dict:
    rec = state.get(job.id)
    rec.update(status="running", attempts=rec.get("attempts", 0) + 1, started_at=time.time(),
               config_hash=job.config_hash(env), cmd=expand(job.cmd, env), reason=None)
    state.put(rec)
    log = state.root / "logs" / f"{job.id}.attempt{rec['attempts']}.log"
    t0 = time.time()
    rc = runner(expand(job.cmd, env), log, job.timeout_min, env)
    dt = time.time() - t0
    rec.update(exit_code=rc, seconds=round(dt, 1), finished_at=time.time(), log=str(log),
               cost_usd_est=round(dt / 3600 * rt.price_usd_h, 4) if (job.gpu and rt.price_usd_h) else 0.0)
    if rc == 0:
        ok, sums = outputs_ok(job, env)
        if ok:
            rec.update(status="done", outputs=sums, failure_class=None)
        else:
            rec.update(status="failed", failure_class="engineering:outputs_invalid", outputs=sums)
    else:
        rec.update(status="failed", failure_class=EXIT_CLASS.get(rc, "engineering"))
    state.put(rec)
    return rec


def shell_runner(cmd: str, log: Path, timeout_min: int, env: dict) -> int:
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        try:
            p = subprocess.run(["bash", "-c", cmd] if os.name == "posix" else cmd, shell=(os.name != "posix"),
                               cwd=REPO, env=e, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_min * 60)
            return p.returncode
        except subprocess.TimeoutExpired:
            return 124


def plan_env(plan: dict) -> dict:
    env = {k: os.path.expandvars(str(v)) for k, v in (plan.get("env") or {}).items()}
    env.setdefault("PROJECT_RUN", plan["project_run"])
    return env


def main(argv=None, runner=shell_runner, runtime: Optional[Runtime] = None, now_fn=time.time) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--price-usd-h", type=float, default=None, help="override the price read from the provider")
    ap.add_argument("--retry-failed", nargs="*", default=[], help="job ids whose failed state may be retried")
    ap.add_argument("--add-spend", type=float, default=None, help="record spend outside this tool (USD), e.g. setup")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    plan, jobs = load_plan(a.plan)
    env = plan_env(plan)
    root = Path(os.path.expandvars(plan.get("runs_root", "data/runs"))) / plan["project_run"]
    state = State(root)
    if a.add_spend is not None:
        b = state.budget()
        b["adjustments"].append({"t": now_fn(), "usd": float(a.add_spend), "note": a.note})
        state.save_budget(b)
        print(f"recorded {a.add_spend} USD ({a.note}); cumulative est {spent_usd(b, now_fn()):.2f} USD")
        return 0
    rt = runtime or default_runtime(plan, a.price_usd_h)
    env["PRICE_USD_H"] = "" if rt.price_usd_h is None else f"{rt.price_usd_h:.6f}"
    now = now_fn()
    rows = []
    for j in jobs:
        rec = state.get(j.id)
        why = None if rec["status"] == "done" else blocked_reason(j, plan, state, env, rt, now)
        rows.append((j, rec, why))
    if a.status or a.dry_run or not a.resume:
        cap = "no financial cap (explicit)" if no_financial_cap(plan) else f"approved {plan.get('approved_total_usd')} USD"
        print(f"project {plan['project_run']} | price {rt.price_usd_h} ({rt.price_source}) | {cap} | spent est "
              f"{spent_usd(state.budget(), now):.2f} USD")
        for j, rec, why in rows:
            st = rec["status"] if rec["status"] in ("done", "failed") else ("eligible" if why is None else "blocked")
            print(f"  {j.id:24s} {j.phase:4s} {'GPU' if j.gpu else 'CPU'} est {j.est_hours if j.est_hours is not None else '?':>5} h"
                  f"  {st:8s} {why or ''}")
        return 0
    lock = acquire_lock(root)
    pidfile = plan.get("workload_pidfile") and Path(expand(os.path.expandvars(plan["workload_pidfile"]), env))
    marker = plan.get("finished_marker") and Path(expand(os.path.expandvars(plan["finished_marker"]), env))
    if marker and marker.exists():
        marker.unlink()                       # a new run: the watchdog must not see the previous run's marker
    if pidfile:
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(f"{os.getpid()}\n")
    b = state.budget()
    session = {"start": now, "end": None, "usd_per_h": rt.price_usd_h or 0.0, "price_source": rt.price_source,
               "instance_id": os.environ.get("CONTAINER_ID")}
    b["sessions"].append(session)
    state.save_budget(b)
    try:
        for j in jobs:
            if a.only and j.id not in a.only:
                continue
            rec = state.get(j.id)
            if rec["status"] == "done":
                ok, _ = outputs_ok(j, env)
                if ok and rec.get("config_hash") == j.config_hash(env):
                    continue
                rec.update(status="pending", reason="provenance changed or outputs invalid: re-run as new attempt")
                state.put(rec)
            if rec["status"] == "failed" and j.id not in a.retry_failed:
                continue
            why = blocked_reason(j, plan, state, env, rt, now_fn())
            if why:
                rec.update(status="blocked" if why != "disabled" else "disabled", reason=why)
                state.put(rec)
                continue
            run_job(j, env, state, rt, runner)
    finally:
        b = state.budget()
        b["sessions"][-1]["end"] = now_fn()
        state.save_budget(b)
        lock.close()
    failed = [j.id for j in jobs if state.get(j.id)["status"] == "failed"]
    summary = {"done": [j.id for j in jobs if state.get(j.id)["status"] == "done"], "failed": failed,
               "blocked": {j.id: state.get(j.id).get("reason") for j in jobs if state.get(j.id)["status"] == "blocked"},
               "spent_usd_est": round(spent_usd(state.budget(), now_fn()), 3)}
    print(json.dumps(summary, indent=1))
    if marker:                                # no eligible job left in this run: the watchdog may stop the instance
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"t": now_fn(), **summary}, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
