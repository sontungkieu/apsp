# Cost watchdog for rented GPU instances

**Why.** diag1's runner finished in 23 minutes, but the instance stayed up about 5 hours because the session that
was supposed to download and stop was lost (`DIAG1_RUN_REPORT.md` §4). Runner timeouts bound the workload, not the
rental. The watchdog bounds the rental.

## Design (`scripts/vast_watchdog.py`, stdlib only)

- **Runs on the instance, detached.** `scripts/arm_watchdog.sh` starts it with `setsid nohup`, stdin from
  `/dev/null` and SIGHUP ignored. It needs nothing from the laptop, the Claude session or the SSH connection after
  arming.
- **Armed before the workload, with a fixed deadline** (an absolute epoch). Arming refuses if:
  - the machine's `CONTAINER_ID` differs from the target instance;
  - the deadline is under 60 s away or more than `--max-hours` (default 6) away;
  - the save margin is longer than the time left.
- **Runner gate.** On a Vast instance (`CONTAINER_ID` set), `scripts/gpu_smoke.sh` exits 3 before any step unless
  the watchdog is armed for this instance, its process is alive and its deadline is in the future. The gate can be
  disabled only explicitly, with `REQUIRE_WATCHDOG=0`.
- **Triggers:**
  - `deadline`, at `deadline − save_margin` (default 300 s);
  - `idle_after_finish`, once the workload's `.finished` marker has been seen for `--idle-grace-s` (default
    900 s). That grace is the download window, and the stop does not wait beyond it.
- **Save phase, bounded.** The `--save-cmd` commands run in order (end the runner, write the tarball and state to
  the instance disk). Each is limited to `min(save_cmd_timeout, time to deadline)`; on timeout its whole process
  group is killed, and commands are skipped once the deadline is reached. Saving never extends past the deadline.
- **Stop.** One instance id, a PUT of `{"state": "stopped"}`. There is no destroy code path; the disk is kept.
  Calls are retried (default 5 attempts) with a per-call timeout, so a hanging API call cannot block the watchdog.
- **Secrets.** The key is read from the environment (`CONTAINER_API_KEY`, injected by Vast) and sent only in an
  `Authorization` header. It never appears on a command line (no `--api-key` argument, unlike `vastai stop … --api-key`
  as used for smoke1), in logs or in `state.json`/`events.jsonl`. Provider answers are reduced to short scalar
  fields, with key/token-like names dropped.

## Three stop levels (recorded separately in `state.json`)

| Field | Meaning | Set when |
|---|---|---|
| `stop_requested` / `stop_requested_at` | The watchdog began a stop call | before the first call |
| `api_accepted` (`api_result` = accepted / rejected / error) | The provider answered 2xx without `success: false` | after each attempt |
| `confirmed_stopped` | A status read returned `stopped` / `exited` / `offline` | only from an actual status read |

- From inside the instance `confirmed_stopped` will normally stay false, because the container halts after
  acceptance.
- **"Stopped" must be confirmed externally:** the Vast console, or `vast_watchdog.py --check-status` run from a
  machine that has API access (in the environment).
- A report may say "stop requested" or "API accepted" from the state file, but "stopped" only after that check.

## Verified, and not verified

**Verified by tests** (`tests/ops/test_vast_watchdog.py`, 14 tests; the runner-gate test runs on Linux only):
- arming refusals;
- deadline and idle triggers;
- the stop request is sent by the deadline even while a save command hangs (the command is killed);
- rejected, failing and hanging API calls are retried, timed out and reported as `stop_failed`;
- `confirmed_stopped` is only set by a status read;
- **session loss**: the launching process and its process group are killed, and the watchdog still stops at the
  deadline;
- **hanging download**: a reader keeps the tarball open forever, and the watchdog still stops after the idle
  grace, while the reader is still running;
- the REST client against a local HTTP mock: path, body, header, and that the key and echoed secrets never reach
  the state or event files;
- the runner refuses to start without an armed watchdog.

**Not verified on the real provider:**
- the Vast endpoint paths and payloads (`PUT /api/v0/instances/{id}/ {"state":"stopped"}`,
  `GET /api/v0/instances/{id}/`), taken from the `vastai` CLI's behaviour;
- whether `CONTAINER_API_KEY` is allowed to stop its own instance through this REST call (smoke1 and diag1 used
  the `vastai` CLI with that key);
- the response shape;
- how long a stop takes;
- whether `setsid`/SIGHUP handling survives the provider's SSH proxy.

**First use on a real instance:** arm with a short deadline on an idle box and watch the console. Treat every
budget as unprotected until that check passes.

## Procedure on the next GPU session

```bash
export APSP_STORE=/workspace/apsp RUN_ID=<run>
DEADLINE=$(( <session_start_epoch> + <budget_minutes>*60 - 300 ))   # leave 5 min for the stop to take effect
bash scripts/arm_watchdog.sh "$CONTAINER_ID" "$DEADLINE" "$APSP_STORE/watchdog" \
  --finished-marker "$APSP_STORE/$RUN_ID.finished" --idle-grace-s 900 \
  --save-cmd "pkill -TERM -f '[s]cripts/gpu_smoke.sh'; sleep 20; pkill -KILL -f '[s]cripts/gpu_smoke.sh'" \
  --save-cmd "tar --exclude='*.lock' -czf $APSP_STORE/${RUN_ID}_artifacts.tgz -C $APSP_STORE/runs $RUN_ID"
HARD_DEADLINE_EPOCH=$(( DEADLINE - 600 )) MAX_MINUTES=<n> bash scripts/gpu_smoke.sh <preset>; touch "$APSP_STORE/$RUN_ID.finished"
```

Then download within the idle grace. If the download is not finished, the instance stops anyway; the tarball stays
on the kept disk.

## No-financial-deadline mode (D47, current)

The user removed the rental cap for the agreed suite. `scripts/launch_project.sh` arms
`vast_watchdog.py --no-financial-deadline` and starts `apsp.run_project` under `setsid`. Triggers (all mandatory when
arming in this mode):

| Trigger | Condition |
|---|---|
| `idle_after_finish` | run_project wrote `$APSP_STORE/run_project.finished` (no eligible job left: done, failed, or PREREG-blocked) and the idle grace passed |
| `workload_died` | the pid in `$APSP_STORE/run_project.pid` is gone without the marker (or never appeared) for the idle grace |
| `stalled` | `--stall-s` without progress: no watched log/state file changed and GPU utilisation below `--gpu-busy-pct` |

On a trigger the watchdog ends the workload by the process group of that pid (TERM, KILL after 60 s; never its own
group, never a command-line pattern), runs the bounded packaging command, and then sends the stop whatever the
packaging result. Stop levels stay separate in `state.json`: `stop_requested`, `api_accepted`, `confirmed_stopped`.
A progressing job is never stopped for time; per-job timeouts in the plan still bound hangs. Live check of this mode:
`bash scripts/watchdog_live_check.sh` (90 s dummy workload, stop request at about +2.5 min).
