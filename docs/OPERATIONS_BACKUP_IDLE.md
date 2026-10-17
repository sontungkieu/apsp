# Off-instance backup, heartbeat and the mentor idle rule (2026-10-17)

The owner's operating condition: the mentor system destroys an instance it considers inactive for 30 minutes.
`/workspace` on these Vast instances is not a volume (`workspace_is_volume: false`), so anything not copied off the
instance is lost on destroy.

## What runs on the instance (independent of the laptop)

| Process | File | Role |
|---|---|---|
| heartbeat | `scripts/heartbeat.py` | every 60 s `heartbeat.json` / `heartbeat.jsonl`: one of `gpu_compute`, `cpu_progress`, `gpu_job_low_util`, `upload`, `cpu_setup`, `waiting_dependency`, `failed_jobs`, `job_dead`, `idle_unknown`, with real counters (bank shards, episode lines, newest output time, running job). New `job_dead` / `failed_jobs` / `idle_unknown` states go to `ALERTS.jsonl` immediately |
| backup daemon | `scripts/backup_daemon.sh` | every 60 s incremental `rclone copy` of `$APSP_STORE` (finished shards, checkpoints, job outputs, logs, heartbeat, alerts; no venvs, HF cache or evaluator weights) and of the PREREG snapshots written on the box, to `$APSP_BACKUP_REMOTE/<instance id>/`. `backup_status.json` records the last successful pass; failures go to `ALERTS.jsonl` |
| supervisor | `scripts/supervise_project.sh` | relaunches `run_project --resume` between phases |
| launch guard | `scripts/launch_project.sh` | starts heartbeat and backup, and refuses to start a workload until one backup pass has completed |

No process creates artificial load or touches any platform idle mechanism.

## Last progress on 56272061 (evidence kept)

Only this session's SSH observations remain (instance logs were lost with the instance):
- 23:04:39Z: newest bank shard written (collector running, 15.7 s per shard); about 89/480 shards at ~23:07Z.
- 23:13:46Z: last successful SSH (HPS adapter check).
- 05:46Z (next attempt): connection refused, then timed out.

What ran between 23:13Z and the deletion, and why the instance was considered idle, is **not established**. The bank
was due to finish around 00:55Z, followed by CPU (tuning) and GPU (dev benchmark) jobs. Whether any of them ran is
unknown.

## Questions for the mentor (needed to keep a legitimate long run alive)

1. Which signal defines "active": GPU utilisation (which threshold, sampled how often, averaged over what window),
   GPU memory in use, CPU load, disk or network I/O, SSH sessions, or a process list?
2. Is a CPU-only phase (replay tuning, RL training on CPU, report generation; GPU at 0 % for up to ~1 h) counted as
   active? If not, how should it be declared?
3. Is an upload phase (rclone copy to off-instance storage, network-bound) counted as active?
4. Is there a supported way to declare activity or progress (an API, a heartbeat file, a tag) that the mentor reads?
   Our heartbeat file is `/workspace/apsp/heartbeat.json`.
5. Is there a warning before destruction (an email or a notice period), and can a run be exempted while waiting for
   a dependency for less than 30 minutes?
6. Does the 30-minute window restart after any activity, or is it cumulative?

## Backup target (owner action)

A remote must be configured by the owner. The agent never sees the credential. Options:
- `rclone config` on the instance, with a remote such as Google Drive, Cloudflare R2, Backblaze B2 or S3, then start
  `APSP_BACKUP_REMOTE=<name>:<bucket-or-folder>`.
- The same remote on the laptop (rclone for Windows), used to verify a restore. Before a new run the agent uploads a
  test artifact, restores it on the laptop and compares sha256.
