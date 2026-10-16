# Experiment status (updated 2026-10-16)

Labels: **CUDA-real** = ran on a real GPU with real models; **synthetic** = fake backend / synthetic bank or costs,
logic only, never evidence; **not run**; **blocked** (by what).

## Machine

| Item | Value | Source |
|---|---|---|
| Instance | Vast **56224386** (Texas, US), image `vastai/base-image:cuda-12.8.1-auto` | `$CONTAINER_ID`, API read |
| GPU | NVIDIA GeForce RTX 4090, UUID GPU-246fa41a-e528-49f8-f888-bf580022a9d3, part 2684-301-A1, 128 SMs, sm_89, power limit 450 W | nvidia-smi + PyTorch |
| GPU memory | **48 GB variant, real**: nvidia-smi 49,140 MiB (550 MiB reserved); torch total 50,950,569,984 B (47.45 GiB); 45 × 1 GiB blocks allocated and pattern-verified, no error; D2D copy 937 GB/s | `scripts/vram_check.py` (prep0) |
| Driver / CUDA | 615.71.09, CUDA UMD 13.4; generator env torch 2.4.0+cu121 (pinned, not upgraded) | nvidia-smi, env |
| CPU / RAM | AMD EPYC 7713, 256 visible cores, **cgroup quota ≈ 61.4 CPUs** (cpu.cfs_quota_us 6,143,999 / 100,000); 1 TB host RAM visible | cgroup |
| Disk | 200 GB container overlay, no separate volume (survives stop/start, not destroy/recycle) | df, Vast guide |
| Price (API) | dph_total **$0.7778/h** (compute $0.6667 + storage share); storage $0.40/GB/month ≈ $2.67/day while stopped; bandwidth $0.0104/GB up, $0.0052/GB down | API `instances/56224386` |

## Sessions on 56224386

**prep0** (2026-10-16 08:49–~09:21Z; ~32 min ≈ $0.41 at $0.778/h, estimate). Code `7becb6f` (bundle sha256
`9283e22f…`), PSP `590f59f`. Watchdog armed (deadline 10:04:13Z, idle grace 180 s after the finished marker).

| Step | Result | Label |
|---|---|---|
| 00_setup (uv, Python 3.10, pinned env) | ok, 40 s | CUDA-real env |
| 01_pytest | first run throttled (torch 215 threads vs 61 CPU quota), killed; rerun with 16 threads: 199 passed, 2 failed — both watchdog tests refused the test instance id because `CONTAINER_ID` is set on the box (fixed in `c251230`) | engineering |
| vram_check | ok (above) | CUDA-real |
| 02_env_check | ok, 4 s | CUDA-real |
| 02b_fingerprint, 03_model | **unknown**: SSH refused from ~09:20:50Z while they ran | — |
| watchdog | **incident**: `pkill -f scripts/gpu_smoke.sh` used to restart the chain also matched the watchdog's own argv (its `--save-cmd`) and killed it; re-armed within ~1 min with the same deadline. The same pattern was in the documented save command, so a deadline stop would have killed the watchdog before the stop call. Fixed: arming refuses self-matching kill patterns; docs use `'[s]cripts/gpu_smoke.sh'` | engineering |
| stop | **unconfirmed**: SSH refused since ~09:20:50Z, earlier than the watchdog's idle-stop could fire (≥ 09:22Z if 03_model had finished). Cause unknown until the disk is read (`/workspace/apsp/watchdog/state.json`, `events.jsonl`) or the console is checked | needs external check |

## Results

| Evidence | Status |
|---|---|
| smoke1 / diag1 (RTX 3090, 2026-10-15) | CUDA-real, **previous machine**; not reused as 4090 latency |
| Calibration M2/M3/M9/chunk/equivalence/M2b on 4090 | **not run** (needs session budget; M3/M9 also need pool approval) |
| Prompt pool full-pool duplicate check (Q16) | **not run** |
| New baselines B4b/B7/B10, RL registry, orchestrator, forecast | **synthetic** only (Windows tests 220 passed, 1 skipped at `c251230`) |
| HPS v2.1 adapter | code + pins only; **not run** |
| Bank, tuning, online benchmarks, controller, val, test | **blocked**: PREREG-A/B/C unsigned; budget not approved |
| Quality–cost table on real data | **not available** (needs P2 + P3) |

## Spend log (estimates; provider bill authoritative)

| Session | Instance | Duration | Price | Est. cost |
|---|---|---|---|---|
| smoke1 | 56044352 | 52 min | $0.378/h | ≈ $0.33 |
| diag1 | 56070930 | ~5.67 h (mostly idle) | $0.359/h | ≈ $2.04 |
| prep0 | 56224386 | ~32 min (if stopped at ~09:21Z) | $0.778/h | ≈ $0.41 |
