# Experiment status (updated 2026-10-16)

Labels: **CUDA-real** = ran on a real GPU with real models; **synthetic** = fake backend / synthetic bank or costs,
logic only, never evidence; **not run**; **blocked** (by what).

Financial mode: **explicit no-financial-cap** (user decision, D47). Spend is still recorded below and in
`budget.json` of the project run.

## Machine (measurement machine, D48)

| Item | Value | Source |
|---|---|---|
| Instance | Vast **56261825** (The Netherlands), image `vastai/base-image:cuda-12.8.1-auto` | `$CONTAINER_ID`, API read 2026-10-16 15:15Z |
| GPU | NVIDIA GeForce RTX 4090 **24 GB** (24,564 MiB), UUID GPU-ab11fd55-23b9-a7ef-f27a-84fab6e78f79, power limit 450 W | nvidia-smi; PyTorch check pending (p1_env) |
| Driver | 590.48.01 | nvidia-smi |
| CPU / RAM | 256 visible cores (quota to be read by p1_env), 1 TB host RAM visible | nproc, free |
| Disk | 200 GB container overlay, empty at first login | df |
| Price (API) | dph_total **$0.527/h** (includes storage share $0.267/h) | API `instances/56261825` |

Previous machine 56224386 (RTX 4090 48 GB, $0.778/h) is no longer used; its facts and the prep0 session are kept
below for the record.

## Sessions

**56261825**

| Step | Result | Label |
|---|---|---|
| first login 15:15Z | identity, GPU, price read; nothing run | — |
| watchdog live check (no-financial-deadline mode) | pending | — |
| P1 calibration via run_project | pending | — |

**56224386 / prep0** (2026-10-16 08:49–~09:21Z; ~32 min ≈ $0.41 at $0.778/h, estimate). Code `7becb6f`.

| Step | Result | Label |
|---|---|---|
| 00_setup | ok, 40 s | CUDA-real env |
| 01_pytest | throttled first run (torch 215 threads vs 61 CPU quota), killed; rerun with 16 threads: 199 passed, 2 failed (watchdog tests vs real `CONTAINER_ID`; fixed in `c251230`) | engineering |
| vram_check | 48 GB variant real: torch total 50,950,569,984 B; 45 GiB pattern-verified; D2D 937 GB/s | CUDA-real (that machine only) |
| 02_env_check | ok | CUDA-real |
| 02b_fingerprint, 03_model | unknown: SSH refused from ~09:20:50Z | — |
| watchdog | incident: `pkill -f scripts/gpu_smoke.sh` killed the watchdog via its own argv; fixed (D46), and the workload is now ended by PGID (D47) | engineering |
| stop | **never confirmed** (Q22) | needs console check |

## Results

| Evidence | Status |
|---|---|
| smoke1 / diag1 (RTX 3090, 2026-10-15) | CUDA-real, previous machine; not reused as 4090 latency |
| Calibration M2/M3/M9/chunk/equivalence/M2b on 56261825 | **not run yet** |
| Prompt pool full-pool duplicate check (Q16) | **not run yet** (first P1 GPU job) |
| Calibration-split quality–cost table (D49, descriptive) | **not run yet** |
| New baselines B4b/B7/B10, RL registry, orchestrator, forecast, no-cap watchdog | **synthetic** only (tests) |
| HPS v2.1 adapter | code + pins only; **not run** (bench_calib reports HPS as missing) |
| Bank, tuning, online train_dev benchmark, controller, val, test | **blocked**: PREREG-A/B/C unsigned |

## Spend log (estimates; provider bill authoritative)

| Session | Instance | Duration | Price | Est. cost |
|---|---|---|---|---|
| smoke1 | 56044352 | 52 min | $0.378/h | ≈ $0.33 |
| diag1 | 56070930 | ~5.67 h (mostly idle) | $0.359/h | ≈ $2.04 |
| prep0 | 56224386 | ~32 min (if stopped at ~09:21Z) | $0.778/h | ≈ $0.41 |
| idle before deploy | 56261825 | running since ≤ 15:15Z, no workload (waiting for permissions) | $0.527/h | to be recorded at the first stop |
