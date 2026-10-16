# Experiment status (updated 2026-10-16)

Labels: **CUDA-real** = ran on a real GPU with real models; **synthetic** = fake backend / synthetic bank or costs,
logic only, never evidence; **not run**; **blocked** (by what).

Financial mode: **explicit no-financial-cap** (user decision, D47). Spend is still recorded below and in
`budget.json` of the project run.

## Machine (measurement machine, D51)

| Item | Value | Source |
|---|---|---|
| Instance | Vast **56272061** (Oregon, US), host 24953, machine 7913, image `vastai/base-image:cuda-12.8.1-auto` | `$CONTAINER_ID`, API read 16:48Z |
| GPU | NVIDIA GeForce RTX 4090 **24 GB** (24,564 MiB), UUID GPU-8d7d5042-94e5-7bd3-27ff-62409aea3aac | nvidia-smi |
| VRAM gate | **PASS** 16:49Z: third-party cuda_memtest `e94e1ee5` Test1/2/4 0 errors; own CUDA test 2 passes 0 mismatches | `scripts/vram_gate.sh` |
| Driver | 575.57.08 | nvidia-smi |
| CPU | 128 visible cores (quota read by p1_env) | nproc |
| Disk | 200 GB overlay, empty at first login | df |
| Price (API) | dph_total **$0.496/h** (storage share $0.200/h) | API `instances/56272061` |
| Stop policy | no stop before the whole project is complete; watchdog monitor-only | D50 |

Excluded: **56267276** (VRAM corruption, D51, docs/VRAM_FAULT_56267276.md). Earlier: 56261825, 56224386.

## Sessions

**56267276 — excluded (VRAM fault, D51)**

| Step | Result | Label |
|---|---|---|
| first login 16:05Z | identity, GPU, price, quota read | — |
| launch 16:22:45Z (`54c7a3b`, monitor-only watchdog, run_project pid 2626) | 00_setup ok; 01_pytest ok (70 s); 02_env_check ok; 02b_fingerprint ok; 03_model ok (62 s) + 03b model check ok | CUDA-real env |
| vram_check (in p1_env) | **FAILED: GPU memory corruption.** 21 × 1 GiB blocks, pattern errors in blocks 4–9 (torch total 25,336,807,424 B, D2D 918.7 GB/s). Diagnostic (3 repetitions of plain `fill_` + strided write, 19 blocks): every repetition has 6 corrupted blocks with 47k–259k wrong float32 values each; wrong values are neighbouring blocks' fill values or bit-flipped values (e.g. 5.5, 9.000129…, 17.000259…); the same in-block offsets recur (8960, 41984, 99072, 238592, 175210696) and move with the allocation → a fixed faulty physical VRAM region. No ECC on this GeForce card (ECC N/A, remapped rows 0, no Xid in dmesg). The same script passed on 56224386 (45 GiB, 0 errors). | CUDA-real, **hardware fault** |
| P1 jobs after p1_env | **not run**: p1_env failed, dependants blocked; no measurement on this GPU is trustworthy (silent corruption of latents/images/scores) | blocked (hardware) |

**56261825** (15:15–16:01Z, ≈ 46 min ≈ $0.40 at $0.527/h): deploy of `d620d6c` + live stop check; dummy workload failed to start (script bug, fixed in `790d7cb`), watchdog took the workload_died path; SSH refused 16:00:59Z. No experiment ran.

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
| **P1 calibration on 56272061** (2026-10-16 16:54–18:43Z, commit `d4cb987`) | **CUDA-real, complete** (11/11 jobs): VRAM gate + torch 21 GiB + CUDA memtest clean; pool full check 0 cross-split; M3 fp16/fp32 0 failures (6,400 previews each); M9 16.1 s per prompt-bank; chunk 1 fastest + smallest; M2 k ∈ {1,2,3,4,8,12,16} (provisional); equivalence 10/10 exact; M2b residual −2.8 %; forecast 32.4 GPU-h. Report: `runs/calib1/report/calibration_report.md` |
| **Calibration-split fixed baselines** (D49, descriptive) | **CUDA-real**: 6 methods × 50 prompts × 2 banks = 600 episodes, 0 failures; `calibration_quality_cost.csv`. HPS not run (adapter unverified) |
| New baselines B4b/B7/B10, RL registry, orchestrator, forecast, no-cap watchdog | **synthetic** only (tests) |
| HPS v2.1 adapter | code + pins only; **not run** (bench_calib reports HPS as missing) |
| Bank, tuning, online train_dev benchmark, controller, val, test | **blocked**: PREREG-A/B/C unsigned |

## Spend log (estimates; provider bill authoritative)

| Session | Instance | Duration | Price | Est. cost |
|---|---|---|---|---|
| smoke1 | 56044352 | 52 min | $0.378/h | ≈ $0.33 |
| diag1 | 56070930 | ~5.67 h (mostly idle) | $0.359/h | ≈ $2.04 |
| prep0 | 56224386 | ~32 min (if stopped at ~09:21Z) | $0.778/h | ≈ $0.41 |
| deploy + stop check | 56261825 | ≤ 15:15–16:01Z (≈ 46 min seen; start time from the console) | $0.527/h | ≈ $0.40 |
| setup + VRAM fault diagnosis | 56267276 | ≤ 16:05Z – (user handles the instance) | $0.626/h | ≈ $0.47 to 16:50Z |
| P1 onward | 56272061 | running from ≤ 16:48Z | $0.496/h | recorded in budget.json |
