# PREREG-A: locked protocol (2026-10-17)

Approval: project owner, messages of 2026-10-17 recorded verbatim in `docs/PREREG_SIGNED/AUTHORIZATION_20261017.md`.
This file is the approved proposal, and its sha256 is pinned in `docs/PREREG_SIGNED/PREREG-A.json`. It must not be
edited after locking; any change needs an amendment file and a new marker. The decision history is in
`PREREG_A_DECISION_PACKAGE.md` (commit `105d624`), and the measured P1 values are in `calibration_report.md`
(calib1, sha256 `eb847c5b…`).

| # | Field | Locked value | Source |
|---|---|---|---|
| A1 | Measurement machine | Vast **56272061**, RTX 4090 24 GB (UUID GPU-8d7d5042-94e5-7bd3-27ff-62409aea3aac), driver 575.57.08, torch 2.4.0+cu121 / CUDA 12.1 / cuDNN 9.1, Python 3.10, OMP/MKL 16 threads, CUBLAS_WORKSPACE_CONFIG `:4096:8`; one GPU worker. VRAM gate + torch 21 GiB + CUDA memtest: 0 errors | calib1 `p0/` |
| A2 | Budget | No rental cap (D47). Cost is recorded | owner |
| A3 | Generator | SD1.5 `451f4fe1…`, DDIM eta 0, T 64, CFG 7.5, 512², delta 4, negative prompt "", UNet fp16, deliver fp16 | `configs/pilot.yaml` |
| A4 | Grammar / caps | G-842-v1, keep [4,2], reference cuts [4,8]; alpha {0.25, 0.5, 1.0, 1.5} × C_ref, primary {0.5, 1.0}; infeasible caps reported as NA | `configs/pilot.yaml` |
| A5 | Seed scheme | apsp-v1-sha256-cpu; 2 banks × 8 seeds per research prompt | `configs/pilot.yaml` |
| A6 | Scoring | ImageReward-v1.0 `5736be03…`, tokenizer bert-base-uncased `86b5e093…`, path ir_pil_v1 | `configs/pilot.yaml` |
| A7 | Prompt source / filter | DiffusionDB 2M metadata (sha `eecd3411…`); fastText lid.176 en p ≥ 0.90; NSFW < 0.1; 4–60 CLIP tokens; apsp-norm-v1 | `reports/prompt_pool/pool_manifest.json` |
| A8 | Dedup / split | `sampled-grouping-v1` + full-pool check (231,516 survivors, 206,782 components, 0 selected prompts across splits): calibration 50, train_fit 190, train_dev 50, val 80, val_ext 220 (sealed). No prompt is moved or dropped based on generation results | calib1 `calib/pool_full_check.json` |
| A9 | Preview precision | **preview_dtype = fp16** (M3: 0 failures / 6,400 previews per dtype) | pilot.yaml |
| A10 | VAE decode chunk | **vae_decode_chunk = 1** (fastest and smallest memory) | pilot.yaml |
| A11 | Generator cost table | Initial accounting table, re-measured under the locked configuration: calibA M2, k ∈ {1,2,3,4,8,12,16}, **GCH `bffb378d71080a2c…`**, file sha256 `b2c688e9748d2e7b…`, EPH = A14. `provisional: true` until controller costs (disp/once/pred) are profiled; those costs are profiled and confirmed before the final evaluation. Missing controller costs do not block bank collection | `runs/calibA/p0/m2/generator_costs.json` |
| A12 | Collector equivalence | Fresh two-sided run under A14: 10/10 calibration prompts, delivered latents exactly equal to the pinned PSP reference (max abs 0), preview score difference 0 (`equivalence_fresh.json` sha256 `e502ca4d…`). No bitwise requirement across batch sizes (native batching) | `runs/calibA/calib/` |
| A13 | M2b residual | Under A11: 20 PSP episodes, predicted 5.758 s, wall 5.769 s (p95 6.090 s), residual +0.19 % | `runs/calibA/calib/m2b.json` (sha256 `206b1869…`) |
| A14 | EPH (bank) | **`097d7a3c01af4c504b3811bece3404ba42cdd5e16d4a12ff0668b91290b13fc7`**, frozen 2026-10-16T22:4xZ on 56272061 in the gpu_smoke runtime env (`frozen_eph.json` sha256 `1229f915…`). The bank collector refuses any other EPH (`--expect-eph-file`). The earlier value `da8a45c6…` came from a run without CUBLAS_WORKSPACE_CONFIG and is invalid; it is kept aside on the instance | `${APSP_STORE}/frozen_eph.json` |
| A15 | Feature definitions | Implementation Guide v1.1.1 / CONTRACT at the pinned commit | Guide |
| A16 | Latent storage | Store x0 preview latents as float16 `latents/x0_<uid>_<bank>.npy` [8 seeds, 16 checkpoints, 4, 64, 64] for **train_fit/train_dev only** (≈ 4 MiB per prompt-bank, ≈ 1.9 GiB in total), checksummed in the BMH. No latents for val/val_ext/test | `apsp/gpu/collect_bank.py` |
| A17 | Amendment 1 rev. 2 | Accepted. Replay is an approximate simulator used for training and screening. Confirmatory conclusions come from online runs. O1-replay is only a bound inside the simulator, not an online upper bound | owner |
| A18 | P0 acceptance | P0 CPU acceptance + the P1 CUDA outputs (calib1) + the calibA re-measure above | `docs/P0_CPU_ACCEPTANCE.md` |

Not re-measured, because they are not affected by the A9/A10 binding: pool check, M3 (fp16 was already the effective
preview dtype), chunk, M9, and the calibration-split fixed-baseline table (descriptive; that run had no
CUBLAS_WORKSPACE_CONFIG and stays descriptive only).

Forecast under A11 at $0.4956/h, an estimate rather than a cap: P2 2.15 h, P3 3.18 h, P4 0.92 h, P5 2.73 h, P6 21.4 h,
P7 1.22 h; total 31.6 GPU-h ≈ $15.7 (`runs/calibA/forecast/phase_forecast.json`, sha256 `79be92c5…`).
