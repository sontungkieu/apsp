# PREREG-A decision package (proposal — NOT signed)

Spec §11.5 A: machine/budget, prompt split, sampler/seed, precision, scoring, feature definitions, EPH, generator
costs/caps, collector equivalence, P0 acceptance. Plan v1.3 §9 rows mapped here. Measured values filled 2026-10-17 from the
P1 run on 56272061 (copies + sha256: `DMRL_artifacts/p1/from_56272061/`); research approves the whole package once. Signing = research places
`docs/PREREG_SIGNED/PREREG-A.json` with this file's sha256 (docs/PREREG.md). The agent never signs.

Status legend: **fixed** = already decided/pinned in the repo, approve as is · **measured** = value comes from a P1
file (filled after P1) · **decide** = research choice needed (agent proposal given).

| # | Field | Proposal / value | Status | Source |
|---|---|---|---|---|
| A1 | Measurement machine | **Vast 56272061**, RTX 4090 24 GB (UUID GPU-8d7d5042-…), driver 575.57.08, torch 2.4.0+cu121 / CUDA 12.1 / cuDNN 9.1, Python 3.10.22, OMP/MKL 16 threads; runtime fingerprint `02fa12c4…`; VRAM gate + torch check (21 GiB, 0 errors) + CUDA memtest (0 mismatches); one GPU worker | **measured** | `calib1/p0/runtime_fingerprint.json`, `vram_check.json` |
| A2 | Budget | No rental cap (user, D47); cost recorded | fixed (user) | D47 |
| A3 | Generator | SD1.5 `451f4fe1…`, DDIM eta 0, T 64, CFG 7.5, 512², delta 4, negative prompt "" , UNet fp16, deliver fp16 | fixed | `configs/pilot.yaml`, M1 hash check |
| A4 | Grammar / caps | G-842-v1, keep [4,2], reference cuts [4,8]; alpha {0.25,0.5,1.0,1.5} × C_ref, primary {0.5,1.0}; infeasible caps reported NA | fixed | `configs/pilot.yaml` |
| A5 | Seed scheme | apsp-v1-sha256-cpu; 2 banks × 8 seeds per research prompt | fixed | `configs/pilot.yaml` |
| A6 | Scoring | ImageReward-v1.0 `5736be03…`, tokenizer bert-base-uncased `86b5e093…` (D22), path ir_pil_v1 | fixed | `configs/pilot.yaml` |
| A7 | Prompt source/filter | DiffusionDB 2M metadata `fb620fbe…` (sha `eecd3411…`); fastText lid.176 top-1 en p ≥ 0.90; NSFW max < 0.1; 4–60 CLIP tokens; apsp-norm-v1 | fixed (Q2/Q3/Q15) | `reports/prompt_pool/pool_manifest.json` |
| A8 | Dedup / split | full-pool connected components cosine > 0.95 (GPU check), group split; GenEval text > 0.9 / template exclusion; calibration 50, train_fit 190, train_dev 50, val 80, val_ext 220 sealed. **Measured: 231,516 survivors, 206,782 components (largest 86), 0 selected prompts in a cross-split component, 0 missing, val_ext list = sealed file** → no move/drop needed; approve `sampled-grouping-v1` + full check (Q16) | **measured**, approve | `calib1/calib/pool_full_check.json` |
| A9 | Preview precision | rule (Q7): keep FP16 unless M3 failure > 1e-3 at any checkpoint (Spec §12). **Measured: fp16 and fp32 on the same x0, 6,400 previews each, 0 latent/image/score failures at all 16 checkpoints, delivered 0; |ΔIR fp16−fp32| mean 0.0103, max 0.108** → proposal **preview_dtype = fp16** | **measured** | `calib1/calib/m3m9.json` |
| A10 | VAE decode chunk | rule: fastest valid chunk within memory; tie → smaller. **Measured: chunk 1 fastest (k8 267.8 ms, k16 533.8 ms; chunks 2–16: 290–611 ms) and smallest memory (5.17 GB reserved)** → proposal **vae_decode_chunk = 1** | **measured** | `calib1/calib/chunk.json` |
| A11 | Generator cost table | M2 primitives k ∈ {1,2,3,4,8,12,16}; GCH `b22457b6…`; e.g. den(8) 134.4 ms, ver(8) 433.3 ms, fin(8) 433.5 ms, prune ≤ 0.17 ms, feat ≈ 0.34 ms, C_fixed ≈ 34–41 ms; `provisional: true` until signed; controller costs (disp/once/pred) not yet measured | **measured** | `calib1/p0/m2/generator_costs.json` |
| A12 | Collector equivalence | same scoring definition; no bitwise requirement across batch sizes (native batching, D34). **Measured: 10/10 calibration prompts exact (`all_A_exact: true`)** | **measured** | `calib1/calib/equivalence.json` |
| A13 | M2b residual | online B2 vs ledger. **Measured: 20 PSP episodes, predicted 5.930 s, wall mean 5.765 s (p95 5.882 s), residual −2.8 %** | **measured** | `calib1/calib/m2b.json` |
| A14 | EPH | Calibration EPH `a70c0c3e…` (56272061) differs from smoke1 `c1351a99…` (3090): EPH v1 hashes the generation code (changed since smoke1, e.g. `lfeat` engines), the env lock and the numeric runtime. It also hashes `preview_dtype` and `vae_decode_chunk`, which are `null` in `configs/pilot.yaml` today → after A9/A10 are approved and written (fp16, 1) the bank EPH is **re-measured on the box and that value is frozen**, not `a70c0c3e…` | measured; re-measure after A9/A10 | `apsp/gpu/backend.py:eph_body`, collector manifest |
| A15 | Feature definitions | as in the Implementation Guide / CONTRACT at the pinned commit | fixed | Guide v1.1.1 |
| A16 | Latent storage (train) | store train latents only if B10/ablation need them; estimate size first; none for held-out | decide | plan v1.3 E2 |
| A17 | Replay vs online (Amendment 1 rev. 2) | replay = simulator for training/screening; confirmatory online; native batching | decide (Q12) | `docs/PREREG_PROPOSALS/AMENDMENT_1_replay_online_DRAFT.md` |
| A18 | P0 acceptance | P0 CPU acceptance + P1 CUDA outputs above | measured | `docs/P0_CPU_ACCEPTANCE.md`, calibration_report |

Fields that belong to PREREG-B/C (not needed to start the bank): anchor, epsilon/tau, sigma_R, r_fail, contrasts/M,
looks, confirmatory N, GenEval/HPS protocol — plan v1.3 §9 proposals are carried into the B/C packages.
