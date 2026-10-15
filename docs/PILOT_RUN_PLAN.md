# Pilot run plan: quality-cost benchmark (prepared; nothing run on GPU)

**Objective.** min E[C_policy] subject to E[Q_policy − Q_ref] ≥ −ε. Q is the ImageReward of the delivered image; Q_ref
is PSP-default by default (`configs/evaluation.yaml`). Native batching 8→4→2 is kept. Replay serves training and
approximate screening; confirmatory conclusions are online (Amendment 1). O1-replay is not an online upper bound.

## 1. Baselines

| Method | Spec ID | Status | Where | Runs |
|---|---|---|---|---|
| Single, 64 steps | B0 | **done** | `single` (F(1): no RM in inference; Q scored after delivery as evaluation-only) | online; replay |
| Single, 32 steps | — | **done** | `single32`: own engine with the DDIM scheduler set to T=32 (32 forwards, own EPH, timesteps in manifest); never derived from the 64-step trajectory | **online only** |
| BoN-2/4/8 (16) | B1 | **done** | `bon:N`, nested seeds of one bank (N=16 = pool 16, one episode per prompt); batch N online | online; replay |
| PSP-default (4,8) | B2 | **done** | `psp` | online; replay |
| Best fixed pair, chosen on train-dev | B3 | **done** | `select_b3_dev` (train_fit top-5, train_dev choice, per cap) → `fixed:J1,J2@alpha` | online; replay |
| Threshold | B6 | **done** | `ThresholdPolicy` + `tune_threshold` (≤ 2000 train_fit evaluations, train_dev choice) → `threshold:PATH@alpha` | online; replay |
| Screen-and-resume, one VERIFY | B5 | **done (new)** | `apsp/policies/screen_resume.py`: M∈{8,16}, screen j_s, keep q∈{1,2}; tuned per cap → `sr:M,JS,Q@alpha`. Our own same-bank baseline, **not Flash-BoN** | online; replay |
| One VERIFY then a fixed schedule | B8 | **done (new)** | `apsp/policies/query_commit.py`: query at j_c∈1..8; preflight before paying; ridge selector (group 5-fold over 13 alphas) on prompt PCA + purchased scores; reuses the query if j1 = j_c; spend checked against the full-trace cost; j_c chosen on train_dev → `b8:PATH@alpha` | online; replay |
| Random feasible | B9 | done | `random:SEED@alpha` | online; replay |
| PSP wide grid | B4b | grid **done**, method **planned** | `apsp/sim/b4b_grid.py` (7,823 strategies, identical to PSP); `to_spec` runs any strategy through `run_schedule`. Missing: per-cap train/dev selection over the grid and a `b4b:` method string. Needs the M2 table with k = 3 and 12 (calibration) | planned |
| Prompt-only selector | B7 | **planned** | needs train_fit bank labels | — |
| Cheap latent predictor | B10-min | **planned** | hooks (`lfeat`) and pred-cut ledger path exist; predictor training/tuning not implemented; needs the bank | — |
| RL controller | RL* | implemented, **not trained** | `apsp/policies/reinforce.py`; real training needs PREREG-B and the G2 marker | — |
| Flash-BoN | — | **not implemented** | would need the original method on the shared backbone; B5 is not Flash-BoN | — |

**Real batches only.** Single and BoN are always run online with their own batch (1/2/4/8/16). No delivered score of
the batch-8 bank is reused as an online result.

## 2. Instrumentation (`apsp/gpu/benchmark.py`)

- **Same GPU, one worker.** Untimed warm-up episodes for every method family on a technical prompt.
- **Model load / setup** is timed once and reported separately.
- **e2e request time.** CUDA-synchronised `perf_counter` from before `prepare_request` (text encoding) until the
  delivered image is on the CPU, including the reward-model selection of selecting methods. PNG I/O and evaluation
  come after the timer.
- **What counts as inference cost.**
  - Reward-model selection is inference cost: in the ledger (ver/fin) and inside e2e.
  - Post-hoc IR of an F(1) delivery and independent metrics (`apsp/gpu/evaluators/independent.py`, HPSv2 when
    pinned) are evaluation-only and recorded separately.
- **Per episode:** prompt_uid, split, bank_id, method, config hash, candidates, winner, image path and sha256,
  success, inference score, q_eval and its source, ledger spend/cap, e2e, eval seconds, forwards,
  candidate-forwards, VERIFY count, prune checkpoints, peak VRAM, method EPH.
- **Manifest:** commit, EPHs, runtime fingerprint, cost table hash and provisional flag, caps, load and warm-up
  times. Resumable.
- **Report** (`apsp/reports/quality_cost.py`):
  - curve (e2e and ledger) with Pareto flags;
  - quality at each shared cap B_alpha;
  - cheapest method reaching each reference level;
  - mean/p50/p95 e2e, failure rate, peak VRAM;
  - paired-by-prompt cluster-bootstrap contrasts;
  - NI (LCB(ΔQ) > −ε) and the cost gate (Spec §11.4, savings 10% / 5%).
- **Unapproved config.** While ε, r_fail or M are not approved, every conclusion is `not_concludable`; the CIs are
  printed as descriptive.
- **Selection.** Configurations are tuned on train_fit/train_dev (`apsp/sim/tune_baselines.py`). The operating point
  is frozen from train_dev only (`--freeze`), and val runs accept only that frozen file.
- **Sample report:** `reports/quality_cost/SAMPLE_SYNTHETIC_report.md` (fake backend, synthetic costs; **not
  evidence**).

## 3. GPU sequence (each session with the watchdog; see `CALIBRATION_PLAN.md` §2–3 for arming)

| # | Session | Preconditions | Command (on the instance, after bundle transfer and `arm_watchdog.sh`) |
|---|---|---|---|
| 0 | Watchdog live check | instance provided | `bash scripts/watchdog_live_check.sh`; the user confirms *Stopped* in the console |
| 1 | Calibration/profiling | #0 passed; pool approved (Q15/Q16) | `RUN_ID=calib1 POOL_DIR=/workspace/apsp/prompts bash scripts/gpu_smoke.sh calib` |
| 2 | Bank + tuning + online train_dev | PREREG-A signed (machine, pool, precision from M3, chunk, cost table from #1) | `RUN_ID=pilot1 COSTS=$APSP_STORE/runs/calib1/p0/m2/generator_costs.json bash scripts/gpu_smoke.sh pilot_dev` |
| 3 | Online val (confirmatory) | PREREG-B signed; `configs/evaluation.yaml` approved (ε, r_fail, M); frozen operating point from #2 | `RUN_ID=pilot1 COSTS=… bash scripts/gpu_smoke.sh pilot_val` |

`pilot_dev` steps:
1. `40_bank`: `collect_bank`, train_fit + train_dev, 2 banks per prompt, resumable.
2. `41_tune`: replay tuning of B3/B5/B6/B8 per cap α ∈ {0.5, 1.0}.
3. `42_bench_dev`: every method online on train_dev.
4. `43_report_dev`: report, then the freeze. This step **stops** if the evaluation config is not approved; that is
   the gate.

`pilot_val` runs the frozen methods online on val and reports. Each step resumes after interruption.

Guards that apply without exception: PREREG-A for train_fit/train_dev, PREREG-B for val, PREREG-C for test/GenEval,
sealed val_ext, failures counted at r_fail.

## 4. Time and cost (RTX 3090, $0.36/h; measured vs assumed)

**Measured on smoke1** (technical prompt, one sample each):
- single, 64 steps: 7.8 s;
- PSP-default online: 28.6 s;
- collector: 81.9 s per prompt-bank.

**Assumed** from smoke1's provisional M2 smoke profile (den is per forward: den_1 0.116 s, den_2 0.209, den_4
0.395, den_8 0.778; ver_8 1.95 s, ver_4 0.98 s):

| Method | Estimated seconds per episode |
|---|---|
| single32 | ≈ 3.9 |
| BoN-2 | ≈ 13.8 |
| BoN-4 | ≈ 26.3 |
| BoN-8 | ≈ 51.8 |
| B3/B5/B6/B8 | ≈ 20–35 |

| Session | Estimate | Basis |
|---|---|---|
| 0 watchdog check | ~10 min, ≈ $0.06 | assumption |
| 1 calibration | **≈ 5 h**: M3/M9 ≈ 2.2 h (50 × ~122 s augmented + 20 × 82 s), equivalence ≈ 0.3 h, M2 ≈ 1.8 h, M2b ≈ 0.2 h, pool check + chunk ≈ 0.5 h → ≈ $1.8 + bandwidth | 82 s measured; rest assumed |
| 2 bank | 240 prompts × 2 banks × 81.9 s ≈ **10.9 h** | measured per prompt-bank |
| 2 online train_dev | 100 instances × (fixed methods ≈ 132 s + 8 tuned × ≈ 30 s) ≈ **10.3 h** | assumed |
| 3 online val | 160 instances × ≈ 150 s (≈ 6 frozen methods) ≈ **6.7 h** | assumed |
| **Total** | **≈ 33 h GPU ≈ $12** + storage while stopped + bandwidth | M2/M9 replace these numbers |

**Correction.** `CALIBRATION_PLAN.md` earlier read den as per 4 forwards and quoted ~46 s per prompt-bank and
"≈ 14 GPU h". The measured collector time is 81.9 s per prompt-bank. Collecting all 540 small-pilot prompts would
take ≈ 24.6 h; banks for train_fit + train_dev only (val evaluated online) take ≈ 10.9 h.

## 5. Decisions that block running

1. **Pool approval** (Q15 English threshold effect, Q16 sampled grouping pending the GPU check): blocks session 1
   (calibration prompts) and everything after.
2. **Machine and budget for sessions 0–1** (instance, ~$2 for calibration).
3. **PREREG-A signature** (needs the results of session 1, the measurement machine Q1, preview dtype and chunk):
   blocks session 2.
4. **PREREG-B and `configs/evaluation.yaml` approval:** ε (δ_IR = τ_prac/2 from the train anchor), r_fail, number of
   primary contrasts M, reference method. Amendment 1 wording (Q12). Blocks the freeze and session 3.
