# Pre-pilot acceptance checklist

Status 2026-10-15. Legend:
- **done**: implemented and checked locally;
- **ready**: implemented, needs the GPU session;
- **research**: needs a research decision;
- **blocked**: waits on another item.

Nothing on this list collects the pilot bank, trains RL on real data, uses val/test, or signs a PREREG.

## Engine

| Item | Status | Evidence / next |
|---|---|---|
| Batch policy: native batching 8→4→2 | **done** (research decision 2026-10-15) | Amendment 1 rev. 2 |
| EPH (`c1351a99…`) + runtime fingerprint v1 recorded per run | done | `apsp/gpu/fingerprint.py`, step `02b_fingerprint` |
| GPU model, driver range, package versions frozen | research (PREREG-A) | `reports/runtime_fingerprints/` |
| Preview dtype (Q7) | ready | M3 in `31_calib_m3m9`; escalate if > 1e-3 |
| VAE decode chunk (Q7) | ready | `34_calib_chunk`; rule Guide CC-4, applied in the PREREG-A proposal |
| M0 / M6 / A/A on the frozen engine | done on technical prompts (smoke1, diag1) | re-check on calibration prompts optional |
| Collector equivalence on 10 calibration prompts | ready | `33_calib_equivalence` (reuses step-31 tensors) |
| Cost watchdog verified on the real provider | ready | `scripts/watchdog_live_check.sh`; the user confirms *Stopped* in the console |

## Cost table

| Item | Status | Evidence / next |
|---|---|---|
| Prune pairs for main, B5 and B4b | done | `required_prune_pairs()` (21 pairs); `reports/b4b/` |
| M2 den/ver/fin k=1..16, feat/lfeat, fixed, prune; n=50 | ready | `35_m2` per k, `36_m2_merge` (coverage check) |
| Table final (not provisional) | blocked | needs Q1 (designated measurement machine) and `M2_MEASUREMENT=1` there |
| M2b residual (B2 online vs primitives, ≥ 20 episodes) | ready | `37_m2b` |
| M8 peak memory (collector, M2 per k, online B2) | ready | steps 31, 35, 37 |
| Caps / C_ref | blocked | after the M2 table |

## Prompt pool

| Item | Status | Evidence / next |
|---|---|---|
| DiffusionDB 2M metadata pinned (revision + sha256) | done | `configs/prompt_pool.yaml` |
| English rule fastText lid.176 top-1 en, p ≥ 0.90 (model pinned) | done (rule applied) / research | finding: removes 82.5% of top-1-English prompts (`reports/prompt_pool/`) |
| NSFW < 0.1, 4–60 CLIP tokens, normalisation, exact dedup | done | pool manifest |
| GenEval manifest pinned, text-only dedup | done | 0 contaminated candidates |
| Near-duplicate groups / group-disjoint split | done for the candidates; full pool ready | `30_pool_full_check` (GPU); deviation `sampled-grouping-v1` |
| Pilot small 190/50/80, val_ext 220 sealed, calibration 50, M9 subset 20 | done (proposal) | exact counts; val_ext text only in `val_ext_sealed.parquet` |
| Test split definition (Q5) | research | open |
| Research approval of pool and split | research | `reports/prompt_pool/PROMPT_POOL_REPORT.md` |

## Methods and protocol

| Item | Status | Evidence / next |
|---|---|---|
| B4b grid from pinned PSP code | done | 7,823 strategies, identical to PSP; all checkpoints on the bank |
| Amendment 1 (native batching, simulator replay, online confirmation) | research (draft rev. 2) | `docs/PREREG_PROPOSALS/AMENDMENT_1_replay_online_DRAFT.md` |
| M9 bank-time and pilot GPU-hour forecast | ready | `32_calib_summary`. Measured smoke1: 81.9 s per prompt-bank, ≈ 10.9 GPU h for train_fit + train_dev banks (`PILOT_RUN_PLAN.md` §4) |
| PREREG-A proposal | blocked | needs the items above plus Q1; agents draft it, research signs |
| Pilot bank collection | blocked | PREREG-A signed |
| Pilot baselines (single, single32, BoN-2/4/8, PSP, B3, B5, B6, B8) | done | `PILOT_RUN_PLAN.md` §1; B4b, B7, B10, RL training planned |
| Online quality-cost benchmark + report + NI/savings status | done | `apsp/gpu/benchmark.py`, `apsp/reports/quality_cost.py`; conclusions `not_concludable` until `configs/evaluation.yaml` is approved |
| Independent evaluation metric (HPSv2) | ready (unpinned) | `apsp/gpu/evaluators/independent.py`; pin package and checkpoint in `configs/evaluators.yaml` |
