# Full experiment plan — implementation map (SD1.5 first)

Source: `docs/plans/Adaptive_PSP_Full_Experiment_Plan_20261016.md` (v1.0, 16/10/2026, Vietnamese; authoritative for
scope and rules). This page maps that plan onto the repo: what exists, what was added on 2026-10-16, what is
blocked and by which signature/decision. Status of each run: `docs/EXPERIMENT_STATUS.md`.

**Objective.** min E[C_π] subject to E[Q_π − Q_ref] ≥ −ε. Q_ref is PSP-default (proposed primary reference; all
other baselines are still reported). Native batching 8→4→2; replay = simulator for training/screening;
confirmatory conclusions are online. No generator fine-tuning, no kernel debugging, no action-space expansion.

## 1. Baseline matrix: gap list (HEAD `c251230` and later)

| ID | Method | Code | Registry string | Tested | CUDA real |
|---|---|---|---|---|---|
| B0 | single 64 | `sim/methods.py` | `single` | fake/synthetic | smoke1 (3090) only; **not on 4090** |
| B0-32 | single 32 (own T=32 scheduler) | `sim/methods.py`, `gpu/benchmark.py` | `single32` (online only) | fake | no |
| B1 | BoN-2/4/8, BoN-16 | `sim/methods.py` | `bon:N` | fake/synthetic | no |
| B2 | PSP-default | `sim/methods.py` | `psp` | synthetic; smoke1 online 28.6 s (3090) | 3090 only |
| B3 | best fixed pair per cap | `policies/fixed.py` | `fixed:J1,J2@a` | synthetic | no |
| B4b | PSP grid (7,823) per cap | `sim/b4b_grid.py`, **`sim/b4b_select.py` (new)** | **`b4b:IDX@a` (new)** | synthetic; numpy replay = `run_schedule` | no |
| B5 | screen-and-resume | `policies/screen_resume.py` | `sr:M,JS,Q@a` | synthetic | no |
| B6 | threshold | `policies/threshold.py` | `threshold:PATH@a` | synthetic | no |
| B7 | prompt-only selector | **`policies/prompt_selector.py` (new)** | **`b7:PATH@a`** | synthetic | no |
| B8 | one query then commit | `policies/query_commit.py` (CV standardisation now per fold) | `b8:PATH@a` | synthetic | no |
| B9 | random feasible | `policies/random_policy.py` | `random:SEED@a` | synthetic | no |
| B10-min | cheap latent predictor, fixed pred cuts | **`policies/latent_predictor.py` (new)**, engines `lfeat()` | **`b10:PATH@a`** | synthetic | no |
| RL-M1 | REINFORCE | `policies/reinforce.py` (+`load_policy`) | **`rl:artifact.json@a` (new)** | synthetic smoke | no; real training needs PREREG-B + G2 |
| External | Flash-BoN / FK-Steering | — | — | — | **deferred**: compatibility matrix in §4; B5 is not Flash-BoN |

Known limitations kept explicit:
- B4b: strategies whose primitives are missing from the measured cost table are reported as *uncovered* and never
  priced by interpolation; calibration measures k ∈ {1,2,3,4,8,12,16} so the whole grid is covered.
- B8: the prompt PCA-16 inside its features is fitted on all train_fit prompts (label-free); the ridge
  standardisation is fitted per fold. B7 refits PCA and standardisation per fold.
- B10-min OOF scores at the second cut use the 8-candidate group mean (approximation, noted in code); pred(k) is the
  predictor's measured CPU time on the machine that froze it (provisional).

## 2. Evaluators

- HPS v2.1 (`gpu/evaluators/independent.py`): every artifact the hpsv2 1.2.0 code really loads is pinned in
  `configs/evaluators.yaml` — package wheel sha256, HPS checkpoint (repo/revision/sha256), the OpenCLIP ViT-H-14
  laion2B-s32B-b79K base weights the package would otherwise download unpinned, and the tokenizer BPE file. The
  adapter never calls `hpsv2.score()` (unpinned latest checkpoint, default v2.0, reload per call). Separate venv:
  `env/setup_hpsv2_linux.sh`. **Not yet run on a GPU** (`verified_on_gpu: false`).
- GenEval: refused before PREREG-C (official repo/weights to be pinned then).

## 3. Orchestration

`python -m apsp.run_project --plan configs/full_experiments.yaml --resume` (new). Jobs P1–P8 in dependency order;
in `approved_budget` mode each GPU job needs an approved total budget, a provider price ≤ the approved ceiling and
enough budget left for its forecast × 1.25 plus a 20 min pack/stop margin; in the explicit `financial_cap_mode: none`
(current, D47) those checks are skipped and spend is only recorded. Always: an armed watchdog for the plan's
instance (**56261825**). `scripts/launch_project.sh` arms it and starts run_project in its own process group. PREREG markers are read, never written. Budget is cumulative across restarts (`budget.json`; external
spend recorded with `--add-spend`). `--status` prints why each job is blocked.

`python -m apsp.reports.forecast` turns the measured M9/M2/M2b of the calibration run into the P2–P7 table (GPU h,
USD central, USD × safety). It refuses synthetic tables.

## 4. External baselines: compatibility matrix (to verify before P8)

| Method | Official code | SD1.5 support | Needs | Status |
|---|---|---|---|---|
| Flash-BoN [R2] | github.com/flash-bon/flash-bon | to check | draft model + multi-stage verifier | deferred; never renamed from B5 |
| FK-Steering | official repo | SD1.5 reported in paper (to check at pinned commit) | stochastic sampler (eta > 0) | deferred; run with the authors' valid sampler, differences reported |
| PSP [R1] | third_party/psp @590f59f | yes | — | B2/B4b reproduce its schedules on our backbone |

## 5. Phase order and gates

| Phase | Content | Gate |
|---|---|---|
| P1 | watchdog live check → env/fingerprint/VRAM → full-pool check → M3/M9 → chunk → M2 (k=1,2,3,4,8,12,16) → equivalence → M2b → forecast | budget for P1; pool approval (Q15/Q16) before M3/M9 |
| P2 | bank train_fit + train_dev, 2 banks × 8 seeds | **PREREG-A** |
| P3 | replay tuning (B3/B4b/B5/B6/B7/B8/B10), online train_dev benchmark, descriptive report; anchor/threshold proposal for B | PREREG-A; freeze needs `evaluation.yaml` approved |
| P4 | RL-M1 (3 training seeds), B7/B8/B10 first | PREREG-B + G2 marker |
| P5 | online val, frozen methods only | PREREG-B |
| P6 | GenEval test | PREREG-C |
| P7 | ablations (list frozen before confirmation) | PREREG-B |
| P8 | SDXL / flow | separate budget |

## 6. Conflicts with older documents (resolved in favour of the 2026-10-16 plan)

- Measurement machine: older plans assume an RTX 3090 at ≤ $0.40/h (56070930), the plan v1.0 names 56224386
  (RTX 4090 48 GB, $0.778/h). The measurement machine is now **56261825, RTX 4090 24 GB at $0.527/h** (D48). Neither
  3090 nor 48 GB latencies are reused; the 3090 numbers in `configs/full_experiments.yaml` are placeholders until
  `p1_forecast` replaces them.
- Budget: plan §2/§8 require an approved total budget and price checks; the user removed the rental cap for the
  agreed suite (D47, explicit `financial_cap_mode: none`). The watchdog stops on completion / no eligible job /
  workload death / stall instead of a money deadline. Scientific gates are unchanged.
- Calibration-split table (D49): an addition approved by the user, descriptive only; plan §P1 still forbids choosing
  pilot size from method results on calibration (the table runs after the forecast).
- The user referred to a "plan v1.1"; the only plan file present is v1.0 (identical copies in the repo root and
  `docs/plans/`), so v1.0 + Spec/Guide v1.1.1 are followed.
- M2 subset: `CALIB1_SESSION_PLAN.md` measured k ∈ {1,2,4,8,16}; B4b needs 3 and 12 too, so the plan adds them.
- Equivalence: was optional in the minimal calibration; the plan puts it in P1 (input to PREREG-A).
- The 33 GPU-hour total of `PILOT_RUN_PLAN.md` is a pilot figure for a 3090, not a full-paper ETA.
