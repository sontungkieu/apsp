# Before the pilot bank: what is still missing

Status 2026-10-15, after diag1. Nothing here collects the pilot bank, trains RL on real data, uses val/test or signs
a PREREG.

## 1. Open items by area

### Engine (freeze for PREREG-A)

| Item | State | Needs |
|---|---|---|
| Batch policy | Proposal: native batching (Amendment 1 draft) | research decision (Q10, Q12); optional backend A/B (§3) first |
| Numeric flags and attention processor | Recorded (runtime fingerprint v1) but not frozen | freeze in PREREG-A together with GPU model, driver range and package versions |
| Preview dtype, VAE decode chunk | `null` in `pilot.yaml` (Q7) | M3 validity runs on calibration prompts |
| M0 hook, M6 batch, equivalence | Measured on technical prompts (smoke1, diag1) | re-run on the frozen engine and calibration prompts (collector equivalence on 10 calibration prompts) |
| Cost watchdog | Implemented and tested locally | one live check on a provider (`COST_WATCHDOG.md`) before any long session |

### Cost table and caps

| Item | State | Needs |
|---|---|---|
| Generator cost table (den/ver/fin/prune/feat, k=1..16) | Only the smoke table (provisional, one machine) | full M2 with `--measurement` on the designated measurement machine (Q1), n=50; prune pairs from the PSP grid audit (Q6) |
| Caps / C_ref | Derived from the cost table | after full M2 |
| M9 throughput forecast | Not run | prompt pool and GPU |

### Prompt pool

| Item | State | Needs |
|---|---|---|
| English filter | Placeholder `en-ascii-v1` | Q2 |
| DiffusionDB metadata file and revision | Open | Q3 |
| Test split definition | Open | Q5 |
| Pool build and freeze (calibration 50 + subset 20, train/dev/val/val_ext) | Code ready | Q2, Q3, Q5, then a GPU for the CLIP filter; research approves |

### PREREG-A proposal (Spec §11.5 A)

Machine and budget (Q1), prompt split, sampler and seed, precision (Q7), scoring, feature definitions, EPH plus
runtime fingerprint, generator costs and caps, collector equivalence, P0 acceptance. It can be drafted once the
rows above are in. Agents write it under `docs/PREREG_PROPOSALS/`; research signs.

**Order.** Research answers Q1–Q3, Q5–Q7, Q10/Q12 (and Q13 if the A/B is wanted). Then: prompt pool (GPU), M3,
full M2 on the measurement machine, collector equivalence, then the PREREG-A draft, then signature, then the pilot
bank.

## 2. GPU work that needs a machine and budget (not started)

| Session | Content | Rough runner time (RTX 3090) | Depends on |
|---|---|---|---|
| Backend A/B (optional) | §3 | 25–30 min | Q13 |
| Prompt pool + M3 | CLIP filter; 6,400 previews per precision | 1–2 h | Q2, Q3, Q5 |
| Full M2 + M9 + equivalence | measurement machine, n=50 | 1.5–3 h | Q1, Q6, pool |

Each session runs with the watchdog armed (deadline = approved minutes; idle grace 15 min). Costs are reported as
compute, idle and bandwidth separately (`DIAG1_RUN_REPORT.md` §4 format).

## 3. Proposed backend A/B (one session, fixed end)

**Purpose.** Decide one question: is there a cheap UNet configuration that makes batch size irrelevant (bitwise),
which would make replay exact and change Amendment 1? If not, keep native batching and stop investigating kernels.

**Configurations.** Fixed list; nothing is added during the run.

| ID | Change from production (C0) |
|---|---|
| C0 | production: native batching, SDPA flash + mem-efficient + math enabled, fp16 reduced-precision reduction on, deterministic algorithms off |
| C1 | SDPA math only (flash and mem-efficient disabled) |
| C2 | `torch.use_deterministic_algorithms(True)` (CUBLAS_WORKSPACE_CONFIG already set) |
| C3 | `allow_fp16_reduced_precision_reduction = False` |
| C4 | C1 + C2 + C3 |

**Measurements per configuration**, technical prompt `m6-technical`, the same 16 CandidateIds:
1. UNet single-forward invariance: the existing `unet` part, b1/b2/b4/b16 vs b8 at forwards 0/16/32/63.
2. den_k timing for k = 1, 2, 4, 8 (M2 smoke subset), with the same warm-up and repeats for every configuration.
3. Only if (1) is bitwise for every size: the `traj` part for b1/b2/b4 (is the full trajectory invariant?).

**Implementation needed first** (local, CPU-tested):
- a `--numeric-variant {C0..C4}` option applied before the backend is built, recorded in the runtime fingerprint
  and in a variant field of the EPH body for non-C0 variants only (so C0 keeps `c1351a99…`);
- an A/B preset in the runner.

**Budget and limits:**
- one instance, one session;
- watchdog deadline 60 min from instance start;
- runner `MAX_MINUTES=40`;
- budget **≤ $0.60** including idle and bandwidth (estimate $0.20–0.35 at $0.36/h, plus up to ~$0.25 bandwidth on
  a fresh disk);
- no reruns of a failed configuration in the same session.

**Stopping and decision rule** (thresholds left to research):
- Each configuration is reported as bitwise batch-invariant (yes/no per size and forward) together with its den_k
  overhead vs C0.
- If no configuration is bitwise invariant, the A/B ends. Native batching stays, Amendment 1 applies, and there is
  no further kernel debugging.
- If one or more are invariant, research chooses whether the overhead is acceptable (a maximum overhead
  percentage, to be set by research before the run).
  - If yes: an engine-change proposal with a new EPH and BMH fields, and M0/M2/M6/M9/equivalence re-measured
    (`BATCH_DRIFT_AUDIT.md` §10).
  - If no: native batching stays.
- **Out of scope:** padding, batch-1, FP32, custom kernels, profiling which GEMM/conv kernel is selected.
