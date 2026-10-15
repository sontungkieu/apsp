# GPU smoke checklist (first session, engineering only)

Runner: `scripts/gpu_smoke.sh` (see `docs/GPU_FIRST_RUN.md`). Each step writes `runs/<RUN_ID>/logs/<step>.log`
and a line in `summary.tsv` (exit code, seconds, failure class). Memory snapshots before and after each step go
to `logs/memory_evidence.txt` (free, cgroup memory.max/peak/events, nvidia-smi).

## Pass criteria per step

| Step | Pass if | Artifacts |
|---|---|---|
| 00_setup | exit 0; version assertion prints torch 2.4.0+cu121, diffusers 0.31.0.dev0, transformers 4.38.2, Python 3.10 | `env/env.lock.linux-gpu.txt`, log |
| 01_pytest | all tests pass (the no-CUDA guard test is skipped on CUDA) | log |
| 02_env_check | exit 0, `cuda_available: true`, GPU name / VRAM / driver / CUDA / cgroup limit recorded | `p0/env_<MACHINE_ID>.json` |
| 03_model | `check_m1.py` prints all PASS: model revision = pinned `451f4fe1…`, 64 timesteps, ε-prediction, RM revision `5736be03…`, RM missing_keys = [], tokenizer revision `86b5e093…` with file hashes, **weight file SHA256 and timesteps equal to the 2026-10-14 laptop report** | `p0/M1_model.json`, `p0/model.json` |
| 04_single | `single.json` status PASS: 64 UNet calls, latent/image/score valid, finite IR, 512×512 PNG | `tsmoke/single.json`, `single_s0.png` |
| 05_psp | `psp.json` PASS: prunes at j=4 and j=8, 8→4→2; for each prune: rows == state.alive == kept, old order kept, latents follow identities, CFG uncond/cond rows exact, fresh scores from the same j, keep == fixed top-k; winner ∈ final survivors with top delivered score; 64 UNet calls | `tsmoke/psp.json`, `psp_winner.png` |
| 06_collector | `collector.json` PASS: bank valid (`validate_bank`), 128 preview rows per prompt-bank, resume skips every shard with 0 extra UNet calls, EPH mismatch refused | `tsmoke/collector.json`, `tsmoke/TSMOKE_BANK/` |
| 07_parity | `parity.json` PASS (feasibility agrees). Report `n_trace_survivor_winner_equal` / `n` and `max_score_abs_diff`. Differences are findings to analyse, not hidden | `tsmoke/parity.json` |
| 08_equivalence | exit 0 and a report with A/B delivered exact-hash flags, max abs diffs, C (j=4,8,16) x0/score diffs. Exactness is reported as measured, no tolerance claimed as bitwise | log (JSON printed) |
| 09_m0_hook | `no_extra_forward: true` (64 calls/run), `mode_none_feature_kernel_calls: 0`; repeat exact flags/diffs reported | `p0/hook.json` |
| 10_m6_batch | report produced for batch 1/2/4/8/16 (latent/preview/delivered deltas, winner flips). Passing means the script ran; it is **not** a batch-invariance acceptance (see BATCH_DRIFT_AUDIT.md) | `p0/batch-parity.json` |
| 11_m2_generator | report produced; `provisional: true`; every key present (den/ver/fin/prune/feat/lfeat/fixed for the ks run) | `p0/SMOKE_generator_costs*.json`, `p0/SMOKE_generator.json` |
| 12_vae_chunk | reports produced per chunk | `p0/chunk_C/…` |

## Stop conditions

- Any of 00–03 fails: the runner stops. Read the log; fix the environment, not scientific pins.
- `check_m1.py` reports different weight hashes or timesteps than the laptop report: stop and escalate (EPH
  inputs differ).
- Failure class `cuda_vram_oom`, `cgroup_oom_kill`, `host_oom_killer` or `sigkill_unknown_source(supervisor?)`:
  do not repeat the same command. Record the class, `memory_evidence.txt` and `nvidia-smi`. A retry may only
  change engineering settings (`--vae-chunk 1`), recorded in the EPH.
- `MAX_MINUTES` reached: the runner stops before the next step. Download artifacts first, then decide.
- Any temptation to point a step at a research split: stop. The guards refuse; do not work around them.

## Failure classes (from `classify()` in the runner)

`ok`, `cuda_vram_oom` (log contains CUDA OOM), `cgroup_oom_kill` (exit 137 and the cgroup `oom_kill` counter
grew during the step: v2 `memory.events`, v1 `memory/memory.oom_control`), `host_oom_killer` (exit 137 and dmesg
OOM), `sigkill_unknown_source(supervisor?)` (exit 137 without OOM evidence), `step_timeout` (exit 124), `no_cuda`,
`error_exit_<n>`. Helpers live in `scripts/gpu_smoke_lib.sh`, tested by `tests/test_gpu_smoke_lib.py`. The run
records its cgroup version in `runs/<RUN_ID>/cgroup.txt`. The runner exits non-zero if packaging fails.

smoke1 (2026-10-15, commit `933e9ac`) ran the earlier runner: its tar call put `--exclude` after the member names,
so GNU tar exited 2 and an empty `.lock` file was archived (contents otherwise complete), and on that cgroup v1
host the OOM classifier could not read `memory.events`. Both are fixed after smoke1 (D28); smoke1 results are
unaffected because every step exited 0.

## Before terminating the Pod

1. `ls -la $APSP_STORE/<RUN_ID>_artifacts.tgz` and copy it locally (scp/runpodctl).
2. Check the local copy opens (`tar -tzf`).
3. Note GPU model, hourly price and total minutes used for the GPU-hours report (Spec §13.3).
