# diag1 run report: provenance, runtime, time and cost

ENGINEERING diagnostic on technical prompts; not G1. Results are in `BATCH_DRIFT_AUDIT.md` §11. This file records
where and how the run happened and what it cost. The diag1 and smoke1 artifacts are unchanged; everything below is
derived from them read-only.

## 1. Machine and why it changed

| | |
|---|---|
| Planned | Restart Vast instance **56044352** (smoke1's machine; disk with venv and weights kept). |
| What happened | The assistant could not start it: the laptop has no Vast CLI, API key file or key variable, and searching for other credentials was not allowed. The assistant asked the user to press Start in the console. |
| Instance used | **56070930** (label `NgocMinh_data`, RTX 3090 24 GB, driver 595.91.07, host machine 30391, Sweden, 200 GB disk, empty at first login). |
| Evidence for the switch | The user's message "`ssh -p 22395 root@194.14.47.19 -L 8080:localhost:8080` connect to new". At first login `CONTAINER_ID=56070930`; the API reported `start_date` 08:56:09Z and price $0.359/h. |
| Not recorded | Who created 56070930, why 56044352 was not restarted, and whether 56044352 is stopped or destroyed. The run does not establish who authorised the machine beyond the user supplying its endpoint. |
| Deviation from the instructions | The instructions named 56044352 as the instance to stop; the instance actually used and stopped was 56070930. The `-L 8080` tunnel in the supplied command was not used. A Jupyter server listening on 0.0.0.0:8080 (outside supervisor) was stopped before the workload. |

## 2. Code and runtime

- Code: `1100a12147856b751c62c230029cbf619ab0d7cb`, transferred by git bundle (sha256 `627ae929…95e8b4`). PSP
  `590f59f58384169c719e431dc01d14006fb0fc0c` (bundle `77cf3767…ef903`). The repo was clean before the run. After
  the run `git status` showed one entry (written by setup; not inspected before the stop).
- Production EPH measured in every part: `c1351a998e37d89ec892abf2f5f3ac16312ca20d3d12fdc359530bb3605c83bd`, the
  same as smoke1.
- Runtime fingerprints reconstructed from artifacts: `reports/runtime_fingerprints/{smoke1,diag1}.json` (schema
  `apsp.runtime_fingerprint/v1`). Differences in numeric-relevant fields
  (`reports/runtime_fingerprints/diff_smoke1_diag1.json`):

| Field | smoke1 (56044352) | diag1 (56070930) |
|---|---|---|
| Python | 3.10.20 | 3.10.21 |
| NVIDIA driver | 580.95.05 | 595.91.07 |
| GPU | RTX 3090, sm 8.6 | RTX 3090, sm 8.6 |
| torch / CUDA / cuDNN | 2.4.0+cu121 / 12.1 / 9.1.0 | same |
| diffusers / transformers | 0.31.0.dev0 / 4.38.2 | same |
| Engine-time flags, attention processor | not recorded by smoke1 | cudnn deterministic, TF32 matmul off, cuDNN TF32 on, SDPA flash+mem-efficient+math on, `AttnProcessor2_0`, deterministic algorithms off, fp16 reduced-precision reduction on |

- Despite the different Python patch and driver versions, the k=8 bank of the astronaut prompt is bitwise identical
  across the two machines (`reports/batch_drift/diag1_crosscheck_smoke1.json`).
- **What EPH v1 covers.** torch, CUDA and cuDNN versions; the cudnn benchmark/deterministic, TF32, SDPA and CUBLAS
  flags; model, weights, sampling, dtypes, scoring, seed scheme, generation code and the requirements file.
- **What EPH v1 does not cover.** GPU model, driver, Python, installed package patch versions, attention processor,
  `deterministic_algorithms`, fp16 reduced-precision reduction, OS.
- Because the EPH bodies hash equal, the flags EPH v1 covers were equal in smoke1 and diag1; the others are equal
  only where recorded.
- From this commit on, every run writes `p0/runtime_fingerprint.json` (step `02b_fingerprint`), and the harness
  stores the fingerprint in each report. EPH and its schema are unchanged, so old manifests stay valid.

## 3. Time

| Interval (UTC, 2026-10-15) | Duration | What |
|---|---|---|
| 08:56:09 → 08:58:00 | 1 min 51 s | Instance up, before the assistant's first login (hand-off) |
| 08:58:00 → 09:00:31 | 2 min 31 s | Identity and price check, Jupyter stop, bundle transfer and verification |
| 09:00:31 → 09:23:10 | **22 min 39 s** | Runner: setup 41 s, pytest 14 s, env 4 s, model 60 s, diagnostic parts 1,235 s |
| 09:23:10 → ~09:24 | ~1 min | Packaging and evidence tarball |
| ~09:24 → 14:35:22 | **~5 h 11 min** | **Idle.** The Claude session ended during the artifact download; the instance kept running |
| 14:35:22 → ~14:36:12 | ~1 min | Download (85 MB, sha256 verified) and stop request |
| **Total** | **5.67 h** | |

Diagnostic parts (wall seconds, including model load per process): audit 14, A/A 56 + 57, A/A compare 2, traj
322, unet 86, vae 68, rm 56, method 574.

## 4. Cost

Rate verified by API: compute $0.267/h plus storage $0.093/h = **$0.359/h**. Bandwidth $0.026/GB in either
direction.

| Item | Hours | Cost |
|---|---|---|
| Runner (compute actually used) | 0.377 | $0.136 |
| Setup, transfer, hand-off, packaging, download | ~0.12 | ~$0.04 |
| **Idle after the run** | ~5.19 | **~$1.86** |
| Total instance time | 5.67 | **$2.04** |
| Bandwidth: weights and wheels downloaded by setup | not measured | est. $0.10–0.25 (4–10 GB) |
| Bandwidth: artifacts to laptop (85 MB) | — | ~$0.002 |

- The approved budget was **$1 and 120 min**. Both were exceeded, almost entirely by idle time.
- **Cause.** The runner's timeouts bound the workload, not the rental. Nothing on the instance stopped it once the
  session that was supposed to download and stop was lost.
- **Fix.** `scripts/vast_watchdog.py` (see `COST_WATCHDOG.md`), armed before the workload and enforced by the
  runner on Vast.
- **Stop status.** A stop request was sent for 56070930 at about 14:36:12Z, and SSH was refused seconds later.
  Whether the provider accepted it and whether the instance is stopped was **not confirmed** through the API. It
  needs a console check.
- **Storage keeps billing while stopped.** About $0.093/h (~$2.2/day) for 56070930 and $0.056/h (~$1.3/day) for
  56044352, until each is destroyed. Destroying deletes the disk; that decision is the user's.

## 5. Inference cost actually measured

- diag1 measured no per-request inference latency. Harness wall times include model loading, hashing and CPU
  copies, so they are not latency evidence.
- The only measured generator primitives are smoke1's M2 smoke profile (`p0/SMOKE_generator.json`,
  `provisional: true`, 1 process, RTX 3090): mean µs per denoise interval `den_k`.

| k | den_k (ms) | k · den_1 (ms) | batch-1 overhead |
|---|---|---|---|
| 1 | 115.5 | 115.5 | — |
| 2 | 208.6 | 231.0 | +10.7% |
| 4 | 395.3 | 462.0 | +16.9% |
| 8 | 778.1 | 924.0 | +18.7% |
| 16 | 1523.7 | 1848.0 | +21.3% |

- The "+16–18%" quoted earlier for a batch-1 engine is the k=4 and k=8 rows of this table.
  - It is a provisional smoke measurement of den only (no VERIFY/decode/score, one sample set).
  - It is not a request-level cost and must be re-measured under the final engine before any decision.
- The "2–4× for FP32" figure was **never measured**; it is an expectation only.
