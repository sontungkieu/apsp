# GPU first run (rented Linux box): technical smoke

Scope: **engineering only**. Up to 3 hand-written technical prompts (`configs/technical_smoke.yaml`, split
`technical_smoke`), checked against the research prompt pool and GenEval text. No research split, no pilot
bank, no RL on real data, no val/test/GenEval, no PREREG signing. The guards in the code refuse those anyway.

Expected box: 1 NVIDIA GPU with 24 GB (RTX 3090 or 4090), ≥64 GB RAM, ~150 GB disk, a single GPU worker.
Plan to stop within **2 hours of rental**, including setup and downloads. That is a target, not a measurement:
`MAX_MINUTES` stops the runner before starting a new step once the budget is spent, and the steps are ordered
so the essential ones run first. Per-step timeouts are in `scripts/gpu_smoke.sh`.

Stage names in the code: `profile --stage model` = M1, `--stage hook` = M0, `--stage generator` = M2,
`--stage validity` = M3, `--stage batch-parity` = M6, `collect_bank --calib-throughput` = M9.

## 0. Storage

Put everything on **persistent** storage. On RunPod that is the network volume mounted at `/workspace`; the
container disk can vanish when the Pod stops.

```bash
export APSP_STORE=/workspace/apsp          # HF cache, venv, runs and the artifact tarball all live here
mkdir -p "$APSP_STORE"
```

## 1. Clone the candidate commit

```bash
cd /workspace
git clone --recurse-submodules -b bootstrap/p0 https://github.com/NguyenNgocMinh30012005/DMRL.git
cd DMRL
git checkout <COMMIT_SHA_FROM_HANDOFF> && git submodule update --init --recursive
git rev-parse HEAD                        # must equal the handoff SHA
git submodule status                      # must show 590f59f58384169c719e431dc01d14006fb0fc0c third_party/psp
git -C third_party/psp status --porcelain # must print nothing
```

The repo is private, so the box needs a GitHub credential with read access to it (deploy key or token), set
up by the user. Never commit or log a token.

## 2. Run the smoke (all steps, resumable)

```bash
APSP_STORE=/workspace/apsp RUN_ID=smoke1 MAX_MINUTES=100 MACHINE_ID=gpu_box_1 bash scripts/gpu_smoke.sh
```

To run single steps: `bash scripts/gpu_smoke.sh 04_single 05_psp`. Re-running with the same `RUN_ID` skips
steps that have a `.done` marker. A failed step is not retried until you read its log and remove
`runs/<RUN_ID>/.failed/<step>`. A second concurrent run with the same `RUN_ID` is refused (flock).

The steps, in order, with the exact commands the runner executes (`PY=$APSP_STORE/venv-gpu/bin/python`,
`TS="--smoke-config configs/technical_smoke.yaml --out $RUN/tsmoke"`):

| Step | Command | Purpose |
|---|---|---|
| 00_setup | `bash env/setup_gpu_linux.sh` (VENV=$APSP_STORE/venv-gpu) | Python 3.10 env with pinned deps; asserts torch 2.4.0 / diffusers 0.31.0.dev0 / transformers 4.38.2; writes `env/env.lock.linux-gpu.txt` |
| 01_pytest | `$PY -m pytest tests -q` | CPU suite inside the GPU env (the no-CUDA guard test skips on a CUDA host) |
| 02_env_check | `$PY -m apsp.gpu.env_check --machine-id $MACHINE_ID --out $RUN/p0/env_$MACHINE_ID.json --require-cuda` | GPU model, VRAM, RAM, cgroup limit, driver, CUDA, numeric runtime |
| 03_model (M1) | `$PY -m apsp.gpu.profile --stage model --out-dir $RUN/p0` then `$PY scripts/check_m1.py $RUN/p0/M1_model.json reports/local_smoke/M1_model.json` | downloads + hashes SD1.5 (pinned revision), scheduler dump, ImageReward revision, missing keys, pinned BERT tokenizer hashes; compares with the 2026-10-14 laptop report |
| 04_single | `$PY -m apsp.gpu.technical_smoke $TS --step single` | one candidate, 64 steps, delivered decode, real ImageReward score |
| 05_psp | `$PY -m apsp.gpu.technical_smoke $TS --step psp` | PSP-default 8→4→2 with identity / CFG-row / fresh-score / final-delivery audit |
| 06_collector | `$PY -m apsp.gpu.technical_smoke $TS --step collector` | 1 prompt × 1 bank × 8 seeds × 16 checkpoints; validate; resume; EPH mismatch refusal |
| 07_parity | `$PY -m apsp.gpu.technical_smoke $TS --step parity` | online vs replay on that bank: traces, survivors, winner, scores (not G1) |
| 08_equivalence | `$PY -m apsp.gpu.collect_bank --test-equivalence --smoke-prompt "a photograph of an astronaut riding a horse"` | our engine vs the pinned PSP reference pipeline (A/B/C comparisons) |
| 09_m0_hook (M0) | `$PY -m apsp.gpu.profile --stage hook --out-dir $RUN/p0` | repeat determinism, 64 UNet calls per run, hook x0 vs DDIM inversion |
| 10_m6_batch (M6) | `$PY -m apsp.gpu.profile --stage batch-parity --out-dir $RUN/p0` | batch 1/2/4/8/16 vs 8 |
| 11_m2_generator (M2) | `$PY -m apsp.gpu.profile --stage generator --smoke --ks 1 2 4 8 16 --out-dir $RUN/p0` | primitive timings, **SMOKE / provisional** (n=5) |
| 12_vae_chunk | `$PY -m apsp.gpu.profile --stage generator --smoke --ks 8 --vae-chunk C --out-dir $RUN/p0/chunk_C` for C in 1 2 4 8 | technical survey for Q7; report only, no freeze |

Steps 00–03 are prerequisites: if one fails, the runner stops. Pass/fail criteria and stop conditions are in
`docs/GPU_SMOKE_CHECKLIST.md`.

## 3. Download the artifacts before terminating the Pod

The runner writes `$APSP_STORE/<RUN_ID>_artifacts.tgz` (logs, JSON reports, summary.tsv, memory evidence,
env freeze, the small technical bank and a few PNGs; no weights). From the local machine:

```bash
scp -P <SSH_PORT> root@<POD_HOST>:/workspace/apsp/smoke1_artifacts.tgz .
# or: runpodctl send /workspace/apsp/smoke1_artifacts.tgz   (then runpodctl receive <code> locally)
```

Commit back only the small JSON reports you want reviewed (under `reports/gpu_smoke/<RUN_ID>/`) plus
`env/env.lock.linux-gpu.txt`. Never commit weights, banks or images.

## 4. What this run does NOT establish

- Cost tables: step 11 is a smoke (n=5, provisional). The ledger in steps 05/07 uses the SYNTHETIC
  `engineering_only` table. No latency, cap or research conclusion may come from either.
- Replay–online parity / G1: step 07 compares a few hand-written prompts. It is a code-path check, not the
  statistical protocol of Spec §10.2.
- Preview dtype / VAE chunk: step 12 only produces numbers for a proposal (QUESTIONS Q7).
