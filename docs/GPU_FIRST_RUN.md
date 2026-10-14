# GPU first run (rented Linux box)

Purpose: the first technical smoke, profiling and parity runs on a real CUDA machine. Run the commands **in
order**. Nothing here touches research splits; prompts are technical prompts outside the research pool until
the prompt pool is approved (QUESTIONS Q2/Q3). Every output lands in `reports/p0/` (small JSON, safe to commit
back) or `data/` (gitignored).

Do not run bank collection on research splits, tuning on val/test, or anything GenEval here. The code
refuses those until PREREG-A/B/C are signed.

## 0. Prerequisites

- Linux x86_64, NVIDIA driver that supports CUDA 12.1 wheels (`nvidia-smi` shows CUDA ≥ 12.1), `git`.
- `uv` (recommended: `curl -LsSf https://astral.sh/uv/install.sh | sh`) or `python3.10`.
- Disk: ~10 GB for the env, ~8 GB HF cache (SD1.5 fp32 safetensors + ImageReward), plus outputs.

## 1. Clone the exact commit

```bash
git clone --recurse-submodules -b bootstrap/p0 https://github.com/NguyenNgocMinh30012005/DMRL.git
cd DMRL
git checkout <COMMIT_SHA_FROM_HANDOFF>
git submodule update --init --recursive
git rev-parse HEAD                                  # must equal the handoff SHA
git submodule status                                # must show 590f59f58384169c719e431dc01d14006fb0fc0c third_party/psp
git -C third_party/psp status --porcelain           # must be empty (read-only submodule)
```

## 2. Build the GPU environment

```bash
bash env/setup_gpu_linux.sh            # creates .venv-gpu (Python 3.10), writes env/env.lock.linux-gpu.txt
source .venv-gpu/bin/activate
export HF_HOME=$PWD/.cache/hf          # keep weights out of the repo tree tracked by git (.cache is ignored)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # recorded in the numeric runtime; helps deterministic cuBLAS
```

If pip cannot resolve a pin, don't upgrade diffusers or transformers silently. Write down the exact error in
`docs/QUESTIONS.md` and stop at this step.

## 3. CPU sanity on the GPU box

```bash
python -m pytest tests -q              # contract/sim/stats + fake-backend engine tests; must all pass
```

## 4. CUDA and machine facts

```bash
python -m apsp.gpu.env_check --machine-id GPU_BOX_1 --out reports/p0/env_GPU_BOX_1.json --require-cuda
```

Exit code 3 means there is no CUDA: stop. Copy the detected facts into `configs/machines.yaml`
(`measurement-gpu`). Research decides whether this box is the measurement machine and sets the budget.

## 5. M1: model, revision, weights, scheduler, ImageReward keys

```bash
python -m apsp.gpu.profile --stage model --out-dir reports/p0
```

Check `reports/p0/M1_model.json`: `model_revision` (40-hex SHA), `weight_files_sha256`, 64 timesteps,
`terminal_update`, `reward_model.missing_keys == []`. Then pin the revision in `configs/pilot.yaml`
(`model_revision: <sha>`) and commit that one-line change. Every later run must report the same revision.

## 6. Technical smoke (prompt outside the research pool)

```bash
P="a photograph of an astronaut riding a horse"
python -m apsp.gpu.online --task smoke --prompt "$P" --synthetic-costs --methods b2 random fixed:2,9 \
       --disp-us 0 --out data/runs/smoke_online
python -m apsp.gpu.collect_bank --smoke-prompt "$P" --bank-name SMOKE_1 --machine-id GPU_BOX_1
python -m apsp.contracts.schema --bank data/banks/SMOKE_1 --validate
```

Expected: every record is `feasible` with `ledger_spend_us <= cap_us`, PNGs exist, and the smoke bank validates
(8 seeds x 16 checkpoints, 8 delivered). The costs here are **synthetic**, so the spend numbers exercise the
ledger and say nothing about real timing.

## 7. M0: hook, determinism, no extra forward

```bash
python -m apsp.gpu.profile --stage hook --out-dir reports/p0
```

Report `no_extra_forward` (UNet calls = 64 per run), `mode_none_feature_kernel_calls == 0`, repeat max abs
diffs / exact flags, and `hook_x0_vs_ddim_inversion_max_abs`. Exact flags are measurements; don't assume them.

## 8. M6: batch-size parity

```bash
python -m apsp.gpu.profile --stage batch-parity --out-dir reports/p0
```

Report the latent / preview / delivered deltas and winner flips for batch 1, 2, 4, 8 and 16 vs batch 8. Treat a
non-zero delta as a finding for G1 planning, not as a failure to hide.

## 9. Collector equivalence against the PSP reference

```bash
python -m apsp.gpu.collect_bank --test-equivalence --smoke-prompt "$P"
```

This needs `clip` and `google-genai` (installed by the setup script) because the reference module imports them.
After the prompt pool exists, rerun on 10 calibration prompts: `--test-equivalence --prompt-pool data/prompts/prompt_pool.parquet`.

## 10. M2: primitive profile (generator cost table)

```bash
python -m apsp.gpu.profile --stage generator --smoke --ks 1 2 4 8 --out-dir reports/p0     # quick check
python -m apsp.gpu.profile --stage generator --out-dir reports/p0                          # full: k=1..16, warm-up 10, n=50
```

Add `--measurement` only if research designated this box as the measurement machine; otherwise the table stays
`provisional: true`. Check `cv_flags` (CV > 0.15 needs a diagnostic) and that every prune pair is present.

## 11. M3 validity and VAE chunk (after the prompt pool, Q2/Q3)

```bash
python -m apsp.gpu.profile --stage validity --preview-dtype fp16 --prompts-file data/prompts/calibration.jsonl --out-dir reports/p0/fp16
python -m apsp.gpu.profile --stage validity --preview-dtype fp32 --prompts-file data/prompts/calibration.jsonl --out-dir reports/p0/fp32
for c in 1 2 4 8 16; do python -m apsp.gpu.profile --stage generator --ks 8 16 --vae-chunk $c --out-dir reports/p0/chunk_$c; done
```

50 calibration prompts x 8 seeds x 16 checkpoints = 6,400 previews per precision. Any rate > 1e-3 escalates
to research before a bank.

## 12. M9 throughput forecast (after the prompt pool)

```bash
python -m apsp.gpu.collect_bank --calib-throughput --prompt-pool data/prompts/prompt_pool.parquet --machine-id GPU_BOX_1
```

## Send back

`reports/p0/*.json`, `env/env.lock.linux-gpu.txt`, the console logs of failed steps, and the `configs/pilot.yaml`
revision pin. Don't send weights, banks or PNGs.

## If something fails

- OOM: keep the run directory and log. For a technical retry, `--vae-chunk 1` (and `--rm-device cpu` on very
  small GPUs) are allowed and recorded in the EPH. Don't change scientific settings.
- Import errors in step 9 only: the reference pipeline's extra dependencies are missing. Steps 5–8 don't need them.
