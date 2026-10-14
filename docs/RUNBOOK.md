# Runbook

Two environments:

| Env | Where | Purpose | Python / key deps |
|---|---|---|---|
| CPU | Windows dev box or any Linux | contracts, DP, replay, policies, stats, reports, synthetic tests, fake-backend engine tests | ≥3.10; numpy/pandas/pyarrow/scipy/pillow/pyyaml; CPU torch for RL |
| GPU | Linux + CUDA (rented box) | generator, ImageReward, profiling, collector, online, parity | 3.10; torch 2.4.0+cu121, diffusers @ af28ae2d, transformers 4.38.2, ImageReward @ 2ca71bac, CLIP @ dcba3cb2 (`env/requirements-gpu-linux.txt`) |

CPU modules import and run without CUDA. GPU commands call `apsp.gpu.runtime.require_cuda()` and exit with
`CudaUnavailableError` when there is no CUDA. They never fall back to CPU silently. The only non-CUDA GPU-code
path is `--fake-backend`, a test fixture whose outputs are labelled `backend: fake`.

## A. Windows local (CPU)

```powershell
git -c core.longpaths=true clone --recurse-submodules -b bootstrap/p0 https://github.com/NguyenNgocMinh30012005/DMRL.git
cd DMRL
uv venv --python 3.11 .venv            # or: py -3.11 -m venv .venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,rl]"
python -m pytest tests -q
python -m apsp.quickstart --out data/quickstart
```

Optional technical GPU smoke on a small Windows GPU (this machine: RTX 3050 Ti Laptop 4 GB). It is not a
measurement machine and its results never feed cost tables:

```powershell
uv venv --python 3.10 .venv-gpu-win
$P = ".venv-gpu-win\Scripts\python.exe"
uv pip install --python $P "setuptools==69.5.1" wheel
uv pip install --python $P torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
# requirements without hpsv2 / google-genai (not needed for the smoke)
Get-Content env\requirements-gpu-linux.txt | Select-String -NotMatch "^(hpsv2|google-genai)" | Set-Content -Encoding utf8 req-win.txt
uv pip install --python $P -r req-win.txt
$env:PYTHONUTF8 = "1"   # ImageReward's setup.py reads its README with the locale codec on Windows
uv pip install --python $P --no-deps --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" "git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3"
uv pip install --python $P --no-deps -e .
$env:HF_HOME = "$PWD\.cache\hf"
& $P -m apsp.gpu.env_check --machine-id local-win-dev --out reports/local_smoke/env_local-win-dev.json --require-cuda
& $P -m apsp.gpu.profile --stage model --rm-device cpu --vae-chunk 1 --out-dir reports/local_smoke
```

`--rm-device cpu` keeps ImageReward off the 4 GB GPU. Both overrides are recorded in the EPH.

## B. Linux GPU (rented box)

The exact first-run sequence is in `docs/GPU_FIRST_RUN.md`. Summary:

```bash
git clone --recurse-submodules -b bootstrap/p0 https://github.com/NguyenNgocMinh30012005/DMRL.git && cd DMRL
git checkout <COMMIT_SHA> && git submodule update --init --recursive
bash env/setup_gpu_linux.sh                       # .venv-gpu, writes env/env.lock.linux-gpu.txt
source .venv-gpu/bin/activate
export HF_HOME=$PWD/.cache/hf
python -m apsp.gpu.env_check --machine-id MEASUREMENT_ID --out reports/env_MEASUREMENT_ID.json --require-cuda
```

## C. CLI map (Guide §13 → implementation)

| Guide command | Status in this commit |
|---|---|
| `python -m apsp.gpu.env_check --machine-id ID --out F` | implemented (+ `--require-cuda`) |
| `python -m apsp.data.prompts --config C --out data/prompts` | implemented; needs `--metadata` (local DiffusionDB metadata, never auto-downloaded) and a GPU env for the CLIP encoder; rules pending Q2/Q3 |
| `python -m apsp.gpu.profile --stage model` | implemented (M1: revision, weight SHA256, scheduler dump, terminal coefficients, RM key check) |
| `python -m apsp.gpu.profile --stage hook` | implemented (M0: repeat determinism, UNet call count, hook-vs-inversion x0) |
| `python -m apsp.gpu.profile --stage validity` | implemented (M3 per precision; run once with `--preview-dtype fp16` and once with `fp32`; needs `--prompts-file` with calibration prompts for the ≥6,400-preview requirement) |
| `python -m apsp.gpu.profile --stage generator` | implemented (M2: den/ver/fin/prune/feat/lfeat/fixed → `generator_costs.json` + raw samples + GCH; `--measurement` only on the designated machine) |
| `python -m apsp.gpu.profile --stage batch-parity` | implemented (M6: batch 1/2/4/8/16 vs 8) |
| `python -m apsp.gpu.collect_bank --test-equivalence` | implemented (A/B/C comparisons against the PSP reference pipeline via a read-only import); needs the full GPU env incl. clip + google-genai |
| `python -m apsp.gpu.collect_bank --calib-throughput` | implemented (M9 forecast, peak VRAM, disk) |
| `python -m apsp.gpu.collect_bank --split ... --resume` | implemented; refused until PREREG-A is signed |
| `python -m apsp.contracts.schema --bank B --validate` | implemented |
| `python -m apsp.reports.anchor --bank B --proposal-only` | implemented (also needs `--costs`); writes an unsigned proposal |
| `python -m apsp.gpu.online --task replay-parity --split train_dev` | implemented (needs `--bank`, `--costs`; PREREG-B guard and an EPH match for research banks) |
| `python -m apsp.reports.replay_vs_online` | implemented (`--records`, `--tau` from signed PREREG-B) |
| `python -m apsp.reports.headroom --look 1` | implemented (needs `--bank`, `--costs`; PREREG-B guard on research banks) |
| `python -m apsp.policies.reinforce --train` | guard implemented (PREREG-B + `G2_EVALUATED`); real-bank training waits for Q4 / PREREG-B values. `--synthetic-smoke` works now |
| `python -m apsp.reports.ctrl_costs` | implemented for synthetic exercise only (`--synthetic`); frozen-artifact profiling comes in P3 |
| `python -m apsp.reports.pilot_report` | **not implemented** (P3) |
| `python -m apsp.gpu.online --split test`, `apsp.gpu.evaluators.geneval/hps` | **not implemented**; `--task test` only checks the PREREG-C guard (P4) |
| `python -m pytest tests/contract tests/sim` | implemented |
| `python -m pytest tests/gpu` | implemented; runs on the fake backend without CUDA. Real-CUDA checks are the profile/collect commands above |

Extra commands: `python -m apsp.quickstart`, `python -m apsp.data.synthetic_bank`,
`python -m apsp.gpu.online --task smoke`, `python -m apsp.gpu.collect_bank --smoke-prompt`.

## D. Baselines status

| Method | Code | Notes |
|---|---|---|
| B0 / B1 | `apsp.sim.schedules.bon_spec`, `apsp.policies.fixed.bon_feasible_n` | F(1) never calls RM (tested) |
| B2 | `fixed_pair_spec(4, 8)` | reference |
| B3 | `apsp.policies.fixed.best_fixed`, `select_b3_dev` | tune train_fit, choose train_dev, per cap |
| B6 | `apsp.policies.threshold` | six parameters, ≤2000 evals, fit/dev selection |
| B9 | `apsp.policies.random_policy` | |
| M1 | `apsp.policies.reinforce` | masked REINFORCE + value net |
| O1/O1-free/O2/O3 | `apsp.sim.oracles` | O1-free = O1 on the table built with `MethodCosts.scaled_verifier(0)` |
| B4b, B5, B7, B8, B10-min, A1-A6 | **not yet** | grammar G-A1 and cost paths (pred cuts, arbitrary keep, pool16 rows) exist; method code is next in dependency order |

## E. Operational rules

- Data, banks, images, latents, weights and HF caches live under `data/` and `.cache/` (gitignored).
- A run that fails or hits OOM keeps its directory. Collector partials move to `<bank>/failed/` with `REASON.txt`.
- Never resume a bank across EPH/config changes; the collector refuses. Start a new `--bank-name`.
- Wall-clock is reported separately from the ledger. A hard ledger cap is not a wall-clock guarantee.
