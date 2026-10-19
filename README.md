# Adaptive PSP (DMRL)

Source được quản lý tại [sontungkieu/apsp](https://github.com/sontungkieu/apsp),
branch `main`. Server tải cùng source commit qua Git bundle trên
[HF public distribution](https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128).
Xem [publish/update workflow](docs/lite/SOURCE_DISTRIBUTION.md); model/data/runtime
giữ riêng, không rebuild theo mỗi thay đổi source.

APSP Lite data/runtime amendment (2026-10-19): see [runbook](docs/lite/RUNBOOK_LITE.md)
and [runtime decision](docs/lite/RUNTIME_DECISION.md). Lite uses a separate `env/lite`
lock, 8/8/16 checkpoints and three pinned execution profiles. The sections below
describe the original v1 workflow; Lite qualification and collection status are
reported separately and do not change the v1 scientific contract.

Budgeted control of **CONTINUE / VERIFY / PRUNE** on top of Progressive Seed Pruning (PSP) for SD1.5 with
deterministic DDIM. The scientific sources of truth are at the repo root:

- `01_Adaptive_PSP_Research_Spec_v1_1_1.md`: formulation, scientific config, baselines, statistics
- `02_Adaptive_PSP_Implementation_Guide_v1_1_1.md`: interfaces, runtime contracts, acceptance

**Status (branch `bootstrap/p0`):** the P0 code is in place. The CPU/synthetic checks have run. The GPU code
paths have run only on a deterministic CPU *fake backend*, so nothing in this repo is yet verified on CUDA with
the real SD1.5 or ImageReward weights. What has and hasn't run is listed in `docs/P0_CPU_ACCEPTANCE.md`. No
PREREG is signed, so no bank exists.

## Layout

```
apsp/contracts   hashing (5-hash provenance), identity/RowMap/seed scheme, ledger (single fee source),
                 EpisodeState/PolicyView, privileged accessor, bank schema + validate_bank, PREREG guards
apsp/sim         G-842-v1 / G-A1-v1 grammars, L^m(q) DP + budget mask, shared closed-loop & schedule runners,
                 replay engine, obs builder (38/42), caps, 105-pair tables, oracles O1/O2/O3
apsp/policies    fixed (B0/B1/B2/B3), threshold B6, random B9, REINFORCE M1
apsp/stats       cluster bootstrap, gate rules, power, variance components, frontier/operating point
apsp/reports     anchor (PREREG-B proposal only), headroom, replay_vs_online (G1), ctrl_costs (M2c)
apsp/data        DiffusionDB-metadata prompt pool, GenEval text (dedup only), synthetic bank fixtures
apsp/gpu         env_check, model/provenance, hooks, validity, decode, scoring (ir_pil_v1), engine,
                 profile (M0/M1/M2/M3/M6), collect_bank (resume + EPH checks), online, reference_psp,
                 fake_backend (CPU test fixtures only)
third_party/psp  PSP @ 590f59f58384169c719e431dc01d14006fb0fc0c (read-only submodule)
configs/         pilot.yaml (spec values), machines.yaml
env/             requirements-gpu-linux.txt + setup_gpu_linux.sh (Linux GPU env)
docs/            RUNBOOK, GPU_FIRST_RUN, P0_CPU_ACCEPTANCE, DECISIONS_LOG, QUESTIONS, PREREG
```

## Install (CPU, Windows or Linux, Python >= 3.10)

```bash
git -c core.longpaths=true clone --recurse-submodules -b bootstrap/p0 https://github.com/NguyenNgocMinh30012005/DMRL.git
cd DMRL
git -C third_party/psp config core.longpaths true   # Windows only: PSP has paths > 260 chars
python -m venv .venv            # or: uv venv --python 3.11 .venv
# Windows: .venv\Scripts\activate   Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU torch for the RL controller
pip install -e ".[dev,rl]"
```

## Quickstart (CPU, synthetic, about a minute)

```bash
python -m pytest tests -q                      # contract, DP, replay, leakage, stats, fake-backend engine tests
python -m apsp.quickstart --out data/quickstart  # SYNTHETIC end-to-end: bank -> caps -> B2/B3/B6/B9/O1/O3 -> REINFORCE
```

Every quickstart number is **synthetic**. It checks code paths only and says nothing about image quality or speed.

Other CPU entry points:

```bash
python -m apsp.data.synthetic_bank --out data/banks/SYN --prompts 60 --costs-out data/costs/synthetic_costs.json
python -m apsp.contracts.schema --bank data/banks/SYN --validate
python -m apsp.reports.anchor --bank data/banks/SYN --costs data/costs/synthetic_costs.json --proposal-only --fit-split syn_train_fit --out data/quickstart/anchor_SYN.json
python -m apsp.reports.headroom --bank data/banks/SYN --costs data/costs/synthetic_costs.json --fit-split syn_train_fit --split syn_train_dev
python -m apsp.policies.reinforce --synthetic-smoke --iterations 40
python -m apsp.reports.ctrl_costs --synthetic --policy random
```

## GPU (Linux)

See `docs/GPU_FIRST_RUN.md` for the commands in order: clone the exact commit, build the env, check CUDA, run
the smoke, profiling and parity. GPU commands refuse to run without CUDA. There is no silent CPU fallback.

## Rules this repo enforces

- Single fee source (`apsp/contracts/ledger.py`); every runner charges through `EpisodeState.charge`, and the
  DP mask guarantees completion within the cap (no free rescue).
- Policies only see `PolicyView`-derived observations: no future, unbought or pruned scores, and no eval data.
- PREREG-A/B/C are signed by research only. Guards refuse research splits until the matching marker exists in
  `docs/PREREG_SIGNED/`.
- Weights, banks, images, latents and credentials stay out of Git (`.gitignore`).
