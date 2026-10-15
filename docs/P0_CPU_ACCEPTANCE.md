# P0 acceptance: CPU / synthetic part (branch `bootstrap/p0`)

Last updated: 2026-10-15.

## Tested commit

- **Code candidate `19c1f3794621016a81713e3dd3c566face01de8c`**: full checks below.
- The handoff commit that follows it changes only `docs/` plus a one-line fix to the failure classifier in
  `scripts/gpu_smoke.sh` (`git diff --stat 19c1f37 -- . ':!docs'` shows exactly that file). The orchestration
  check was rerun after that change, and the handoff SHA itself is re-checked in the handoff message.
- Earlier handoff `16cf0fb` vs the clean-clone-tested `cb5bebd`: the code difference was `--rm-revision` plus
  `reward_revision: null` in `pilot.yaml`. Behaviour was unchanged while the value was null. The new key
  changes `scientific_config_hash`, but no bank or cost table existed, so nothing was invalidated. Both values
  are now pinned (D23).

## Environments and results

| Env | What | Result |
|---|---|---|
| Windows 11, Python 3.11.15, numpy 2.4.6, pandas 3.0.6, torch 2.14.1+cpu | clean clone from GitHub at `19c1f37` (`git -c core.longpaths=true clone --recurse-submodules`), `uv pip install -e ".[dev,rl]"`, `pytest tests -q`, `python -m apsp.quickstart` | submodule `590f59f5…`, clean; **118 passed**; quickstart bank valid, all dev episodes within cap (68 s) |
| WSL2 Ubuntu 24.04 (Linux 5.15), **Python 3.10.20, numpy 1.26.3, pandas 2.2.1, scipy 1.13.1, pyarrow 18.1.0, torch 2.4.0+cpu** (the GPU-env versions) | clone of the same SHA, editable install, `pytest tests -q`, quickstart | submodule clean; **118 passed**; quickstart bank valid, within cap (38 s) |
| WSL2 | `VENV=<python3.12 venv> bash env/setup_gpu_linux.sh` | exit 2, "expected 3.10 … not deleting it", venv kept |
| WSL2 | `bash scripts/gpu_smoke.sh 01_pytest 02_env_check` twice (CPU-only box) | 01 ok (log, summary row, `.done`); 02 failed with exit 3 (no CUDA) → `.failed`, runner stopped; rerun skipped 01 and refused to retry 02; memory evidence and tarball written |
| Windows | `uv pip compile env/requirements-gpu-linux.txt --python-version 3.10 --python-platform x86_64-manylinux_2_28 …cu121` | resolves; torch 2.4.0+cu121, triton 3.0.0, diffusers @ af28ae2d, transformers 4.38.2 (`env/resolved-linux-py310.txt`) |
| Windows GPU env (Python 3.10, torch 2.4.0+cu121, transformers 4.38.2, ImageReward @ 2ca71bac) | tokenizer pin check (no model weights loaded) | `bert-base-uncased` → `google-bert/bert-base-uncased` @ `86b5e093…`; pinned snapshot gives ids identical to BLIP's floating path; BLIP_Pretrain calls `init_tokenizer` by module-global name, so the swap applies and is restored |

Windows note: set `git config --global core.longpaths true`. The PSP submodule has paths longer than 260
characters, and checkout/status fail without it. Linux is unaffected.

## Four categories

| Category | Content |
|---|---|
| **Implemented** | everything in `README.md` → Layout, plus `apsp.gpu.technical_smoke`, `scripts/gpu_smoke.sh`, `scripts/check_m1.py`, ImageReward tokenizer pin |
| **Run on CPU / synthetic** | the suite above (incl. DP vs brute force, budget invariant, leakage sentinels, PREREG guards, technical split guards, replay e2e, REINFORCE smoke, fake-backend engine / collector resume and mismatch / online==replay / technical smoke steps), quickstart, setup-script guard, runner orchestration |
| **Run on a real GPU** | only on the 4 GB dev laptop on 2026-10-14: `env_check --require-cuda` and `profile --stage model` (M1). That report predates the tokenizer pin, so `check_m1.py` reports the two tokenizer checks as FAIL for it, as expected. The M0 run was stopped by the session for host RAM pressure and is not repeated on the laptop. |
| **Not verified** | denoising, decode, ImageReward scoring, collector, online and profiling with real weights; PSP reference equivalence; `env/setup_gpu_linux.sh` end to end (only resolution and the venv guard have run; the CUDA wheel install has not) |

## Limits

- Synthetic costs and rewards test code paths only. The `engineering_only` cost table of the technical
  smoke is the same synthetic table.
- The fake backend is row-independent by construction; online==replay is exact there and says nothing about CUDA.
- B4b/B5/B7/B8/B10 and ablations are backlog items (`docs/BACKLOG.md`). No PREREG is signed and no research
  bank exists.
