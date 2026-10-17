# PREREG-A amendment 2: measurement machine 56272061 → 56353913 (2026-10-17)

Approval source: the project owner's instruction of 2026-10-17 in the Claude Code session
`c8081ba0-a4d8-4c0d-baa8-d73f41b08105`: run the whole agreed plan on instance 56353913, complete the machine-change
amendment by measuring provenance, cost primitives, M2b and equivalence, and fill the protocol fields by the agreed
rules. The agent records the approval. Nobody else's signature is implied.

Reason: instance 56272061 was deleted before the bank finished, and its files were lost except the laptop copies of
calib1. Every other PREREG-A field (`PREREG_A_LOCKED.md`, sha256 `25ef58a6…`) is unchanged: generator, grammar/caps,
seeds, scoring, prompt filter and split, preview_dtype fp16, vae_decode_chunk 1, train-only x0 latents (A16) and
Amendment 1 rev. 2 (A17).

| Field | Rule under this amendment |
|---|---|
| A1 machine | Vast **56353913**, RTX 4090 (48 GB variant, UUID GPU-c19b628e-19db-5b3c-cc29-1d2a60972e27), driver 615.71.09, same pinned env (torch 2.4.0+cu121), one GPU worker. VRAM gate passed on 2026-10-17 06:1xZ (third-party cuda_memtest 0 error blocks, own CUDA test 0 mismatches) |
| A11 cost table | re-measured on 56353913 (`runs/calibA/p0/m2/generator_costs.json`, k ∈ {1,2,3,4,8,12,16}); this value is used from the bank onward; provisional until controller costs are profiled (unchanged rule) |
| A12 equivalence | fresh two-sided run on 56353913 (`runs/calibA/calib/equivalence_fresh.json`); collection proceeds only if every delivered latent matches the pinned reference exactly (the job exits non-zero otherwise) |
| A13 M2b | re-measured on 56353913 against the new cost table |
| A14 EPH | measured in the gpu_smoke runtime env on 56353913 and frozen in `${APSP_STORE}/frozen_eph.json` by `scripts/freeze_eph.py`; the collector refuses any other EPH |
| A8 / A9 / A10 evidence | pool check, M3 and chunk measured on 56272061 (calib1) stay the evidence; they do not depend on this GPU's identity |
| Forecast | recomputed from the new M2/M2b; M9 throughput still from calib1 until the bank's own timings replace it |

Operations, not science: cloud backup is optional in this session (owner, 2026-10-17). Artifacts are pulled to the
laptop after each job, and the pull never gates the workload. Data checks, leakage guards and validity checks are
unchanged.
