# Backup inventory per instance (for the keep/destroy decision)

Status 2026-10-15. **Nothing has been destroyed; destroying is the user's decision.** Instance contents are as last
observed during the sessions; they were not re-listed, because both instances are believed to be stopped and the
assistant has no API access from the laptop. The local copies were re-hashed today.

## Instance 56044352 (smoke1; RTX 3090, $0.0556/h storage while stopped, ~$1.33/day)

State: stop request sent 2026-10-15 ~05:35Z; **stopped state never confirmed by API or console.**

| On the instance (last observed) | Local copy | Unique to the instance? | Recreate cost |
|---|---|---|---|
| `/workspace/apsp/runs/smoke1/` (logs, p0, tsmoke incl. TSMOKE_BANK images) | `D:\Workspace\DMRL_artifacts\smoke1\smoke1_artifacts.tgz` (4,706,661 B, sha256 `1c5e4bd2…a7ca4`, 100 entries) and `smoke1_final.tgz` (4,709,110 B, `5ff2267e…ebce9`, 107 entries, includes evidence); extracted in `x\` | no | — |
| runner evidence (driver script, runner times, watchdog log, resource snapshots) | inside `smoke1_final.tgz` (`evidence/`) | no | — |
| `/workspace/DMRL` at `933e9ac`, PSP at `590f59f` | GitHub `bootstrap/p0` history; PSP bundle `diag1\xfer\psp_590f59f.bundle` | no | — |
| `/workspace/xfer/*.bundle` (933e9ac, PSP) | reproducible from git | no | — |
| `/workspace/apsp/venv-gpu` (Python 3.10.20, torch 2.4.0+cu121) | — | re-creatable | ~1–2 min (`00_setup`) |
| `/workspace/apsp/hf` (SD1.5 @451f4fe, ImageReward @5736be03, BERT tokenizer @86b5e09) | — | re-downloadable; hashes in `p0/M1_model.json` | ~4 min, est. 5–8 GB bandwidth (on 56070930's price, $0.026/GB ≈ $0.13–0.21) |

**Everything needed is on the laptop.** Keeping this disk only saves the venv/weights setup (a few minutes and
<$0.25 of bandwidth). That saving is smaller than one day of storage.

## Instance 56070930 (diag1; RTX 3090, $0.0926/h storage while stopped, ~$2.22/day)

State: stop request sent 2026-10-15 ~14:36Z; **stopped state never confirmed by API or console.**

| On the instance (last observed) | Local copy | Unique to the instance? | Recreate cost |
|---|---|---|---|
| `/workspace/apsp/runs/diag1/` (diag JSON, A/A tensors, rm inputs, logs) | `D:\Workspace\DMRL_artifacts\diag1\download\diag1_artifacts.tgz` (85,220,826 B, `f57d0646…885a`, 73 entries); extracted in `x\` | no | — |
| `diag1_evidence.tgz` (runner times, driver log, git provenance, nvidia-smi, driver script) | `download\diag1_evidence.tgz` (2,114 B, `b1e525fe…38b2`) | no | — |
| `/workspace/apsp/stop_result.txt`: output of the `vastai stop` call at 14:36Z | **not downloaded** (written after the evidence tarball) | **yes**: the only record of the provider's answer to the diag1 stop | requires starting the instance to read it |
| `/workspace/DMRL` at `1100a12` (+1 untracked file from setup), `/workspace/xfer/*` | git history; bundles in `diag1\xfer\` | no | — |
| `/workspace/apsp/venv-gpu` (Python 3.10.21), `/workspace/apsp/hf` (same pinned weights) | — | re-creatable / re-downloadable | as above |

## Decision options (cost per day while stopped)

| Option | Cost | Loses |
|---|---|---|
| Destroy both | $0 | `stop_result.txt` of diag1 (provider answer to a past stop; evidence only, nothing scientific) |
| Destroy 56044352; keep 56070930 for the watchdog live check and calibration | ~$2.22/day until used | nothing; saves ~5 min setup and <$0.25 bandwidth when the session starts |
| Keep both | ~$3.55/day | — |

- **Before destroying anything:** check each instance's real state in the console. If either is still running,
  compute charges apply ($0.27–0.32/h) in addition to storage.
- **Reading `stop_result.txt`** needs a Start: a few minutes of compute. It can be combined with the watchdog live
  check on 56070930 (`docs/CALIBRATION_PLAN.md` §2).
