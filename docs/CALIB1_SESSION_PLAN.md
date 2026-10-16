# Session plan calib1 (P1 on 56261825, no-financial-cap mode)

**Machine (D48).** Vast **56261825**, RTX 4090 24 GB, API price $0.527/h. Fresh disk. 48 GB / 3090 timings are not
reused. **Financial mode (D47):** no budget gate and no price ceiling; spend recorded. The watchdog has no money
deadline and stops on completion / no eligible job / run_project death / stall.

**A. Live stop check** (control path, before the long workload): `bash scripts/watchdog_live_check.sh` arms the
no-financial-deadline mode around a 90 s dummy workload; stop request at about +2.5 min. The user confirms
*Stopped* in the console and starts the instance again; then read `/workspace/apsp/watchdog_live_check/state.json`.

**B. P1** (one command; jobs p1_env → p1_pool_check → p1_m3m9 → p1_chunk → p1_m2 → p1_equivalence → p1_m2b →
p1_forecast → p1_bench_calib → p1_report_calib; P2+ stay blocked by PREREG-A, so run_project then writes its finished
marker and the watchdog stops the instance after the idle grace):

```bash
cd /workspace/DMRL && git fetch /workspace/xfer/dmrl_<sha>.bundle && git checkout <sha>
# prompt files (sha256 in reports/prompt_pool/pool_manifest.json): /workspace/apsp/prompts/{prompt_pool,val_ext_sealed,survivors}.parquet
export APSP_STORE=/workspace/apsp
bash scripts/launch_project.sh
```

3090-based upper bounds for the job list (replaced by `p1_forecast`): ≈ 4.25 h + bench_calib ≈ 1.5 h.

### Previous 3090 plan (kept for the record)

## Order on the instance

**A. Watchdog live check** (no model):
1. Start the instance, log in, and transfer the bundle (code at the handoff SHA).
2. Run `bash scripts/watchdog_live_check.sh`. It probes the API read-only and arms the watchdog with a deadline 6 min
   ahead.
3. Log out. **The user confirms *Stopped* in the Vast console.**
4. The user starts the instance again. The SSH port may change.
5. Read `state.json` (stop_requested / api_accepted) and diag1's `stop_result.txt`.
6. If the check fails, stop here: no long workload.

**B. Calibration** (after A passes):

```bash
cd /workspace/DMRL && git fetch /workspace/xfer/dmrl_<sha>.bundle xfer && git checkout <sha> && git status --porcelain
# prompt files (from the laptop, sha256 in reports/prompt_pool/pool_manifest.json):
#   /workspace/apsp/prompts/{prompt_pool.parquet, val_ext_sealed.parquet, survivors.parquet}
export APSP_STORE=/workspace/apsp RUN_ID=calib1 MACHINE_ID=vast_${CONTAINER_ID}_rtx3090 POOL_DIR=/workspace/apsp/prompts
START=$(date +%s); BUDGET_MIN=<approved minutes>          # minimal plan 270, full plan 360
DEADLINE=$(( START + BUDGET_MIN*60 - 300 ))
bash scripts/arm_watchdog.sh "$CONTAINER_ID" "$DEADLINE" "$APSP_STORE/watchdog" \
  --finished-marker "$APSP_STORE/calib1.finished" --idle-grace-s 900 \
  --save-cmd "pkill -TERM -f '[s]cripts/gpu_smoke.sh'; sleep 20; pkill -KILL -f '[s]cripts/gpu_smoke.sh'" \
  --save-cmd "tar --exclude='*.lock' -czf $APSP_STORE/calib1_artifacts.tgz -C $APSP_STORE/runs calib1"
export HARD_DEADLINE_EPOCH=$(( DEADLINE - 600 )) MAX_MINUTES=$(( BUDGET_MIN - 30 ))
# minimal plan (M2 on the k the pilot methods use):
M2_KS="1 2 4 8 16" CHUNK_N=50 bash scripts/gpu_smoke.sh 00_setup 01_pytest 02_env_check 02b_fingerprint 03_model \
  30_pool_full_check 31_calib_m3m9 32_calib_summary 34_calib_chunk 35_m2 36_m2_merge 37_m2b
# full plan instead: bash scripts/gpu_smoke.sh calib            (adds equivalence and M2 for every k = 1..16)
touch "$APSP_STORE/calib1.finished"
```

**Artifacts.**
- On the instance: `/workspace/apsp/runs/calib1/` and `/workspace/apsp/calib1_artifacts.tgz`. The calibration bank
  stays on the kept disk.
- On the laptop: `D:\Workspace\DMRL_artifacts\calib1\` (download, sha256, test extraction).
- The watchdog stops the instance 15 min after `calib1.finished` at the latest, and at the deadline in any case.

## Estimates per task (RTX 3090)

Measured on smoke1 or diag1, one sample each:
- collector 81.9 s per prompt-bank;
- PSP online 28.6 s;
- setup + model check on a fresh disk 1.8 min.

Everything else is assumed from smoke1's provisional M2 smoke profile (den per forward: k=1 0.116 s, 2 0.209,
4 0.395, 8 0.778, 16 1.524; ver_8 1.95 s; fin_8 1.97 s).

| Task | Prompts × banks × runs | Time | Basis |
|---|---|---|---|
| A. Watchdog check + restart | — | ~15 min instance time | assumed |
| 00–03 setup, tests, fingerprint, M1 | — | 2–5 min | measured (diag1: 2 min) |
| 30 full-pool duplicate check | 231,516 survivor prompts, CLIP fp32 | ~10 min | assumed |
| 31 M3 + M9 (+ equivalence tensors) | 50 calibration × bank 0, collector + fp32 preview of the same x0 (M3, 6,400 previews per precision); 20 subset × bank 1, plain collector (M9) | 50 × ~110–120 s + 20 × 82 s ≈ **2.0–2.2 h** | 82 s measured; +30–40 s fp32 preview assumed |
| 32 summary | — | <1 min | — |
| 33 collector equivalence | 10 calibration prompts × 2 reference runs (A, B) | ~20 min | assumed |
| 34 VAE chunk | chunks {1,2,4,8,16} × preview/delivered × k ∈ {8,16}, 60 reps each | ~20 min | assumed |
| 35 M2 pilot subset | k ∈ {1,2,4,8,16}, warm-up 10 + n 50 per primitive | ~30 min | assumed |
| 35 M2 remaining k | k ∈ {3,5,6,7,9…15} (needed for B4b/B1-any-N; not for this pilot's methods) | ~1.5 h | assumed |
| 37 M2b | 20 subset prompts × B2 online | ~10 min | 28.6 s per episode measured |

**Minimum to fix M2, M3, M9, the chunk and the pilot budget:** steps 00–03, 30, 31, 32, 34, 35 with
`M2_KS="1 2 4 8 16"`, 36, 37. That is **≈ 3.3–3.6 h runner**. With A and overheads, a **270 min (4.5 h)** session cap
at $0.359/h is **≈ $1.62**, plus ≤ $0.30 bandwidth on a fresh disk → **budget ≤ $2.0**.

**Full plan** (adds equivalence and all k): ≈ 5.2 h runner, 360 min cap → **≈ $2.15 + bandwidth ≤ $2.5**.
Equivalence is a PREREG-A acceptance item, so it is needed before the bank but not for the speed measurements.

## What comes back after the session

- measured M3 failure rates for fp16 and fp32 (same trajectories), with the proposed preview dtype;
- chunk time and memory, with the proposed chunk (Guide CC-4 rule);
- merged cost table (`provisional: true` until Q1, with the missing-k list if the minimal plan ran);
- peak VRAM (collector, per k, online B2);
- M9 s/prompt-bank and an updated pilot budget;
- M2b residual;
- full-pool duplicate check for Q16.

No pilot step runs in this session.
