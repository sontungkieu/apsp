# Calibration / profiling session plan (prepared; not run)

Nothing here has run on a GPU. All times and costs are **estimates** from smoke1's provisional smoke profile on an
RTX 3090 (`DIAG1_RUN_REPORT.md` §5). They are not measurements, and the cost table produced by this session stays
`provisional: true` until research designates the measurement machine (Q1).

## 1. Order

1. **Watchdog live check** (§2). It must pass before any long session. No model is loaded.
2. **Calibration session** (§3), with the watchdog armed. It is ordered by value; every step resumes after
   interruption (`.done` markers, per-instance records, per-k M2 parts).
3. Results go to research: M3 decision (preview dtype), VAE chunk proposal, cost table (provisional), M8 peaks, M9
   bank-collection time and pilot GPU-hour forecast, collector equivalence, M2b residual, full-pool duplicate check.

## 2. Watchdog live check (real provider; external confirmation required)

On the instance (either the kept 56070930 disk or a new one the user provides), after login:

```bash
cd /workspace/DMRL                              # code transferred by bundle as before
bash scripts/watchdog_live_check.sh
```

What it does:
1. Probes the API read-only (`vast_watchdog.py --check-status`, own instance, key from the environment). If the
   probe fails, it aborts: the watchdog cannot be trusted on this provider.
2. Arms the watchdog detached, with a deadline 6 min ahead (save margin 60 s).
3. The operator logs out.

| Level | Evidence | Who |
|---|---|---|
| stop requested | `state.json` `stop_requested_at` (read after the next Start) | assistant |
| API accepted | `state.json` `api_accepted: true`, `stop_attempts[].http_status` | assistant |
| **instance stopped** | Vast console shows *Stopped* at about deadline − 60 s | **user (console)** |

**Pass:** the console shows *Stopped* without anyone having issued a manual stop, and `state.json` shows
`api_accepted`. **If anything is missing:** no long session is started; the rental is protected only by manual stops.

Estimated cost: about 10 min of instance time, ≈ $0.06.

## 3. Calibration session

**Transfer before starting** (outside git, verified by sha256 from `pool_manifest.json`):
- `data/prompts/prompt_pool.parquet`;
- `data/prompts/val_ext_sealed.parquet`;
- `data/prompts/work/survivors_<key>.parquet`, renamed to `survivors.parquet`.

They go to `$APSP_STORE/prompts/`.

```bash
cd /workspace/DMRL && git fetch /workspace/xfer/dmrl_<sha>.bundle xfer && git checkout <sha> && git status --porcelain
export APSP_STORE=/workspace/apsp RUN_ID=calib1 MACHINE_ID=vast_<id>_rtx3090 POOL_DIR=/workspace/apsp/prompts
START=$(date +%s); DEADLINE=$(( START + <approved_minutes>*60 - 300 ))
bash scripts/arm_watchdog.sh "$CONTAINER_ID" "$DEADLINE" "$APSP_STORE/watchdog" \
  --finished-marker "$APSP_STORE/$RUN_ID.finished" --idle-grace-s 900 \
  --save-cmd "pkill -TERM -f '[s]cripts/gpu_smoke.sh'; sleep 20; pkill -KILL -f '[s]cripts/gpu_smoke.sh'" \
  --save-cmd "tar --exclude='*.lock' --exclude='banks' -czf $APSP_STORE/${RUN_ID}_artifacts.tgz -C $APSP_STORE/runs $RUN_ID"
HARD_DEADLINE_EPOCH=$(( DEADLINE - 600 )) MAX_MINUTES=<approved_minutes - 20> bash scripts/gpu_smoke.sh calib
touch "$APSP_STORE/$RUN_ID.finished"
```

If the deadline interrupts the session, the same command resumes it in the next approved session: done steps are
skipped, finished calibration instances and M2 k-parts are reused.

| Step | Content | Trajectory reuse | Estimated time |
|---|---|---|---|
| 00–03, 02b | setup, tests, env, runtime fingerprint, M1 check | — | 2 min (kept disk) to 6 min (fresh) |
| 30_pool_full_check | embed every filtered survivor; full-pool components of cosine > 0.95; selected prompts crossing splits | — | 20–30 min |
| 31_calib_m3m9 | 50 calibration prompts × bank 0 through the production collector. At each checkpoint the same x0 is also decoded with the other preview precision (M3). The 10 equivalence prompts keep x0@4/8/16, final latents, initial latents and embeddings. The 20 subset prompts × bank 1 run the plain collector (clean M9) | each trajectory computed once; M3, M9 and equivalence share it | ≈ 2.2 h (50 × ~122 s + 20 × ~82 s) |
| 32_calib_summary | M3 rates per j and per dtype, delivered failures, escalation flag, M9 seconds/prompt-bank and pilot GPU hours, collector peak VRAM | reads records | <1 min |
| 33_calib_equivalence | reference PSP pipeline A (our embeddings) and B (own encoding) vs stored tensors: exact-hash flags, max abs, x0/score at j = 4, 8, 16 | our side comes from step 31 (no second UNet pass) | ≈ 18 min |
| 34_calib_chunk | VAE decode chunk {1,2,4,8,16} for preview and delivered at k = 8 and 16: mean/p95 time and peak VRAM (Q7) | stored final latents | 5 min |
| 35_m2 (k01…k16) | M2 generator primitives k = 1..16, warm-up 10, n = 50: den, ver, fin, feat/lfeat, fixed and all 21 prune pairs (main, B5, B4b); peak VRAM per k (M8) | — | ~2 h |
| 36_m2_merge | merge, key coverage (den/ver/fin 1..16, prune pairs, feat/lfeat), CV flags; `provisional` with reasons | — | <1 min |
| 37_m2b | B2 (4,8) online on the 20 subset prompts: wall clock vs ledger prediction, residual, peak VRAM | — | ~8 min |

- **Total runner estimate (corrected 2026-10-15):** ≈ 5 h. As one session: watchdog deadline 5.5 h, about
  **$2.0 compute + ≤ $0.25 bandwidth**.
- **As two sessions:** A = steps 00–34 (~3 h); B = 35–37 (~2 h).
- **Not included:** idle time (bounded by the watchdog: 15 min idle grace) and storage while stopped.

**Expected outputs**, all labelled provisional or engineering:
- **Preview dtype.** M3 failure rates for fp16 and fp32 on the same trajectories. If either exceeds 1e-3 (any
  level) or delivered failures exceed 1e-3, the result is escalated to research (no automatic change).
- **VAE chunk.** Measured time and memory per chunk; the selection rule (Guide CC-4) is applied in the PREREG-A
  proposal, not here.
- **Cost table.** den/ver/fin k = 1..16, all prune pairs for G-842, B5 and B4b; `provisional: true` with reasons.
- **Peak memory (M8).** Collector, M2 per k, online B2.
- **Bank collection time (M9)** and pilot GPU hours for collection: s/prompt-bank × 2 banks × 540 prompts (small
  pilot, val_ext at its maximum). smoke1 **measured 81.9 s per prompt-bank** (an earlier ~46 s figure read den
  as per 4 forwards and was wrong). That gives ≈ 24.6 GPU h for all 540 prompts, or ≈ 10.9 h for train_fit +
  train_dev only (`PILOT_RUN_PLAN.md` §4).
- **Collector equivalence** report and **full-pool duplicate check** report.

## 4. Not in this session

Pilot bank collection, research splits, val/test, RL training, GenEval images, PREREG signing, kernel debugging,
backend A/B.
