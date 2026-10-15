# Amendment 1 (DRAFT): native batching, replay as an approximation, online confirmation

**Status: DRAFT proposal by a coding agent. Not signed, not frozen, not part of PREREG-A/B/C.** Research decides
whether to adopt, change or reject it; signing follows `docs/PREREG.md`. Nothing here changes split, reward,
grammar, caps, gate thresholds or τ_prac.

## 1. Evidence

From `docs/BATCH_DRIFT_AUDIT.md` §11 and `docs/DIAG1_RUN_REPORT.md`; diagnostic prompts only.

- **Fixed batch size:** the engine is deterministic.
  - Two processes: bitwise identical.
  - Across two machines and drivers: the k=8 bank is bitwise identical.
  - Composition or order within a batch of 8: bitwise identical.
- **Changed batch size:** numbers change from the first UNet forward (fp16 ulp level); 64 steps amplify that to
  final-latent differences of 1.4–2.4.
  - VAE: ≤ 2 u8 levels.
  - ImageReward: ~2e-6.
- **Online fixed schedules vs replay on the same instance** (36 episodes):
  - winners 36/36 equal; first keep sets 36/36 (same batch, k=8); second keep sets 32/36;
  - |Y_online − Y_replay| max 0.049, mean 0.0036;
  - paired-effect error vs (4,8) max 0.058.
- Under native batching, PSP-style online pruning changes the batch size after every prune. Online trajectories
  therefore equal the bank only up to the first prune.

## 2. Proposed main branch

1. **Production and online engine keep native batching** (batch = number of alive candidates). EPH stays
   `c1351a99…` for this configuration; a runtime fingerprint is recorded next to it. No batch-1, padding or FP32
   engine is adopted. Those options stay in `BATCH_DRIFT_AUDIT.md` §9 with their re-measurement lists.
2. **Replay (the k=8 bank) is an approximation**, used for:
   - training controllers;
   - screening and ranking candidate methods on train-fit/train-dev;
   - computing diagnostics (O1-replay, O2, O3).
3. **Every confirmatory statement is evaluated online** with the production engine, on the same machine class
   (GPU model fixed in PREREG-A): gates G3/G3b/G4, val/test endpoints, and any G2 claim that replay cannot carry
   (§4).

## 3. Text changes proposed to the Research Spec

### 3.1 Mệnh đề 2 (replay equivalence): explicit assumptions

Replace the assumption list with:

> Replay of a pruning trace equals online execution **if** for every candidate (a) the initial latent, timesteps,
> model, weights and dtypes are identical; (b) every kernel call that touches the candidate produces the same
> per-row result regardless of the other rows in the call: the same batch size and shape (or kernels proven
> batch-invariant), the same algorithm selection and the same reduction order; (c) VERIFY scoring and terminal
> decode satisfy (b) as well.

Add the consequence:

> With native batching on CUDA, (b) holds only while the batch size equals the bank's (k=8). Replay is exact up to
> the first prune and approximate afterwards. Its error is measured (G1), not assumed.

The algebraic content of Hệ quả 2a (105 schedules, at most 8 delivered images per bank) holds **within replay**.
Online, the delivered image of schedule s is a different sample, close to the bank image but not equal to it.

### 3.2 Mệnh đề 3 (O1): no online upper-bound claim

Rename the bank quantity **O1-replay**:

O1ʳᵉᵖ_i(B) = max over s ∈ 𝒮 with C₀(s) ≤ B of Yʳᵉᵖ_i(W_i(s)).

- Keep the proof, scoped to replay: it bounds every **replayed** policy trace in the same grammar and paid ranking.
- Add: O1-replay is **not** an upper bound on online policy quality. Online, Y_i(s) differs from Yʳᵉᵖ_i(s) (diag1:
  up to 0.049 per episode), so Y_policy,online ≤ O1ʳᵉᵖ is not guaranteed.
- An empirical gap (for example a high quantile of |Y_online(s) − Y_replay(s)| on train-dev) may be reported, but
  it is not a bound.
- Wording such as "O1 is the ceiling of online performance" is not used.

### 3.3 G1: paired effect discrepancy only

Keep §10.2's statistic and bands unchanged:

Δ_disc = mean(Y_m − Y_ref)_online − mean(Y_m − Y_ref)_replay

with CI inside ±0.25 τ_prac meaning replay may serve gates, and inside ±0.5 τ_prac meaning train only. Add:

- Y_ref is the PSP-default schedule (cuts j=4, j=8) on the same (prompt, bank). Each method m and Y_ref are run
  online on the same instances with the production engine, and paired by instance.
- Winner agreement, keep-set agreement and effect-sign agreement are **reported as diagnostics and never used as
  G1 or as a substitute for it**.
  - Winner agreement can be perfect while Y differs (diag1: 36/36 winners, |ΔY| up to 0.049).
  - Sign agreement is undefined in effect when the replay effect is exactly 0.
- Train-dev instances, the method list for G1 and the number of online episodes are fixed in PREREG-B before G1 is
  computed. A wide CI is "not established", not parity (as §10.2 says).

### 3.4 If G1 does not pass: online evidence for headroom and quality

- **G1 within ±0.5 τ_prac but not ±0.25 τ_prac (replay trains, gates online).**
  - **G2-quality (headroom).** For each train-dev instance, the replay-optimal schedule
    s*_i = argmax Yʳᵉᵖ_i(s) under the cap is evaluated **online**.
  - Y_online,i(s*_i) is attained by a feasible schedule, so it is a lower bound on the online hindsight optimum.
    Headroom is claimed only if mean(Y_online(s*) − Y_online(B3)) clears the prespecified threshold, with a paired
    CI.
  - If it does not clear, G2-quality is **INC**, not STOP: s* may be suboptimal online.
  - O1-replay − B3 is reported as a diagnostic only.
  - **G2-cost.** The primal witness (schedule assignment reaching the mean target) must be evaluated online. A
    replay-only witness is INC.
  - G3/G3b/G4 run online, as already required.
- **G1 outside ±0.5 τ_prac (or not established within budget): STOP-ENG** (§10.2).
  - No headroom or quality conclusion is drawn from replay.
  - Options are an engine change (`BATCH_DRIFT_AUDIT.md` §9, with new EPH/BMH and re-measured primitives) or
    research re-scoping.
- **In every case,** any claim about quality or headroom in the paper states whether it is a replay diagnostic or
  an online result.

## 4. Engineering consequences (no scientific change)

- **Online episodes cost GPU time per evaluation.** The PREREG-B budget must include G1's online runs and, if needed,
  the online evaluation of s* per train-dev instance (one online run per instance per cap, plus the B3 run).
- **Runtime fingerprint v1** (`apsp/gpu/fingerprint.py`) is recorded for every bank shard and online run. PREREG-A
  fixes the GPU model, driver range and package versions. A fingerprint change between bank collection and online
  evaluation is reported and, for confirmatory runs, needs research approval.
- The bank collector stays at k=8 (native), so bank and online coincide up to the first prune.

## 5. Decisions needed from research

1. Adopt the main branch (native batching, replay approximate, online confirmation)?
2. Approve the wording changes to Mệnh đề 2/3 and the O1-replay naming.
3. Approve the G2-quality online witness rule in §3.4 and its INC (not STOP) outcome.
4. Online episode budget for G1 and G2 in PREREG-B, and the train-dev subset used.
5. Whether a short backend A/B (`docs/NEXT_STEPS_PRE_PILOT.md` §3) should run before PREREG-A, to check whether a
   batch-invariant UNet configuration is cheap enough to make replay exact instead.
