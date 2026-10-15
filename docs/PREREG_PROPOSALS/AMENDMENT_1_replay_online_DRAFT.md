# Amendment 1 (DRAFT, revision 2): native batching, replay as a simulator, online confirmation

**Status: DRAFT by a coding agent, revised after research decisions of 2026-10-15. Not signed, not frozen, not part
of PREREG-A/B/C.** Signing follows `docs/PREREG.md`. Nothing here changes split, reward, grammar, caps, gate
thresholds or τ_prac.

**Research decisions recorded (2026-10-15):**
- keep native batching 8→4→2;
- the backend A/B is dropped and `--numeric-variant` is not implemented;
- replay serves training and approximate screening; confirmatory conclusions rest on online runs;
- no PREREG is signed and no pilot bank is collected yet.

## 1. Evidence

From `docs/BATCH_DRIFT_AUDIT.md` §11 and `docs/DIAG1_RUN_REPORT.md`, diagnostic prompts only.

- **Fixed batch size: bitwise identical results**
  - across two processes;
  - across two machines and driver versions (bank k=8, same EPH);
  - across composition and order within a batch.
- **Changed batch size:** UNet outputs differ from the first forward (fp16 ulp level); after 64 steps, final latents
  differ by 1.4–2.4.
  - VAE: ≤ 2 u8 levels.
  - ImageReward: ~2e-6.
- **Online fixed schedules vs replay** (36 episodes):
  - winners 36/36 equal; first keep sets 36/36; second keep sets 32/36;
  - |Y_online − Y_replay| max 0.049, mean 0.0036;
  - paired-effect error vs (4,8) max 0.058.

## 2. Main branch

1. **Production and online engine: native batching** (batch = number of alive candidates), EPH `c1351a99…` for this
   configuration, plus a runtime fingerprint (`apsp.runtime_fingerprint/v1`).
2. **Replay is a simulator** built from the k=8 bank. It serves controller training, screening and the simulator
   oracles/diagnostics (O1-replay, O2, O3).
3. **Confirmatory statements are online** with the production engine on the machine class fixed in PREREG-A:
   G3/G3b/G4, val/test endpoints, and every G2 claim that replay cannot carry (§3.4–3.5).

## 3. Text changes proposed to the Research Spec

### 3.1 Mệnh đề 2: all assumptions explicit

> Replay of a pruning trace equals online execution if **all** of the following hold for every surviving candidate:
>
> (a) **Candidate independence.** Each candidate's update, preview score and delivered score depend only on that
> candidate's own latent, the prompt, the timestep and the frozen model (no cross-row interaction; CFG pairs the
> uncond/cond rows of the same candidate).
>
> (b) **Identical inputs.** Initial latent, prompt embeddings, timesteps, weights and dtypes.
>
> (c) **Identical numerical execution** of every operation that touches the candidate: the same kernels and
> algorithm selection, reduction order, precision and math flags, library versions and hardware/driver behaviour.
>
> (d) (a)–(c) also hold for VERIFY decoding/scoring and for the terminal decode/score.

Add:

> **Equal batch shape is not a substitute for (a)–(c).** It removes one known cause of differing kernels, but (c)
> still requires the same runtime (recorded by the runtime fingerprint), and (a) is a property of the model and
> pipeline that must hold independently.
>
> In this configuration (native batching, CUDA fp16), (a) holds by construction of the UNet/CFG path. (c) holds
> empirically at equal batch size (diag1: bitwise identity across processes and two machines). (c) fails once the
> batch size changes, so replay is exact only up to the first prune and is an approximation afterwards. The
> approximation error is measured (G1), not assumed.

Hệ quả 2a (105 schedules, at most 8 delivered images per bank) holds **inside the simulator**. Online, schedule s
delivers a different, nearby sample.

### 3.2 Mệnh đề 3: O1-replay is a simulator oracle only

Rename the bank quantity **O1-replay**:

O1ʳᵉᵖ_i(B) = max over s ∈ 𝒮 with C₀(s) ≤ B of Yʳᵉᵖ_i(W_i(s)).

- It is the hindsight optimum **of the simulator**. The proof bounds simulator (replayed) policy traces in the same
  grammar and paid ranking, and nothing else.
- It is **not** an upper bound on online policy quality.
- It is **not** a policy that can be deployed: it uses each instance's outcomes in hindsight.
- Any empirical replay-vs-online gap may be reported as a description, never as a bound.

### 3.3 G1: paired effect discrepancy

The statistic and bands of §10.2 are unchanged:

Δ_disc = mean(Y_m − Y_ref)_online − mean(Y_m − Y_ref)_replay

with CI inside ±0.25 τ_prac meaning replay may serve gates, and inside ±0.5 τ_prac meaning train only.

- Y_ref is PSP-default (4,8) on the same (prompt, bank). The online runs use the production engine, paired by
  instance.
- Winner, keep-set and effect-sign agreement are diagnostics only and never replace G1.
- **CI too wide to decide: G1 = INC** (not established, not parity). The train-dev budget may be extended within
  PREREG-B limits.
- The train-dev instances, the method list and the number of online episodes are fixed in PREREG-B before G1 is
  computed.

### 3.4 Simulator discrepancy vs engine error

A replay-online difference has two possible sources. They are reported separately and lead to different actions.

| | Simulator discrepancy (expected) | Engine error (a bug) |
|---|---|---|
| Cause | (c) fails after a batch-size change: numeric drift amplified along the trajectory | identity/row mapping errors, wrong costs or ledger, non-determinism at fixed batch, a bank that does not match the engine, a changed runtime fingerprint |
| Checks that separate them | the engine checks below pass | any check below fails |
| Action | handle by online confirmation (this amendment); do not "fix" by retuning | **STOP-ENG**: fix the engine, re-collect what the fix invalidates |

**Engine checks, run on train-dev before G1:**
- A/A repeatability at fixed batch (bitwise);
- bank vs online bitwise equality up to the first prune (k=8);
- PRUNE identity checks;
- ledger spend equal to the schedule cost;
- same EPH and runtime fingerprint for bank and online runs.

### 3.5 When G1 does not establish equivalence: online evidence

**G1 within ±0.5 τ_prac but not ±0.25 τ_prac, or INC.** Replay may train; every gate uses online evidence.

- **G2-quality (headroom).**
  - For each train-dev instance and cap, take the simulator-optimal schedule s*_i (argmax of Yʳᵉᵖ_i(s) under the
    cap). **Check its budget:** C₀(s*_i) ≤ B under the frozen ledger.
  - Run s*_i online and record the actual ledger spend. A run whose spend exceeds the cap is infeasible and counted
    as a failure under the frozen failure rule.
  - Y_online,i(s*_i) is a **hindsight-feasible solution**: a lower bound on the online hindsight optimum for that
    instance. It is **not** a policy that can be deployed and must not be presented as one.
  - Headroom is claimed only if mean(Y_online(s*) − Y_online(B3)) clears the prespecified threshold with a paired
    CI. Otherwise G2-quality is INC (s* may be suboptimal online), not STOP. O1-replay − B3 is reported as a
    simulator diagnostic only.
- **G2-cost.** The primal witness (an assignment of schedules reaching the mean-quality target with the stated
  saving) must be **confirmed online**: every schedule of the witness is run online on its instances, with budget
  checks as above. A replay-only witness is INC.
- **G3/G3b/G4:** online, as already required.

**G1 outside ±0.5 τ_prac.**
- If the engine checks of §3.4 fail: STOP-ENG.
- If they pass (pure simulator discrepancy): replay is not used even for training decisions that feed gates.
- No headroom or quality conclusion is drawn from replay in either case.

**In every case,** each reported number states whether it is a simulator (replay) result or an online result.

## 4. Engineering consequences (no scientific change)

- **PREREG-B budget** must cover:
  - the online runs of G1;
  - the engine checks (§3.4);
  - if needed, the online runs of s* (one per train-dev instance per cap, plus B3);
  - the online confirmation of the G2-cost witness.
- **Runtime fingerprint.** Recorded for every bank shard and online run (`02b_fingerprint`; harness and calibration
  reports carry it). PREREG-A fixes the GPU model, driver range and package versions. A fingerprint change between
  bank and online runs is an engine-check failure until research rules otherwise.
- **Collector.** Stays at k=8 (native), so bank and online coincide up to the first prune.

## 5. Decisions still needed from research

1. Approve the wording of §3.1–3.5. In particular: O1-replay as a simulator oracle; s* online as a
   hindsight-feasible witness with budget checks; INC for a wide G1 CI; online confirmation of the G2-cost witness.
2. Approve the engine-check list that separates simulator discrepancy from engine error (§3.4).
3. Online episode budget for G1, the engine checks and G2 in PREREG-B, and the train-dev subset used.
