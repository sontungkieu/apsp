# Batch drift audit (after smoke1)

Last updated: 2026-10-15. ENGINEERING ONLY. Nothing here is a G1 result, and no threshold, split, reward, grammar
or cap was changed.

## Provenance

| Item | Value |
|---|---|
| Run analysed | smoke1, RTX 3090 (Vast.ai 56044352), commit **`933e9ac`** (PSP `590f59f`), EPH `c1351a99…` |
| Artifacts (laptop, read-only) | `smoke1_final.tgz` sha256 `5ff2267e…bebce9`; `smoke1_artifacts.tgz` sha256 `1c5e4bd2…a7c4` |
| Audit script | `scripts/m6_artifact_audit.py` → `reports/batch_drift/smoke1_audit.json` (input dir hashed before and after: unchanged) |
| Harness | `apsp/gpu/drift_harness.py`, `configs/drift_diagnostic.yaml`, runner preset `scripts/gpu_smoke.sh diag`, commit **`122d0c8`** |
| Runner fix | `73d3b25` (written after smoke1; smoke1 ran `933e9ac` unchanged) |

## 1. Script success is not batch-invariance acceptance

| Question | Answer |
|---|---|
| Did the M6 script (step 10) run? | Yes: exit 0, 619 s, report written. The checklist criterion was "report produced". |
| Is the pipeline batch-invariant? | **No.** Changing the batch size changes latents, preview scores, delivered scores, and sometimes the B2 winner. |
| Is there an acceptance criterion for batch drift? | No. Q10 deliberately defers any tolerance until per-candidate evidence exists. Nothing here proposes one. |

## 2. What the M6 comparison actually held fixed

Source: `apsp/gpu/profile.py::stage_batch_parity` at `933e9ac`. "By code" means guaranteed by construction but **not
recorded** in the report.

| Item | Same across batch sizes? | Recorded in M6 report? |
|---|---|---|
| CandidateId | By code: every plan slices `pool_rows(2)`; group rows map to tensor rows in order (no pruning in M6) | No |
| Initial latent bytes | By code: `derive_seed("m6-technical", bank, seed)` → CPU `torch.Generator`, float32 randn → fp16, per candidate | No (`init_latent_sha` only exists in banks) |
| Prompt embeddings | Expected: each group re-encodes the prompt at batch 1 and tiles it | No hash; not verified |
| Scheduler / timesteps | Expected: `set_timesteps(64)` per engine; the EPH pins the 64 timesteps | EPH only |
| CFG row mapping | By code: `emb = [uncond×k; cond×k]`, `x_in = cat([x, x])`, `eps.chunk(2)` | No |
| Seed integer | Same, but not sufficient on its own | — |

Empirical support from other artifacts: with the **same batch composition**, numbers are reproducible across
processes. The collector (k=8) and the online PSP run (k=8 until j=4) produced **bitwise-equal preview scores at
j=4 for all 8 candidates** (bank stores float32; equal as float32, the RM output dtype). M0 shows bitwise repeat at
k=8 within a process, and equivalence variant A is exact. This supports identical inputs and deterministic
kernels for a fixed batch. It does not prove that M6's per-group embeddings were identical; `drift_harness unet`
hashes them.

## 3. M6 table (smoke1)

Prompt "a photograph of an astronaut riding a horse", uid `m6-technical`, 16 candidates (two banks), reference =
batch 8 (two groups of 8).

| Batch | Groups | Latent x^64 max abs | Latent RMSE / rel. | Preview IR max abs (j=4,8,16) | Delivered IR max abs | Per-candidate rows | Ranking gaps | B2 winner flips |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 1.7068 | not measured | 0.2867 | 0.1787 | not measured | not measured | 0 / 2 |
| 2 | 8 | 2.4443 | not measured | 0.7866 | 0.1167 | not measured | not measured | **1 / 2** |
| 4 | 4 | 2.0454 | not measured | 0.3439 | 0.3663 | not measured | not measured | 0 / 2 |
| 8 (ref) | 2 | 0 (trivial: compared with itself) | — | 0 | 0 | — | — | — |
| 16 | 1 | 1.4354 | not measured | 0.2793 | 0.2667 | not measured | not measured | **1 / 2** |

Metric definitions:
- **Latent max abs:** the max over the 16 CandidateIds and all 4×64×64 elements of |x^64_b − x^64_ref| on the
  final DDIM latent. Values are fp16 cast to fp32, in scaled latent space, before /0.18215.
- **Preview IR max abs:** the max over j∈{4,8,16} and candidates. Decode and ImageReward ran on the same batch b,
  so the UNet, VAE and RM effects are **confounded**.
- **Delivered IR max abs:** the same as preview IR, on delivered images.
- **Flips:** B2 replayed on the batch-b score table vs on the batch-8 table, counted per bank (2 banks, one prompt).

Limits:
- Batch 8 is not a repeat test.
- Two banks of one prompt give no flip rate.
- M6 changes the batch **from the first forward**. Online PSP keeps k=8 until the first cut, so M6 overstates the
  drift that replay actually faces (§4).

## 4. Per-candidate evidence that does exist (technical smoke, same uid `1f64b1c6…`, bank 0)

| Candidate | Quantity | Online batch history | Online | Bank (k=8) | Diff |
|---|---|---|---|---|---|
| (0,0)…(0,7) | preview IR j=4 | k=8 from j=0 (same as bank) | — | — | **0 for all 8** |
| (0,0) | preview IR j=8 | k=8 → j=4, then k=4 | 1.82584 | 1.83003 | −0.00419 |
| (0,4) | preview IR j=8 | same | 1.34598 | 1.36252 | −0.01654 |
| (0,5) | preview IR j=8 | same | 1.58805 | 1.56829 | +0.01976 |
| (0,6) | preview IR j=8 | same | 1.69733 | 1.68004 | +0.01728 |
| (0,0) | delivered IR | k=8→4→2 | 1.89406 | 1.89245 | +0.00161 |
| (0,6) | delivered IR | k=8→4→2 | 1.86552 | 1.86584 | −0.00031 |
| (0,0) | delivered IR | k=1 for all 64 forwards (single step) | 1.88777 | 1.89245 | −0.00469 |
| (0,0) | delivered IR, online fixed:2,9 / random | schedule-dependent | 1.89265 / 1.89114 | 1.89245 | +0.00020 / −0.00131 |

B2 ranking gaps on the k=8 bank, compared with the drift above:

| Cut | Last kept vs first dropped | Gap | Largest online−bank |diff| at that stage | Decision |
|---|---|---|---|---|
| j=4, keep 4 | (0,6) −1.0844 vs (0,2) −1.1128 | **0.0284** | 0 (same batch) | same keep set |
| j=8, keep 2 | (0,6) 1.6800 vs (0,5) 1.5683 | 0.1118 | 0.0198 | same keep set |
| final, keep 1 | (0,0) 1.8925 vs (0,6) 1.8658 | 0.0266 | 0.0016 | same winner |

Pixels of candidate (0,0), uint8 RGB, against the k=8 bank image:

| Image | Max abs | RMSE | Pixels changed | Pixels changed by > 8 |
|---|---|---|---|---|
| k=1 for the whole trajectory (`single_s0.png`) | 228 | 7.24 | 77.8% | 4.46% |
| k=8 → 4 → 2 (`psp_winner.png`) | 12 | 0.44 | 26.6% | 0% |

## 5. What is established and what is not

**Established from artifacts:**
1. **Fixed batch, deterministic.** For a fixed batch composition, the pipeline is reproducible bit for bit across
   processes (§2).
2. **Batch size changes the numbers.** Changing the batch size changes latents (max abs up to 2.44), preview IR (up
   to 0.79) and delivered IR (up to 0.37). With the batch changed from forward 1, a B2 winner flipped in 2 of 8
   bank-comparisons (b=2 and b=16, one bank each).
3. **Earlier change, larger effect.** The later the batch changes, the smaller the effect: changing at j=4/j=8
   (the online B2 pattern) moved scores by ≤ 0.02, and the image by RMSE 0.44 px. Batch 1 from the start moved
   the image by RMSE 7.2 px.
4. **Online decisions held in this one case.** In the 3 online episodes, all decisions matched replay. The ranking
   gaps (≥ 0.027) were larger than the drift at the stage where the batch had changed. The j=4 gap (0.028),
   however, is the same order of magnitude as the j=8 drift (0.02). A cut placed after an earlier prune, with a
   similarly small gap, could flip.

**Not established:**
- Which component causes the drift (UNet vs VAE vs ImageReward).
- Per-candidate error distributions, and the error-growth curve over forwards.
- The flip rate or the size of the effect on Y and on method comparisons.
- Whether M6's embeddings were bitwise equal across groups.

Three episodes and two M6 flips can establish neither "G1 passes" nor "replay is unusable".

## 6. Suspected causes and the evidence each needs

| # | Suspect | Why plausible | Evidence missing | Harness part |
|---|---|---|---|---|
| 1 | UNet GEMM / conv kernel selection depends on the batch dimension (cuBLAS heuristics / split-K, cuDNN algorithm per shape) → different fp16 reduction order | `cudnn.deterministic=True` fixes run-to-run, not cross-shape; fp16 UNet | Per-candidate eps diff for **one forward on hashed identical inputs** (f=0,16,32,63) | `unet` (A1) |
| 2 | Chaotic amplification over 64 DDIM steps with CFG 7.5 | Late batch change → small effect; whole-trajectory change → large | First divergent forward per candidate; error-vs-checkpoint curves | `traj` (A2) |
| 3 | VAE decode depends on the batch (decode chunk = batch when `vae_decode_chunk` is null) | Same kernel-selection mechanism in fp16 | Same latent, chunks 1/2/4/8/16 and orders: tensor and uint8 diffs | `vae` (B) |
| 4 | ImageReward depends on the batch (fp32; `cudnn.allow_tf32=True` for the ViT patch conv; BLIP attention) | Scores are computed on the batch of survivors | Same PNG pixels, preprocessed-tensor hash, repeats, sizes/compositions/orders | `rm` (C) |
| 5 | A module left in train mode (dropout) | Ruled out by code (`eval()` + `requires_grad_(False)` on UNet/VAE/text encoder; RM `.eval()`), but not observed at runtime | Runtime flags of every submodule | `audit` |
| 6 | Identity / mapping bug | Ruled out for fixed batches (bitwise equality, PSP identity checks). For M6, ruled out by reading the code only | Hashed inputs and embeddings per plan | `unet`, `traj` |

`matmul.allow_tf32=False`, while `cudnn.allow_tf32=True` (EPH `numeric_runtime`). The attention backend is diffusers'
default processor, with flash and mem-efficient SDPA enabled; the `audit` part records the processor class.

## 7. Harness (prepared, CPU-tested, not yet run on a GPU)

`python -m apsp.gpu.drift_harness --diag-config configs/drift_diagnostic.yaml --part <audit|unet|traj|vae|rm|method|all> --out DIR`

- **Plans.** Batch sizes 1, 2, 4, 8, 16. Orders: `fwd` (pool order), `rev` (changes composition and in-group
  order) and `interleave` (mixes banks inside a group). The reference plan is `b8_fwd`, as in M6. Every result is
  keyed by CandidateId.
- **Same code as production.** The parts reuse the production engine, decode and scoring code unchanged (the EPH is
  unaffected). The only addition is a wrapper that reads `pred_original_sample` at non-checkpoint forwards in A1.
- **Inputs are hashed.** Initial latents, per-forward latent rows, embeddings (uncond/cond per group), decode-input
  latents (saved as `vae_input_latents.pt`), PNG pixels and ImageReward-preprocessed tensors are all hashed.
- **Metrics:**
  - tensors: bitwise flag, max abs, RMSE, relative L2, reference RMS;
  - pixels: max abs, RMSE, fraction of pixels changed and changed by > 8;
  - scores: per-candidate diffs;
  - B2: keep sets, winner, ranking gaps at j=4, j=8 and final, flips;
  - method: online vs replay winner, keep sets, Y, and the effect versus the (4,8) reference pair together with
    its sign agreement.
- **Prompts.**
  - Hand-written diagnostic prompts with a technical split label, checked against GenEval and the prompt pool.
  - The M6 reproduction uses the M6 prompt **and uid** (`m6-technical`), so the seeds are identical.
  - Research calibration prompts can replace the diagnostic prompts only after the pool is frozen (Q2/Q3) and
    research approves that use.
- **A/A repeatability (added before diag1).** Part `aa` runs the reference plan `b8_fwd` end to end in its own
  process and saves initial-latent, embedding, per-forward, final-latent, decoded-tensor and pixel hashes plus all
  scores (and the tensors). The runner runs it twice in separate processes (`20a`, `20b`), and `--compare-aa`
  (`20c`) compares them, refuses two records from the same process, and checks both EPHs against `EXPECT_EPH`
  (smoke1's measured production EPH `c1351a99…`). A plan compared with its own tensors is never used as
  repeatability evidence.
- **Method part, extra fields.** The schedule list is written to `method_plan.json` before the part runs. Each
  episode also records the online ranking gaps at its own cuts, and both winners' delivered score on the same k=8
  table (`bank_y_gap_between_winners`), so a winner change can be read against its quality cost. The k=8 tables and
  delivered pixel hashes are stored per instance.
- **Cross-session A/A.** `scripts/diag_crosscheck.py` compares the `method` k=8 collection of the astronaut prompt
  (bank 0) with smoke1's technical-smoke bank shard of the same (prompt, uid, bank): initial latents, the 8×16
  preview table, delivered scores and PNG pixels. It runs on the laptop after download.
- **Tests.** 13 fake-backend tests at `122d0c8` (18 after the A/A additions), part of 151 (156) passing. They cover: a row-independent fake gives bitwise zero diffs
  under every permutation (which validates the id mapping); injected batch-, chunk- and RM-dependence is detected
  and attributed to the right plans; the CLI; and the audit script leaves its input untouched.

## 8. Proposed GPU diagnostic (needs a machine and budget from you; nothing is rented)

Order of the runner preset `diag`: setup → pytest → env check → M1 check → 20 audit → **21 traj (reproduce the M6
case)** → 22 unet → 23 vae → 24 rm → 25 method (fixed pairs (4,8), (2,9), (1,2), (2,4), (6,12), (8,14) on 3
diagnostic prompts × 2 banks).

Commands. On the laptop, build bundles for the new commit as for smoke1; then on the box:
```bash
cd /workspace/DMRL && git fetch /workspace/xfer/dmrl_<sha>.bundle xfer && git checkout <sha> && git status --porcelain
export APSP_STORE=/workspace/apsp RUN_ID=diag1 MAX_MINUTES=75
export HARD_DEADLINE_EPOCH=$(( $(date +%s) + 85*60 ))
bash scripts/gpu_smoke.sh diag
```

Time estimate on an RTX 3090, based on smoke1 timings:

| Step | Estimate |
|---|---|
| Setup on a fresh box (wheels and weights) | ~8 min; ~3 min if instance 56044352 is restarted with its disk |
| audit | ~1.5 min |
| traj (7 plans; M6 averaged ~110 s per plan including scoring) | ~13 min |
| unet | ~3 min |
| vae | ~2 min |
| rm | ~2 min |
| method (6 instances × (82 s collection + 6 × 27 s online)) | ~25 min |
| Packaging and download | ~5 min |
| **Total** | **~60–70 min wall** |

Budget: at the verified $0.378/h, expect about **$0.40–0.45**. A 100-minute cap is about $0.63, plus storage while
stopped ($1.33/day for 200 GB).

Stop conditions:
- Steps 00–03 fail.
- A memory-class failure (do not repeat the same command).
- `HARD_DEADLINE_EPOCH`, or the runner's minute budget.
- Any prompt guard refusal.

If time runs short, steps 21–24 are the priority, because they locate the source.

What the run will and will not decide. It localises the drift (UNet, VAE or RM) and measures, on diagnostic prompts,
the size of the drift relative to ranking gaps and how far method effects move. It **does not** set a tolerance,
does not make a G1 decision, and does not select a configuration. Any choice of engine policy must then go through
research using train/dev data, never val/test.

## 9. Options if drift matters (not adopted; production unchanged)

Costs come from the smoke1 provisional M2 table (per Δ=4 forwards, RTX 3090): den(1)=115 ms, den(2)=209, den(4)=395,
den(8)=778, den(16)=1524. ver(2)=493 ms, ver(4)=977, ver(8)=1950.

| Option | Compute cost | Effect on replay | Must be re-measured |
|---|---|---|---|
| **Batch 1 everywhere** (each candidate alone; decode and RM per image) | den: 8×115 = 920 ms vs 778 at k=8 (+18%); 4×115 = 460 vs 395 at k=4 (+16%). ver(1) and fin(1) per image not measured yet (fin(1)=162 ms). Collector throughput drops by about the same factor | Bank and online compute each candidate identically, so with deterministic kernels replay = online bitwise. Candidate independence then holds by construction | M2 (all primitives, all k), M0, M6 (becomes a check of exact equality), M9 throughput, equivalence (variant A no longer equals the base's batched path), caps (C_ref is derived from costs) |
| **Pad to a fixed batch 8** (keep survivors, fill the rest with dummy rows) | Real compute stays at den(8) after every prune: pruning saves no UNet compute (den(4) and den(2) become 778 ms). Cost-aware methods lose the saving they are designed to exploit. Costs must be charged as actually computed | The batch stays the same, but the composition changes (dummy rows); drift is reduced only if kernels are row-independent at a fixed shape, which needs measuring | M2 with padded costs, M6 with padding, caps, all cost-sensitive conclusions |
| **FP32 UNet / VAE** | UNet roughly 2–4× slower and about 2× VRAM (not measured) | Smaller drift, but kernel selection still depends on shape: no invariance guarantee | Everything above, plus M3 validity and the env/EPH dtype fields |
| **Keep the engine; treat drift as measured noise** | none | Replay is approximate; G1 analysis would need the drift distribution against ranking gaps on train/dev, as research decides (Q10) | — |

## 10. EPH / BMH changes required if the engine policy changes

Today the EPH body records: generation code sha, env lock, model and weights, timesteps, sampling, dtypes,
`vae_decode_chunk`, scoring path, RM provenance, seed scheme, and `numeric_runtime` (cudnn
deterministic/benchmark, both TF32 flags, SDPA flash/mem-efficient/math flags, `CUBLAS_WORKSPACE_CONFIG`).

To add if any option in §9 is adopted:
- **EPH:**
  - `batch_policy` (`natural` | `batch1` | `pad8`) for the UNet;
  - `decode_batch_policy` and `rm_batch_policy`;
  - the UNet/VAE attention processor class;
  - `torch.are_deterministic_algorithms_enabled()`;
  - `allow_fp16_reduced_precision_reduction`;
  - the GPU model, because kernel selection is device-specific.
- **BMH:**
  - the batch context of the stored numbers: `collection_batch` (8 today), `decode_chunk_effective` and `rm_batch`
    (8 today);
  - a statement that previews were computed without pruning (already `mode: full-8-seed-no-prune`).

  Online runs should log the batch size at every VERIFY and delivery, so replay comparisons can condition on it.

Any of these changes yields a new EPH. Existing banks (only the technical smoke bank) cannot be mixed with new ones.

## 11. diag1 results (2026-10-15, GPU run of the harness)

ENGINEERING ONLY: diagnostic prompts in a technical split, synthetic costs, **not G1**, no threshold set.

**Run.**
- Machine: Vast instance **56070930** (RTX 3090 24 GB, driver 595.91.07, Sweden), provided by the user. It is not
  56044352.
- Price: $0.359/h compute plus storage (verified via API); bandwidth $0.026/GB.
- Code: commit `1100a12147856b751c62c230029cbf619ab0d7cb`, PSP `590f59f58384169c719e431dc01d14006fb0fc0c`, transferred
  by bundle (sha256 verified on the box). Repo clean before the run.
- Runner: `gpu_smoke.sh diag`, RUN_ID `diag1`, started 09:00:31Z and exited 0 at 09:23:10Z. All 14 steps exited 0.
- Peak VRAM: 12.6 GB allocated and 24.1 GB reserved (traj b16).
- Artifacts: `diag1_artifacts.tgz`, sha256 `f57d0646…146885a`. Digest in `reports/batch_drift/diag1_summary.json`.
- EPH measured in every part: `c1351a99…`, the same as smoke1. EPH does not include the driver or Python patch
  version (3.10.21 here, 3.10.20 in smoke1), so matching EPHs say nothing about those.

**A/A repeatability.**
- *Two processes, same batch 8:* bitwise identical for all 16 candidates. This covers initial latents, embeddings,
  all 64 forwards, the decoded tensor, uint8 pixels, and preview and delivered scores (`aa_compare.json`).
- *Across sessions, machines and drivers:* the `method` k=8 collection of the astronaut prompt (bank 0) equals
  smoke1's bank shard. Initial latents 8/8, preview table 128/128, delivered 8/8 (all compared as float32) and PNG
  pixels 8/8 (`diag1_crosscheck_smoke1.json`).
- *M6 reproduced:* traj reproduced smoke1's M6 numbers to every digit (latent max 1.7068 / 2.4443 / 2.0454 /
  1.4354; flips at b2 and b16).

**Drift sources.** At a fixed batch size the pipeline is deterministic; changing the batch size changes numbers in
all three components.

| Component | Test (identical, hashed inputs) | Batch size ≠ 8 | Same size, other composition/order |
|---|---|---|---|
| UNet | one forward at f = 0, 16, 32, 63 | 0/16 bitwise; ε max 0.002–0.005 (fp16 ulp level) | identical to the same size `fwd` |
| Trajectory | 64 forwards | first divergent forward = **1** for all 16; final latent max 1.4–2.4, RMSE 0.07–0.13 | b8 rev/interleave bitwise = b8 |
| VAE | same latent bytes, chunk 1/2/4/16 | 0/16 bitwise; tensor max ≤ 0.013; **pixels ≤ 2 u8 levels** | chunk 8 rev/interleave bitwise |
| ImageReward | same PNG pixels and preprocessed tensor; 3 repeats of b8 bitwise | score max **2.0e-6** | identical to the same size `fwd` |

- *Reading.* The dominant source is UNet batch-size-dependent kernels. Ulp-level ε differences at forward 1 are
  amplified by the 64-step trajectory into visible changes.
- *Not decisive.* VAE (≤ 2 u8 levels) and ImageReward (~1e-6) are measurable but tiny compared with ranking gaps
  (smallest seen about 0.0015).
- *Not established.* Which kernel (GEMM, conv or SDPA backend) varies with batch size. This run did not toggle
  backends.

**Impact on the method** (6 fixed pairs × 3 prompts × 2 banks = 36 episodes, online vs replay on the same
instance's k=8 table; plan written to `method_plan.json` before running).
- *Winners:* 36/36 equal, so `winner_changes` is empty and there is no quality cost to report.
- *Keep sets:* first prune 36/36 equal. That prune runs at k=8, the same batch as the bank, so this is expected.
  Second prune 32/36 equal. The 4 differences, (2,4) on two instances and (6,12) on two, all had small replay gaps
  at j2 (0.0076–0.059), and the winner still matched.
- *Delivered Y online − replay:* max |·| 0.049, mean 0.0036.
- *Effect vs (4,8):* online − replay error max 0.058. 7 sign disagreements, mostly where the replay effect is exactly
  0 (same winner as (4,8)) and the online effect is the delivered-score noise of a different final batch size.
  Sign agreement is therefore not a useful statistic when the replay effect is 0.
- *Scope:* 3 diagnostic prompts. This says nothing about how often flips occur on the research distribution, where
  some j=4 gaps are as small as 0.0015.

**Engine changes worth trying next (not adopted; production unchanged).**
1. Pin UNet kernels per batch shape. Test the SDPA backend (math vs flash/mem-efficient), `torch.use_deterministic_algorithms`
   and `fp16_reduced_precision_reduction=False`, and check whether any setting makes b1/b2/b4/b16 equal b8.
   Primitives to re-measure: den(k) for each k, M0, M6, equivalence.
2. Padding only for the final delivered decode and score (VAE+RM effect ≤ 2 u8 and 1e-6). It would not remove UNet
   drift, so it is low value.
3. Treat drift as measurement noise in replay, sized by the Y differences above.

Any of these needs the proposal and EPH/BMH changes of §9–10.
