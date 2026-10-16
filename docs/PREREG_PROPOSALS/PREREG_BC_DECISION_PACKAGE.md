# PREREG-B / PREREG-C decision package (proposal — NOT signed)

From plan v1.3 §9 and Spec §11.5/§12. Formulas are fixed now; numbers come only from train_fit / train_dev after the
bank (job `p3_anchor_proposal`), never from val/test. Research approves the formulas once; the agent fills the
values by these rules and re-submits the filled file for signature (`docs/PREREG.md`). The agent never signs.

## PREREG-B (before G1 / headroom / val)

| # | Field | Proposal (rule) | Data | Open question |
|---|---|---|---|---|
| B1 | Reference | PSP-default (8→4→2, cuts j=4,8); single64 and BoN4 reported in full | — | — |
| B2 | Anchor | Δ_anchor = mean over train_fit prompts of (IR_PSP − IR_BoN4), paired by prompt/bank, **online** train runs (batch drift known); replay value reported as approximation | online train_fit | online vs replay anchor (plan proposes online) |
| B3 | Margins | τ_prac = max(0.2·Δ_anchor, 0.02 IR); ε_IR = τ_prac/2; θ_RL = θ_query = τ_prac/4 | B2 | "0.02 IR": 0.02 in IR units (absolute) — confirm |
| B4 | r_fail | Q_0.001(valid IR_deliv, train_fit) − SD(same set); population quantile (numpy `quantile`, method "linear") and population SD (ddof=0); STOP-ENG if too few valid | train_fit bank | — |
| B5 | σ_R | population SD of valid IR_deliv over all train_fit candidates, floor 1e-6, frozen before RL (plan v1.3); alternative Q4: SD of Y(O1,α) − Y(B3,α) | train_fit bank | choose one (Q4) |
| B6 | Normalisation / PCA | NormStats and PCA-16 fitted on train_fit only; CV refits preprocessing per fold (B7/B8 done in code) | train_fit | — |
| B7 | Caps | {0.25, 0.5, 1.0, 1.5} × C_ref (same absolute caps for all methods); primary 0.5 and 1.0; infeasible → NA | cost table | — |
| B8 | Selection on dev | operating point = lowest actual mean e2e cost among points meeting the quality target on train_dev | train_dev | — |
| B9 | RL | REINFORCE/MLP as in handoff; ≥ 3 training seeds for the final config; checkpoint rule fixed on dev; all seeds reported | — | — |
| B10 | Endpoints | primary quality = delivered IR; primary efficiency = actual e2e cost; HPS v2.1 diagnostic on val | — | — |
| B11 | Decision rule | GO: IR NI (LCB(ΔQ) ≥ −ε) and LCB(savings) ≥ 10 %; STOP: quality inferior beyond margin or UCB(savings) < 5 %; else INC | — | — |
| B12 | Contrasts / multiplicity | `contrast_manifest.json` lists primary scalar contrasts → M; two looks (val 80, then val_ext ≤ 220) with α₁ = α₂ = 0.025; per-contrast CI level 1 − 0.025/M; bootstrap ≥ 10,000 (more for tail resolution) | — | — |
| B13 | Confirmatory N | power 80 % from paired train/dev variance and target effect, fixed before the first val look; if N > 300 an amendment before any outcome is seen | train/dev | — |
| B14 | Amendment 1 rev. 2 | replay = simulator; confirmatory online; G1 = CI of online−replay paired effect discrepancy vs PSP-default; wide CI → INC | — | Q12 wording |

## PREREG-C (before any GenEval image is generated)

| # | Field | Proposal | Open question |
|---|---|---|---|
| C1 | Frozen artifacts | methods, operating points, controller checkpoints, norm/PCA, cost tables with hashes | — |
| C2 | GenEval | official manifest @af4902f and official aggregation; detector/code/weights pinned at that time | — |
| C3 | Test protocol | 4 independent delivered images per prompt per method, same seed protocol across methods | confirm 4 |
| C4 | HPS / GenEval role | reported as generalisation metrics; no NI claim without a separately fixed margin/power | — |
| C5 | Failure handling | failed generation counts as unsuccessful on every metric | — |
| C6 | DiffusionDB test (Q5) | GenEval only, or also a sealed DiffusionDB test (proposal 200 prompts) for IR/HPS | Q5 |
