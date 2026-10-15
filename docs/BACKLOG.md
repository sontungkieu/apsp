# Backlog (after the technical smoke)

Ordered by dependency. Nothing here blocks the first GPU smoke.

| Item | Depends on | Notes |
|---|---|---|
| Batch-drift diagnostic run (`gpu_smoke.sh diag`) | Q11 (machine/budget) | localises drift; feeds Q10. Harness ready (BATCH_DRIFT_AUDIT.md) |
| Engine batch-policy decision | diagnostic results, research (Q10) | options/costs in BATCH_DRIFT_AUDIT §9; any change = new EPH + M2/M0/M6/M9/equivalence re-measured |
| Prompt pool build (`apsp.data.prompts`) | Q2 (English rule), Q3 (DiffusionDB metadata revision), GPU env for CLIP | produces calibration 50 + subset 20, train/dev/val/val_ext; research approves |
| M3 validity per precision (6,400 previews each) | prompt pool (calibration), GPU | `profile --stage validity --preview-dtype fp16|fp32 --prompts-file …` |
| Full M2 generator cost table (k=1..16, n=50, `--measurement`) | measurement machine designated (Q1), Q6 prune pairs | the smoke table is provisional |
| M9 throughput forecast | prompt pool, GPU | `collect_bank --calib-throughput` |
| Collector equivalence on 10 calibration prompts | prompt pool | `collect_bank --test-equivalence --prompt-pool …` |
| PREREG-A proposal | M1/M2/M3/M6/M9 + equivalence reports, Q1–Q3, Q6–Q7 | agents write the proposal; research signs |
| Pilot bank collection | PREREG-A signed | `collect_bank --split train_fit train_dev val --resume` |
| B8 one-query selector | DP + runners (done); ridge/GBM with group folds | needs preflight before query and query-reuse cost path (`schedule_cost` supports the trace) |
| B7 prompt-only selector | PCA-16 (done), train_fit labels | |
| B5 screen-and-resume | pool16 rows (identity done), prune pairs 8/16→1/2 profiled | |
| B4b PSP grid | Q6 (grid audit at the pinned commit) | arbitrary keep is supported by `run_schedule` |
| B10-min predictor | `lfeat` hook (done), pred-cut cost path (done) | ridge default |
| Ablations A1–A6 | G-A1 grammar (done), `scaled_verifier` (done) | |
| `pilot_report`, GenEval/HPS evaluators | P3/P4, PREREG-C | evaluators live outside env/policy modules |
| Real-bank RL training | PREREG-B, Q4 (σ_R), `G2_EVALUATED` marker | `policies.reinforce --train` currently only checks guards |
