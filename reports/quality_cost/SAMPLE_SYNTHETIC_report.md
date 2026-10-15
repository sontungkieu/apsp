# Quality-cost report

**SYNTHETIC / FAKE BACKEND - plumbing check only; NOT evidence of quality or savings**

Split: `technical_smoke` · reference: `psp` · cost measure: `e2e_seconds` · CI level 0.9500

Conclusions: **not_concludable** (evaluation config not approved by research; epsilon not set (PREREG-B); number of primary contrasts M not fixed (descriptive CI level used))

## Per method

| method | n | fail | mean Q | mean e2e s | p50 | p95 | ledger us | fwd | cand-fwd | VERIFY | RM in inference | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single32 | 6 | 0.000 | 0.3075 | 0.0162 | 0.0159 | 0.0181 | 456000 | 32 | 32 | 0.0 | 0 | n/a |
| bon:2 | 6 | 0.000 | 0.3137 | 0.0290 | 0.0290 | 0.0302 | 1173000 | 64 | 128 | 0.0 | 1 | n/a |
| single | 6 | 0.000 | 0.3068 | 0.0300 | 0.0289 | 0.0333 | 872000 | 64 | 64 | 0.0 | 0 | n/a |
| bon:4 | 6 | 0.000 | 0.3498 | 0.0334 | 0.0329 | 0.0355 | 1715000 | 64 | 256 | 0.0 | 1 | n/a |
| sr:8,3,2@0.5 | 6 | 0.000 | 0.3584 | 0.0351 | 0.0348 | 0.0382 | 1587360 | 64 | 200 | 1.0 | 1 | n/a |
| bon:8 | 6 | 0.000 | 0.3617 | 0.0371 | 0.0376 | 0.0399 | 2799000 | 64 | 512 | 0.0 | 1 | n/a |
| fixed:3,8@1.0 | 6 | 0.000 | 0.3617 | 0.0388 | 0.0387 | 0.0405 | 1825640 | 64 | 240 | 2.0 | 1 | n/a |
| sr:16,2,2@1.0 | 3 | 0.000 | 0.3720 | 0.0390 | 0.0396 | 0.0400 | 1843520 | 64 | 240 | 1.0 | 1 | n/a |
| psp | 6 | 0.000 | 0.3617 | 0.0400 | 0.0393 | 0.0451 | 1889640 | 64 | 256 | 2.0 | 1 | n/a |
| fixed:2,6@0.5 | 6 | 0.000 | 0.3617 | 0.0411 | 0.0407 | 0.0433 | 1697640 | 64 | 208 | 2.0 | 1 | n/a |
| sr:16,2,2 | 3 | 0.000 | 0.3720 | 0.0427 | 0.0402 | 0.0494 | 1843520 | 64 | 240 | 1.0 | 1 | n/a |
| bon:16 | 3 | 0.000 | 0.3764 | 0.0441 | 0.0441 | 0.0447 | 4967000 | 64 | 1024 | 0.0 | 1 | n/a |
| b8:data/sample/tuned/b8_selector_alpha1.0.json@1.0 | 6 | 0.000 | 0.3617 | 0.0481 | 0.0463 | 0.0556 | 1825640 | 64 | 240 | 2.0 | 1 | n/a |
| b8:data/sample/tuned/b8_selector_alpha0.5.json@0.5 | 6 | 0.000 | 0.3617 | 0.0485 | 0.0481 | 0.0505 | 1633640 | 64 | 192 | 2.0 | 1 | n/a |
| threshold:data/sample/tuned/b6_threshold_alpha0.5.json@0.5 | 6 | 0.000 | 0.3617 | 0.0503 | 0.0510 | 0.0520 | 1697640 | 64 | 208 | 2.0 | 1 | n/a |
| threshold:data/sample/tuned/b6_threshold_alpha1.0.json@1.0 | 6 | 0.000 | 0.3617 | 0.0530 | 0.0498 | 0.0647 | 1793640 | 64 | 232 | 2.0 | 1 | n/a |

## Contrasts vs reference (paired by prompt, cluster bootstrap)

| method | ΔQ | CI | savings | CI | NI | cost decision |
|---|---:|---|---:|---|---|---|
| single32 | -0.0542 | [-0.0856, -0.0242] | +0.595 | [+0.558, +0.632] | not_concludable | not_concludable |
| bon:2 | -0.0480 | [-0.0832, -0.0250] | +0.276 | [+0.251, +0.305] | not_concludable | not_concludable |
| single | -0.0549 | [-0.0832, -0.0280] | +0.251 | [+0.198, +0.317] | not_concludable | not_concludable |
| bon:4 | -0.0119 | [-0.0356, +0.0000] | +0.165 | [+0.078, +0.216] | not_concludable | not_concludable |
| sr:8,3,2@0.5 | -0.0033 | [-0.0098, +0.0000] | +0.124 | [+0.114, +0.136] | not_concludable | not_concludable |
| bon:8 | +0.0000 | [+0.0000, +0.0000] | +0.072 | [-0.002, +0.140] | not_concludable | not_concludable |
| fixed:3,8@1.0 | +0.0000 | [+0.0000, +0.0000] | +0.030 | [-0.028, +0.090] | not_concludable | not_concludable |
| sr:16,2,2@1.0 | +0.0103 | [-0.0066, +0.0292] | +0.025 | [-0.006, +0.061] | not_concludable | not_concludable |
| fixed:2,6@0.5 | +0.0000 | [+0.0000, +0.0000] | -0.026 | [-0.092, +0.019] | not_concludable | not_concludable |
| sr:16,2,2 | +0.0103 | [-0.0066, +0.0292] | -0.069 | [-0.239, +0.045] | not_concludable | not_concludable |
| bon:16 | +0.0147 | [+0.0066, +0.0292] | -0.103 | [-0.186, -0.061] | not_concludable | not_concludable |
| b8:data/sample/tuned/b8_selector_alpha1.0.json@1.0 | +0.0000 | [+0.0000, +0.0000] | -0.202 | [-0.402, -0.067] | not_concludable | not_concludable |
| b8:data/sample/tuned/b8_selector_alpha0.5.json@0.5 | +0.0000 | [+0.0000, +0.0000] | -0.213 | [-0.295, -0.138] | not_concludable | not_concludable |
| threshold:data/sample/tuned/b6_threshold_alpha0.5.json@0.5 | +0.0000 | [+0.0000, +0.0000] | -0.257 | [-0.349, -0.212] | not_concludable | not_concludable |
| threshold:data/sample/tuned/b6_threshold_alpha1.0.json@1.0 | +0.0000 | [+0.0000, +0.0000] | -0.325 | [-0.603, -0.174] | not_concludable | not_concludable |

## Quality at equal budget (shared caps B_alpha)

- alpha 0.25 (cap 1601640 us): bon:2 0.3137, single32 0.3075, single 0.3068
- alpha 0.5 (cap 1697640 us): fixed:2,6@0.5 0.3617, b8:data/sample/tuned/b8_selector_alpha0.5.json@0.5 0.3617, threshold:data/sample/tuned/b6_threshold_alpha0.5.json@0.5 0.3617, sr:8,3,2@0.5 0.3584, bon:2 0.3137, single32 0.3075, single 0.3068
- alpha 1.0 (cap 1889640 us): sr:16,2,2@1.0 0.3720, sr:16,2,2 0.3720, fixed:3,8@1.0 0.3617, psp 0.3617, b8:data/sample/tuned/b8_selector_alpha1.0.json@1.0 0.3617, threshold:data/sample/tuned/b6_threshold_alpha1.0.json@1.0 0.3617, bon:4 0.3498, bon:2 0.3137, single32 0.3075, single 0.3068
- alpha 1.5 (cap 2081640 us): sr:16,2,2 0.3720, psp 0.3617, bon:4 0.3498, bon:2 0.3137, single32 0.3075, single 0.3068

## Cost to reach reference quality

- level `single` (Q 0.3068, 0.0300 s): cheapest `single32`, savings +0.459 [+0.424, +0.488] — DESCRIPTIVE ONLY (point estimate dQ >= 0); not an NI conclusion
- level `psp` (Q 0.3617, 0.0400 s): cheapest `bon:8`, savings +0.072 [-0.002, +0.140] — DESCRIPTIVE ONLY (point estimate dQ >= 0); not an NI conclusion
- level `bon:8` (Q 0.3617, 0.0371 s): cheapest `fixed:3,8@1.0`, savings -0.045 [-0.058, -0.026] — DESCRIPTIVE ONLY (point estimate dQ >= 0); not an NI conclusion
