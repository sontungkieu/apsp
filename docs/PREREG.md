# Preregistration status

| PREREG | Content (Spec §11.5) | Status |
|---|---|---|
| A | machine/budget, prompt split, sampler/seed, precision, scoring, feature definitions, EPH, generator costs/caps, collector equivalence, P0 acceptance | **not signed** (and no proposal yet: needs GPU P0 measurements) |
| B | anchor, thresholds, failure penalty, reward scale, normalisation/PCA, candidate selection, primary contrasts, multiplicity | **not signed** |
| C | frozen artifacts/config/operating point, evaluator/seed protocol, test endpoints/correction | **not signed** |

Signing (research only): place `docs/PREREG_SIGNED/PREREG-<X>.json`:

```json
{"prereg": "A", "role": "research", "signed_by": "<name>", "proposal_path": "docs/PREREG_PROPOSALS/<file>",
 "proposal_sha256": "<sha256 of that proposal file>"}
```

`apsp.contracts.prereg` refuses research splits (`train_fit`/`train_dev` → A, `val`/`val_ext` → B,
`test`/`geneval` → C) until a valid marker exists and the proposal file is unchanged. Coding agents write
proposals under `docs/PREREG_PROPOSALS/` only.
