# Preregistration status

| PREREG | Content (Spec §11.5) | Status |
|---|---|---|
| A | machine/budget, prompt split, sampler/seed, precision, scoring, feature definitions, EPH, generator costs/caps, collector equivalence, P0 acceptance | **locked 2026-10-17** (`PREREG_SIGNED/PREREG-A.json`, approval by the project owner) |
| B | anchor, thresholds, failure penalty, reward scale, normalisation/PCA, candidate selection, primary contrasts, multiplicity | protocol-lock snapshot pending (values from train_fit/train_dev) |
| C | frozen artifacts/config/operating point, evaluator/seed protocol, test endpoints/correction | protocol-lock snapshot pending (before any GenEval image) |

## Process (from 2026-10-17, `PREREG_SIGNED/AUTHORIZATION_20261017.md`)

The project owner approved PREREG-A and authorised the agent to complete the remaining protocol details per the
HANDOFF and plan v1.3. B and C are no longer signed separately per phase. They become **protocol-lock snapshots**
under that authorization:

1. Every value is fixed in advance by an agreed formula, or computed from train_fit/train_dev only.
2. The filled proposal is committed under `docs/PREREG_PROPOSALS/`, and its sha256 and timestamp are written to the
   marker before any outcome on the protected split is read: val/val_ext for B, test/GenEval for C.
3. After a snapshot is locked, methods, operating points, margins, sample size, contrasts and statistical rules do
   not change. Test data is used only for evaluation and never feeds back into model selection. Negative and
   inconclusive results are reported as they are.

Marker `docs/PREREG_SIGNED/PREREG-<X>.json`:

```json
{"prereg": "A", "role": "research", "signed_by": "<who approved; for B/C: snapshot under the owner's authorization>",
 "approval_kind": "owner_approval | protocol_lock_snapshot", "authorization_record": "docs/PREREG_SIGNED/AUTHORIZATION_20261017.md",
 "authorization_sha256": "<sha256>", "proposal_path": "docs/PREREG_PROPOSALS/<file>",
 "proposal_sha256": "<sha256 of that proposal file>", "recorded_at": "<UTC>", "recorded_by": "<agent>"}
```

`role: research` means research authority, here the project owner's. The marker never names anyone else as a
signer. `apsp.contracts.prereg` refuses research splits (`train_fit`/`train_dev` → A, `val`/`val_ext` → B,
`test`/`geneval` → C) until a valid marker exists and the proposal file is unchanged.
