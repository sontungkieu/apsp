# P0 acceptance: CPU / synthetic part (branch `bootstrap/p0`)

Date: 2026-10-14. Dev machine: Windows 11, AMD Ryzen 7 6800HS (16 logical), 13.7 GB RAM, RTX 3050 Ti Laptop
4 GB (driver 531.88). It is **not** a measurement machine.

## Four categories, kept separate

| Category | Content |
|---|---|
| **Implemented** | contracts (hashing, identity/RowMap, seed scheme, ledger, EpisodeState/PolicyView, accessor, bank schema, PREREG guards); DP + mask (G-842-v1, G-A1-v1); shared closed-loop/schedule runners; replay engine; obs 38/42; caps; 105-pair tables; O1/O1-free/O2 (dual + primal witness)/O3; B0/B1/B2/B3/B6/B9; REINFORCE M1; stats (bootstrap, gate rules, power, variance, frontier); reports (anchor, headroom, G1, ctrl_costs); prompt pool; GPU env_check/model/hooks/validity/decode/scoring/engine/profile/collector/online/reference equivalence |
| **Run on CPU / synthetic** | full pytest suite (113 tests), `apsp.quickstart`, the README CPU commands, fake-backend runs of every GPU code path (engine, collector resume/mismatch, online==replay, profile stages) |
| **Run on a real GPU** | only on the 4 GB dev laptop: `env_check --require-cuda` (CUDA OK, torch 2.4.0+cu121) and `profile --stage model` (M1: SD1.5 mirror revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, 12 weight files SHA256, 64 DDIM timesteps 946→1 leading/offset 1, terminal coefficients, ImageReward revision `5736be03b2652728fb87788c9797b0570450ab72` with missing_keys = [] and unexpected_keys = []). Reports: `reports/local_smoke/`. The M0 hook stage was started but the session stopped it because the host ran low on RAM; it did not finish. |
| **Not verified** | denoising, decode, scoring, collector, online and profiling on real weights; M0/M2/M3/M6/M9; PSP reference equivalence; the Linux GPU env (`env/setup_gpu_linux.sh` has not run on Linux) |

## Checks executed and results

| Check | Command | Result |
|---|---|---|
| Clean install, imports, CLI | clean clone (temp dir) → `uv venv` → `pip install -e ".[dev,rl]"` (CPU torch) | OK |
| Submodule pin | `git submodule status` | `590f59f58384169c719e431dc01d14006fb0fc0c`, working tree clean. **Windows needs `core.longpaths=true`** (a deep path under `baselines/` exceeds MAX_PATH); Linux is unaffected |
| Full suite | `python -m pytest tests -q` | **113 passed** (dev venv and clean clone) |
| Hash / schema / identity | `tests/contract` | type separation, pinned golden bytes, NFC key collision, self-reference/abs-path/timestamp rejection, CRLF-invariant code hash; pool16 row order, ranking ties/invalid last, seed derivation; bank validation catches seed, validity-chain, missing-row, finite-iff and BMH corruption |
| DP vs brute force | `tests/sim/test_budget_dp.py` | L and mask equal brute force on every reachable state for G-842 and G-A1 (J=5, 12 random cost tables each, budgets at threshold ±1); budget invariant on 20 random traces at J=16; forced action still pays disp; once prepaid; no terminal dispatch |
| Ledger regressions | `tests/contract/test_ledger_state.py` | CONT = Δ·den + feat (43, not 52) + disp; schedule_cost equals the closed form for all 105 pairs; F(1) fin(1); pred cut; no charge beyond cap |
| Leakage / prereg guards | `test_replay_leakage.py`, `test_prereg_schema.py` | preview reads only at the current j for alive candidates; sentinel overwrite of every unbought / future / pruned / non-final value leaves observations, actions and Y unchanged; unsigned research splits refused; marker must match the proposal hash; val log refuses look 3 and config changes |
| Replay e2e | fixed / threshold / random policies | spend ≤ cap, trace sums equal spend, exactly two prunes at distinct j ≤ 15; closed-loop fixed pair == schedule runner (Y, spend, winner); O1 ≥ every main-grammar policy per instance |
| REINFORCE smoke | `python -m apsp.policies.reinforce --synthetic-smoke` | trains on the synthetic bank, loss decreases, all dev episodes within cap; masked actions have p=0; forced steps carry no policy gradient; learns the known optimum of a bandit |
| Collector resume / mismatch (fixtures) | `tests/gpu` (fake backend) | valid bank; resume skips valid shards; partial shard quarantined to `failed/`; EPH mismatch, no-`--resume` reuse and shard checksum tamper all rejected |
| Replay ↔ online integration | `tests/gpu` (fake backend) | identical actions, spend, winner and Y for b2 / random / fixed:2,9 at α=0.5 and 1.0 on a bank collected by the same engine |
| GPU guard | `tests/gpu` | collect_bank / profile / online exit with `CudaUnavailableError` on a host without CUDA (no fallback) |
| Quickstart | `python -m apsp.quickstart` | runs in about 45–55 s; bank valid; per-cap B2/B3/B6/B9/O1 and a REINFORCE smoke (all synthetic) |

## Limits

- All synthetic costs and rewards are generated numbers. They test code paths and are not evidence about
  image quality or speed.
- The fake backend is row-independent by construction, so online==replay is exact there. On CUDA, batch effects
  are measured by M6/G1 and are not assumed away.
- B4b, B5, B7, B8, B10-min and ablations A1–A6 are not implemented yet; the grammar and cost paths they need
  exist. `pilot_report` and the GenEval/HPS evaluators are not implemented (P3/P4).
- No PREREG is signed and no bank exists.
