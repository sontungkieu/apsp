# Prompt pool v1: report for research review

**Status: PROPOSAL.** Research approves the pool and split before any collection. Nothing has been collected and no
image has been generated.

Built 2026-10-15 by `scripts/build_prompt_pool.py` (`apsp/data/pool_v1.py`) on the laptop CPU. Pinned inputs are in
`configs/prompt_pool.yaml`, and every input was sha256-verified before use.

## Pinned inputs

| Input | Revision | sha256 |
|---|---|---|
| DiffusionDB 2M `metadata.parquet` (2,000,000 rows) | `poloclub/diffusiondb` @ `fb620fbe49fa4420e0734bd9c0df11f51176b61f` | `eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f` (= HF LFS hash) |
| fastText `lid.176.bin` (131,266,198 B) | HF mirror `julien-c/fasttext-language-id` @ `9f2e94d2…`; the official `dl.fbaipublicfiles.com` URL returned 403 from this network | `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e` |
| GenEval `evaluation_metadata.jsonl` (553 prompts, **text only**) | `djghosh13/geneval` @ `af4902f24d3ca90ebbb446dd9891a59e0f82725f` | `5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13` |
| CLIP tokenizer + text encoder | `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `451f4fe1…` (the pinned generator) | `text_encoder/model.safetensors` `d008943c…88334` |

- The GenEval copy vendored in the PSP submodule has the same 553 records. On Windows its file hash differs only by
  CRLF line endings.
- The official manifest is now the pinned source.

## Counts and exclusion reasons (sequential)

| Stage | Rule | Removed | Remaining |
|---|---|---:|---:|
| Metadata rows | — | — | 2,000,000 |
| Empty after normalisation | apsp-norm-v1 (NFC, collapse whitespace, strip) | 603 | 1,999,397 |
| Exact duplicates merged | one row per prompt_uid = sha1(normalised text) | 473,383 rows | 1,526,014 unique |
| NSFW | max `prompt_nsfw` over the uid's rows < 0.1 (18 uids had inconsistent values) | 90,081 | 1,435,933 |
| Language: top-1 not English | fastText lid.176 | 15,220 | 1,420,713 |
| Language: English with p < 0.90 | `fasttext-lid176-top1en-p0.90` | **1,172,804** | 247,909 |
| Tokens < 4 | content CLIP tokens, BOS/EOS excluded | 2,483 | |
| Tokens > 60 | | 13,910 | |
| Technical/diagnostic prompts | uids from `technical_smoke.yaml` and `drift_diagnostic.yaml` | 0 | **231,516 survivors** |
| Candidates | first 2,000 survivors in sha256(salt\|uid) order (salt `apsp-split-v1`) | | 2,000 |
| GenEval | cosine > 0.9 to any GenEval text, or full GenEval-template match | 0 | 2,000 |
| Near-duplicate groups | cosine > 0.95, connected components | 1,994 singletons + 3 pairs; no group spans two splits | |
| Split (group-exact-fill-v1) | groups in sha256(salt\|group) order | 1,410 unused | 590 selected |

**Split counts (exact):** calibration 50 (M9 subset 20 = the 20 smallest uids), train_fit 190, train_dev 50,
val 80, val_ext 220. There is no shortfall and no group was split.

## Findings for research

1. **The English rule removes 82.5% of top-1-English prompts** (`lid_threshold_analysis.json`).
   - Among the 1,420,713 top-1-English prompts, fastText p has median 0.81 (quartiles 0.73 / 0.88). Only 247,909
     reach p ≥ 0.90.
   - **The dropped prompts are typical DiffusionDB style:** longer (median 22 vs 15 words) and keyword lists (median
     5 vs 1 commas). Examples: "alita battle angel. intricate artwork. octane render, trending on artstation…" (p
     0.785), "dystopian forest on fire with pandemic warning signs" (0.762). Even the project's own technical prompt
     "a photograph of an astronaut riding a horse" scores 0.868.
   - **Effect:** the pool leans toward short, sentence-like prompts.
   - **Counts other thresholds would keep (information only; the rule is unchanged):** 0.85: 499,686; 0.80:
     777,398; 0.70: 1,154,294.
2. **NSFW filter.** Selected prompts have prompt_nsfw ≤ 0.099 (median 0.002). The filter is text-score based, so
   suggestive or celebrity prompts still pass (2 of the 370 visible prompts contain "sexy" or similar). The spec rule
   is applied as written.
3. **GenEval contamination: none among the candidates**, by cosine (max 0.675) or by template.
4. **Deviation: near-duplicate grouping covers the candidates only** (`sampled-grouping-v1`).
   - The CLIP text encoder runs at about 3.2 prompts/s on the laptop CPU; all 231,516 survivors would take about
     20 hours.
   - Groups are exact among the 2,000 candidates, so no two selected prompts in different splits have cosine > 0.95.
     Chains through non-candidate prompts are not followed.
   - `scripts/full_pool_check.py` (first step `30_pool_full_check` of the calibration session, GPU, est. 5–10 min
     for 231k prompts) computes full-pool components and lists any selected prompts that would share a component
     across splits. **The pool is not frozen until that check and research approval.**
5. **Sampling unit is the group.** A group larger than the remaining slots of a split would go to `unused`. None did.

## Files

| File | Where | Content |
|---|---|---|
| `prompt_pool.parquet` | `data/prompts/` (not in git), sha256 `5246859b…a66a8` | 590 rows: uid, text (**NULL for val_ext**), split, group, tokens, M9 flag, lang_p, prompt_nsfw, GenEval max cosine |
| `val_ext_sealed.parquet` | `data/prompts/` (not in git), sha256 `f6d2408d…cf9a5116dcf` | the 220 val_ext texts; uid-list sha256 `f60554f0…52ea2fc`. Opened only after PREREG-B, as the protocol allows |
| `survivors_8681118963d80f9e.parquet` | `data/prompts/work/` (not in git) | 231,516 filtered survivors (input to the full-pool check) |
| `pool_manifest.json` | here and in `data/prompts/` | rules, inputs, counts, split report, file hashes |
| `prompt_pool_review.csv` | here | the 370 non-sealed prompts with text, plus the 220 val_ext rows without text |
| `excluded_candidates.csv` | here | GenEval-contaminated candidates (none) |
| `lid_threshold_analysis.json` | here | language-ID distribution, alternative-threshold counts, samples |

## Decisions needed

1. Approve the English rule as it acts on DiffusionDB (finding 1), or adjust it. The pool would then be rebuilt
   deterministically.
2. Approve `sampled-grouping-v1` pending the GPU full-pool check, or require the full-pool grouping before any use.
3. Approve the pool and split (`prompt_pool_review.csv`) before calibration (calibration prompts are used on GPU in
   the calibration session).
