"""Prompt pool v1 stages on synthetic metadata: filter reasons/counts, GenEval templates, exact-fill group split."""
import numpy as np
import pandas as pd
import pytest

from apsp.data import pool_v1
from apsp.data.prompts import normalize, prompt_uid


def lang(texts):
    lab = ["__label__fr" if "chat" in t else "__label__en" for t in texts]
    return lab, [0.5 if "maybe" in t else 0.99 for t in texts]


def toks(texts):
    return [len(t.split()) for t in texts]


def test_filter_stage_counts_each_reason():
    meta = pd.DataFrame({"prompt": ["a cat on a mat today", "a cat on a mat today", "a  cat on a mat today ",
                                    "nsfw words here ok", "le chat noir est la", "maybe english text here",
                                    "too short", " ".join(["w"] * 61), "", "a photograph of an astronaut riding a horse",
                                    "line one\nline two here"],
                         "prompt_nsfw": [0.0, 0.0, 0.05, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    tech = [prompt_uid(normalize("a photograph of an astronaut riding a horse"))]
    seen = {}

    def lang_spy(texts):
        seen["texts"] = texts
        return lang(texts)
    surv, rep = pool_v1.filter_stage(meta, lang_fn=lang_spy, count_tokens=toks, exclude_uids=tech, tok_range=(4, 60))
    assert rep["n_rows"] == 11 and rep["drop_empty"] == 1
    assert rep["n_unique_uid"] == 8 and rep["exact_duplicate_rows_merged"] == 2   # 3 raw variants -> 1 uid
    assert rep["uids_with_raw_variants"] == 1 and rep["uids_with_inconsistent_nsfw"] == 1
    assert rep["drop_nsfw"] == 1 and rep["drop_lang_not_en_top1"] == 1 and rep["drop_lang_en_below_threshold"] == 1
    assert rep["drop_tokens_lt_min"] == 1 and rep["drop_tokens_gt_max"] == 1 and rep["drop_technical_prompt"] == 1
    assert set(surv["text"]) == {"a cat on a mat today", "line one line two here"}   # apsp-norm-v1 collapses newlines
    assert all("\n" not in t for t in seen["texts"])            # LID sees one line; stored text unchanged
    assert float(surv.loc[surv["text"] == "a cat on a mat today", "prompt_nsfw"].iloc[0]) == pytest.approx(0.05)


def test_geneval_templates():
    recs = [{"prompt": "a photo of a bench", "include": [{"class": "bench", "count": 1}]},
            {"prompt": "a photo of three dogs", "include": [{"class": "dog", "count": 3}]},
            {"prompt": "a photo of a red car and a blue bird",
             "include": [{"class": "car", "color": "red"}, {"class": "bird", "color": "blue"}]}]
    regs = pool_v1.geneval_template_regexes(recs)
    hits = pool_v1.template_hits(["A photo of a cat", "a photo of two teddy bears", "a photo of a green bus and a pink kite",
                                  "a photo of a cat sitting on a sofa, octane render", "portrait of a bench"], regs)
    assert hits.tolist() == [True, True, True, False, False]


def _cand(n, rng):
    texts = [f"prompt {i}" for i in range(n)]
    return pd.DataFrame({"prompt_uid": [prompt_uid(t) for t in texts], "text": texts, "n_clip_tokens": 5}), rng.normal(size=(n, 16))


def test_group_and_split_exact_disjoint_and_geneval_groups_dropped():
    rng = np.random.default_rng(0)
    cand, emb = _cand(800, rng)
    emb[1] = emb[0] + 1e-4               # near-duplicate pair -> one group
    emb[5] = emb[4] + 1e-4
    g_emb = np.stack([emb[4] + 2e-4])    # prompt 4 (and its group mate 5) contaminated
    tmask = np.zeros(len(cand), bool)
    tmask[7] = True                       # template hit
    table, rep = pool_v1.group_and_split(cand, emb, g_emb, tmask, "small", "salt")
    c = table["split"].value_counts()
    assert {k: c[k] for k in pool_v1.SPLIT_ORDER} == {"calibration": 50, "train_fit": 190, "train_dev": 50, "val": 80,
                                                      "val_ext": 220}
    assert rep["split_shortfall"] == {} and rep["cross_split_groups"] == 0
    assert set(table.loc[[4, 5, 7], "split"]) == {"excluded"} and rep["groups_dropped_geneval"] == 2
    assert table.loc[0, "group_id"] == table.loc[1, "group_id"] and table.loc[0, "split"] == table.loc[1, "split"]
    assert table["throughput_subset"].sum() == 20 and set(table.loc[table["throughput_subset"], "split"]) == {"calibration"}


def test_split_reports_shortfall_when_pool_too_small():
    cand, emb = _cand(300, np.random.default_rng(1))
    _, rep = pool_v1.group_and_split(cand, emb, np.zeros((0, 16)), np.zeros(300, bool), "small", "s")
    assert rep["split_shortfall"]["val_ext"] > 0


def test_cross_split_full_pool_check():
    sel = pd.DataFrame({"prompt_uid": ["a", "b", "c"], "split": ["train_fit", "val", "val"]})
    assert pool_v1.cross_split_full_pool(sel, {"a": "g1", "b": "g1", "c": "g2"}) == [
        {"prompt_uid": "a", "split": "train_fit", "full_group": "g1"}, {"prompt_uid": "b", "split": "val", "full_group": "g1"}]
