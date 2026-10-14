import numpy as np
import pandas as pd

from apsp.data.prompts import build_pool, normalize, prompt_uid


def test_normalize_no_lowercase_and_uid():
    assert normalize("  A  Cat on\tmat ") == "A Cat on mat"
    assert prompt_uid(normalize("A cat")) != prompt_uid(normalize("a cat"))


def test_pool_groups_are_split_disjoint_and_geneval_filtered():
    rng = np.random.default_rng(0)
    n = 900
    texts = [f"prompt number {i} with words" for i in range(n)]
    texts += [texts[0] + "!", "a photo of a bench", "日本語のプロンプト"]
    meta = pd.DataFrame({"prompt": texts, "prompt_nsfw": [0.0] * len(texts)})
    base = {t: rng.normal(size=32) for t in texts}
    base[texts[0] + "!"] = base[texts[0]] + 1e-4           # near duplicate of prompt 0
    base["a photo of a bench"] = rng.normal(size=32)

    def embed(ts):
        return np.stack([base.get(t, base.get(normalize(t), np.ones(32))) for t in ts])

    res = build_pool(meta, count_tokens=lambda ts: [5] * len(ts), embed=embed,
                     geneval_texts=["a photo of a bench"], pilot="small")
    t = res.table
    assert "日本語のプロンプト" not in set(t["text"])                 # English filter
    assert "a photo of a bench" not in set(t["text"])                # GenEval contamination removed
    assert t.groupby("group_id")["split"].nunique().max() == 1       # group-disjoint
    g0 = t[t["text"].isin([texts[0], texts[0] + "!"])]
    assert g0["group_id"].nunique() == 1
    counts = t["split"].value_counts()
    assert counts["calibration"] >= 50 and counts["train_fit"] >= 190
    assert t["throughput_subset"].sum() == 20 and set(t[t["throughput_subset"]]["split"]) == {"calibration"}
