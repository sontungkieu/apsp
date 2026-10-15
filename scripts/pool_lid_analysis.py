"""Diagnostic for research: how the fastText English rule (top-1 en, p >= 0.90) shapes the DiffusionDB pool.

Read-only; does not change the pool or the rule. Reports the p distribution among top-1-English prompts (after exact
dedup and the NSFW filter, i.e. the population the LID rule sees), the counts that alternative thresholds WOULD keep
(information only), and deterministic samples of prompts dropped for p < 0.90 and kept.

  python scripts/pool_lid_analysis.py --external D:/Workspace/DMRL_external --out reports/prompt_pool/lid_threshold_analysis.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from apsp.data.prompts import normalize, prompt_uid  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompt_pool.yaml")
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import fasttext
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    ext = Path(a.external)
    meta = pd.read_parquet(ext / cfg["inputs"]["diffusiondb_metadata"]["local_path"], columns=["prompt", "prompt_nsfw"])
    meta = meta[meta["prompt"].notna()]
    meta["text"] = meta["prompt"].map(normalize)
    meta = meta[meta["text"].str.len() > 0]
    meta["uid"] = meta["text"].map(prompt_uid)
    u = meta.groupby("uid").agg(text=("text", "first"), nsfw=("prompt_nsfw", "max")).reset_index()
    u = u[u["nsfw"] < 0.1]
    m = fasttext.load_model(str(ext / cfg["inputs"]["fasttext_lid176"]["local_path"]))
    lab, pr = m.predict(list(u["text"]), k=1)
    u["label"] = [x[0] for x in lab]
    u["p"] = [float(x[0]) for x in pr]
    en = u[u["label"] == "__label__en"]
    qs = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
    hist, edges = np.histogram(en["p"], bins=[0, .3, .4, .5, .6, .7, .8, .85, .9, .95, 1.0001])
    order = lambda d: d.assign(_h=d["uid"].map(lambda x: hashlib.sha256(f"lid|{x}".encode()).hexdigest())).sort_values("_h")
    dropped = order(en[en["p"] < 0.90]).head(30)
    kept = order(en[en["p"] >= 0.90]).head(15)
    words = en["text"].str.split().str.len()
    rep = {"population": "unique normalised prompts after exact dedup and prompt_nsfw < 0.1",
           "n_population": int(len(u)), "n_top1_en": int(len(en)),
           "en_p_quantiles": {str(q): float(en["p"].quantile(q)) for q in qs},
           "en_p_histogram": {f"[{edges[i]:.2f},{min(edges[i + 1], 1):.2f})": int(h) for i, h in enumerate(hist)},
           "would_keep_at_threshold_information_only": {str(t): int((en["p"] >= t).sum()) for t in (0.5, 0.7, 0.8, 0.85, 0.9)},
           "median_words": {"p<0.90": float(words[en["p"] < 0.9].median()), "p>=0.90": float(words[en["p"] >= 0.9].median())},
           "comma_fraction_median": {"p<0.90": float(en.loc[en["p"] < 0.9, "text"].str.count(",").median()),
                                     "p>=0.90": float(en.loc[en["p"] >= 0.9, "text"].str.count(",").median())},
           "samples_dropped_p_lt_0.90": [{"p": round(r.p, 3), "text": r.text[:200]} for r in dropped.itertuples()],
           "samples_kept": [{"p": round(r.p, 3), "text": r.text[:200]} for r in kept.itertuples()],
           "rule_in_force": "fasttext-lid176-top1en-p0.90 (research decision); this report changes nothing"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_population", "n_top1_en", "en_p_quantiles", "would_keep_at_threshold_information_only",
                                          "median_words", "comma_fraction_median")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
