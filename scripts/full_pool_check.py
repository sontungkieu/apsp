"""Full-pool near-duplicate check for the prompt pool v1 proposal (GPU; first step of the calibration session).

Embeds EVERY filtered survivor (survivors_<key>.parquet from build_prompt_pool.py) with the pinned CLIP text encoder,
forms connected components of cosine > 0.95 over the whole pool (Spec 10.1) and reports selected prompts whose
full-pool component spans more than one split. Read-only for the pool files; the result goes to research.

  python scripts/full_pool_check.py --survivors data/prompts/work/survivors_KEY.parquet \
      --pool data/prompts/prompt_pool.parquet --sealed data/prompts/val_ext_sealed.parquet --out RUN/pool_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from apsp.data.pool_v1 import cross_split_full_pool  # noqa: E402
from apsp.data.prompts import UnionFind  # noqa: E402


def embed_all(texts, cfg, device, bs=512):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    e = cfg["encoder"]
    tok = CLIPTokenizer.from_pretrained(e["repo"], subfolder="tokenizer", revision=e["revision"])
    enc = CLIPTextModel.from_pretrained(e["repo"], subfolder="text_encoder", revision=e["revision"]).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding="max_length", max_length=tok.model_max_length, truncation=True,
                    return_tensors="pt").to(device)
            out.append(torch.nn.functional.normalize(enc(**b).pooler_output.float(), dim=1).cpu())
    return torch.cat(out)


def components(emb, thr, device, block=4096):
    """Connected components of cosine > thr. Screen in fp16 with a margin, confirm each pair in fp32."""
    import torch
    n = emb.shape[0]
    uf = UnionFind(n)
    E32 = emb.to(device)
    E = E32.half() if str(device).startswith("cuda") else E32
    for s in range(0, n, block):
        sims = E[s:s + block] @ E.T
        ii, jj = torch.nonzero(sims > thr - 2e-3, as_tuple=True)
        ii = ii + s
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        if len(ii):
            exact = (E32[ii] * E32[jj]).sum(1) > thr
            for a, b in zip(ii[exact].tolist(), jj[exact].tolist()):
                uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompt_pool.yaml")
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="TEST ONLY")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    surv = pd.read_parquet(a.survivors)
    if a.limit:
        surv = surv.head(a.limit)
    pool = pd.read_parquet(a.pool)
    sealed = pd.read_parquet(a.sealed)
    t0 = time.time()
    emb = embed_all(list(surv["text"]), cfg, a.device)
    t1 = time.time()
    comp = components(emb, 0.95, a.device)
    t2 = time.time()
    uid = surv["prompt_uid"].tolist()
    full = {uid[i]: uid[c] for i, c in enumerate(comp)}
    sel = pool[["prompt_uid", "split"]]
    missing = sorted(set(sel["prompt_uid"]) - set(full))
    bad = cross_split_full_pool(sel[sel["prompt_uid"].isin(full)], full)
    sizes = pd.Series(comp).value_counts()
    rep = {"n_survivors": len(surv), "n_components": int(len(sizes)), "largest_component": int(sizes.max()),
           "selected_not_in_survivors": missing, "cross_split_full_pool": bad, "n_cross_split": len(bad),
           "val_ext_uid_list_matches_sealed": sorted(sealed["prompt_uid"]) == sorted(pool[pool["split"] == "val_ext"]["prompt_uid"]),
           "seconds": {"embed": t1 - t0, "components": t2 - t1}, "device": a.device,
           "verdict": "no selected prompts share a full-pool component across splits" if not bad else
                      "research decision needed: move or drop the listed prompts (pool not frozen)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "cross_split_full_pool"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
