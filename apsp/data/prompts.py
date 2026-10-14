"""Prompt pool from DiffusionDB METADATA only (Guide CX-1). Never downloads data by itself.

Pipeline (rule ids are recorded in the pool manifest and frozen in PREREG-A):
  normalise (apsp-norm-v1: NFC + collapse whitespace + strip; no lowercasing/reordering)
  -> prompt_uid = sha1(normalised)  -> English filter (en-ascii-v1, see QUESTIONS Q2)
  -> prompt_nsfw < 0.1 -> 4..60 content CLIP tokens (BOS/EOS excluded, actual tokenizer)
  -> near-duplicate groups: connected components of cosine > 0.95 (L2-normalised pooler_output)
  -> drop groups with cosine > 0.9 to any GenEval text
  -> group-disjoint deterministic split: calibration(50, subset 20) / train_fit / train_dev / val / val_ext.

CLI (GPU env for the encoder):
  python -m apsp.data.prompts --config configs/pilot.yaml --metadata PATH/metadata.parquet --pilot small --out data/prompts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

NORM_RULE = "apsp-norm-v1"
LANG_RULE = "en-ascii-v1"
PILOT_SIZES = {"small": {"train_fit": 190, "train_dev": 50, "val": 80, "val_ext": 220},
               "large": {"train_fit": 320, "train_dev": 80, "val": 200, "val_ext": 100}}
CALIB, SUBSET = 50, 20

_WS = re.compile(r"\s+")


def normalize(raw: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", raw)).strip()


def prompt_uid(normalized: str) -> str:
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_english(text: str, min_ascii: float = 0.97) -> bool:
    """Heuristic placeholder rule en-ascii-v1 (>=97% ASCII letters among letters). Research to approve (Q2)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= min_ascii


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def near_duplicate_groups(emb: np.ndarray, thr: float = 0.95, block: int = 2048) -> np.ndarray:
    """Connected components of cosine > thr. Returns group index per row (smallest member index)."""
    u = _unit(emb)
    uf = UnionFind(len(u))
    for s in range(0, len(u), block):
        sims = u[s:s + block] @ u.T
        ii, jj = np.nonzero(sims > thr)
        for i, j in zip(ii + s, jj):
            if i < j:
                uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(len(u))])


def geneval_contaminated(emb: np.ndarray, geneval_emb: np.ndarray, thr: float = 0.9) -> np.ndarray:
    return (_unit(emb) @ _unit(geneval_emb).T).max(axis=1) > thr


@dataclass
class PoolResult:
    table: pd.DataFrame
    manifest: dict


def build_pool(meta: pd.DataFrame, *, count_tokens: Callable[[Sequence[str]], Sequence[int]],
               embed: Callable[[Sequence[str]], np.ndarray], geneval_texts: Sequence[str], pilot: str,
               nsfw_max: float = 0.1, tok_range=(4, 60), seed_salt: str = "apsp-split-v1") -> PoolResult:
    if pilot not in PILOT_SIZES:
        raise ValueError(pilot)
    df = meta[["prompt"] + (["prompt_nsfw"] if "prompt_nsfw" in meta else [])].copy()
    df["text"] = df["prompt"].astype(str).map(normalize)
    df = df[df["text"].str.len() > 0]
    df["prompt_uid"] = df["text"].map(prompt_uid)
    n_raw = len(df)
    raw_by_uid = df.groupby("prompt_uid")["prompt"].apply(lambda s: sorted(set(s))).to_dict()
    collisions = {u: v for u, v in raw_by_uid.items() if len(set(normalize(x) for x in v)) > 1}
    if collisions:
        raise RuntimeError(f"prompt_uid collision across distinct normalised texts: {list(collisions)[:3]}")
    df = df.drop_duplicates("prompt_uid")
    df = df[df["text"].map(is_english)]
    if "prompt_nsfw" in df:
        df = df[df["prompt_nsfw"].astype(float) < nsfw_max]
    df["n_clip_tokens"] = np.asarray(count_tokens(list(df["text"])), dtype=np.int16)
    df = df[(df["n_clip_tokens"] >= tok_range[0]) & (df["n_clip_tokens"] <= tok_range[1])].reset_index(drop=True)
    emb = np.asarray(embed(list(df["text"])), np.float32)
    g_emb = np.asarray(embed(list(geneval_texts)), np.float32)
    gidx = near_duplicate_groups(emb)
    df["group_id"] = [df["prompt_uid"].iloc[i] for i in gidx]
    contaminated = geneval_contaminated(emb, g_emb)
    bad_groups = set(df.loc[contaminated, "group_id"])
    df = df[~df["group_id"].isin(bad_groups)].reset_index(drop=True)
    # deterministic group order
    groups = sorted(df["group_id"].unique(), key=lambda g: hashlib.sha256(f"{seed_salt}|{g}".encode()).hexdigest())
    targets = [("calibration", CALIB)] + list(PILOT_SIZES[pilot].items())
    assign, ti, filled = {}, 0, 0
    sizes = df.groupby("group_id").size().to_dict()
    for g in groups:
        while ti < len(targets) and filled >= targets[ti][1]:
            ti, filled = ti + 1, 0
        if ti >= len(targets):
            assign[g] = "unused"
            continue
        assign[g] = targets[ti][0]
        filled += sizes[g]
    df["split"] = df["group_id"].map(assign)
    calib = df[df["split"] == "calibration"].sort_values("prompt_uid")
    subset = set(calib["prompt_uid"].iloc[:SUBSET])
    df["throughput_subset"] = df["prompt_uid"].isin(subset)
    df["source"] = "diffusiondb-metadata"
    counts = df["split"].value_counts().to_dict()
    manifest = {"norm_rule": NORM_RULE, "lang_rule": LANG_RULE, "nsfw_max": nsfw_max, "token_range": list(tok_range),
                "dup_threshold": 0.95, "geneval_threshold": 0.9, "split_salt": seed_salt, "pilot": pilot,
                "n_raw": n_raw, "n_after_filters": int(len(df)), "split_counts": counts,
                "geneval_groups_removed": len(bad_groups),
                "status": "PROPOSAL - research approves the pool/split before collection"}
    out_cols = ["prompt_uid", "text", "split", "source", "group_id", "n_clip_tokens", "throughput_subset"]
    return PoolResult(df[out_cols], manifest)


def _clip_functions(model_repo: str, revision: Optional[str], device: str = "cuda"):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(model_repo, subfolder="tokenizer", revision=revision)
    enc = CLIPTextModel.from_pretrained(model_repo, subfolder="text_encoder", revision=revision).to(device).eval()

    def count(texts):
        return [len(tok(t, add_special_tokens=False, truncation=False)["input_ids"]) for t in texts]

    @torch.inference_mode()
    def embed(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding="max_length", max_length=tok.model_max_length, truncation=True,
                    return_tensors="pt").to(device)
            out.append(enc(**b).pooler_output.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    return count, embed


def main(argv=None) -> int:
    from apsp.config import load_config
    from apsp.data.geneval_text import load_geneval_texts
    ap = argparse.ArgumentParser(description="Build the DiffusionDB prompt pool proposal (metadata only)")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--metadata", required=False, help="local DiffusionDB metadata parquet (not downloaded by this tool)")
    ap.add_argument("--pilot", choices=sorted(PILOT_SIZES), default="small")
    ap.add_argument("--out", default="data/prompts")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if not a.metadata:
        print("ERROR: --metadata is required. Download DiffusionDB metadata (poloclub/diffusiondb, metadata.parquet) "
              "with a pinned revision yourself; this tool never fetches data.", file=sys.stderr)
        return 2
    meta = pd.read_parquet(a.metadata)
    if a.dry_run:
        print(json.dumps({"rows": len(meta), "columns": list(meta.columns)}, indent=2))
        return 0
    count, embed = _clip_functions(cfg["model_repo"], cfg.get("model_revision"))
    texts, gsha = load_geneval_texts()
    res = build_pool(meta, count_tokens=count, embed=embed, geneval_texts=texts, pilot=a.pilot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_parquet(out / "prompt_pool.parquet", index=False)
    res.manifest.update({"geneval_text_sha256": gsha, "encoder_repo": cfg["model_repo"],
                         "encoder_revision": cfg.get("model_revision")})
    (out / "pool_manifest.json").write_text(json.dumps(res.manifest, indent=2), encoding="utf-8")
    print(json.dumps(res.manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
