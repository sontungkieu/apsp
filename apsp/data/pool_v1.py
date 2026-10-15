"""Prompt pool v1 (Spec §10.1, Guide CX-1): staged build with counts and exclusion reasons.

Rules (ids recorded in the manifest):
  normalise            apsp-norm-v1 (NFC, collapse whitespace, strip; no lowercasing / reordering)
  exact dedup          one row per prompt_uid = sha1(normalised)
  NSFW                 prompt_nsfw < 0.1, using the MAX prompt_nsfw over all metadata rows of the uid
  English              fasttext-lid176-top1en-p0.90: fastText lid.176.bin top-1 label __label__en with p >= 0.90.
                       (apsp-norm-v1 already collapses newlines, which fastText requires)
  tokens               4..60 content CLIP tokens (BOS/EOS excluded) with the SD1.5 CLIPTokenizer at the pinned revision
  technical            uids of technical / diagnostic prompts (configs/technical_smoke.yaml, drift_diagnostic.yaml)
                       are excluded so those prompts stay outside every research split
  candidates           sampled-grouping-v1: the first N survivors in sha256(salt|uid) order are embedded
                       (CLIP text encoder pooler_output, L2-normalised); see the deviation note below
  GenEval              drop a candidate whose cosine to any GenEval prompt text > 0.9, or whose lowercased text fully
                       matches a GenEval template (class names, colours and count words replaced by wildcards)
  near duplicates      connected components of cosine > 0.95 among the candidates; the whole group is dropped if any
                       member is GenEval-contaminated
  split                groups in sha256(salt|group_id) order fill calibration 50, train_fit, train_dev, val, val_ext
                       exactly; a group larger than the remaining slots of the current split goes to `unused`
                       (groups are never split). M9 subset = the 20 calibration prompts with the smallest uid.

DEVIATION (needs research approval): Spec §10.1 groups near duplicates over the whole filtered pool. The CLIP text
encoder runs at ~3.4 prompts/s on the laptop CPU, so the whole pool (~10^6 prompts) is out of reach locally. Groups
are therefore exact among the N candidates (so no two selected prompts in different splits have cosine > 0.95), but
chains through non-candidate prompts are not followed. `full_pool_check` (GPU, first step of the calibration session)
embeds the whole filtered pool and reports any selected prompts that fall into one full-pool component across splits.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from apsp.data.prompts import CALIB, NORM_RULE, PILOT_SIZES, SUBSET, UnionFind, _unit, normalize, prompt_uid

LANG_RULE = "fasttext-lid176-top1en-p0.90"
GROUP_RULE = "sampled-grouping-v1"
SPLIT_RULE = "group-exact-fill-v1"
SPLIT_ORDER = ("calibration", "train_fit", "train_dev", "val", "val_ext")
COUNT_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _h(salt: str, s: str) -> str:
    return hashlib.sha256(f"{salt}|{s}".encode()).hexdigest()


# ------------------------------------------------------------------ stage 1: filters --------------------------------
def filter_stage(meta: pd.DataFrame, *, lang_fn: Callable[[Sequence[str]], tuple[list[str], list[float]]],
                 count_tokens: Callable[[Sequence[str]], Sequence[int]], exclude_uids: Iterable[str] = (),
                 nsfw_max: float = 0.1, tok_range=(4, 60), lang_label: str = "__label__en",
                 lang_min: float = 0.90) -> tuple[pd.DataFrame, dict]:
    """Sequential filters on unique normalised prompts. Returns survivors and a counts/reasons report."""
    rep: dict = {"n_rows": int(len(meta))}
    df = pd.DataFrame({"prompt": meta["prompt"].astype("string"), "prompt_nsfw": meta["prompt_nsfw"].astype(float)})
    df = df[df["prompt"].notna()]
    df["text"] = df["prompt"].map(normalize)
    empty = df["text"].str.len() == 0
    rep["drop_empty"] = int(empty.sum())
    df = df[~empty]
    df["prompt_uid"] = df["text"].map(prompt_uid)
    g = df.groupby("prompt_uid", sort=False)
    nsfw = g["prompt_nsfw"].agg(["max", "min"])
    u = g.agg(text=("text", "first"), n_rows=("text", "size"), n_raw_variants=("prompt", "nunique")).join(nsfw)
    u = u.reset_index()
    rep["n_unique_uid"] = int(len(u))
    rep["exact_duplicate_rows_merged"] = int(len(df) - len(u))
    rep["uids_with_raw_variants"] = int((u["n_raw_variants"] > 1).sum())
    rep["uids_with_inconsistent_nsfw"] = int((u["max"] - u["min"] > 1e-6).sum())
    bad = u["max"].isna() | (u["max"] >= nsfw_max)
    rep["drop_nsfw"] = int(bad.sum())
    u = u[~bad]
    labels, probs = lang_fn([t.replace("\n", " ").replace("\r", " ") for t in u["text"]])
    u = u.assign(lang_label=labels, lang_p=np.asarray(probs, float))
    not_en = u["lang_label"] != lang_label
    low = (~not_en) & (u["lang_p"] < lang_min)
    rep["drop_lang_not_en_top1"] = int(not_en.sum())
    rep["drop_lang_en_below_threshold"] = int(low.sum())
    rep["lang_top_labels_dropped"] = u.loc[not_en, "lang_label"].value_counts().head(10).to_dict()
    rep["lang_en_p_quantiles_kept"] = {}
    u = u[~not_en & ~low]
    rep["lang_en_p_quantiles_kept"] = {str(q): float(u["lang_p"].quantile(q)) for q in (0.0, 0.05, 0.5)} if len(u) else {}
    u = u.assign(n_clip_tokens=np.asarray(count_tokens(list(u["text"])), dtype=np.int32))
    short, long_ = u["n_clip_tokens"] < tok_range[0], u["n_clip_tokens"] > tok_range[1]
    rep["drop_tokens_lt_min"], rep["drop_tokens_gt_max"] = int(short.sum()), int(long_.sum())
    u = u[~short & ~long_]
    tech = u["prompt_uid"].isin(set(exclude_uids))
    rep["drop_technical_prompt"] = int(tech.sum())
    u = u[~tech].reset_index(drop=True)
    rep["n_survivors"] = int(len(u))
    cols = ["prompt_uid", "text", "n_rows", "max", "lang_p", "n_clip_tokens"]
    return u[cols].rename(columns={"max": "prompt_nsfw"}), rep


def select_candidates(survivors: pd.DataFrame, n: int, salt: str) -> pd.DataFrame:
    order = survivors["prompt_uid"].map(lambda x: _h(salt, x))
    return survivors.assign(_o=order).sort_values("_o").head(n).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------ GenEval -----------------------------------------
def geneval_template_regexes(records: Sequence[dict]) -> list[re.Pattern]:
    """One regex per GenEval prompt: its class names, colour words and count words become wildcards."""
    pats = set()
    for r in records:
        t = normalize(r["prompt"]).lower()
        words = set()
        for inc in r.get("include", []):
            words.add(str(inc.get("class", "")).lower())
            if inc.get("color"):
                words.add(str(inc["color"]).lower())
        words |= set(COUNT_WORDS)
        p = re.escape(t)
        for w in sorted((w for w in words if w), key=len, reverse=True):
            # the class may appear pluralised (counting prompts): allow an optional trailing s/es
            p = re.sub(rf"(?<![a-z]){re.escape(re.escape(w))}(e?s)?(?![a-z])", "WILDCARD", p)
        p = p.replace("WILDCARD", r"[a-z]+(?: [a-z]+)?")
        pats.add(f"^{p}$")
    return [re.compile(x) for x in sorted(pats)]


def template_hits(texts: Sequence[str], regexes: Sequence[re.Pattern]) -> np.ndarray:
    return np.array([any(r.match(normalize(t).lower()) for r in regexes) for t in texts], dtype=bool)


# ------------------------------------------------------------------ groups and split --------------------------------
def groups_by_cosine(emb: np.ndarray, thr: float = 0.95, block: int = 512) -> np.ndarray:
    u = _unit(emb)
    uf = UnionFind(len(u))
    for s in range(0, len(u), block):
        ii, jj = np.nonzero(u[s:s + block] @ u.T > thr)
        for i, j in zip(ii + s, jj):
            if i < j:
                uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(len(u))])


def group_and_split(cand: pd.DataFrame, emb: np.ndarray, geneval_emb: np.ndarray, template_mask: np.ndarray, pilot: str,
                    salt: str, dup_thr: float = 0.95, geneval_thr: float = 0.9) -> tuple[pd.DataFrame, dict]:
    if pilot not in PILOT_SIZES:
        raise ValueError(pilot)
    rep: dict = {"n_candidates": int(len(cand))}
    gidx = groups_by_cosine(emb, dup_thr)
    df = cand.copy()
    df["group_id"] = [df["prompt_uid"].iloc[i] for i in gidx]
    gmax = (_unit(emb) @ _unit(geneval_emb).T).max(axis=1) if len(geneval_emb) else np.zeros(len(df))
    df["geneval_max_cos"] = gmax
    df["geneval_cos_hit"] = gmax > geneval_thr
    df["geneval_template_hit"] = np.asarray(template_mask, bool)
    sizes = df.groupby("group_id").size()
    rep["n_groups"] = int(len(sizes))
    rep["group_size_hist"] = {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    rep["candidates_geneval_cos_hit"] = int(df["geneval_cos_hit"].sum())
    rep["candidates_geneval_template_hit"] = int(df["geneval_template_hit"].sum())
    bad = set(df.loc[df["geneval_cos_hit"] | df["geneval_template_hit"], "group_id"])
    rep["groups_dropped_geneval"] = len(bad)
    rep["prompts_dropped_geneval_groups"] = int(df["group_id"].isin(bad).sum())
    df["excluded_reason"] = np.where(df["group_id"].isin(bad), "geneval_contaminated_group", "")
    ok = df[df["excluded_reason"] == ""]
    order = sorted(ok["group_id"].unique(), key=lambda g: _h(salt, g))
    size = ok.groupby("group_id").size().to_dict()
    targets = [("calibration", CALIB)] + [(k, PILOT_SIZES[pilot][k]) for k in SPLIT_ORDER[1:]]
    assign, ti, left = {}, 0, targets[0][1]
    skipped_big = 0
    for g in order:
        while ti < len(targets) and left == 0:
            ti += 1
            left = targets[ti][1] if ti < len(targets) else 0
        if ti >= len(targets):
            assign[g] = "unused"
            continue
        if size[g] > left:
            assign[g] = "unused"
            skipped_big += 1
            continue
        assign[g] = targets[ti][0]
        left -= size[g]
    df["split"] = df["group_id"].map(assign).fillna("excluded")
    calib = df[df["split"] == "calibration"].sort_values("prompt_uid")
    df["throughput_subset"] = df["prompt_uid"].isin(set(calib["prompt_uid"].iloc[:SUBSET]))
    counts = df["split"].value_counts().to_dict()
    missing = {k: n - counts.get(k, 0) for k, n in targets if counts.get(k, 0) != n}
    rep.update({"groups_skipped_too_big_for_remaining": skipped_big, "split_counts": counts,
                "split_targets": dict(targets), "split_shortfall": missing,
                "cross_split_groups": int(df[df["split"].isin(SPLIT_ORDER)].groupby("group_id")["split"].nunique().gt(1).sum())})
    return df, rep


def manifest_rules(pilot: str, n_candidates: int, salt: str) -> dict:
    return {"norm_rule": NORM_RULE, "lang_rule": LANG_RULE, "nsfw_rule": "max prompt_nsfw over rows < 0.1",
            "token_rule": "4..60 content CLIP tokens (BOS/EOS excluded)", "group_rule": GROUP_RULE,
            "split_rule": SPLIT_RULE, "dup_threshold": 0.95, "geneval_threshold": 0.9,
            "geneval_template_rule": "full match after replacing include classes, colours and count words",
            "n_candidates": n_candidates, "salt": salt, "pilot": pilot, "pilot_sizes": PILOT_SIZES[pilot],
            "calibration": CALIB, "m9_subset": SUBSET}


def cross_split_full_pool(selected: pd.DataFrame, full_groups: dict[str, str]) -> list[dict]:
    """For full_pool_check: selected prompts whose FULL-pool component spans more than one split."""
    s = selected[selected["split"].isin(SPLIT_ORDER)].assign(full_group=lambda d: d["prompt_uid"].map(full_groups))
    bad = s.groupby("full_group")["split"].nunique()
    bad = set(bad[bad > 1].index)
    return s[s["full_group"].isin(bad)][["prompt_uid", "split", "full_group"]].to_dict("records")
