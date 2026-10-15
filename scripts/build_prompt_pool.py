"""Build the prompt pool v1 proposal (apsp.data.pool_v1) from pinned local inputs, with resumable stages.

Never downloads anything. Inputs must match the pinned sha256 values in configs/prompt_pool.yaml. Stages cache their
outputs under <work>/ (keyed by the input hashes), so a rerun resumes. CPU only (CUDA hidden).

  python scripts/build_prompt_pool.py --config configs/prompt_pool.yaml --external D:/Workspace/DMRL_external \
      --work data/prompts/work --out data/prompts --report reports/prompt_pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from apsp.contracts.hashing import file_sha256  # noqa: E402
from apsp.data import pool_v1  # noqa: E402
from apsp.data.prompts import normalize, prompt_uid  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_inputs(cfg: dict, ext: Path) -> dict:
    out = {}
    for key, spec in cfg["inputs"].items():
        p = ext / spec["local_path"]
        if not p.exists():
            raise SystemExit(f"missing input {key}: {p} (download it from {spec['url']} yourself)")
        sha = file_sha256(p)
        if sha != spec["sha256"]:
            raise SystemExit(f"{key}: sha256 {sha} != pinned {spec['sha256']}")
        out[key] = {"path": str(p), "sha256": sha, "bytes": p.stat().st_size}
    return out


def technical_uids() -> list[str]:
    uids = set()
    for f in ("configs/technical_smoke.yaml", "configs/drift_diagnostic.yaml"):
        d = yaml.safe_load((ROOT / f).read_text(encoding="utf-8"))
        for p in d.get("prompts", []):
            uids.add(prompt_uid(normalize(p)))
        if d.get("repro", {}).get("prompt"):
            uids.add(prompt_uid(normalize(d["repro"]["prompt"])))
    return sorted(uids)


def clip(cfg: dict, threads: int):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    torch.set_num_threads(threads)
    e = cfg["encoder"]
    tok = CLIPTokenizer.from_pretrained(e["repo"], subfolder="tokenizer", revision=e["revision"])
    enc = None

    def count(texts):
        return [len(tok(t, add_special_tokens=False, truncation=False)["input_ids"]) for t in texts]

    def embed(texts, bs=64):
        nonlocal enc
        if enc is None:
            enc = CLIPTextModel.from_pretrained(e["repo"], subfolder="text_encoder", revision=e["revision"]).eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                b = tok(list(texts[i:i + bs]), padding="max_length", max_length=tok.model_max_length, truncation=True,
                        return_tensors="pt")
                out.append(enc(**b).pooler_output.float().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)
    return count, embed


def fasttext_fn(path: str):
    import fasttext
    m = fasttext.load_model(path)

    def lang(texts, chunk=100_000):
        labels, probs = [], []
        for i in range(0, len(texts), chunk):
            lab, pr = m.predict(list(texts[i:i + chunk]), k=1)
            labels += [x[0] if x else "" for x in lab]
            probs += [float(p[0]) if len(p) else 0.0 for p in pr]
        return labels, probs
    return lang


def cached_embed(embed, texts, uids, cache: Path, chunk=256) -> np.ndarray:
    """Embeddings saved per chunk so an interrupted run resumes."""
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in range(0, len(texts), chunk):
        key = hashlib.sha256("|".join(uids[s:s + chunk]).encode()).hexdigest()[:16]
        f = cache / f"emb_{s:07d}_{key}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        t0 = time.time()
        e = embed(texts[s:s + chunk])
        np.save(f, e)
        parts.append(e)
        log(f"embedded {s + len(e)}/{len(texts)} ({len(e) / (time.time() - t0):.1f}/s)")
    return np.concatenate(parts) if parts else np.zeros((0, 768), np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompt_pool.yaml")
    ap.add_argument("--external", required=True)
    ap.add_argument("--work", default="data/prompts/work")
    ap.add_argument("--out", default="data/prompts")
    ap.add_argument("--report", default="reports/prompt_pool")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    ext, work, out, rpt = Path(a.external), Path(a.work), Path(a.out), Path(a.report)
    for d in (work, out, rpt):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ext / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    inputs = check_inputs(cfg, ext)
    log("inputs verified")
    count, embed = clip(cfg, a.threads)
    tech = technical_uids()
    key = hashlib.sha256(json.dumps([inputs[k]["sha256"] for k in sorted(inputs)] + [cfg["encoder"], tech,
                                                                                     pool_v1.LANG_RULE]).encode()).hexdigest()[:16]
    surv_p, frep_p = work / f"survivors_{key}.parquet", work / f"filter_report_{key}.json"
    if surv_p.exists() and frep_p.exists():
        surv, frep = pd.read_parquet(surv_p), json.loads(frep_p.read_text(encoding="utf-8"))
        log(f"filter stage resumed: {len(surv)} survivors")
    else:
        meta = pd.read_parquet(inputs["diffusiondb_metadata"]["path"], columns=["prompt", "prompt_nsfw"])
        log(f"metadata rows {len(meta)}")
        surv, frep = pool_v1.filter_stage(meta, lang_fn=fasttext_fn(inputs["fasttext_lid176"]["path"]),
                                          count_tokens=count, exclude_uids=tech)
        surv.to_parquet(surv_p, index=False)
        frep_p.write_text(json.dumps(frep, indent=1), encoding="utf-8")
        log(f"filter stage done: {len(surv)} survivors")
    salt, n_cand = cfg["salt"], int(cfg["n_candidates"])
    cand = pool_v1.select_candidates(surv, n_cand, salt)
    emb = cached_embed(embed, list(cand["text"]), list(cand["prompt_uid"]), work / f"emb_{key}_{salt}")
    gen = [json.loads(x) for x in Path(inputs["geneval_manifest"]["path"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    g_texts = [g["prompt"] for g in gen]
    g_emb = cached_embed(embed, g_texts, [prompt_uid(normalize(t)) for t in g_texts], work / f"emb_geneval_{key}")
    regs = pool_v1.geneval_template_regexes(gen)
    tmask = pool_v1.template_hits(list(cand["text"]), regs)
    table, srep = pool_v1.group_and_split(cand, emb, g_emb, tmask, cfg["pilot"], salt)
    # ---- outputs
    sel = table[table["split"].isin(pool_v1.SPLIT_ORDER)].copy()
    sel["source"] = "diffusiondb-2m-metadata"
    sealed = sel[sel["split"] == "val_ext"][["prompt_uid", "text", "group_id", "n_clip_tokens"]].sort_values("prompt_uid")
    sealed.to_parquet(out / "val_ext_sealed.parquet", index=False)
    pool = sel[["prompt_uid", "text", "split", "source", "group_id", "n_clip_tokens", "throughput_subset", "lang_p",
                "prompt_nsfw", "geneval_max_cos"]].copy()
    pool.loc[pool["split"] == "val_ext", "text"] = None          # sealed: text only in val_ext_sealed.parquet
    pool = pool.sort_values(["split", "prompt_uid"]).reset_index(drop=True)
    pool.to_parquet(out / "prompt_pool.parquet", index=False)
    manifest = {"schema": "apsp.prompt_pool/v1", "status": "PROPOSAL - research approves pool and split before collection",
                "rules": pool_v1.manifest_rules(cfg["pilot"], n_cand, salt), "inputs": {k: {kk: v[kk] for kk in ("sha256", "bytes")}
                                                                                       | {"url": cfg["inputs"][k]["url"],
                                                                                          "revision": cfg["inputs"][k].get("revision")}
                                                                                       for k, v in inputs.items()},
                "encoder": cfg["encoder"], "technical_uids_excluded": tech, "filter_report": frep, "split_report": srep,
                "deviation": "sampled-grouping-v1: near-duplicate groups are exact among the candidates only; full-pool "
                             "grouping to be checked on GPU (full_pool_check) before freezing",
                "files": {"prompt_pool.parquet": file_sha256(out / "prompt_pool.parquet"),
                          "val_ext_sealed.parquet": file_sha256(out / "val_ext_sealed.parquet")},
                "val_ext_sealed": {"n": int(len(sealed)),
                                   "uid_list_sha256": hashlib.sha256("\n".join(sealed["prompt_uid"]).encode()).hexdigest()}}
    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    # ---- review copies (texts of val_ext are not exported)
    pool.to_csv(rpt / "prompt_pool_review.csv", index=False, encoding="utf-8")
    exc = table[table["excluded_reason"] != ""][["prompt_uid", "text", "group_id", "geneval_max_cos", "geneval_cos_hit",
                                                  "geneval_template_hit", "excluded_reason"]]
    exc.to_csv(rpt / "excluded_candidates.csv", index=False, encoding="utf-8")
    (rpt / "pool_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(json.dumps(srep["split_counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
