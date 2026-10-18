"""CPU checks of the B200 collector environment with the real scorer weights (no GPU needed).

  uv run --locked python scripts/cpu_checks.py roundtrip  --out r.json
  uv run --locked python scripts/cpu_checks.py ir         --out r.json [--v1-ref DIR]
  uv run --locked python scripts/cpu_checks.py pickscore  --out r.json [--v1-ref DIR]
  uv run --locked python scripts/cpu_checks.py prompt_emb --out r.json [--v1-ref DIR]

roundtrip   tensor -> safetensors file -> tensor for fp32 / fp16 / bf16 (values and dtype bit-exact), including bf16
            values outside the fp16 range (shows why bf16 latents must not be cast to fp16 for storage).
ir          ImageReward through apsp.gpu.scoring.ImageRewardScorer: tokenizer from the pinned snapshot (vocab 30524,
            [ENC] 30523, [DEC] 30522, file sha256), missing keys = [] (enforced), unexpected keys reported,
            checkpoint sha256, finite scores, image -> score mapping (permutation, duplicates, batch = single),
            equality with the package's own ImageReward.score(); with --v1-ref, scores of v1 PNGs vs the IR recorded
            in the v1 bank (RTX 4090, torch 2.4.0, transformers 4.38.2).
pickscore   PickScore through run_shard.PickScorer: missing keys = [] (enforced), stored value = raw
            logit_scale * cos (recomputed independently), not a softmax; mapping checks as above.
prompt_emb  common prompt encoder (SD1.5 CLIP text pooler, pinned revision): shape, finiteness; with --v1-ref,
            agreement with the v1 bank embeddings of the same prompts.
Exit 0 only if every assertion of the mode passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = yaml.safe_load((HERE.parent / "models_b200.yaml").read_text(encoding="utf-8"))
IR_CKPT_SHA256_V1 = "d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6"   # v1 HF cache blob name


def synthetic_images():
    from PIL import Image, ImageDraw
    ims = []
    for i, (bg, fg) in enumerate([("white", "red"), ("black", "yellow"), ("skyblue", "green"), ("gray", "blue")]):
        im = Image.new("RGB", (512, 512), bg)
        d = ImageDraw.Draw(im)
        (d.rectangle if i % 2 == 0 else d.ellipse)([128, 128, 384, 384], fill=fg)
        ims.append(im)
    rng = np.random.default_rng(0)
    ims.append(Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)))
    return ims


def mapping_checks(score, prompt, ims, tol):
    """score(prompt, list[PIL]) -> list[float]. Checks finiteness and that each score belongs to its image."""
    base = np.array(score(prompt, ims), dtype=np.float64)
    perm = [3, 0, 4, 1, 2]
    permuted = np.array(score(prompt, [ims[i] for i in perm]))
    single = np.array([score(prompt, [im])[0] for im in ims])
    dup = np.array(score(prompt, [ims[0], ims[0], ims[1]]))
    r = {"scores": base.tolist(), "finite": bool(np.isfinite(base).all()),
         "perm_max_abs_diff": float(np.abs(permuted - base[perm]).max()),
         "single_vs_batch_max_abs_diff": float(np.abs(single - base).max()),
         "duplicate_abs_diff": float(abs(dup[0] - dup[1])), "distinct_images_distinct_scores": len(set(np.round(base, 4))) == len(base)}
    r["pass"] = (r["finite"] and r["perm_max_abs_diff"] < tol and r["single_vs_batch_max_abs_diff"] < tol
                 and r["duplicate_abs_diff"] < tol and r["distinct_images_distinct_scores"])
    return r


def v1_ref(path):
    p = Path(path)
    ref = json.loads((p / "v1_ref.json").read_text(encoding="utf-8"))
    return p, ref


def mode_roundtrip(a):
    import torch
    from safetensors.torch import load_file, save_file
    g = torch.Generator().manual_seed(0)
    res, ok = {}, True
    with tempfile.TemporaryDirectory() as d:
        for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
            x0 = (torch.randn(4, 3, 16, 8, 8, generator=g) * 3).to(dt)
            fin = torch.randn(4, 16, 8, 8, generator=g).to(dt)
            if dt == torch.bfloat16:
                x0[0, 0, 0, 0, :4] = torch.tensor([1e5, -3e6, 7e-8, 65504.0 * 2])   # outside / at the fp16 limits
            f = Path(d) / f"t_{name}.safetensors"
            save_file({"x0": x0, "final": fin}, str(f), metadata={"dtype": name})
            back = load_file(str(f))
            same = all(back[k].dtype == t.dtype and back[k].shape == t.shape and torch.equal(back[k], t)
                       for k, t in (("x0", x0), ("final", fin)))
            res[name] = {"bit_exact": same, "dtype_back": str(back["x0"].dtype), "bytes": f.stat().st_size}
            ok &= same
        bf = torch.tensor([1e5, -3e6, 7e-8], dtype=torch.bfloat16)
        as16 = bf.to(torch.float16)
        res["bf16_to_fp16_would_lose"] = {"bf16": bf.float().tolist(), "fp16": as16.float().tolist(),
                                          "finite_after_cast": torch.isfinite(as16).tolist()}
        # numpy cannot hold bf16: show that the old npy path is not an option for bf16
        try:
            np.save(Path(d) / "x.npy", bf.numpy())
            res["numpy_bf16"] = "unexpectedly supported"
        except TypeError as e:
            res["numpy_bf16"] = f"TypeError: {e}"
    res["pass"] = bool(ok)
    return res


def mode_ir(a):
    import torch
    from apsp.gpu.scoring import ImageRewardScorer
    c = CFG["scorers"]["imagereward"]
    sc = ImageRewardScorer(torch.device("cpu"), revision=c["revision"], tokenizer_revision=c["tokenizer_revision"])
    tok = sc.model.blip.tokenizer
    prov = sc.provenance
    r = {"provenance": prov,
         "tokenizer": {"name_or_path": str(tok.name_or_path), "vocab": len(tok), "enc_token_id": tok.enc_token_id,
                       "dec_token_id": tok.convert_tokens_to_ids("[DEC]")}}
    r["tokenizer_ok"] = (r["tokenizer"]["vocab"] == 30524 and r["tokenizer"]["enc_token_id"] == 30523
                         and r["tokenizer"]["dec_token_id"] == 30522 and prov["bert_tokenizer"]["revision"] == c["tokenizer_revision"])
    r["missing_keys_empty"] = prov["missing_keys"] == []
    r["ckpt_sha256_matches_v1"] = prov["ckpt_sha256"] == IR_CKPT_SHA256_V1
    ims = synthetic_images()
    prompt = "a red square on a white background"
    r["mapping"] = mapping_checks(sc.score, prompt, ims, tol=1e-4)
    official = [float(sc.model.score(prompt, im)) for im in ims]
    r["official_score_max_abs_diff"] = float(np.abs(np.array(official) - np.array(r["mapping"]["scores"])).max())
    ok = r["tokenizer_ok"] and r["missing_keys_empty"] and r["ckpt_sha256_matches_v1"] and r["mapping"]["pass"] \
        and r["official_score_max_abs_diff"] < 1e-4
    if a.v1_ref:
        from PIL import Image
        p, ref = v1_ref(a.v1_ref)
        rows = []
        for im in ref["images"]:
            f = p / im["png"]
            assert hashlib.sha256(f.read_bytes()).hexdigest() == im["image_sha"], f
            s = sc.score(ref["prompts"][im["prompt_uid"]], [Image.open(f).convert("RGB")])[0]
            rows.append({"png": im["png"], "ir_v1_4090": im["ir_v1"], "ir_now_cpu": s, "abs_diff": abs(s - im["ir_v1"])})
        md = max(x["abs_diff"] for x in rows)
        r["v1_cross_check"] = {"n": len(rows), "max_abs_diff": md, "rows": rows, "tolerance": 2e-3}
        ok = ok and md < 2e-3
    r["pass"] = bool(ok)
    return r


def mode_pickscore(a):
    import torch
    from run_shard import PickScorer
    ps = PickScorer(torch, torch.device("cpu"), CFG["scorers"]["pickscore"])
    ims = synthetic_images()
    prompt = "a red square on a white background"
    r = {"provenance": ps.provenance, "missing_keys_empty": ps.provenance["missing_keys"] == []}
    r["mapping"] = mapping_checks(ps.score, prompt, ims, tol=1e-3)
    # independent recomputation of the raw value (no softmax)
    with torch.inference_mode():
        m, pr = ps.model, ps.processor
        ii = pr(images=ims, return_tensors="pt")
        ti = pr(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt")
        ie = torch.nn.functional.normalize(m.get_image_features(**ii), dim=-1)
        te = torch.nn.functional.normalize(m.get_text_features(**ti), dim=-1)
        cos = (te @ ie.T)[0]
        raw = (m.logit_scale.exp() * cos).tolist()
        r["logit_scale_exp"] = float(m.logit_scale.exp())
    stored = np.array(r["mapping"]["scores"])
    r["raw_recomputed_max_abs_diff"] = float(np.abs(stored - np.array(raw)).max())
    r["cos"] = cos.tolist()
    r["softmax_over_images_for_reference_only"] = torch.softmax(torch.tensor(stored), 0).tolist()
    r["stored_is_raw_not_softmax"] = bool(abs(stored.sum() - 1.0) > 1e-3 and (stored > 1.0).any())
    ok = r["missing_keys_empty"] and r["mapping"]["pass"] and r["raw_recomputed_max_abs_diff"] < 1e-3 and r["stored_is_raw_not_softmax"]
    if a.v1_ref:
        from PIL import Image
        p, ref = v1_ref(a.v1_ref)
        by = {}
        for im in ref["images"]:
            by.setdefault(im["prompt_uid"], []).append(im)
        rows = []
        for uid, lst in by.items():
            s = ps.score(ref["prompts"][uid], [Image.open(p / x["png"]).convert("RGB") for x in lst])
            rows += [{"png": x["png"], "pickscore_raw": v, "ir_v1": x["ir_v1"]} for x, v in zip(lst, s)]
        r["v1_images"] = {"rows": rows, "finite": bool(np.isfinite([x["pickscore_raw"] for x in rows]).all()),
                          "note": "no PickScore was recorded in v1; this only shows finite raw scores on real SD1.5 images"}
        ok = ok and r["v1_images"]["finite"]
    r["pass"] = bool(ok)
    return r


def mode_prompt_emb(a):
    from apsp.data.prompts import _clip_functions
    pe = CFG["prompt_encoder"]
    _, embed = _clip_functions(pe["repo"], pe["revision"], device="cpu")
    texts = ["a red bicycle leaning against a brick wall", "a watercolor painting of a fox"]
    e = embed(texts)
    r = {"shape": list(e.shape), "finite": bool(np.isfinite(e).all())}
    ok = r["shape"] == [2, 768] and r["finite"]
    if a.v1_ref:
        _, ref = v1_ref(a.v1_ref)
        uids = list(ref["prompts"])
        now = embed([ref["prompts"][u] for u in uids]).astype(np.float16).astype(np.float32)
        v1 = np.array([ref["prompt_emb_v1"][u] for u in uids], dtype=np.float32)
        cos = (now * v1).sum(1) / np.linalg.norm(now, axis=1) / np.linalg.norm(v1, axis=1)
        r["v1_cross_check"] = {"n": len(uids), "max_abs_diff": float(np.abs(now - v1).max()), "min_cosine": float(cos.min()),
                               "tolerance": "max_abs_diff < 1e-2 (float16 storage, GPU vs CPU)"}
        ok = ok and r["v1_cross_check"]["max_abs_diff"] < 1e-2
    r["pass"] = bool(ok)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["roundtrip", "ir", "pickscore", "prompt_emb"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1-ref", default=None, help="directory with v1_ref.json + PNGs from the v1 bank")
    a = ap.parse_args(argv)
    r = {"mode": a.mode, **globals()[f"mode_{a.mode}"](a)}
    Path(a.out).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k not in ("provenance",)}, indent=1, default=str)[:6000])
    print("PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
