"""Validate one shard directory written by run_shard.py.

  uv run --locked python scripts/validate_shard.py OUT/<model>/shardXXXXXofYYYYY [--redecode 1] [--device cuda]
      [--require-scores] [--out report.json]

Checks:
  - manifest.json present; every file listed has the recorded sha256; no unlisted file (parts/ excluded);
  - prompts.parquet / prompt_emb.npy (float16[N,768]) consistent;
  - seeds.parquet: one row per (prompt, bank, seed); previews.parquet: every seed has exactly the manifest checkpoints;
  - latents/*.safetensors: keys x0 [K,J,C,h,w] and final [K,C,h,w], dtype = manifest latent_dtype, metadata
    checkpoints = manifest checkpoints;
  - PNG for every seed, sha256 as recorded;
  - latent_ok / img_ok all true; IR and pickscore_raw finite for valid images when the scorer ran (NaN otherwise);
  - --redecode N: reload the VAE of the manifest model and decode the stored final latents of N prompt-banks; the
    result must reproduce the stored PNGs (max |pixel diff| <= --pixel-tol, 0 expected on CPU).
Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard")
    ap.add_argument("--redecode", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pixel-tol", type=int, default=0)
    ap.add_argument("--require-scores", action="store_true", help="fail if IR or PickScore did not run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = Path(a.shard)
    E: list[str] = []
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    rep = {"shard": str(d), "model": m["model"], "checks": {}}

    # files and hashes
    listed = m["files_sha256"]
    bad = [f for f, h in listed.items() if not (d / f).exists() or sha256_file(d / f) != h]
    present = {str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*")
               if p.is_file() and p.name != "manifest.json" and "parts" not in p.relative_to(d).parts}
    unlisted = sorted(present - set(listed))
    rep["checks"]["files"] = {"listed": len(listed), "hash_mismatch_or_missing": bad[:20], "unlisted": unlisted[:20]}
    if bad or unlisted:
        E.append("file hashes / listing")

    K, J, ck = m["k"], len(m["checkpoints"]), m["checkpoints"]
    C, h, w = m["latent_shape"]
    prompts = pd.read_parquet(d / "prompts.parquet")
    seeds = pd.read_parquet(d / "seeds.parquet")
    prev = pd.read_parquet(d / "previews.parquet")
    n_pb = m["n_prompt_banks"]
    if (d / "prompt_emb.npy").exists():
        emb = np.load(d / "prompt_emb.npy")
        if emb.dtype != np.float16 or emb.shape != (len(prompts), 768) or not np.isfinite(emb).all():
            E.append(f"prompt_emb {emb.dtype}{emb.shape}")
    exp_pb = len(prompts) * len(m["banks"])
    rows_ok = (n_pb == exp_pb and len(seeds) == exp_pb * K and len(prev) == exp_pb * K * J
               and not seeds.duplicated(["prompt_uid", "bank_id", "seed_idx"]).any())
    per_seed = prev.groupby(["prompt_uid", "bank_id", "seed_idx"])["step"].apply(lambda s: sorted(s.tolist()))
    ck_ok = bool((per_seed.apply(lambda s: s == ck)).all()) and len(per_seed) == len(seeds)
    rep["checks"]["tables"] = {"prompt_banks": n_pb, "expected": exp_pb, "seeds_rows": len(seeds), "preview_rows": len(prev),
                               "all_checkpoints_per_seed": ck_ok}
    if not rows_ok:
        E.append("table row counts")
    if not ck_ok:
        E.append("checkpoints per seed")

    # latents
    from safetensors import safe_open
    lat_bad = []
    for (uid, b), _ in seeds.groupby(["prompt_uid", "bank_id"]):
        f = d / "latents" / f"{uid}_b{b}.safetensors"
        try:
            with safe_open(str(f), framework="pt") as so:
                md = so.metadata()
                x0, fin = so.get_tensor("x0"), so.get_tensor("final")
            if (list(x0.shape) != [K, J, C, h, w] or list(fin.shape) != [K, C, h, w]
                    or str(x0.dtype).replace("torch.", "") != m["latent_dtype"] or json.loads(md["checkpoints"]) != ck):
                lat_bad.append(f"{f.name}: {tuple(x0.shape)} {x0.dtype} {md.get('checkpoints')}")
        except Exception as e:  # noqa: BLE001
            lat_bad.append(f"{f.name}: {type(e).__name__}: {e}")
    rep["checks"]["latents"] = {"files": n_pb, "bad": lat_bad[:20], "dtype": m["latent_dtype"], "shape_x0": [K, J, C, h, w]}
    if lat_bad:
        E.append("latent files")

    # images
    png_bad = [r.png for r in seeds.itertuples() if not (d / r.png).exists() or sha256_file(d / r.png) != r.png_sha256]
    rep["checks"]["png"] = {"n": len(seeds), "bad": png_bad[:20]}
    if png_bad:
        E.append("png")

    # validity and scores
    v = {"latent_ok_final": int(seeds.latent_ok.sum()), "img_ok_final": int(seeds.img_ok.sum()),
         "latent_ok_previews": int(prev.latent_ok.sum()), "img_ok_previews": int(prev.img_ok.sum()),
         "n_final": len(seeds), "n_previews": len(prev)}
    rep["checks"]["validity"] = v
    if v["latent_ok_final"] != len(seeds) or v["img_ok_final"] != len(seeds) or v["img_ok_previews"] != len(prev):
        E.append("invalid latents/images")
    for col, ran in (("ir", m["imagereward"] is not None), ("pickscore_raw", m["pickscore"] is not None)):
        allv = pd.concat([seeds[[col, "img_ok"]], prev[[col, "img_ok"]]])
        fin_ok = bool(np.isfinite(allv.loc[allv.img_ok, col]).all())
        rep["checks"][col] = {"ran": ran, "finite_on_valid": fin_ok if ran else None,
                              "mean_final": float(seeds[col].mean()) if ran else None}
        if ran and not fin_ok:
            E.append(f"{col} not finite")
        if not ran and a.require_scores:
            E.append(f"{col} did not run")
        if not ran and allv[col].notna().any():
            E.append(f"{col} has values but the scorer did not run")

    # re-decode stored latents -> PNG
    if a.redecode:
        import torch
        import diffusers
        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image
        from safetensors.torch import load_file
        spec = m["spec"]
        vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[m["vae_dtype"]]
        vae = diffusers.AutoencoderKL.from_pretrained(spec["repo"], subfolder="vae", revision=spec["revision"],
                                                      torch_dtype=vdt).to(a.device).eval()
        proc = VaeImageProcessor()
        vc = vae.config
        diffs = []
        for (uid, b), g in list(seeds.groupby(["prompt_uid", "bank_id"]))[:a.redecode]:
            z = load_file(str(d / "latents" / f"{uid}_b{b}.safetensors"))["final"].float().to(a.device)
            if getattr(vc, "latents_mean", None) is not None and getattr(vc, "latents_std", None) is not None:
                mean = torch.tensor(vc.latents_mean).view(1, -1, 1, 1).to(z)
                std = torch.tensor(vc.latents_std).view(1, -1, 1, 1).to(z)
                z = z * std / vc.scaling_factor + mean
            else:
                z = z / vc.scaling_factor + (getattr(vc, "shift_factor", None) or 0.0)
            with torch.inference_mode():
                img = vae.decode(z.to(vdt), return_dict=False)[0].float()
            pil = proc.postprocess(img, output_type="pil")
            for r in g.sort_values("seed_idx").itertuples():
                a_ = np.asarray(pil[r.seed_idx], dtype=np.int16)
                b_ = np.asarray(Image.open(d / r.png).convert("RGB"), dtype=np.int16)
                diffs.append(int(np.abs(a_ - b_).max()) if a_.shape == b_.shape else 999)
        rep["checks"]["redecode"] = {"prompt_banks": a.redecode, "images": len(diffs), "max_pixel_diff": max(diffs),
                                     "tolerance": a.pixel_tol}
        if max(diffs) > a.pixel_tol:
            E.append("re-decode does not reproduce the PNGs")

    rep["errors"], rep["pass"] = E, not E
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    print(s)
    print("PASS" if not E else "FAIL: " + "; ".join(E))
    return 0 if not E else 1


if __name__ == "__main__":
    sys.exit(main())
