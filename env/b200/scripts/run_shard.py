"""One inference shard of the multi-model collector (B200 environment env/b200).

For each prompt of the shard and each bank: K seeds (apsp-v1 seed scheme, CPU float32 draw cast to the model dtype)
are denoised in one batch by the chosen generator; the predicted clean latent x0 is taken at the checkpoint steps
from the scheduler step itself (DDIM / Euler: pred_original_sample; flow matching: x0 = x_t - sigma * v), decoded
with the model's VAE, and scored with ImageReward (apsp.gpu.scoring.ImageRewardScorer: pinned BERT tokenizer,
missing keys refused) and PickScore (missing keys refused). The final latent is decoded and scored the same way.

Outputs (one directory per shard, resumable per prompt-bank):
  prompts.parquet, prompt_emb.npy (float16[N,768], SD1.5 CLIP text pooler = the v1 common prompt encoder)
  seeds.parquet     one row per (prompt, bank, seed): seed, init hash, final IR / PickScore, validity, PNG path + sha256
  previews.parquet  one row per (prompt, bank, seed, checkpoint): step, timestep, sigma, IR, PickScore, validity
  timings.parquet   seconds per prompt-bank (generation, decode, scoring)
  latents/{uid}_b{bank}.safetensors  "x0" [K, J, C, h, w] (x0 at the J checkpoints, unpacked for FLUX) and
                                     "final" [K, C, h, w], in the storage dtype (--latent-dtype: the model dtype by
                                     default, bf16 kept as bf16; or fp32). The decoded images and their scores are
                                     computed from exactly these stored tensors, so a re-decode reproduces them.
  images/{uid}_b{bank}_s{seed}.png (final), and _j{j}.png previews with --save-preview-png
  manifest.json     config, model/scorer provenance, environment, GPU, git commit, file sha256

  uv run --project env/b200 python env/b200/scripts/run_shard.py --model sd15 \
      --prompts prompts.parquet --split val --shard-index 0 --num-shards 64 --out /data/collect

Sampling defaults come from env/b200/models_b200.yaml and are model-card defaults for smoke runs, not study values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CFG_PATH = HERE.parent / "models_b200.yaml"
SEED_SCHEME = "apsp-v1-sha256-cpu"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_prompts(path: str, split: str | None, shard_index: int, num_shards: int, max_prompts: int | None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
    else:
        df = pd.DataFrame({"text": [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]})
    if "prompt_uid" not in df.columns:
        df["prompt_uid"] = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in df["text"]]
    if split:
        df = df[df["split"] == split]
    df = df.sort_values("prompt_uid").reset_index(drop=True).iloc[shard_index::num_shards]
    if max_prompts:
        df = df.head(max_prompts)
    return df[["prompt_uid", "text"]].reset_index(drop=True)


class X0Tap:
    """Wraps scheduler.step: records x0 (and timestep / sigma) after the configured numbers of completed steps."""

    def __init__(self, torch, scheduler, capture_steps):
        self.torch, self.s, self.capture = torch, scheduler, set(capture_steps)
        self.orig = scheduler.step
        scheduler.step = self.step
        self.reset()

    def reset(self):
        self.n, self.x0, self.meta = 0, {}, {}

    def step(self, model_output, timestep, sample, *args, **kwargs):
        torch = self.torch
        return_dict = kwargs.pop("return_dict", True)
        sigma = None
        if type(self.s).__name__.startswith("FlowMatch"):
            if self.s.step_index is None:
                self.s._init_step_index(timestep)
            sigma = self.s.sigmas[self.s.step_index]
        out = self.orig(model_output, timestep, sample, *args, return_dict=True, **kwargs)
        self.n += 1
        if self.n in self.capture:
            x0 = getattr(out, "pred_original_sample", None)
            if x0 is None:
                if sigma is None:
                    raise RuntimeError(f"{type(self.s).__name__} gives no pred_original_sample and is not flow matching")
                x0 = sample.float() - sigma.to(sample.device).float() * model_output.float()
            self.x0[self.n] = x0.detach().float()
            self.meta[self.n] = {"timestep": float(timestep), "sigma": None if sigma is None else float(sigma)}
        return out if return_dict else (out.prev_sample,)


class PickScorer:
    """PickScore v1, official usage (CLIP-H processor, logit_scale * cosine); refuses missing checkpoint keys."""

    def __init__(self, torch, device, c: dict):
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(c["processor_repo"], revision=c["processor_revision"])
        model, info = AutoModel.from_pretrained(c["model_repo"], revision=c["model_revision"], output_loading_info=True)
        if info["missing_keys"]:
            raise RuntimeError(f"PickScore checkpoint has missing keys: {info['missing_keys'][:10]}")
        self.model = model.eval().to(device).requires_grad_(False)
        self.provenance = {**c, "class": type(model).__name__, "missing_keys": list(info["missing_keys"]),
                           "unexpected_keys": list(info["unexpected_keys"]), "device": str(device)}

    def score(self, prompt: str, images: list) -> list[float]:
        torch = self.torch
        with torch.inference_mode():
            ii = self.processor(images=images, return_tensors="pt").to(self.device)
            ti = self.processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            ie = self.model.get_image_features(**ii)
            te = self.model.get_text_features(**ti)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            s = self.model.logit_scale.exp() * (te @ ie.T)[0]
        return s.float().cpu().tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="key in models_b200.yaml: sd15 sdxl sd35m flux_dev (tiny_* = CPU code test)")
    ap.add_argument("--prompts", required=True, help=".parquet (prompt_uid,text[,split]) / .jsonl / .txt (one prompt per line)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--banks", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=8, help="seeds per prompt-bank (one batch)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    ap.add_argument("--checkpoints", default=None, help="comma list of completed-step counts; default every steps/8 steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ir", action="store_true")
    ap.add_argument("--no-pickscore", action="store_true")
    ap.add_argument("--latent-dtype", choices=["model", "fp32"], default="model",
                    help="dtype of the stored x0/final latents (decode and scores use the stored tensor)")
    ap.add_argument("--no-prompt-emb", action="store_true")
    ap.add_argument("--save-preview-png", action="store_true")
    a = ap.parse_args(argv)

    import torch
    import diffusers
    from apsp.contracts.identity import derive_seed

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    spec = dict(cfg["generators"][a.model])
    steps = a.steps or spec["steps"]
    guidance = spec["guidance"] if a.guidance is None else a.guidance
    H, W = a.height or spec["height"], a.width or spec["width"]
    dtype_name = a.dtype or spec["dtype"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    store_dtype = dtype if a.latent_dtype == "model" else torch.float32
    from safetensors.torch import save_file
    ckpts = sorted({int(x) for x in a.checkpoints.split(",")}) if a.checkpoints else \
        sorted(set(range(max(1, steps // 8), steps + 1, max(1, steps // 8))) | {steps})
    if not all(1 <= c <= steps for c in ckpts):
        raise SystemExit(f"checkpoints must be in 1..{steps}")
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    out = Path(a.out) / a.model / f"shard{a.shard_index:05d}of{a.num_shards:05d}"
    for d in ("latents", "images", "parts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(a.prompts, a.split, a.shard_index, a.num_shards, a.max_prompts)
    print(f"[shard] {a.model} {len(prompts)} prompts x banks {a.banks} x k={a.k}, steps {steps}, ckpts {ckpts}, {dtype_name}, {H}x{W}")

    # ---- common prompt encoder (v1: SD1.5 CLIP text pooler, float16[N,768])
    if not a.no_prompt_emb and not (out / "prompt_emb.npy").exists():
        from apsp.data.prompts import _clip_functions
        pe = cfg["prompt_encoder"]
        _, embed = _clip_functions(pe["repo"], pe["revision"], device=str(device))
        np.save(out / "prompt_emb.npy", embed(list(prompts["text"])).astype(np.float16))
        torch.cuda.empty_cache() if device.type == "cuda" else None
    prompts.assign(emb_row=np.arange(len(prompts))).to_parquet(out / "prompts.parquet", index=False)

    # ---- generator
    kind = spec["pipeline"]
    kw = dict(revision=spec["revision"], torch_dtype=dtype)
    if kind == "StableDiffusionPipeline":
        kw.update(safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe = getattr(diffusers, kind).from_pretrained(spec["repo"], **kw)
    if spec.get("scheduler") == "DDIMScheduler":
        pipe.scheduler = diffusers.DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae_dtype = dtype
    if getattr(pipe.vae.config, "force_upcast", False) and dtype == torch.float16:
        pipe.vae.to(torch.float32)          # SDXL VAE overflows in fp16
        vae_dtype = torch.float32
    tap = X0Tap(torch, pipe.scheduler, ckpts)
    vsf = pipe.vae_scale_factor
    if kind == "FluxPipeline":
        C, h, w = pipe.transformer.config.in_channels // 4, 2 * (H // (vsf * 2)), 2 * (W // (vsf * 2))
    elif kind == "StableDiffusion3Pipeline":
        C, h, w = pipe.transformer.config.in_channels, H // vsf, W // vsf
    else:
        C, h, w = pipe.unet.config.in_channels, H // vsf, W // vsf

    def unpack(z):
        return pipe._unpack_latents(z, H, W, vsf) if kind == "FluxPipeline" else z

    def decode(z):
        vc = pipe.vae.config
        z = z.float()
        if getattr(vc, "latents_mean", None) is not None and getattr(vc, "latents_std", None) is not None:
            mean = torch.tensor(vc.latents_mean).view(1, -1, 1, 1).to(z)
            std = torch.tensor(vc.latents_std).view(1, -1, 1, 1).to(z)
            z = z * std / vc.scaling_factor + mean
        else:
            z = z / vc.scaling_factor + (getattr(vc, "shift_factor", None) or 0.0)
        with torch.inference_mode():
            img = pipe.vae.decode(z.to(device, vae_dtype), return_dict=False)[0].float()
        ok = torch.isfinite(img).flatten(1).all(1).cpu().tolist()
        return pipe.image_processor.postprocess(img, output_type="pil"), ok

    # ---- scorers
    ir = ps = None
    if not a.no_ir:
        from apsp.gpu.scoring import ImageRewardScorer
        c = cfg["scorers"]["imagereward"]
        ir = ImageRewardScorer(device, revision=c["revision"], tokenizer_revision=c["tokenizer_revision"])
    if not a.no_pickscore:
        ps = PickScorer(torch, device, cfg["scorers"]["pickscore"])

    def scores(prompt, ims, ok):
        idx = [i for i, o in enumerate(ok) if o]
        r_ir, r_ps = [float("nan")] * len(ims), [float("nan")] * len(ims)
        if idx and ir is not None:
            for i, v in zip(idx, ir.score(prompt, [ims[i] for i in idx])):
                r_ir[i] = v
        if idx and ps is not None:
            for i, v in zip(idx, ps.score(prompt, [ims[i] for i in idx])):
                r_ps[i] = v
        return r_ir, r_ps

    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    t_run = time.time()
    for _, row in prompts.iterrows():
        uid, text = row["prompt_uid"], row["text"]
        for b in a.banks:
            part = out / "parts" / f"{uid}_b{b}.json"
            if part.exists():
                continue
            seeds = [derive_seed(uid, b, s) for s in range(a.k)]
            x = [torch.randn((1, C, h, w), generator=torch.Generator("cpu").manual_seed(sd), dtype=torch.float32) for sd in seeds]
            init_sha = [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in x]
            lat = torch.cat(x).to(dtype)
            if kind == "FluxPipeline":
                lat = pipe._pack_latents(lat, a.k, C, h, w)
            call = dict(prompt=text, num_inference_steps=steps, guidance_scale=guidance, height=H, width=W,
                        latents=lat.to(device), num_images_per_prompt=a.k, output_type="latent")
            if spec.get("max_sequence_length"):
                call["max_sequence_length"] = spec["max_sequence_length"]
            tap.reset()
            sync(); t0 = time.time()
            with torch.inference_mode():
                final = pipe(**call).images
            sync(); t_gen = time.time() - t0
            if sorted(tap.x0) != ckpts:
                raise RuntimeError(f"captured steps {sorted(tap.x0)} != {ckpts} (scheduler ran {tap.n} steps)")
            # stored tensors = decoded tensors (no silent cast: bf16 stays bf16 in safetensors)
            x0 = torch.stack([unpack(tap.x0[k]) for k in ckpts], 1).to(store_dtype).contiguous()   # [K, J, C, h, w]
            fin = unpack(final).to(store_dtype).contiguous()                                      # [K, C, h, w]
            lat_ok_x0 = torch.isfinite(x0).flatten(2).all(2).cpu()           # [K, J]
            lat_ok_fin = torch.isfinite(fin).flatten(1).all(1).cpu().tolist()
            lat_file = out / "latents" / f"{uid}_b{b}.safetensors"
            save_file({"x0": x0.cpu(), "final": fin.cpu()}, str(lat_file.with_suffix(".tmp")),
                      metadata={"layout": "x0[K,J,C,h,w] final[K,C,h,w]", "checkpoints": json.dumps(ckpts),
                                "dtype": str(store_dtype).replace("torch.", ""), "model": a.model,
                                "prompt_uid": uid, "bank_id": str(b), "seeds": json.dumps(seeds)})
            os.replace(lat_file.with_suffix(".tmp"), lat_file)
            t_dec = t_sc = 0.0
            prev_rows, seed_rows = [], []
            for j, k in enumerate(ckpts):
                t0 = time.time(); ims, iok = decode(x0[:, j]); sync(); t_dec += time.time() - t0
                t0 = time.time(); r_ir, r_ps = scores(text, ims, iok); sync(); t_sc += time.time() - t0
                for s in range(a.k):
                    if a.save_preview_png:
                        ims[s].save(out / "images" / f"{uid}_b{b}_s{s}_j{j}.png")
                    prev_rows.append({"model": a.model, "prompt_uid": uid, "bank_id": b, "seed_idx": s, "j": j, "step": k,
                                      "timestep": tap.meta[k]["timestep"], "sigma": tap.meta[k]["sigma"],
                                      "ir": r_ir[s], "pickscore_raw": r_ps[s], "latent_ok": bool(lat_ok_x0[s, j]), "img_ok": bool(iok[s])})
            t0 = time.time(); ims, iok = decode(fin); sync(); t_dec += time.time() - t0
            t0 = time.time(); r_ir, r_ps = scores(text, ims, iok); sync(); t_sc += time.time() - t0
            for s in range(a.k):
                fn = out / "images" / f"{uid}_b{b}_s{s}.png"
                ims[s].save(fn)
                seed_rows.append({"model": a.model, "prompt_uid": uid, "bank_id": b, "seed_idx": s, "seed": seeds[s],
                                  "init_sha256": init_sha[s], "ir": r_ir[s], "pickscore_raw": r_ps[s], "latent_ok": bool(lat_ok_fin[s]),
                                  "img_ok": bool(iok[s]), "png": f"images/{fn.name}", "png_sha256": sha256_file(fn)})
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps({"seeds": seed_rows, "previews": prev_rows,
                                       "timing": {"model": a.model, "prompt_uid": uid, "bank_id": b, "gen_s": t_gen,
                                                  "decode_s": t_dec, "score_s": t_sc}}), encoding="utf-8")
            os.replace(tmp, part)
            print(f"[shard] {uid} b{b}: gen {t_gen:.1f}s decode {t_dec:.1f}s score {t_sc:.1f}s "
                  f"final IR {np.nanmean(r_ir):.3f} PS {np.nanmean(r_ps):.3f}", flush=True)

    # ---- tables + manifest
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "parts").glob("*.json"))]
    pd.DataFrame([r for p in parts for r in p["seeds"]]).to_parquet(out / "seeds.parquet", index=False)
    pd.DataFrame([r for p in parts for r in p["previews"]]).to_parquet(out / "previews.parquet", index=False)
    pd.DataFrame([p["timing"] for p in parts]).to_parquet(out / "timings.parquet", index=False)
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import importlib.metadata as md
    env = {p: md.version(p) for p in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors",
                                       "image-reward", "numpy", "pillow", "pyarrow")}
    env.update(python=sys.version.split()[0], platform=platform.platform(), torch_cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    files = {str(p.relative_to(out)).replace(os.sep, "/"): sha256_file(p)
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json" and "parts" not in p.parts}
    manifest = {"schema": "apsp.b200_shard/v1", "model": a.model, "spec": spec, "steps": steps, "guidance": guidance,
                "height": H, "width": W, "dtype": dtype_name, "latent_dtype": str(store_dtype).replace("torch.", ""),
                "latent_format": "safetensors", "pickscore_column": "pickscore_raw = logit_scale * cos(text, image), before softmax", "vae_dtype": str(vae_dtype).replace("torch.", ""),
                "checkpoints": ckpts, "k": a.k, "banks": a.banks, "seed_scheme": SEED_SCHEME, "latent_shape": [C, h, w],
                "scheduler": type(pipe.scheduler).__name__, "shard_index": a.shard_index, "num_shards": a.num_shards,
                "prompts_file": str(a.prompts), "split": a.split, "n_prompts": len(prompts), "n_prompt_banks": len(parts),
                "imagereward": None if ir is None else ir.provenance, "pickscore": None if ps is None else ps.provenance,
                "prompt_encoder": None if a.no_prompt_emb else cfg["prompt_encoder"],
                "git_commit": commit, "env": env, "seconds": time.time() - t_run, "files_sha256": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"[shard] done: {out} ({len(parts)} prompt-banks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
