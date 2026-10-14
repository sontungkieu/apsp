"""Generator loading, scheduler dump and provenance (CC-1 / M1).

We load the components of the SD1.5 mirror with diffusers (the same components FKDStableDiffusion wraps:
it subclasses StableDiffusionPipeline) and drive them step-by-step in ``apsp.gpu.engine``. The DDIM
scheduler is built exactly like the base (``DDIMScheduler.from_config(pipe.scheduler.config)``).
Revision and file SHA256 values come from the actual snapshot on disk; nothing is invented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from apsp.contracts.hashing import file_sha256, sha256_hex
from apsp.gpu.runtime import import_torch

HASHED_SUBFOLDERS = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")


def resolve_revision(repo: str, revision: Optional[str]) -> str:
    """Resolve a branch/tag to the exact commit SHA (network). A 40-hex revision is returned unchanged."""
    if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, revision=revision or "main").sha


def snapshot_hashes(snapshot_dir: Path) -> dict:
    """sha256 of every file actually present in the used subfolders + model_index.json (relative paths)."""
    out = {}
    for sub in HASHED_SUBFOLDERS:
        d = snapshot_dir / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    out[p.relative_to(snapshot_dir).as_posix()] = file_sha256(p)
    mi = snapshot_dir / "model_index.json"
    if mi.exists():
        out["model_index.json"] = file_sha256(mi)
    return out


def scheduler_dump(scheduler, steps: int) -> dict:
    torch = import_torch()
    scheduler.set_timesteps(steps)
    ts = [int(t) for t in scheduler.timesteps]
    ac = scheduler.alphas_cumprod
    fac = scheduler.final_alpha_cumprod
    fac = float(fac) if not isinstance(fac, torch.Tensor) else float(fac.item())
    ratio = scheduler.config.num_train_timesteps // steps
    # terminal update: x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps_hat with a_prev = final_alpha_cumprod
    t_last = ts[-1]
    a_prev = fac if t_last - ratio < 0 else float(ac[t_last - ratio])
    return {
        "class": type(scheduler).__name__, "config": {k: (list(v) if isinstance(v, tuple) else v)
                                                      for k, v in dict(scheduler.config).items()},
        "timesteps": ts, "alphas_cumprod_at_timesteps": [float(ac[t]) for t in ts],
        "final_alpha_cumprod": fac, "prediction_type": scheduler.config.prediction_type,
        "terminal_update": {"t_last": t_last, "alpha_prev": a_prev, "coef_x0": a_prev ** 0.5,
                            "coef_eps": (1 - a_prev) ** 0.5,
                            "note": "delivered latent != preview x0 at T unless alpha_prev == 1"},
    }


@dataclass
class GeneratorBundle:
    """Thin adapter used by the engine; the fake backend implements the same surface."""

    pipe: object
    unet: object
    vae: object
    vae_preview: object
    text_encoder: object
    tokenizer: object
    scheduler: object
    image_processor: object
    device: object
    dtype: object
    provenance: dict = field(default_factory=dict)
    backend: str = "diffusers"

    @property
    def latent_channels(self) -> int:
        return int(self.unet.config.in_channels)

    @property
    def vae_scaling(self) -> float:
        return float(self.vae.config.scaling_factor)

    def encode(self, prompt: str, negative_prompt: str = ""):
        """(cond [1,77,768], uncond [1,77,768]) through the pipeline's own encode_prompt (base path)."""
        cond, uncond = self.pipe.encode_prompt(prompt, self.device, 1, True, negative_prompt)
        return cond, uncond

    def pooler(self, prompt: str):
        torch = import_torch()
        tok = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        with torch.inference_mode():
            out = self.text_encoder(tok.input_ids.to(self.device))
        return out.pooler_output[0].float().cpu().numpy()

    def unet_eps(self, x, t, emb):
        return self.unet(x, t, encoder_hidden_states=emb, return_dict=False)[0]

    def decode(self, z_scaled, preview: bool):
        """VAE decode of latents already divided by the scaling factor; preview may use a separate FP32 instance."""
        vae = self.vae_preview if preview else self.vae
        return vae.decode(z_scaled.to(vae.dtype), return_dict=False)[0]

    def to_pil(self, images):
        return self.image_processor.postprocess(images, output_type="pil", do_denormalize=[True] * images.shape[0])


def load_generator(cfg: dict, device, *, revision: Optional[str] = None, preview_dtype: Optional[str] = None,
                   local_files_only: bool = False) -> GeneratorBundle:
    torch = import_torch()
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from huggingface_hub import snapshot_download

    repo = cfg["model_repo"]
    rev = resolve_revision(repo, revision or cfg.get("model_revision")) if not local_files_only else (revision or cfg.get("model_revision"))
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[cfg.get("unet_dtype", "fp16")]
    variant = cfg.get("weights_variant")  # None follows the base (fp32 files cast to fp16)
    pipe = StableDiffusionPipeline.from_pretrained(repo, revision=rev, torch_dtype=dtype, safety_checker=None,
                                                   feature_extractor=None, requires_safety_checker=False,
                                                   variant=variant, local_files_only=local_files_only)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for m in (pipe.unet, pipe.vae, pipe.text_encoder):
        m.eval()
        m.requires_grad_(False)
    vae_prev = pipe.vae
    pdt = preview_dtype or cfg.get("preview_dtype") or cfg.get("deliver_dtype", "fp16")
    if pdt == "fp32":
        import copy
        vae_prev = copy.deepcopy(pipe.vae).to(dtype=torch.float32)   # weights AND activations in FP32
        vae_prev.requires_grad_(False)
    snap = Path(snapshot_download(repo, revision=rev, local_files_only=True,
                                  allow_patterns=[f"{s}/*" for s in HASHED_SUBFOLDERS] + ["model_index.json"]))
    hashes = snapshot_hashes(snap)
    sched = scheduler_dump(DDIMScheduler.from_config(pipe.scheduler.config), cfg["steps"])
    if len(sched["timesteps"]) != cfg["steps"]:
        raise AssertionError("scheduler did not produce the configured number of steps")
    import diffusers
    import transformers
    prov = {"model_repo": repo, "model_revision": rev, "weights_variant": variant, "unet_dtype": cfg.get("unet_dtype", "fp16"),
            "preview_dtype": pdt, "deliver_dtype": cfg.get("deliver_dtype", "fp16"), "weight_files_sha256": hashes,
            "weights_digest": sha256_hex(hashes), "scheduler": sched, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "vae_scaling_factor": float(pipe.vae.config.scaling_factor)}
    return GeneratorBundle(pipe, pipe.unet, pipe.vae, vae_prev, pipe.text_encoder, pipe.tokenizer, pipe.scheduler,
                           pipe.image_processor, device, dtype, prov)


def write_model_report(bundle: GeneratorBundle, rm_prov: dict, out: str | Path) -> dict:
    rep = {"M1": bundle.provenance, "reward_model": rm_prov}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep
