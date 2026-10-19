"""GPU adapters and a shared decode/scoring path for Lite bank and future online use."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .contracts import file_hash


def old_module():
    # Reuse the pinned, missing-key-checked PickScore implementation without touching Full.
    path = Path(__file__).resolve().parents[2] / "env/b200/scripts/run_shard.py"
    spec = importlib.util.spec_from_file_location("_apsp_b200_shard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CleanTap:
    def __init__(self, scheduler, capture):
        self.scheduler = scheduler
        self.capture = set(capture)
        self.original = scheduler.step
        scheduler.step = self.step
        self.reset()

    def reset(self):
        self.n, self.x0, self.meta = 0, {}, {}
        self.initial = None

    def step(self, model_output, timestep, sample, *args, **kwargs):
        s = self.scheduler
        if self.n == 0:
            self.initial = sample.detach().cpu().clone()
        want_dict = kwargs.pop("return_dict", True)
        if hasattr(s, "_init_step_index") and s.step_index is None:
            s._init_step_index(timestep)
        index = getattr(s, "step_index", None)
        sigmas = getattr(s, "sigmas", None)
        sigma = sigmas[index] if sigmas is not None and index is not None else None
        out = self.original(model_output, timestep, sample, *args, return_dict=True, **kwargs)
        self.n += 1
        if self.n in self.capture:
            x0 = getattr(out, "pred_original_sample", None)
            origin = "scheduler.pred_original_sample"
            if x0 is None:
                if not type(s).__name__.startswith("FlowMatch") or sigma is None:
                    raise RuntimeError("unsupported clean prediction contract")
                x0 = sample.float() - sigma.to(sample.device).float() * model_output.float()
                origin = "flow.x_sigma_minus_sigma_v"
            # Detach and move to CPU promptly: all K*J previews need not live on GPU.
            self.x0[self.n] = x0.detach().cpu()
            next_sigma = sigmas[index + 1] if sigmas is not None and index is not None and index + 1 < len(sigmas) else None
            self.meta[self.n] = {"forward_idx": self.n, "timestep_input": float(timestep),
                "sigma_input": None if sigma is None else float(sigma),
                "sigma_output": None if next_sigma is None else float(next_sigma), "preview_origin": origin}
        return out if want_dict else (out.prev_sample,)


class LiteRuntime:
    def __init__(self, profile, registry, device="cuda"):
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.torch, self.spec, self.device = torch, dict(profile), device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.dtype = getattr(torch, profile["dtype"])
        kind = profile["pipeline"]
        kw = {"revision": profile["revision"], "torch_dtype": self.dtype}
        adapter = profile.get("adapter")
        self.adapter_provenance = None
        if adapter and adapter["kind"] == "unet":
            unet = diffusers.UNet2DConditionModel.from_config(profile["repo"], subfolder="unet", revision=profile["revision"])
            path = hf_hub_download(adapter["repo"], adapter["filename"], revision=adapter["revision"])
            unet.load_state_dict(load_file(path, device="cpu"), strict=True)
            kw["unet"] = unet.to(dtype=self.dtype)
            self.adapter_provenance = {**adapter, "sha256": file_hash(path)}
        self.pipe = getattr(diffusers, kind).from_pretrained(profile["repo"], **kw)
        if adapter and adapter["kind"] == "lora":
            path = hf_hub_download(adapter["repo"], adapter["filename"], revision=adapter["revision"])
            self.pipe.load_lora_weights(str(Path(path).parent), weight_name=Path(path).name, local_files_only=True)
            self.pipe.fuse_lora(lora_scale=adapter["fuse_scale"])
            self.adapter_provenance = {**adapter, "sha256": file_hash(path)}
        if profile["scheduler"] != "native":
            opts = {"timestep_spacing": profile["timestep_spacing"]} if "timestep_spacing" in profile else {}
            self.pipe.scheduler = getattr(diffusers, profile["scheduler"]).from_config(self.pipe.scheduler.config, **opts)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.vae_dtype = torch.float32 if getattr(self.pipe.vae.config, "force_upcast", False) and self.dtype == torch.float16 else self.dtype
        self.pipe.vae.to(dtype=self.vae_dtype)
        # Decoder chunks are fixed at one image, shared with re-decode and online.
        self.decode_chunk = 1
        vsf = self.pipe.vae_scale_factor
        self.H, self.W = profile["height"], profile["width"]
        self.flux = kind == "FluxPipeline"
        self.C = self.pipe.transformer.config.in_channels // 4 if self.flux else self.pipe.unet.config.in_channels
        self.h, self.w = self.H // vsf, self.W // vsf
        self.denoiser_calls = 0
        denoiser = self.pipe.transformer if self.flux else self.pipe.unet
        def count_call(module, args, output):
            self.denoiser_calls += 1
        self.counter_hook = denoiser.register_forward_hook(count_call)
        self.scorer_provenance = {}
        self.ir = self.ps = None
        self.registry = registry

    def load_scorers(self):
        from .compat import imagereward_import_bridge
        self.scorer_provenance["compatibility_bridge"] = imagereward_import_bridge()
        from apsp.gpu.scoring import ImageRewardScorer
        self.ir = ImageRewardScorer(self.device, **self.registry["scorers"]["imagereward"])
        self.scorer_provenance["imagereward"] = self.ir.provenance
        self.ps = old_module().PickScorer(self.torch, self.device, self.registry["scorers"]["pickscore"])
        self.scorer_provenance["pickscore"] = self.ps.provenance

    def sync(self):
        self.torch.cuda.synchronize()

    def unpack(self, z):
        return self.pipe._unpack_latents(z, self.H, self.W, self.pipe.vae_scale_factor) if self.flux else z

    def decode_score(self, prompt, stored_latents):
        import math
        import time
        torch = self.torch
        images, flags, irs, pss = [], [], [], []
        t_dec = t_ir = t_ps = 0.0
        for z in stored_latents.split(self.decode_chunk):
            self.sync(); start = time.perf_counter()
            valid_latent = bool(torch.isfinite(z).all())
            pil, valid_image = None, False
            if valid_latent:
                vc = self.pipe.vae.config
                z = z.float().to(self.device)
                if getattr(vc, "latents_mean", None) is not None and getattr(vc, "latents_std", None) is not None:
                    mean = torch.tensor(vc.latents_mean, device=self.device).view(1, -1, 1, 1)
                    std = torch.tensor(vc.latents_std, device=self.device).view(1, -1, 1, 1)
                    z = z * std / vc.scaling_factor + mean
                else:
                    z = z / vc.scaling_factor + (getattr(vc, "shift_factor", None) or 0.0)
                with torch.inference_mode():
                    decoded = self.pipe.vae.decode(z.to(self.vae_dtype), return_dict=False)[0].float()
                valid_image = bool(torch.isfinite(decoded).all())
                if valid_image:
                    pil = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            self.sync(); t_dec += time.perf_counter() - start
            r_ir = r_ps = None
            if valid_image:
                self.sync(); start = time.perf_counter()
                r_ir = float(self.ir.score(prompt, [pil])[0])
                self.sync(); t_ir += time.perf_counter() - start
                self.sync(); start = time.perf_counter()
                r_ps = float(self.ps.score(prompt, [pil])[0])
                self.sync(); t_ps += time.perf_counter() - start
                r_ir = r_ir if math.isfinite(r_ir) else None
                r_ps = r_ps if math.isfinite(r_ps) else None
            failure = ("LATENT_NONFINITE" if not valid_latent else "DECODE_NONFINITE" if not valid_image else
                       "IR_NONFINITE" if r_ir is None else "PICKSCORE_NONFINITE" if r_ps is None else None)
            images.append(pil)
            flags.append({"latent_ok": valid_latent, "img_ok": valid_image, "valid": failure is None, "failure_code": failure})
            irs.append(r_ir); pss.append(r_ps)
        return images, flags, irs, pss, {"decode_s": t_dec, "imagereward_s": t_ir, "pickscore_s": t_ps}
