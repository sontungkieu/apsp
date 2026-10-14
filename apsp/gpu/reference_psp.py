"""Collector equivalence against the pinned PSP reference (CC-6 ``--test-equivalence``). Read-only use of
third_party/psp: we import ``FKDStableDiffusion`` through sys.path and load it from the pinned mirror (the base
hard-codes the dead ``runwayml`` repo, so the loader is wrapped; the algorithm is untouched).

Comparisons on calibration prompts, same prompt / initial latents / config / dtype:
  A. algorithm path: reference pipeline fed OUR prompt embeddings -> delivered latents exact-hash + max |diff|
  B. text-encoder batch effect: reference pipeline encoding the 8-prompt list itself -> same metrics
  C. checkpoint previews: x0 recomputed the reference way (extra forward on prev latents, diagnostic only)
     vs our hook x0, both scored through our single PIL scoring path -> max |score diff|, x0 max |diff|
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from apsp.config import REPO_ROOT
from apsp.contracts.hashing import array_sha256
from apsp.contracts.identity import bank_rows
from apsp.gpu.runtime import import_torch

PSP_T2I = REPO_ROOT / "third_party" / "psp" / "Fk-Diffusion-Steering" / "text_to_image"


def load_reference_pipeline(cfg: dict, bundle):
    torch = import_torch()
    for p in (str(PSP_T2I), str(PSP_T2I / "fkd_diffusers")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"reference PSP pipeline import failed ({type(e).__name__}: {e}); install the full GPU env "
                           "including clip and google-genai (env/requirements-gpu-linux.txt)") from e
    from diffusers import DDIMScheduler
    rev = bundle.provenance["model_revision"]
    ref = FKDStableDiffusion.from_pretrained(cfg["model_repo"], revision=rev, torch_dtype=bundle.dtype,
                                             safety_checker=None, requires_safety_checker=False,
                                             variant=bundle.provenance.get("weights_variant"))
    ref.scheduler = DDIMScheduler.from_config(ref.scheduler.config)   # exactly as fks_utils.get_model
    ref = ref.to(bundle.device)
    ref.set_progress_bar_config(disable=True)
    return ref


def _ref_x0(ref, scheduler, prev_latents, t, prompt_embeds, guidance, eta):
    """Reference collector's x0 recomputation (compute_x0_preds_non_sd35): an EXTRA UNet forward. Diagnostic only."""
    torch = import_torch()
    with torch.inference_mode():
        x_in = scheduler.scale_model_input(torch.cat([prev_latents] * 2), t)
        eps = ref.unet(x_in, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        u, c = eps.chunk(2)
        eps = u + guidance * (c - u)
        return scheduler.step(eps, t, prev_latents, eta=eta, return_dict=True).pred_original_sample


def run_reference(ref, cfg, prompt: str, init_latents, cond=None, uncond=None, capture_js=()):
    torch = import_torch()
    from diffusers import DDIMScheduler
    k = init_latents.shape[0]
    delta = cfg["delta"]
    want = {delta * j - 1 for j in capture_js}
    x0s, prev = {}, {"x": init_latents}
    sched2 = DDIMScheduler.from_config(ref.scheduler.config)
    sched2.set_timesteps(cfg["steps"], device=init_latents.device)

    def cb(_p, i, t, kw):
        if i in want:
            x0s[(i + 1) // delta] = _ref_x0(ref, sched2, prev["x"], t, kw["prompt_embeds"], cfg["cfg"], cfg["eta"])
        prev["x"] = kw["latents"]
        return {}

    kwargs = dict(num_inference_steps=cfg["steps"], guidance_scale=cfg["cfg"], eta=cfg["eta"], latents=init_latents.clone(),
                  output_type="latent", callback_on_step_end=cb,
                  callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"], fkd_args=None)
    with torch.inference_mode():
        if cond is not None:
            out = ref(prompt_embeds=cond.expand(k, -1, -1), negative_prompt_embeds=uncond.expand(k, -1, -1), **kwargs)
        else:
            out = ref(prompt=[prompt] * k, negative_prompt=[cfg["negative_prompt"]] * k, **kwargs)
    lat = out.images if hasattr(out, "images") else out[0]
    return lat, x0s


def equivalence_report(cfg, bundle, scoring, engine, prompts, args) -> dict:
    torch = import_torch()
    if getattr(bundle, "backend", "") == "fake":
        return {"status": "error", "reason": "equivalence needs the real reference pipeline (no fake backend)"}
    ref = load_reference_pipeline(cfg, bundle)
    js = (4, 8, 16)
    rows_out = []
    for row in prompts.itertuples(index=False):
        engine.prepare_request(row.text, row.prompt_uid)
        engine.start(bank_rows(0, cfg["k0"]))
        ours_x0 = {}
        for j in range(1, cfg["steps"] // cfg["delta"] + 1):
            engine.advance(None, cfg["delta"], False)
            if j in js:
                ours_x0[j] = engine.hook.x0.clone()
        ours = engine.x
        init = (engine.init * bundle.scheduler.init_noise_sigma).to(bundle.device)
        latA, x0A = run_reference(ref, cfg, row.text, init, engine.cond, engine.uncond, capture_js=js)
        latB, _ = run_reference(ref, cfg, row.text, init)
        rec = {"prompt_uid": row.prompt_uid,
               "A_delivered_exact_hash": array_sha256(ours.float().cpu().numpy()) == array_sha256(latA.float().cpu().numpy()),
               "A_delivered_max_abs": float((ours.float() - latA.float()).abs().max()),
               "B_delivered_exact_hash": array_sha256(ours.float().cpu().numpy()) == array_sha256(latB.float().cpu().numpy()),
               "B_delivered_max_abs": float((ours.float() - latB.float()).abs().max())}
        for j in js:
            so = [r.score for r in scoring.score_preview(row.text, ours_x0[j]).records]
            sr = [r.score for r in scoring.score_preview(row.text, x0A[j]).records]
            diffs = [abs(a - b) for a, b in zip(so, sr) if a is not None and b is not None]
            rec[f"C_j{j}_x0_max_abs"] = float((ours_x0[j].float() - x0A[j].float()).abs().max())
            rec[f"C_j{j}_score_max_abs"] = max(diffs) if diffs else None
        rows_out.append(rec)
    return {"status": "measured", "n_prompts": len(rows_out), "records": rows_out,
            "all_A_exact": all(r["A_delivered_exact_hash"] for r in rows_out),
            "note": "exact-hash comparisons are reported as measured; tolerance alone is not called bitwise parity"}
