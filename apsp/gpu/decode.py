"""Preview and delivered decode paths (CC-3). Separate functions; both use the base latent scaling
(latents / vae.config.scaling_factor, SD1.5 has no latents_mean/std) and a fixed chunking."""
from __future__ import annotations

from typing import Optional

from apsp.gpu.runtime import import_torch


def _decode(bundle, latents, preview: bool, chunk: Optional[int]):
    torch = import_torch()
    z = latents / bundle.vae_scaling
    n = z.shape[0]
    step = n if not chunk else int(chunk)
    outs = []
    with torch.inference_mode():
        for s in range(0, n, step):
            outs.append(bundle.decode(z[s:s + step], preview=preview))
    return torch.cat(outs, 0)


def decode_preview(bundle, x0, chunk: Optional[int] = None):
    """x0_hat of the checkpoint forward -> image tensor in [-1,1] (unclamped). May use the FP32 preview VAE."""
    return _decode(bundle, x0, True, chunk)


def decode_delivered(bundle, latents, chunk: Optional[int] = None):
    """Final latent x^64 -> image tensor (unclamped). Never substitute preview@T for the delivered image."""
    return _decode(bundle, latents, False, chunk)
