"""Deterministic CPU stand-ins for UNet / DDIM / VAE / ImageReward. TEST FIXTURES ONLY.

They implement the same surface as ``GeneratorBundle`` / ``ImageRewardScorer`` so that the real engine,
scoring path, collector and online runner code can be exercised without CUDA. Every operation is
row-independent, so collector-bank replay and online execution must agree exactly. Outputs produced with
this backend are labelled ``backend: fake`` and hash into a different EPH; they are never research data.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from apsp.gpu.runtime import import_torch

FAKE_CFG_OVERRIDES = {"height": 64, "width": 64}


def _h(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class FakeDDIM:
    """DDIM eta=0 with 'leading' spacing on a linear-beta schedule (scaled_linear-like)."""

    def __init__(self, num_train_timesteps: int = 1000):
        torch = import_torch()
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, 0)
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        self.T = num_train_timesteps
        self.init_noise_sigma = 1.0
        self.timesteps = None
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps, prediction_type="epsilon")

    def set_timesteps(self, n: int, device=None):
        torch = import_torch()
        ratio = self.T // n
        self.n = n
        self.timesteps = torch.from_numpy((np.arange(0, n) * ratio).round()[::-1].copy().astype(np.int64) + 1)

    def scale_model_input(self, x, t):
        return x

    def step(self, eps, t, x, eta: float = 0.0, return_dict: bool = True):
        t = int(t)
        prev = t - self.T // self.n
        a_t = self.alphas_cumprod[t]
        a_p = self.alphas_cumprod[prev] if prev >= 0 else self.final_alpha_cumprod
        x0 = (x - (1 - a_t) ** 0.5 * eps) / a_t ** 0.5
        xp = a_p ** 0.5 * x0 + (1 - a_p) ** 0.5 * eps
        return SimpleNamespace(prev_sample=xp, pred_original_sample=x0)


class FakeBundle:
    backend = "fake"

    def __init__(self, nan_after_forward: dict | None = None):
        torch = import_torch()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.scheduler = FakeDDIM()
        self.latent_channels = 4
        self.vae_scaling = 0.18215
        self.provenance = {"backend": "fake", "model_repo": "fake", "model_revision": "fake"}
        self.unet_calls = 0
        self.nan_after_forward = nan_after_forward or {}   # {row_value_key: forward} not used by default

    def encode(self, prompt: str, negative_prompt: str = ""):
        torch = import_torch()
        c = torch.full((1, 77, 768), 0.5 + _h(prompt), dtype=self.dtype)
        u = torch.full((1, 77, 768), 0.1 + 0.1 * _h("neg|" + negative_prompt), dtype=self.dtype)
        return c, u

    def pooler(self, prompt: str):
        rng = np.random.default_rng(int(_h(prompt) * 1e9))
        return rng.normal(size=768).astype(np.float32)

    def unet_eps(self, x, t, emb):
        torch = import_torch()
        self.unet_calls += 1
        e = emb.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return torch.tanh(0.7 * x + 0.3 * e * torch.sin(x * (1.0 + float(t) / 1000.0)))

    def decode(self, z_scaled, preview: bool):
        torch = import_torch()
        img = torch.tanh(z_scaled[:, :3] * self.vae_scaling * 2.0)
        return img.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)   # VAE factor 8

    def to_pil(self, images):
        from PIL import Image
        arr = ((images / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        return [Image.fromarray(a) for a in arr]


class FakeScorer:
    def __init__(self):
        self.n_calls = 0
        self.provenance = {"backend": "fake"}

    def score(self, prompt: str, images: list) -> list[float]:
        self.n_calls += 1
        out = []
        for im in images:
            a = np.asarray(im, dtype=np.float64) / 255.0
            out.append(float(a[..., 0].mean() - 0.5 * a[..., 2].mean() + 0.1 * _h(prompt)))
        return out
