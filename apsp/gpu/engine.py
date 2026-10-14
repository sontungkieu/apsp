"""Denoising engine (CC-7) implementing the ``apsp.sim.env.Engine`` protocol on real tensors.

Per forward (identical math to the FKD/StableDiffusionPipeline loop of the pinned base):
  x_in = scale_model_input(cat([x, x]), t);  eps = unet(x_in, t, [uncond rows; cond rows])
  eps_u, eps_c = eps.chunk(2);  eps_cfg = eps_u + cfg * (eps_c - eps_u)
  out = DDIM.step(eps_cfg, t, x, eta=0);  x <- out.prev_sample;  hook(out.pred_original_sample, eps_u, eps_c)
The engine never charges fees; the shared runners own the ledger. Candidate identity is kept by RowMap:
tensor rows always follow ``state.alive`` order, and CFG rows are gathered as [idx; idx + k].
"""
from __future__ import annotations

from typing import Optional, Sequence

from apsp.contracts.identity import CandidateId, cfg_row_indices, derive_seed
from apsp.gpu.hooks import CheckpointHook
from apsp.gpu.runtime import import_torch


def initial_latent(seed: int, shape=(1, 4, 64, 64), dtype=None):
    """Seed scheme apsp-v1-sha256-cpu: CPU torch.Generator, float32 normal draw, then cast to the model dtype."""
    torch = import_torch()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(shape, generator=g, dtype=torch.float32)
    return x.to(dtype) if dtype is not None else x


class DenoiseEngine:
    def __init__(self, bundle, scoring, cfg: dict, hook_mode: str = "none"):
        self.b, self.scoring, self.cfg = bundle, scoring, cfg
        self.delta = int(cfg["delta"])
        self.steps = int(cfg["steps"])
        self.guidance = float(cfg["cfg"])
        self.hook = CheckpointHook(hook_mode, self.delta)
        self.prompt: Optional[str] = None
        self.uid: Optional[str] = None
        self.n_unet_calls = 0
        self.last_images: dict = {}
        self.last_scores: dict = {}

    # --- request setup (C_fixed: outside the cap, inside the e2e timer) -------------------
    def prepare_request(self, prompt: str, uid: str) -> None:
        torch = import_torch()
        self.prompt, self.uid = prompt, uid
        with torch.inference_mode():
            self.cond, self.uncond = self.b.encode(prompt, self.cfg.get("negative_prompt", ""))
        self.b.scheduler.set_timesteps(self.steps, device=self.b.device)
        self.timesteps = self.b.scheduler.timesteps

    def latent_for(self, cid: CandidateId):
        shape = (1, self.b.latent_channels, self.cfg["height"] // 8, self.cfg["width"] // 8)
        return initial_latent(derive_seed(self.uid, cid[0], cid[1]), shape, self.b.dtype)

    def start(self, rows: Sequence[CandidateId]) -> None:
        torch = import_torch()
        if self.prompt is None:
            raise RuntimeError("prepare_request() first")
        self.rows = list(rows)
        k = len(rows)
        self.init = torch.cat([self.latent_for(c) for c in rows], 0)
        self.x = (self.init * self.b.scheduler.init_noise_sigma).to(self.b.device)
        # embeddings encoded once and tiled: every candidate sees bit-identical conditioning
        self.emb = torch.cat([self.uncond.expand(k, -1, -1), self.cond.expand(k, -1, -1)], 0).contiguous()
        self.forward_idx = 0
        self.hook.reset()
        self.last_eps = None
        self.last_images, self.last_scores = {}, {}

    def _forward(self) -> None:
        torch = import_torch()
        t = self.timesteps[self.forward_idx]
        with torch.inference_mode():
            x_in = self.b.scheduler.scale_model_input(torch.cat([self.x, self.x], 0), t)
            eps = self.b.unet_eps(x_in, t, self.emb)
            self.n_unet_calls += 1
            eps_u, eps_c = eps.chunk(2)
            eps_cfg = eps_u + self.guidance * (eps_c - eps_u)
            out = self.b.scheduler.step(eps_cfg, t, self.x, eta=float(self.cfg["eta"]), return_dict=True)
            self.x = out.prev_sample
        self.forward_idx += 1
        self.last_eps = (eps_u, eps_c)
        self.hook.on_forward(self.forward_idx, out.pred_original_sample, eps_u, eps_c)

    def _check_rows(self, state=None) -> None:
        k = len(self.rows)
        assert self.x.shape[0] == k, "latent rows != RowMap"
        assert self.emb.shape[0] == 2 * k, "CFG embeds must be [uncond k; cond k]"
        if self.hook.x0 is not None:
            assert self.hook.x0.shape[0] == k
        if state is not None:
            assert tuple(self.rows) == tuple(state.alive) and state.k == k

    # --- Engine protocol -----------------------------------------------------------------
    def advance(self, state, n_forwards: int, want_traj: bool) -> Optional[dict]:
        for _ in range(n_forwards):
            self._forward()
        if not want_traj:
            return None
        d, g = self.hook.last["drift"].tolist(), self.hook.last["cfg_gap"].tolist()
        return {c: (float(d[i]), float(g[i])) for i, c in enumerate(self.rows)}

    def verify(self, state) -> dict:
        self._check_rows(state)
        if self.hook.j != state.j or self.forward_idx != self.delta * state.j:
            raise AssertionError("VERIFY must score the x0 of the current checkpoint forward")
        sb = self.scoring.score_preview(self.prompt, self.hook.x0)
        return dict(zip(self.rows, sb.records))

    def prune(self, state, retained: tuple) -> None:
        torch = import_torch()
        k = len(self.rows)
        idx = torch.as_tensor(retained, device=self.x.device, dtype=torch.long)
        self.x = self.x[idx]
        self.emb = self.emb[torch.as_tensor(cfg_row_indices(retained, k), device=self.emb.device)]
        self.hook.gather(idx)
        self.rows = [self.rows[i] for i in retained]
        self._check_rows(state)

    def finalize_scores(self, state) -> dict:
        self._check_rows(state)
        assert self.forward_idx == self.steps
        sb = self.scoring.score_delivered(self.prompt, self.x)
        self.last_images = dict(zip(self.rows, sb.images))
        self.last_scores = dict(zip(self.rows, sb.records))
        return dict(self.last_scores)

    def finalize_validity(self, state) -> dict:
        self._check_rows(state)
        assert self.forward_idx == self.steps
        sb = self.scoring.validate_delivered(self.prompt, self.x)
        self.last_images = dict(zip(self.rows, sb.images))
        return {c: (r.latent_ok, r.img_ok) for c, r in zip(self.rows, sb.records)}
