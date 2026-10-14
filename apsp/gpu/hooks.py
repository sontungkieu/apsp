"""Checkpoint hook that reuses the existing forward (CC-2 / M0). It never runs an extra UNet forward.

Inputs per forward: ``pred_original_sample`` from the scheduler result and the cond/uncond noise predictions
already computed for CFG. Modes:
  none    keep the current x0 reference for a possible VERIFY; no norm/drift/gap kernels
  traj    at checkpoints compute drift / cfg_gap (paid ``feat``)
  lfeat   keep the previous checkpoint x0; B10 features computed only at a cut (paid ``lfeat``)
  collect compute every checkpoint feature for the bank (collector overhead, never charged to policies)
Reductions are FP32.
"""
from __future__ import annotations

from typing import Optional

from apsp.gpu.runtime import import_torch

EPS = 1e-6
MODES = ("none", "traj", "lfeat", "collect")


def _rms(x):
    return x.float().pow(2).flatten(1).mean(1).sqrt()


def drift(x0, prev):
    return _rms(x0.float() - prev.float()) / (_rms(prev) + EPS)


def cfg_gap(eps_text, eps_uncond):
    return _rms(eps_text.float() - eps_uncond.float()) / (_rms(eps_uncond) + EPS)


def channel_stats(x0):
    x = x0.float().flatten(2)            # [k, 4, H*W]
    return x.mean(-1), x.std(-1, unbiased=False)   # ddof 0


class CheckpointHook:
    def __init__(self, mode: str = "none", delta: int = 4):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.delta = mode, delta
        self.reset()

    def reset(self):
        self.x0 = None            # x0 of the latest checkpoint forward
        self.prev_x0 = None       # x0 of the previous checkpoint (traj/lfeat/collect)
        self.j = 0
        self.last: dict = {}      # features at the latest checkpoint
        self.feature_kernel_calls = 0

    def on_forward(self, forward_idx: int, x0, eps_uncond, eps_text) -> Optional[int]:
        """Called after each forward (1-based index). Returns j when a checkpoint is reached."""
        if forward_idx % self.delta:
            return None
        torch = import_torch()
        j = forward_idx // self.delta
        if self.mode != "none" and self.x0 is not None:
            self.prev_x0 = self.x0
        # clone: scheduler outputs are fresh tensors, but a clone keeps a later in-place op from altering history
        self.x0 = x0.detach().clone() if self.mode != "none" else x0.detach()
        self.j = j
        self.last = {}
        if self.mode in ("traj", "collect"):
            self.feature_kernel_calls += 1
            with torch.no_grad():
                d = drift(self.x0, self.prev_x0) if self.prev_x0 is not None else torch.full((x0.shape[0],), float("nan"), device=x0.device)
                g = cfg_gap(eps_text, eps_uncond)
                self.last.update({"drift": d.cpu(), "cfg_gap": g.cpu()})
                if self.mode == "collect":
                    m, s = channel_stats(self.x0)
                    self.last.update({"mean_c": m.cpu(), "std_c": s.cpu()})
        return j

    def lfeat(self, eps_uncond=None, eps_text=None) -> dict:
        """B10 features at a cut: drift, cfg_gap (if eps given), channel means/stds."""
        torch = import_torch()
        self.feature_kernel_calls += 1
        with torch.no_grad():
            out = {"drift": drift(self.x0, self.prev_x0).cpu() if self.prev_x0 is not None else None}
            m, s = channel_stats(self.x0)
            out.update({"mean_c": m.cpu(), "std_c": s.cpu()})
            if eps_text is not None:
                out["cfg_gap"] = cfg_gap(eps_text, eps_uncond).cpu()
        return out

    def gather(self, idx_tensor) -> None:
        if self.x0 is not None:
            self.x0 = self.x0[idx_tensor]
        if self.prev_x0 is not None:
            self.prev_x0 = self.prev_x0[idx_tensor]
        self.last = {k: v[idx_tensor.cpu()] for k, v in self.last.items()}
