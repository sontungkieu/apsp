"""CUDA requirement and numeric-runtime facts for GPU commands.

GPU commands call ``require_cuda`` and fail loudly without CUDA. There is no silent CPU fallback;
the only non-CUDA path is the explicit ``--fake-backend`` used by integration tests, whose outputs
are labelled ``backend: fake`` and hashed into a different EPH.
"""
from __future__ import annotations

import os


class CudaUnavailableError(RuntimeError):
    pass


def import_torch():
    try:
        import torch
    except ImportError as e:
        raise CudaUnavailableError("torch is not installed; install the GPU environment (env/requirements-gpu-linux.txt)") from e
    return torch


def require_cuda(min_vram_gb: float | None = None):
    torch = import_torch()
    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "CUDA is not available on this host (torch.cuda.is_available() == False). GPU commands refuse to run; "
            "nothing was executed. Use the Linux GPU box (docs/GPU_FIRST_RUN.md) or --fake-backend for code-path tests.")
    dev = torch.device("cuda:0")
    if min_vram_gb is not None:
        tot = torch.cuda.get_device_properties(dev).total_memory / 2**30
        if tot < min_vram_gb:
            raise CudaUnavailableError(f"GPU has {tot:.1f} GiB < required {min_vram_gb} GiB")
    return dev


def configure_determinism() -> dict:
    """cudnn.benchmark=False, cudnn.deterministic=True. Flags are not a bitwise guarantee (M0/M6 measure it)."""
    torch = import_torch()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return numeric_runtime()


def numeric_runtime() -> dict:
    torch = import_torch()
    info = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_num_threads": torch.get_num_threads(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        info["flash_sdp"] = bool(torch.backends.cuda.flash_sdp_enabled())
        info["mem_efficient_sdp"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        info["math_sdp"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:  # noqa: BLE001
        pass
    return info


def sync(device) -> None:
    torch = import_torch()
    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.synchronize(device)
