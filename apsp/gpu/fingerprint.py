"""Runtime fingerprint (schema apsp.runtime_fingerprint/v1): the execution environment that EPH v1 does not cover.

EPH v1 (apsp.gpu.backend.eph_body) hashes the model/weights, sampling, dtypes, scoring, seed scheme, the generation
code, the requirements file and `numeric_runtime` (torch / CUDA runtime / cuDNN versions and the cudnn, TF32, SDPA
and CUBLAS flags). It does NOT include the GPU model, NVIDIA driver, Python version, installed package versions,
attention processor, `deterministic_algorithms`, fp16 reduced-precision reduction or the OS. Two runs with the same
EPH can therefore differ in those. The fingerprint records them next to the EPH, as a separate hash, so old EPHs and
manifests stay valid (no EPH schema change).

  python -m apsp.gpu.fingerprint --out runs/<RUN_ID>/p0/runtime_fingerprint.json [--with-model]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCHEMA = "apsp.runtime_fingerprint/v1"
EPH_V1_COVERS = ["torch version", "CUDA runtime version", "cuDNN version", "cudnn.benchmark", "cudnn.deterministic",
                 "matmul/cudnn allow_tf32", "SDPA flash/mem_efficient/math enabled", "CUBLAS_WORKSPACE_CONFIG",
                 "requirements-gpu-linux.txt sha256 (pins, not the installed patch versions)"]
EPH_V1_MISSES = ["GPU model and compute capability", "NVIDIA driver", "Python version", "installed diffusers / "
                 "transformers / ImageReward / numpy / PIL versions", "attention processor class",
                 "torch.use_deterministic_algorithms", "allow_fp16_reduced_precision_reduction", "OS / kernel / glibc",
                 "CPU model and thread settings"]
# fields that can change numbers; the rest is context
NUMERIC_KEYS = ("python", "gpu_name", "gpu_capability", "driver", "torch", "cuda_runtime", "cudnn", "packages",
                "flags", "attention_processors", "backend")


def _pkg(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _driver() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def collect(bundle=None, backend: Optional[str] = None) -> dict:
    """Facts of this process. `bundle` (optional) adds the attention processor classes actually installed."""
    fp = {"schema": SCHEMA, "python": platform.python_version(), "os": platform.platform(),
          "cpu": platform.processor() or platform.machine(), "cpu_threads_env": {k: os.environ.get(k) for k in
                                                                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
          "driver": _driver(), "backend": backend or getattr(bundle, "backend", None),
          "packages": {n: _pkg(n) for n in ("diffusers", "transformers", "image-reward", "numpy", "pillow", "accelerate")}}
    try:
        import torch
        fp.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                   "cuda_available": torch.cuda.is_available()})
        flags = {"cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                 "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                 "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                 "fp16_reduced_precision_reduction": bool(getattr(torch.backends.cuda.matmul,
                                                                  "allow_fp16_reduced_precision_reduction", False)),
                 "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                 "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        try:
            flags.update({"flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
                          "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                          "math_sdp": bool(torch.backends.cuda.math_sdp_enabled())})
        except Exception:  # noqa: BLE001
            pass
        fp["flags"] = flags
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            fp.update({"gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}", "gpu_total_mem_bytes": int(p.total_memory),
                       "gpu_count": torch.cuda.device_count()})
    except ImportError:
        fp["torch"] = None
    procs = set()
    for m in ("unet", "vae", "vae_preview"):
        mod = getattr(bundle, m, None)
        if mod is not None and hasattr(mod, "attn_processors"):
            procs.update(type(v).__name__ for v in mod.attn_processors.values())
    fp["attention_processors"] = sorted(procs) if procs else None
    fp["fingerprint_sha"] = numeric_sha(fp)
    return fp


def numeric_sha(fp: dict) -> str:
    """sha256 over the fields that can change numbers (context fields such as OS string are excluded)."""
    body = {k: fp.get(k) for k in NUMERIC_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def diff(a: dict, b: dict) -> dict:
    """Numeric-relevant fields that differ between two fingerprints (None = not recorded on that side)."""
    out = {}
    for k in NUMERIC_KEYS:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            sub = {kk: [va.get(kk), vb.get(kk)] for kk in sorted(set(va) | set(vb)) if va.get(kk) != vb.get(kk)}
            if sub:
                out[k] = sub
        elif va != vb:
            out[k] = [va, vb]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the runtime fingerprint of this process")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-model", action="store_true", help="load the generator to record attention processors")
    a = ap.parse_args(argv)
    bundle = None
    if a.with_model:
        from apsp.gpu.backend import add_backend_args, build_backend
        bp = argparse.ArgumentParser()
        add_backend_args(bp)
        _, bundle, *_ = build_backend(bp.parse_args([]))
    else:
        from apsp.gpu.runtime import configure_determinism
        try:
            configure_determinism()       # same process flags as the engine sets
        except ImportError:
            pass
    fp = collect(bundle)
    fp["eph_v1_covers"], fp["eph_v1_misses"] = EPH_V1_COVERS, EPH_V1_MISSES
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(fp, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": a.out, "fingerprint_sha": fp["fingerprint_sha"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
