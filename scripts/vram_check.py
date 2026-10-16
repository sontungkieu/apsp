#!/usr/bin/env python3
"""Check the GPU's usable memory with PyTorch (the UI may show a non-standard size, e.g. a 48 GB RTX 4090).

Allocates 1 GiB blocks until `--target-gib` (default: total memory minus 2 GiB) or an OOM, writes a distinct
pattern into every block, reads all blocks back and verifies them, and times one device-to-device copy.
Engineering evidence only; no model is loaded.

  python scripts/vram_check.py --out /workspace/apsp/runs/prep0/p0/vram_check.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

GIB = 1 << 30


def nvidia_smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.reserved,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"n/a ({e})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-gib", type=float, default=None)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    p = torch.cuda.get_device_properties(0)
    free0, total = torch.cuda.mem_get_info(0)
    target = a.target_gib if a.target_gib is not None else total / GIB - 2.0
    n_elem = GIB // 4                                    # float32 elements per block
    blocks, oom_at = [], None
    t0 = time.perf_counter()
    try:
        while len(blocks) < int(target):
            b = torch.empty(n_elem, dtype=torch.float32, device="cuda")
            b.fill_(float(len(blocks) + 1))
            b[:: 4096] = torch.arange(0, n_elem, 4096, device="cuda", dtype=torch.float32) % 977
            blocks.append(b)
    except torch.cuda.OutOfMemoryError:
        oom_at = len(blocks)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0
    bad = []
    ref_sparse = torch.arange(0, n_elem, 4096, device="cuda", dtype=torch.float32) % 977
    for i, b in enumerate(blocks):
        v = float(i + 1)
        dense_ok = bool((b.view(-1, 4096)[:, 1:] == v).all())      # every element except the sparse column
        sparse_ok = bool(torch.equal(b[:: 4096], ref_sparse))
        if not (dense_ok and sparse_ok):
            bad.append(i)
    copy_gbps = None
    if len(blocks) >= 2:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(5):
            blocks[1].copy_(blocks[0])
        torch.cuda.synchronize()
        copy_gbps = 5 * 2 * GIB / (time.perf_counter() - t1) / 1e9  # read + write
    res = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_name": p.name, "total_memory_bytes": p.total_memory, "sm_count": p.multi_processor_count,
        "compute_capability": f"{p.major}.{p.minor}", "mem_get_info_free_bytes_at_start": free0,
        "mem_get_info_total_bytes": total, "nvidia_smi": nvidia_smi(),
        "allocated_gib": len(blocks), "target_gib": int(target), "oom_at_gib": oom_at,
        "pattern_errors": bad, "alloc_fill_seconds": round(t_alloc, 2),
        "d2d_copy_gb_per_s": None if copy_gbps is None else round(copy_gbps, 1),
        "verdict": "ok" if (not bad and len(blocks) >= int(target)) else "check",
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
