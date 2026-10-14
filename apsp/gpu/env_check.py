"""Record the actual machine (CC-0): OS, CPU, RAM + cgroup limit, GPU/driver/CUDA/torch, attention backend, threads.

  python -m apsp.gpu.env_check --machine-id MEASUREMENT_ID --out reports/env.json [--require-cuda]
Exit code 3 with --require-cuda when CUDA is unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys
    return None


def _cgroup_limit():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            return None if v == "max" else int(v)
        except (OSError, ValueError):
            continue
    return None


def _cpu_name():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return None


def collect(machine_id: str) -> dict:
    rep = {"machine_id": machine_id, "os": platform.platform(), "python": sys.version.split()[0],
           "cpu": _cpu_name(), "cpu_logical": os.cpu_count(),
           "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "ram_bytes": _ram_bytes(), "cgroup_mem_limit_bytes": _cgroup_limit(),
           "disk_free_bytes_cwd": shutil.disk_usage(".").free, "nvidia_smi": _nvidia_smi(),
           "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            from apsp.gpu.runtime import numeric_runtime
            p = torch.cuda.get_device_properties(0)
            rep["gpu"] = {"name": p.name, "total_mem_bytes": p.total_memory, "capability": f"{p.major}.{p.minor}",
                          "count": torch.cuda.device_count()}
            rep["numeric_runtime"] = numeric_runtime()
    except ImportError:
        rep["torch"] = None
        rep["cuda_available"] = False
    for mod in ("diffusers", "transformers", "ImageReward", "numpy", "PIL"):
        try:
            m = __import__(mod)
            rep.setdefault("packages", {})[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            rep.setdefault("packages", {})[mod] = None
    rep["note"] = "detected facts only; measurement role and GPU-hour budget are set by research in configs/machines.yaml"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record actual machine facts")
    ap.add_argument("--machine-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-cuda", action="store_true")
    a = ap.parse_args(argv)
    rep = collect(a.machine_id)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if a.require_cuda and not rep.get("cuda_available"):
        print("ERROR: CUDA not available on this host", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
