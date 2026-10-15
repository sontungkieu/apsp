"""Reconstruct runtime fingerprints (apsp.runtime_fingerprint/v1) of past GPU runs from their artifacts (read-only).

Fields the artifacts never recorded are written as "not recorded"; nothing is inferred. The source artifacts are not
modified (they keep their original manifests and EPH).

  python scripts/retro_fingerprint.py smoke1=<smoke1 run dir> diag1=<diag1 run dir> --out reports/runtime_fingerprints
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apsp.gpu.fingerprint import EPH_V1_COVERS, EPH_V1_MISSES, SCHEMA, diff, numeric_sha  # noqa: E402

NR = "not recorded"


def _freeze(run: Path) -> dict:
    p = run / "env_freeze.txt"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line.strip())
            if m:
                out[m.group(1).lower()] = m.group(2)
    return out


def reconstruct(run: Path) -> dict:
    env = next(iter(sorted((run / "p0").glob("env_*.json"))), None)
    e = json.loads(env.read_text(encoding="utf-8")) if env else {}
    fr = _freeze(run)
    smi = (e.get("nvidia_smi") or [""])[0].split(", ")
    audit_p = run / "diag" / "audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else None
    eph_files = sorted(p for p in run.rglob("*.json") if '"eph"' in p.read_text(encoding="utf-8", errors="replace")[:100000])
    ephs = set()
    for p in eph_files:
        for m in re.finditer(r'"eph":\s*"([0-9a-f]{64})"', p.read_text(encoding="utf-8", errors="replace")):
            ephs.add(m.group(1))
    nr = (audit or {}).get("numeric_runtime") or {}
    flags = {"cudnn_benchmark": nr.get("cudnn_benchmark", NR), "cudnn_deterministic": nr.get("cudnn_deterministic", NR),
             "matmul_allow_tf32": nr.get("matmul_allow_tf32", NR), "cudnn_allow_tf32": nr.get("cudnn_allow_tf32", NR),
             "fp16_reduced_precision_reduction": nr.get("fp16_reduced_precision_reduction", NR),
             "deterministic_algorithms": nr.get("deterministic_algorithms", NR),
             "CUBLAS_WORKSPACE_CONFIG": nr.get("CUBLAS_WORKSPACE_CONFIG", NR),
             "flash_sdp": nr.get("flash_sdp", NR), "mem_efficient_sdp": nr.get("mem_efficient_sdp", NR),
             "math_sdp": nr.get("math_sdp", NR)}
    procs = sorted({p for m in ("unet", "vae", "vae_preview") for p in ((audit or {}).get(m) or {}).get("attn_processors", [])})
    fp = {"schema": SCHEMA, "source": "reconstructed from artifacts (read-only)", "run_dir": run.name,
          "python": e.get("python", NR), "os": e.get("os", NR), "cpu": e.get("cpu", NR),
          "cpu_threads_env": e.get("env_threads", NR), "driver": smi[2] if len(smi) > 2 else NR,
          "backend": (audit or {}).get("backend", "diffusers (per EPH-bearing reports)"),
          "packages": {"diffusers": fr.get("diffusers", NR), "transformers": fr.get("transformers", NR),
                       "image-reward": fr.get("image-reward", fr.get("imagereward", NR)), "numpy": fr.get("numpy", NR),
                       "pillow": fr.get("pillow", NR), "accelerate": fr.get("accelerate", NR)},
          "torch": e.get("torch", NR), "cuda_runtime": (e.get("numeric_runtime") or {}).get("cuda_runtime", NR),
          "cudnn": (e.get("numeric_runtime") or {}).get("cudnn", NR),
          "gpu_name": (e.get("gpu") or {}).get("name", NR), "gpu_capability": (e.get("gpu") or {}).get("capability", NR),
          "gpu_total_mem_bytes": (e.get("gpu") or {}).get("total_mem_bytes", NR),
          "flags": flags if audit else {"note": "engine-time flags not recorded in this run (env_check records process "
                                                 "defaults before configure_determinism; EPH holds the engine-time "
                                                 "subset)", **{k: NR for k in flags}},
          "attention_processors": procs or NR, "machine_id": e.get("machine_id", NR), "ephs_in_artifacts": sorted(ephs),
          "eph_v1_covers": EPH_V1_COVERS, "eph_v1_misses": EPH_V1_MISSES}
    fp["fingerprint_sha"] = numeric_sha(fp)
    return fp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fps = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        fps[name] = reconstruct(Path(d))
        (out / f"{name}.json").write_text(json.dumps(fps[name], indent=1), encoding="utf-8")
    names = list(fps)
    if len(names) >= 2:
        d = {"a": names[0], "b": names[1], "numeric_fields_differing": diff(fps[names[0]], fps[names[1]]),
             "note": "'not recorded' on either side means the difference cannot be judged from the artifacts"}
        (out / f"diff_{names[0]}_{names[1]}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
