"""B200 node check for the Adaptive PSP collector environment (env/b200). Read-only, about 1 minute.

  uv run --project env/b200 python env/b200/scripts/check_gpu.py --out gpu_check.json [--vram-gb 40] [--allow-cpu]

Checks, in order:
  1. versions: python, torch, torch.version.cuda, cuDNN, driver (nvidia-smi), diffusers, transformers, ...
  2. CUDA: device count, name, compute capability, memory, and that this torch build has kernels for the GPU
     (sm_100 must be in torch.cuda.get_arch_list() on B200);
  3. numerics: fp16 / bf16 / tf32-off matmul and conv2d against an fp32 reference, and which SDPA backends run
     (flash, mem-efficient, cuDNN, math) with their error against the math backend;
  4. optional VRAM pattern test (--vram-gb): index-derived, word-varying pattern (the kind that exposed the faulty
     RTX 4090 of Vast 56267276), written then verified after a device synchronize;
  5. imports used by the collector: the four diffusers pipelines, transformers CLIP/T5, ImageReward, CLIP, apsp.
Exit 0 only if every check passes. Results go to --out as JSON.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time

PKGS = ["torch", "torchvision", "triton", "diffusers", "transformers", "tokenizers", "accelerate", "huggingface-hub",
        "safetensors", "sentencepiece", "protobuf", "image-reward", "clip", "timm", "fairscale", "datasets",
        "numpy", "pandas", "pyarrow", "pillow", "adaptive-psp", "xformers", "flash-attn"]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,uuid",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["nvidia_smi"] = [l.strip() for l in q.stdout.splitlines() if l.strip()] or q.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        out["nvidia_smi"] = f"unavailable: {e}"
    return out


def cuda_info(torch) -> dict:
    d = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
         "cuda_available": torch.cuda.is_available(), "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
    d["devices"] = []
    if d["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cap = f"sm_{p.major}{p.minor}"
            d["devices"].append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                                 "total_mem_gb": round(p.total_memory / 2**30, 2), "multiprocessors": p.multi_processor_count,
                                 "kernels_in_build": cap in d["arch_list"] or f"compute_{p.major}{p.minor}" in d["arch_list"]})
    return d


def numerics(torch, dev) -> dict:
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(2048, 2048, generator=g)
    b = torch.randn(2048, 2048, generator=g)
    ref = (a.double() @ b.double()).float()
    res = {}
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        c = (a.to(dev, dt) @ b.to(dev, dt)).float().cpu()
        res[f"matmul_{name}_rel_err"] = float((c - ref).norm() / ref.norm())
    x = torch.randn(4, 64, 64, 64, generator=g)
    w = torch.randn(128, 64, 3, 3, generator=g) * 0.05
    cref = F.conv2d(x.double(), w.double(), padding=1).float()
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        c = F.conv2d(x.to(dev, dt), w.to(dev, dt), padding=1).float().cpu()
        res[f"conv_{name}_rel_err"] = float((c - cref).norm() / cref.norm())
    # SDPA backends (diffusers uses F.scaled_dot_product_attention)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = (torch.randn(2, 16, 1024, 64, generator=g) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        sref = F.scaled_dot_product_attention(q.to(dev, torch.float32), k.to(dev, torch.float32), v.to(dev, torch.float32)).cpu()
    sd = {}
    for bk in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        if not hasattr(SDPBackend, bk):
            continue
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            try:
                with sdpa_kernel(getattr(SDPBackend, bk)):
                    o = F.scaled_dot_product_attention(q.to(dev, dt), k.to(dev, dt), v.to(dev, dt)).float().cpu()
                sd[f"{bk}_{name}"] = {"ok": True, "rel_err": float((o - sref).norm() / sref.norm())}
            except Exception as e:  # noqa: BLE001
                sd[f"{bk}_{name}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    res["sdpa"] = sd
    res["pass"] = (res["matmul_fp32_rel_err"] < 1e-5 and res["matmul_fp16_rel_err"] < 5e-3 and res["matmul_bf16_rel_err"] < 3e-2
                   and res["conv_fp16_rel_err"] < 5e-3 and res["conv_bf16_rel_err"] < 3e-2
                   and any(v["ok"] for k_, v in sd.items() if not k_.startswith("MATH")))
    return res


def vram_test(torch, dev, gb: float) -> dict:
    """Blocks of 1 GiB int32 filled with a word-varying pattern derived from (block, index); verified after sync."""
    n = (1 << 30) // 4
    blocks, bad = [], {}
    t0 = time.time()
    try:
        for i in range(int(gb)):
            idx = torch.arange(n, device=dev, dtype=torch.int64)
            blk = ((idx * 2654435761 + (i + 1) * 40503) & 0x7FFFFFFF).to(torch.int32)
            del idx
            blocks.append(blk)
        torch.cuda.synchronize()
        for i, blk in enumerate(blocks):
            idx = torch.arange(n, device=dev, dtype=torch.int64)
            exp = ((idx * 2654435761 + (i + 1) * 40503) & 0x7FFFFFFF).to(torch.int32)
            nbad = int((blk != exp).sum())
            if nbad:
                bad[i] = nbad
            del idx, exp
    finally:
        del blocks
        torch.cuda.empty_cache()
    return {"gb_tested": int(gb), "blocks_with_errors": bad, "pass": not bad, "seconds": round(time.time() - t0, 1)}


def imports() -> dict:
    mods = {"diffusers": ["StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusion3Pipeline", "FluxPipeline",
                          "DDIMScheduler", "FlowMatchEulerDiscreteScheduler"],
            "transformers": ["CLIPModel", "CLIPTextModel", "CLIPTextModelWithProjection", "T5EncoderModel", "AutoProcessor",
                             "BertTokenizer"]}
    out = {}
    for m, names in mods.items():
        try:
            mod = importlib.import_module(m)
            missing = [n for n in names if not hasattr(mod, n)]
            out[m] = "ok" if not missing else f"missing {missing}"
        except Exception as e:  # noqa: BLE001
            out[m] = f"{type(e).__name__}: {e}"
    for m in ("ImageReward", "clip", "fairscale", "apsp.gpu.scoring", "apsp.data.prompts", "apsp.contracts.identity"):
        try:
            importlib.import_module(m)
            out[m] = "ok"
        except Exception as e:  # noqa: BLE001
            out[m] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="gpu_check.json")
    ap.add_argument("--vram-gb", type=float, default=0.0, help="GiB to pattern-test on GPU 0 (0 = skip)")
    ap.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is unavailable (env-only check)")
    ap.add_argument("--expect-sm", default="100", help="required compute capability, e.g. 100 for B200; '' = any")
    a = ap.parse_args(argv)
    import torch
    rep = {"versions": versions(), "cuda": cuda_info(torch), "imports": imports()}
    ok = all(v == "ok" for v in rep["imports"].values())
    if rep["cuda"]["cuda_available"]:
        dev = torch.device("cuda:0")
        rep["numerics"] = numerics(torch, dev)
        ok &= rep["numerics"]["pass"]
        ok &= all(d["kernels_in_build"] for d in rep["cuda"]["devices"])
        if a.expect_sm:
            caps = {d["capability"].replace(".", "") for d in rep["cuda"]["devices"]}
            rep["expected_sm_present"] = caps == {a.expect_sm}
            ok &= rep["expected_sm_present"]
        if a.vram_gb > 0:
            rep["vram"] = vram_test(torch, dev, a.vram_gb)
            ok &= rep["vram"]["pass"]
    elif not a.allow_cpu:
        ok = False
    rep["pass"] = bool(ok)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
