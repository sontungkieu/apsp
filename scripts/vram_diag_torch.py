#!/usr/bin/env python3
"""Detailed re-run of the PyTorch VRAM pattern check (scripts/vram_check.py) with full evidence per mismatch.

Each block is a separate torch.empty(268,435,456 float32 = 1 GiB) allocation, all held alive at once (no reuse by
the caching allocator while referenced). Writes: block.fill_(i+1) on the default stream. Synchronisation:
torch.cuda.synchronize() after ALL writes, before ANY read. Check 1 on GPU (b != expected, chunked); check 2 on the
CPU: the first corrupted 1 MiB window is copied to host memory (.cpu()) and compared with numpy, so the comparison
does not depend on a GPU kernel. Records data_ptr, byte offset, expected/actual value and raw bits.

  python scripts/vram_diag_torch.py --blocks 19 --reps 2 --out diag.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

GIB = 1 << 30
N = GIB // 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=19)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "dtype": "float32",
           "block_bytes": GIB, "sync": "torch.cuda.synchronize() after all fills, before all reads", "reps": []}
    for rep in range(a.reps):
        blocks = []
        for i in range(a.blocks):
            b = torch.empty(N, dtype=torch.float32, device="cuda")
            b.fill_(float(i + 1))
            blocks.append(b)
        torch.cuda.synchronize()
        ptrs = [b.data_ptr() for b in blocks]
        assert len(set(ptrs)) == len(ptrs), "aliased blocks"
        overlaps = any(abs(p - q) < GIB for x, p in enumerate(ptrs) for q in ptrs[x + 1:])
        out = []
        for i, b in enumerate(blocks):
            exp = float(i + 1)
            nbad, first = 0, None
            for c in range(0, N, 1 << 24):
                part = b[c:c + (1 << 24)]
                m = part != exp
                k = int(m.sum())
                if k and first is None:
                    first = c + int(m.nonzero()[0, 0])
                nbad += k
            if not nbad:
                continue
            w0 = (first // (1 << 18)) * (1 << 18)                      # 1 MiB window (262,144 floats) on the host
            host = b[w0:w0 + (1 << 18)].cpu().numpy()
            hb = np.nonzero(host != np.float32(exp))[0]
            samples = [{"elem": int(w0 + j), "byte_off": int((w0 + j) * 4), "abs_addr": hex(ptrs[i] + (w0 + j) * 4),
                        "expected": exp, "actual": float(host[j]),
                        "expected_bits": f"{np.float32(exp).view(np.uint32):08x}",
                        "actual_bits": f"{host[j:j + 1].view(np.uint32)[0]:08x}"} for j in hb[:8]]
            out.append({"block": i, "data_ptr": hex(ptrs[i]), "expected": exp, "n_bad_gpu_check": nbad,
                        "n_bad_host_window": int(len(hb)), "first_bad_elem": first, "samples": samples})
        res["reps"].append({"rep": rep, "blocks": a.blocks, "ptr_overlap": overlaps, "corrupted_blocks": out})
        del blocks
        torch.cuda.empty_cache()
    res["verdict"] = "CORRUPTION" if any(r["corrupted_blocks"] for r in res["reps"]) else "clean"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"verdict": res["verdict"], "per_rep": [len(r["corrupted_blocks"]) for r in res["reps"]]}))
    return 0 if res["verdict"] == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
