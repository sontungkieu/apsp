"""Collector-vs-pinned-PSP equivalence on the first 10 calibration prompts, both sides generated now under the current
EPH (no stored M3 tensors). Wraps `collect_bank --test-equivalence` and writes the report to a file.

  python scripts/equivalence_fresh.py --prompt-pool POOL/prompt_pool.parquet --out RUN/calib/equivalence_fresh.json

Exit 5 unless status == measured and every delivered latent matches the reference exactly (A).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apsp.gpu import collect_bank as cb  # noqa: E402
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph  # noqa: E402
from apsp.gpu.engine import DenoiseEngine  # noqa: E402
from apsp.gpu.reference_psp import equivalence_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--prompt-pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args(argv)
    a.test_equivalence, a.calib_throughput, a.smoke_prompt, a.split = True, False, None, []
    prompts = cb._prompts_from_args(a)
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, _ = compute_eph(cfg, bundle, scorer, rt)
    engine = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    rep = equivalence_report(cfg, bundle, scoring, engine, prompts, a)
    rep.update({"eph": eph, "prompt_uids": list(prompts["prompt_uid"]), "vae_decode_chunk": cfg.get("vae_decode_chunk"),
                "preview_dtype": cfg.get("preview_dtype"), "source_of_ours": "fresh collector run (this process)"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "records"}, indent=1, default=str))
    return 0 if rep.get("status") == "measured" and rep.get("all_A_exact") else 5


if __name__ == "__main__":
    sys.exit(main())
