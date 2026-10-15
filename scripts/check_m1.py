"""Compare an M1 report from the GPU box with the pinned config and a previous M1 report (provenance check).

  python scripts/check_m1.py NEW_M1.json [REFERENCE_M1.json]
Exit 0 if every check passes; prints each check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apsp.config import load_config  # noqa: E402


def main(argv) -> int:
    new = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    ref = json.loads(Path(argv[2]).read_text(encoding="utf-8")) if len(argv) > 2 and Path(argv[2]).exists() else None
    cfg = load_config()
    m, rm = new["M1"], new["reward_model"]
    tok = rm.get("bert_tokenizer") if isinstance(rm.get("bert_tokenizer"), dict) else {}
    checks = {
        "model_revision == config": m["model_revision"] == cfg["model_revision"],
        "64 timesteps": len(m["scheduler"]["timesteps"]) == cfg["steps"],
        "prediction_type epsilon": m["scheduler"]["prediction_type"] == "epsilon",
        "weights hashed": bool(m["weight_files_sha256"]),
        "RM revision == config": rm["revision"] == cfg["reward_revision"],
        "RM missing_keys empty": rm["missing_keys"] == [],
        "RM tokenizer revision == config": tok.get("revision") == cfg["reward_tokenizer_revision"],
        "RM tokenizer files hashed": bool(tok.get("files_sha256")),
    }
    if ref:
        rm0 = ref["M1"]
        checks["weight file sha256 == reference report"] = m["weight_files_sha256"] == rm0["weight_files_sha256"]
        checks["timesteps == reference report"] = m["scheduler"]["timesteps"] == rm0["scheduler"]["timesteps"]
        checks["RM ckpt sha256 == reference report"] = rm["ckpt_sha256"] == ref["reward_model"]["ckpt_sha256"]
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if rm.get("unexpected_keys"):
        print("NOTE RM unexpected keys (explain before PREREG-A):", rm["unexpected_keys"][:10])
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
