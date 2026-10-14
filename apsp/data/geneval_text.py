"""GenEval prompt TEXT only (for de-duplication before PREREG-C). Never generates or scores GenEval images.

Source: the GenEval copy vendored in the pinned PSP submodule (553 prompts). The official manifest
(djghosh13/geneval) must be re-checked against this file before PREREG-C (docs/QUESTIONS.md).
"""
from __future__ import annotations

import json
from pathlib import Path

from apsp.contracts.hashing import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
PSP_GENEVAL = REPO_ROOT / "third_party" / "psp" / "geneval" / "prompts" / "evaluation_metadata.jsonl"


def load_geneval_texts(path: str | Path = PSP_GENEVAL) -> tuple[list[str], str]:
    p = Path(path)
    texts = [json.loads(l)["prompt"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return texts, file_sha256(p)
