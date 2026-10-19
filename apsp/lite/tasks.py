"""Immutable distributed task list. No test or held-out sampling in data jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import TRAIN_PROFILES, canonical_hash, file_hash, load_registry, noise_seed
from .prompts import read_jsonl, write_jsonl
from .storage import atomic_json


def make_tasks(prompts, registry=None):
    registry = registry or load_registry()
    result, seen = [], set()
    for row in prompts:
        if row["split"] not in ("train", "dev", "calibration"):
            raise ValueError("test sampling is locked until method freeze")
        if row["prompt_uid"] in seen:
            raise ValueError("duplicate prompt UID")
        seen.add(row["prompt_uid"])
        rank = row.get("train_rank")
        wave = 250 if rank and rank <= 250 else 1000 if rank and rank <= 1000 else 2000
        for profile in TRAIN_PROFILES:
            for bank in (0, 1):
                identity = [row["prompt_uid"], profile, bank]
                result.append({"task_id": canonical_hash(identity), "prompt": row, "profile": profile,
                    "bank_id": bank, "seeds": [noise_seed(row["prompt_uid"], bank, s) for s in range(8)],
                    "wave": wave, "profile_hash": canonical_hash(registry["profiles"][profile])})
    return sorted(result, key=lambda t: (t["wave"], t["task_id"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prompts = [r for p in a.prompts for r in read_jsonl(p)]
    tasks = make_tasks(prompts)
    write_jsonl(a.out, tasks)
    atomic_json(Path(a.out).with_suffix(".manifest.json"), {"schema": "apsp.lite.tasks/v1", "n_tasks": len(tasks),
        "n_candidates": len(tasks) * 8, "tasks_sha256": file_hash(a.out),
        "prompt_inputs": {p: file_hash(p) for p in a.prompts}, "registry_hash": canonical_hash(load_registry())})
    print(json.dumps({"tasks": len(tasks), "candidates": len(tasks) * 8}))


if __name__ == "__main__":
    main()
