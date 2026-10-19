"""Exact release accounting. Missing/failed tasks never count as complete."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .contracts import canonical_hash, load_registry
from .prompts import read_jsonl
from .storage import atomic_json, verify_complete


def audit(tasks, root, backup_root=None):
    root = Path(root)
    counts = Counter()
    failures = []
    for task in tasks:
        path = root / "tasks" / task["task_id"]
        try:
            if not verify_complete(path):
                counts["missing_tasks"] += 1
                continue
            marker = json.loads((path / "COMPLETE.json").read_text())
            if marker["task_id"] != task["task_id"]:
                raise ValueError("marker identity mismatch")
            counts["terminal_tasks"] += 1
            counts["candidate_records"] += marker["n_candidates"]
            counts["valid_final_candidates"] += marker["n_valid_final"]
            counts["preview_records"] += marker["n_previews"]
            counts[marker["status"]] += 1
            if backup_root and verify_complete(Path(backup_root) / "tasks" / task["task_id"]):
                counts["backed_up_tasks"] += 1
        except Exception as exc:
            counts["corrupt_tasks"] += 1
            failures.append({"task_id": task["task_id"], "error_type": type(exc).__name__})
    expected_preview = sum(load_registry()["profiles"][t["profile"]]["checkpoints"] * 8 for t in tasks)
    return {"schema": "apsp.lite.release_audit/v1", "expected_tasks": len(tasks), "expected_candidates": len(tasks) * 8,
            "expected_previews": expected_preview, "counts": dict(counts), "failures": failures,
            "complete": counts["terminal_tasks"] == len(tasks) and not counts["corrupt_tasks"],
            "durable_backup_complete": bool(backup_root) and counts["backed_up_tasks"] == len(tasks),
            "efficacy_claim": False}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backup-root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = audit(read_jsonl(a.tasks), a.root, a.backup_root)
    atomic_json(a.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] and report["durable_backup_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
