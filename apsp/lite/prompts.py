"""Build Lite manifests from provenance-bearing text-only JSONL input.

Input columns: text_raw, source, source_row_id, source_revision, parent_group.
COCO requires image_id; structured requires template_family_id, scene_spec.
All source acquisition happens separately. Never fills a source quota with fake rows.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import QUOTAS, canonical_hash, file_hash, normalize, uid
from .storage import atomic_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A frozen manifest is immutable; explicit new output directory on rerun.
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _grams(text):
    words = re.findall(r"\w+", text.casefold())
    return set(zip(words, words[1:], words[2:])) or {tuple(words)}


def group_rows(rows, exclusions=(), near_threshold=0.85):
    # Connected components of lexical 3-gram Jaccard + declared parent groups.
    # Reports this detector's scope; does not claim semantic decontamination.
    prepared, by_uid = [], {}
    for row in rows:
        for key in ("text_raw", "source", "source_row_id", "source_revision", "parent_group"):
            if not row.get(key):
                raise ValueError(f"missing prompt provenance field {key}")
        text = normalize(row["text_raw"])
        if not text:
            raise ValueError("empty prompt")
        if row["source"] == "coco_captions" and "image_id" not in row:
            raise ValueError("COCO row needs image_id")
        if row["source"] == "structured" and not all(row.get(k) for k in ("template_family_id", "scene_spec")):
            raise ValueError("structured row needs template family and scene specification")
        p = {**row, "text": text, "prompt_uid": uid(text)}
        if p["prompt_uid"] in by_uid:
            by_uid[p["prompt_uid"]]["provenance_aliases"].append(row)
        else:
            p["provenance_aliases"] = [row]
            by_uid[p["prompt_uid"]] = p
            prepared.append(p)
    prepared.sort(key=lambda r: r["prompt_uid"])
    parents = list(range(len(prepared)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)

    inverted, declared = defaultdict(list), {}
    gs = [_grams(r["text"]) for r in prepared]
    ex = [_grams(normalize(t)) for t in exclusions]
    excluded = set()
    for i, r in enumerate(prepared):
        keys = set()
        for alias in r["provenance_aliases"]:
            keys.add((alias["source"], "parent", str(alias["parent_group"])))
            if alias["source"] == "coco_captions":
                keys.add(("coco_captions", "image", str(alias["image_id"])))
            if alias["source"] == "structured":
                keys.add(("structured", "template", str(alias["template_family_id"])))
        for key in keys:
            if key in declared:
                union(i, declared[key])
            declared[key] = i
        possible = set(j for g in gs[i] for j in inverted[g])
        for j in possible:
            if len(gs[i] & gs[j]) / len(gs[i] | gs[j]) >= near_threshold:
                union(i, j)
        for g in gs[i]:
            inverted[g].append(i)
        if any(len(gs[i] & eg) / len(gs[i] | eg) >= near_threshold for eg in ex):
            excluded.add(i)
    bad_roots = {find(i) for i in excluded}
    groups = defaultdict(list)
    for i, r in enumerate(prepared):
        if find(i) not in bad_roots:
            r["group_id"] = prepared[find(i)]["prompt_uid"]
            groups[r["group_id"]].append(r)
    return groups, {"raw_rows": len(rows), "unique_rows": len(prepared),
                    "excluded_groups": len(bad_roots), "exclusion_texts": len(ex),
                    "detector": "NFC-whitespace/exact/parent/lexical-3gram-Jaccard>=0.85",
                    "semantic_coverage": "not_measured"}


def build(rows, exclusions=()):
    groups, report = group_rows(rows, exclusions)
    assigned, taken = [], set()
    # Whole groups belong to a single split. A quota may select a subset of a group;
    # remaining group members never enter any other split.
    for split, targets in QUOTAS.items():
        for source, count in targets.items():
            pool = sorted(groups, key=lambda g: canonical_hash(["lite-split-v1", source, g]))
            selected = []
            for g in pool:
                if g in taken:
                    continue
                candidates = [r for r in groups[g] if r["source"] == source]
                if not candidates:
                    continue
                taken.add(g)
                selected.extend(candidates[:count - len(selected)])
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f"insufficient disjoint {source} groups for {split}: {len(selected)}/{count}")
            assigned.extend({**r, "split": split} for r in selected)
    train = sorted([r for r in assigned if r["split"] == "train"],
                   key=lambda r: canonical_hash(["nested-lite-v1", r["prompt_uid"]]))
    # Exact source proportions at every nested scale.
    ranked = []
    queues = {s: [r for r in train if r["source"] == s] for s in QUOTAS["train"]}
    pattern = ["diffusiondb"] * 5 + ["coco_captions"] * 2 + ["structured"] * 3
    for i in range(200):
        for s in pattern:
            ranked.append(queues[s].pop(0))
    order = {r["prompt_uid"]: i + 1 for i, r in enumerate(ranked)}
    for r in assigned:
        r["train_rank"] = order.get(r["prompt_uid"])
    report["counts"] = dict(Counter(r["split"] for r in assigned))
    report["full_subset_status"] = "independent_lite_manifest_full_not_locked"
    return assigned, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--exclusion", action="append", default=[], help="JSONL containing text/text_raw; GenEval/Parti/v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError("use a new immutable manifest directory")
    rows = read_jsonl(a.input)
    excludes = [r.get("text", r.get("text_raw")) for f in a.exclusion for r in read_jsonl(f)]
    # All Parti rows are exclusions for train/dev and ID, even unselected rows.
    # Reserve them by source before grouping to retain test_shift candidates.
    parti = [r for r in rows if r["source"] == "parti"]
    nonparti = [r for r in rows if r["source"] != "parti"]
    groups, coverage = group_rows(nonparti, excludes + [r["text_raw"] for r in parti])
    clean = [alias for group in groups.values() for r in group for alias in r["provenance_aliases"]]
    assigned, report = build(clean + parti)
    report["benchmark_exclusion"] = coverage
    report["v1_coverage"] = "provided_exclusion_files_only; missing_files_not_assumed"
    out.mkdir(parents=True)
    for split in QUOTAS:
        write_jsonl(out / f"{split}.jsonl", [r for r in assigned if r["split"] == split])
    for size in (250, 1000, 2000):
        write_jsonl(out / f"train{size}.jsonl", [r for r in assigned if r["split"] == "train" and r["train_rank"] <= size])
    write_jsonl(out / "exclusion_index.jsonl", [{"prompt_uid": r["prompt_uid"], "split": r["split"], "group_id": r["group_id"]} for r in assigned])
    hashes = {p.name: file_hash(p) for p in sorted(out.glob("*.jsonl"))}
    atomic_json(out / "manifest.json", {**report, "schema": "apsp.lite.prompts/v1", "input_sha256": file_hash(a.input),
                "exclusion_files": {str(p): file_hash(p) for p in a.exclusion}, "files_sha256": hashes,
                "manifest_hash": canonical_hash(hashes)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
