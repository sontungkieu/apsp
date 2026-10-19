"""Remote-only metadata preparation from official pinned/public sources; no image downloads.

Produces a candidate corpus plus benchmark/v1 exclusions. Text QA is still required.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import urllib.request
import zipfile
from pathlib import Path

from .contracts import canonical_hash, file_hash, normalize, uid
from .prompts import build, group_rows, write_jsonl
from .storage import atomic_json

DIFF_REV = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
GENEVAL_REV = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"


def download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as src, path.with_suffix(".partial").open("wb") as dst:
            while chunk := src.read(4 << 20): dst.write(chunk)
        path.with_suffix(".partial").replace(path)
    return file_hash(path)


def structured_rows():
    # Template ids hash the actual unfilled string, not an arbitrary row index.
    # Clause combinations create distinct template families; all variants stay together.
    contexts = ["in a sunlit courtyard", "inside a quiet library", "beside a rocky shoreline", "at a weekend craft market",
                "in a spacious workshop", "under a glass pavilion", "on a covered terrace", "near a mountain cabin"]
    finishes = ["Soft side lighting reveals the surface texture.", "The background is rendered as a delicate watercolor.",
                "A low camera angle emphasizes the foreground.", "Warm evening light falls across the scene."]
    patterns = {
        "subject_attributes": ["A {color} {object} with {material} details is displayed {context}. {finish}",
                               "The focal subject {context} is a {color} {object}; its trim is {material}. {finish}"],
        "relations": ["A {object} stands to the left of {other} {context}. {finish}",
                      "{context}, place {other} behind a {object}, leaving a clear gap between them. {finish}"],
        "count_binding": ["Exactly three {color} {plural} and two white bowls sit {context}. {finish}",
                          "{context}, a row contains two blue {plural} and one red {object}. {finish}"],
        "material_lighting_style": ["A still life of {material} {object} and {other} is arranged {context}. {finish}",
                                    "{context}, show {object} made from {material}, beside {other}. {finish}"],
        "multiple_constraints": ["{context}, a {color} {object} sits on a round table; {other} is behind it and a leaf lies in front. {finish}",
                                 "Place a {material} {object} below a shelf {context}; add {other} at the right and a folded cloth at the left. {finish}"],
    }
    objects = [("cup", "cups"), ("toy rabbit", "toy rabbits"), ("lantern", "lanterns"),
               ("carved bird", "carved birds"), ("vase", "vases"), ("miniature chair", "miniature chairs")]
    colors = ["red", "blue", "green", "amber", "violet", "white"]
    materials = ["brass", "wood", "ceramic", "silver", "bamboo", "copper"]
    for category, forms in patterns.items():
        for form, context, finish in itertools.product(forms, contexts, finishes):
            template = form.replace("{context}", context).replace("{finish}", finish)
            family = canonical_hash(["structured-lite-v1", category, template])
            for i, (obj, plural) in enumerate(objects):
                scene = {"object": obj, "plural": plural, "other": "a " + objects[(i + 2) % len(objects)][0],
                         "color": colors[i], "material": materials[i]}
                text = template.format(**scene)
                yield {"text_raw": text, "source": "structured", "source_revision": "structured-lite-v1",
                       "source_row_id": f"{family}:{i}", "parent_group": family, "template_family_id": family,
                       "scene_spec": {**scene, "context": context, "rendering_instruction": finish},
                       "primary_category": category, "template": template, "generation_seed": 0}


def main(argv=None):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    meta = Path(hf_hub_download("poloclub/diffusiondb", "metadata.parquet", repo_type="dataset", revision=DIFF_REV))
    if file_hash(meta) != "eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f":
        raise ValueError("DiffusionDB metadata hash mismatch")
    df = pd.read_parquet(meta, columns=["prompt", "prompt_nsfw"])
    candidates = []
    seen = set()
    for i, row in df.iterrows():
        text = normalize(str(row.prompt))
        letters = [c for c in text if c.isalpha()]
        if not 20 <= len(text) <= 700 or not letters or sum(c.isascii() for c in letters) / len(letters) < 0.97:
            continue
        if not pd.notna(row.prompt_nsfw) or row.prompt_nsfw >= 0.1 or uid(text) in seen:
            continue
        seen.add(uid(text))
        candidates.append({"text_raw": str(row.prompt), "source": "diffusiondb", "source_revision": DIFF_REV,
            "source_row_id": str(i), "parent_group": uid(text), "language_rule": "ascii-heuristic-review-required"})
        if len(candidates) == 6000: break
    del df
    # Same official COCO S3 bucket through a valid TLS hostname. The vanity
    # images.cocodataset.org HTTPS endpoint has a hostname-mismatched certificate.
    coco_url = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
    coco = raw / "annotations_trainval2017.zip"
    coco_sha = download(coco_url, coco)
    with zipfile.ZipFile(coco) as z:
        annotations = json.loads(z.read("annotations/captions_train2017.json"))["annotations"]
    images = set()
    for r in annotations:
        if r["image_id"] in images: continue
        images.add(r["image_id"])
        candidates.append({"text_raw": r["caption"], "source": "coco_captions", "source_revision": coco_sha,
            "source_row_id": str(r["id"]), "parent_group": str(r["image_id"]), "image_id": r["image_id"], "source_url": coco_url})
        if len(images) == 1800: break
    del annotations
    api = "https://api.github.com/repos/google-research/parti/commits/main"
    parti_rev = json.load(urllib.request.urlopen(api, timeout=30))["sha"]
    parti_path = raw / f"parti-{parti_rev}.tsv"
    parti_sha = download(f"https://raw.githubusercontent.com/google-research/parti/{parti_rev}/PartiPrompts.tsv", parti_path)
    parti = []
    for i, r in enumerate(csv.DictReader(parti_path.read_text().splitlines(), delimiter="\t")):
        parti.append({"text_raw": r["Prompt"], "source": "parti", "source_revision": parti_rev,
                      "source_row_id": str(i), "parent_group": uid(r["Prompt"]), "category": r["Category"], "challenge": r["Challenge"]})
    candidates.extend(structured_rows())
    geneval = raw / "geneval.jsonl"
    download(f"https://raw.githubusercontent.com/djghosh13/geneval/{GENEVAL_REV}/prompts/evaluation_metadata.jsonl", geneval)
    excludes = [json.loads(l)["prompt"] for l in geneval.read_text().splitlines() if l.strip()]
    v1_path = Path(__file__).resolve().parents[2] / "reports/prompt_pool/prompt_pool_review.csv"
    v1_table = pd.read_csv(v1_path, keep_default_na=False) if v1_path.exists() else None
    v1 = [t for t in v1_table.text if isinstance(t, str) and normalize(t)] if v1_table is not None else []
    excludes.extend(v1)
    # Reserve the separately authored GPU qualification prompt too.
    excludes.append("A red ceramic mug on a wooden table beside a blue book, natural window light.")
    groups, coverage = group_rows(candidates, excludes + [r["text_raw"] for r in parti])
    clean = [alias for group in groups.values() for r in group for alias in r["provenance_aliases"]]
    assigned, report = build(clean + parti)
    release = out / "manifests"
    release.mkdir(exist_ok=False)
    from .contracts import QUOTAS
    for split in QUOTAS:
        write_jsonl(release / f"{split}.jsonl", [r for r in assigned if r["split"] == split])
    for size in (250, 1000, 2000):
        write_jsonl(release / f"train{size}.jsonl", [r for r in assigned if r["split"] == "train" and r["train_rank"] <= size])
    hashes = {p.name: file_hash(p) for p in release.glob("*.jsonl")}
    report.update(schema="apsp.lite.prompts/v1", files_sha256=hashes, manifest_hash=canonical_hash(hashes),
        corpus_qa="pending_manual_and_language_review", exclusion_coverage=coverage,
        v1_available_prompts=len(v1), v1_missing_text_rows=0 if v1_table is None else len(v1_table) - len(v1),
        v1_coverage="bundle_review_csv_only_other_v1_manifests_unknown",
        source_hashes={"diffusiondb": file_hash(meta), "coco2017_official_annotations": coco_sha, "parti_tsv": parti_sha, "geneval": file_hash(geneval)},
        source_revisions={"diffusiondb": DIFF_REV, "parti": parti_rev, "geneval": GENEVAL_REV})
    atomic_json(release / "manifest.json", report)
    from .tasks import make_tasks
    tasks = make_tasks([r for r in assigned if r["split"] in ("train", "dev")])
    write_jsonl(release / "tasks.jsonl", tasks)
    print("APSP_PROMPTS_JSON=" + json.dumps(report), flush=True)


if __name__ == "__main__": main()
