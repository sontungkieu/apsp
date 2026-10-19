"""Stage pinned weights on remote CPU, isolate failures by generator/scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_registry, file_hash
from .storage import atomic_json


def main(argv=None):
    from huggingface_hub import snapshot_download, hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    registry = load_registry()
    jobs = {}
    for name, p in registry["profiles"].items():
        jobs[name] = [(p["repo"], p["revision"], None)]
        if p.get("adapter"):
            ad = p["adapter"]
            jobs[name].append((ad["repo"], ad["revision"], ad["filename"]))
    pe = registry["prompt_encoder"]
    jobs["prompt_encoder"] = [(pe["repo"], pe["revision"], "ENCODER")]
    ir = registry["scorers"]["imagereward"]
    jobs["imagereward"] = [("THUDM/ImageReward", ir["revision"], "ImageReward.pt"),
                           ("THUDM/ImageReward", ir["revision"], "med_config.json"),
                           ("google-bert/bert-base-uncased", ir["tokenizer_revision"], "TOKENIZER")]
    ps = registry["scorers"]["pickscore"]
    jobs["pickscore"] = [(ps["model_repo"], ps["model_revision"], None),
                         (ps["processor_repo"], ps["processor_revision"], "PROCESSOR")]
    report = {}
    order = ["prompt_encoder", "imagereward", "pickscore", "sdxl_lightning8", "sdxl_base50", "hyper_flux8"]
    for name in order:
        assets = jobs[name]
        paths = []
        try:
            for repo, revision, filename in assets:
                if filename == "ENCODER":
                    path = snapshot_download(repo, revision=revision, allow_patterns=["tokenizer/*", "text_encoder/*.json", "text_encoder/*.safetensors", "model_index.json"])
                elif filename == "TOKENIZER":
                    path = snapshot_download(repo, revision=revision, allow_patterns=["vocab.txt", "tokenizer_config.json", "config.json"])
                elif filename == "PROCESSOR":
                    path = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.txt", "*.model"])
                elif filename:
                    path = hf_hub_download(repo, filename, revision=revision)
                else:
                    ignore = ["*.onnx*", "*openvino*", "*.msgpack", "*.h5", "*.ckpt", "*.md", "*.png", "*.jpg", "*fp16*"]
                    allow = None
                    if name in registry["profiles"]:
                        ignore.append("*.bin")  # generators publish safetensors; PickScore may publish .bin only
                        allow = ["model_index.json", "scheduler/*.json", "tokenizer*/*", "text_encoder*/*.json",
                                 "text_encoder*/*.safetensors", "unet/config.json", "unet/*.safetensors",
                                 "transformer/*.json", "transformer/*.safetensors", "vae/config.json", "vae/*.safetensors"]
                    path = snapshot_download(repo, revision=revision, allow_patterns=allow, ignore_patterns=ignore)
                p = Path(path)
                hashes = {f.relative_to(p).as_posix(): file_hash(f) for f in p.rglob("*") if f.is_file()} if p.is_dir() else {p.name: file_hash(p)}
                paths.append({"repo": repo, "revision": revision, "files_sha256": hashes})
            report[name] = {"pass": True, "assets": paths}
        except Exception as exc:
            # HTTP exception messages may contain authenticated request details: type only.
            report[name] = {"pass": False, "error_type": type(exc).__name__, "stage": "asset_download", "assets": paths}
        atomic_json(a.out, report)
        print("APSP_ASSET_JSON=" + json.dumps({"name": name, "pass": report[name]["pass"]}), flush=True)
    # Failures remain explicit; a failing FLUX access check does not erase SDXL successes.
    print(json.dumps(report))


if __name__ == "__main__":
    main()
