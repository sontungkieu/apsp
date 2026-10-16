"""Evaluation-only metrics independent of ImageReward (Guide CC-8). Lives outside env / features / policies / trainer:
nothing in apsp.sim, apsp.policies or apsp.contracts may import this module (tests/contract/test_evaluator_isolation.py).

Scores the DELIVERED images of a benchmark run after the fact. The values never feed a decision and are never added
to inference cost. Guards: the split of every episode passes `require_split_access(split, "evaluation")`; GenEval
(images or scoring) is refused before PREREG-C and is not implemented here.

Evaluators:
  hpsv2   Human Preference Score v2.1 via the `hpsv2` package code with every artifact pinned in
          configs/evaluators.yaml (package, HPS checkpoint, OpenCLIP base weights, tokenizer BPE); unpinned -> refused.
  fake    deterministic image statistic, TEST ONLY (labelled; never a result)

  python -m apsp.gpu.evaluators.independent --run DIR --metric hpsv2 --config configs/evaluators.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


class EvaluatorError(RuntimeError):
    pass


class FakeEvaluator:
    name = "fake"
    manifest = {"name": "fake", "note": "TEST ONLY: mean pixel intensity in [0,1]; not a quality metric"}

    def score(self, prompt: str, images: Sequence) -> list[float]:
        return [float(np.asarray(im.convert("L"), np.float64).mean() / 255.0) for im in images]


HPS_REQUIRED = ("package_version", "checkpoint_repo", "checkpoint_file", "checkpoint_revision", "checkpoint_sha256",
                "base_repo", "base_file", "base_revision", "base_sha256", "tokenizer_bpe_sha256")


def _sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HPSv2Evaluator:
    """HPS v2.1 with EVERY loaded artifact pinned (audit of hpsv2 1.2.0, 2026-10-16):
      - `hpsv2.score()` must NOT be used: without `cp` it downloads the latest HPS checkpoint from the hub, it builds
        the model with the OpenCLIP tag 'laion2B-s32B-b79K' (an unpinned ~3.9 GB download of
        laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin at the hub's current revision), its default
        hps_version is "v2.0", and it reloads the checkpoint on every call.
      - Here: the base OpenCLIP weights and the HPS checkpoint are fetched at pinned revisions and verified by sha256;
        the model is built from the LOCAL base file (same create_model_and_transforms arguments as the package), then
        the HPS state_dict is loaded once; the BPE vocabulary file of the package tokenizer is verified by sha256.
      - Scoring math is the package's: amp autocast, model(image, text) -> image_features @ text_features.T, diagonal.
    Runs in a separate venv (hpsv2 1.2.0 pins protobuf<4 and pytest==7.2.0)."""
    name = "hpsv2"

    def __init__(self, spec: dict, device: str | None = None):
        missing = [k for k in HPS_REQUIRED if not spec.get(k)]
        if missing:
            raise EvaluatorError(f"hpsv2 is not pinned ({', '.join(missing)} null in configs/evaluators.yaml); "
                                 "refusing to run an unpinned evaluator")
        try:
            import hpsv2  # noqa: F401
            from importlib.metadata import version
        except ImportError as e:
            raise EvaluatorError("hpsv2 is not installed in this environment") from e
        got = version("hpsv2")
        if got != spec["package_version"]:
            raise EvaluatorError(f"hpsv2 {got} installed, {spec['package_version']} pinned")
        if spec.get("hps_version", "v2.1") != "v2.1":
            raise EvaluatorError("only HPS v2.1 is declared for this project")
        import torch
        from huggingface_hub import hf_hub_download
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.src.open_clip.tokenizer import default_bpe
        self.spec, self.hps_version = spec, "v2.1"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base = hf_hub_download(spec["base_repo"], spec["base_file"], revision=spec["base_revision"])
        ckpt = hf_hub_download(spec["checkpoint_repo"], spec["checkpoint_file"], revision=spec["checkpoint_revision"])
        for path, want, what in ((base, spec["base_sha256"], "base OpenCLIP weights"),
                                 (ckpt, spec["checkpoint_sha256"], "HPS checkpoint"),
                                 (default_bpe(), spec["tokenizer_bpe_sha256"], "tokenizer BPE vocabulary")):
            got_sha = _sha256_file(path)
            if got_sha != want:
                raise EvaluatorError(f"{what} sha256 {got_sha} != pinned {want}")
        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14", base, precision="amp", device=self.device, jit=False, force_quick_gelu=False,
            force_custom_text=False, force_patch_dropout=False, force_image_size=None, pretrained_image=False,
            image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False)
        sd = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(sd["state_dict"])
        self.model, self.preprocess, self.tokenizer = model.to(self.device).eval(), preprocess_val, get_tokenizer("ViT-H-14")
        self.manifest = {"name": "hpsv2", "package_version": got, "hps_version": self.hps_version,
                         **{k: spec[k] for k in HPS_REQUIRED if k != "package_version"}, "precision": "amp",
                         "device": self.device, "note": "evaluation-only; never used for selection or inference cost"}

    def score(self, prompt: str, images: Sequence) -> list[float]:
        import torch
        out = []
        text = self.tokenizer([prompt]).to(device=self.device)
        for im in images:
            with torch.no_grad():
                x = self.preprocess(im).unsqueeze(0).to(device=self.device)
                with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                    o = self.model(x, text)
                    v = torch.diagonal(o["image_features"] @ o["text_features"].T).float().cpu().numpy()
            out.append(float(v[0]))
        return out


def make_evaluator(metric: str, cfg: dict):
    if metric == "fake":
        return FakeEvaluator()
    if metric == "hpsv2":
        return HPSv2Evaluator(cfg.get("hpsv2", {}))
    if metric.startswith("geneval"):
        raise EvaluatorError("GenEval is not run before PREREG-C (Spec §10.1); not implemented in this evaluator")
    raise EvaluatorError(f"unknown metric {metric}")


def evaluate_run(run: Path, metric: str, cfg: dict) -> dict:
    from PIL import Image
    from apsp.contracts.prereg import require_split_access
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for s in sorted({e["split"] for e in eps}):
        require_split_access("test" if s == "test" else s, "evaluation")
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    texts = cfg.get("_texts") or {}
    ev = make_evaluator(metric, cfg)
    out_p = run / "eval_metrics.jsonl"
    done = set()
    if out_p.exists():
        for l in out_p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["prompt_uid"], r["bank_id"], r["method"], r["metric"]))
    n = 0
    t0 = time.perf_counter()
    with open(out_p, "a", encoding="utf-8") as fh:
        for e in eps:
            key = (e["prompt_uid"], e["bank_id"], e["method"], ev.name)
            if key in done or not e.get("success") or not e.get("image"):
                continue
            p = run / e["image"]
            if hashlib.sha256(p.read_bytes()).hexdigest() != e["image_png_sha256"]:
                raise EvaluatorError(f"image {p} does not match the episode record (identity join failed)")
            text = texts.get(e["prompt_uid"])
            if text is None:
                raise EvaluatorError(f"prompt text for {e['prompt_uid']} not provided")
            v = ev.score(text, [Image.open(p).convert("RGB")])[0]
            fh.write(json.dumps({"prompt_uid": e["prompt_uid"], "bank_id": e["bank_id"], "method": e["method"],
                                 "metric": ev.name, "value": v, "image_png_sha256": e["image_png_sha256"],
                                 "evaluation_only": True}) + "\n")
            n += 1
    m = {"metric": ev.name, "evaluator": ev.manifest, "n_scored": n, "seconds": time.perf_counter() - t0,
         "run_split": man.get("split"), "note": "evaluation-only; never used for decisions or inference cost"}
    (run / f"eval_manifest_{ev.name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def prompt_texts(run: Path, prompt_pool: str | None) -> dict:
    import pandas as pd
    if prompt_pool:
        df = pd.read_parquet(prompt_pool)
        return dict(zip(df["prompt_uid"], df["text"]))
    from apsp.data.prompts import normalize, prompt_uid
    sc = yaml.safe_load(Path("configs/technical_smoke.yaml").read_text(encoding="utf-8"))
    return {prompt_uid(normalize(p)): normalize(p) for p in sc["prompts"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evaluation-only metrics independent of ImageReward")
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--config", default="configs/evaluators.yaml")
    ap.add_argument("--prompt-pool", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) if Path(a.config).exists() else {}
    cfg["_texts"] = prompt_texts(Path(a.run), a.prompt_pool)
    print(json.dumps(evaluate_run(Path(a.run), a.metric, cfg), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
