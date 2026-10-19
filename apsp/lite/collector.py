"""Resident-model Lite collector: eight full trajectories, variable J, atomic task commit.

No controller training, pruning or test sampling. Retries retain the exact task/seed.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

from .contracts import SEED_SCHEME, canonical_hash, checkpoints, file_hash, load_registry, noise_seed
from .prompts import read_jsonl
from .storage import TaskBusy, atomic_json, backup_task, claim, commit_task, verify_complete


def fingerprint():
    import torch
    root = Path(__file__).resolve().parents[2]
    packages = ("torch", "torchvision", "diffusers", "transformers", "peft", "safetensors", "image-reward")
    code = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "apsp").rglob("*.py"))}
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    return {"packages": {p: md.version(p) for p in packages}, "python": platform.python_version(),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "lock_sha256": file_hash(root / "env/lite/uv.lock"), "source_hash": canonical_hash(code),
        "source_commit": commit, "hostname": socket.gethostname(), "seed_scheme": SEED_SCHEME}


def runtime_identity(fp):
    # Host and source commit labels can vary across equivalent overlay mounts; content hash cannot.
    return {k: v for k, v in fp.items() if k not in ("hostname", "source_commit")}


def task_contract(task, runtime):
    return canonical_hash({"task": task, "runtime": runtime_identity(runtime), "schema": "apsp.lite.task/v1"})


def collect_task(runtime, task, root, fp):
    import numpy as np
    from safetensors.torch import save_file
    from .runtime import CleanTap

    torch, p = runtime.torch, runtime.spec
    expected_seeds = [noise_seed(task["prompt"]["prompt_uid"], task["bank_id"], s) for s in range(8)]
    if task["seeds"] != expected_seeds or task["profile_hash"] != canonical_hash(p):
        raise ValueError("task seeds/profile differ from frozen contract")
    if task["prompt"]["split"] not in ("train", "dev", "calibration"):
        raise ValueError("test sampling is forbidden")
    root = Path(root)
    tasks_dir = root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    dest = tasks_dir / task["task_id"]
    contract_hash = task_contract(task, fp)
    if verify_complete(dest, contract_hash):
        return dest
    ck = checkpoints(p["steps"], p["checkpoints"])
    stage = root / "staging" / f"{task['task_id']}-{time.time_ns()}"
    stage.mkdir(parents=True, exist_ok=False)
    # Failures leave staging evidence. A new attempt never reuses partial files.
    latents = [torch.randn((1, runtime.C, runtime.h, runtime.w), generator=torch.Generator("cpu").manual_seed(seed),
                          dtype=torch.float32) for seed in expected_seeds]
    cpu_hashes = [__import__("hashlib").sha256(z.numpy().tobytes()).hexdigest() for z in latents]
    initial = torch.cat(latents).to(runtime.dtype)
    if runtime.flux:
        initial = runtime.pipe._pack_latents(initial, 8, runtime.C, runtime.h, runtime.w)
    tap = CleanTap(runtime.pipe.scheduler, ck)
    before = runtime.denoiser_calls
    call = dict(prompt=task["prompt"]["text"], num_inference_steps=p["steps"], guidance_scale=p["guidance"],
                height=runtime.H, width=runtime.W, latents=initial.to(runtime.device),
                num_images_per_prompt=8, output_type="latent")
    if p.get("max_sequence_length"):
        call["max_sequence_length"] = p["max_sequence_length"]
    runtime.sync(); t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            final = runtime.pipe(**call).images
    finally:
        runtime.pipe.scheduler.step = tap.original
    runtime.sync(); generation_s = time.perf_counter() - t0
    calls = runtime.denoiser_calls - before
    if calls != p["steps"] or tap.n != p["steps"] or sorted(tap.x0) != ck:
        raise RuntimeError(f"forward/checkpoint contract failed: {calls}/{tap.n}/{sorted(tap.x0)}")
    previews = torch.stack([runtime.unpack(tap.x0[k]) for k in ck], 1).to(runtime.dtype).contiguous().cpu()
    final = runtime.unpack(final).to(runtime.dtype).contiguous().cpu()
    actual_initial = runtime.unpack(tap.initial).to(runtime.dtype).contiguous().cpu()
    (stage / "images").mkdir()
    t0 = time.perf_counter()
    save_file({"x0": previews, "final": final, "initial": actual_initial}, str(stage / "latents.safetensors"),
              metadata={"schema": "apsp.lite.latent/v1", "checkpoint_steps": json.dumps(ck),
                "layout": "unpacked_NCHW", "dtype": p["dtype"], "contract_hash": contract_hash,
                "vae_scaling_factor": str(runtime.pipe.vae.config.scaling_factor),
                "vae_shift_factor": str(getattr(runtime.pipe.vae.config, "shift_factor", None)),
                "preview_origin": "scheduler_clean_prediction_from_existing_forward_cast_to_decode_storage_dtype"})
    dump_s = time.perf_counter() - t0
    seed_rows, preview_rows = [], []
    samples = {"generation_s": generation_s, "decode_s": 0., "imagereward_s": 0., "pickscore_s": 0., "artifact_dump_s": dump_s}
    for j, tensor in [(i + 1, previews[:, i]) for i in range(len(ck))] + [(None, final)]:
        images, flags, irs, pss, cost = runtime.decode_score(task["prompt"]["text"], tensor)
        for key, value in cost.items(): samples[key] += value
        for s in range(8):
            row = {"prompt_uid": task["prompt"]["prompt_uid"], "profile": task["profile"], "bank_id": task["bank_id"],
                   "seed_idx": s, "candidate_id": [task["bank_id"], s], "ir": irs[s], "pickscore_raw": pss[s], **flags[s]}
            if j is not None:
                preview_rows.append({**row, "j": j, "step": ck[j - 1], "progress": ck[j - 1] / p["steps"], **tap.meta[ck[j - 1]],
                                     "cfg_branches_available": p["cfg_branches_available"]})
            else:
                fn = stage / "images" / f"s{s}.png"
                if images[s] is not None:
                    t0 = time.perf_counter(); images[s].save(fn); samples["artifact_dump_s"] += time.perf_counter() - t0
                init_hash = __import__("hashlib").sha256(actual_initial[s].view(torch.uint8).numpy().tobytes()).hexdigest()
                seed_rows.append({**row, "seed": expected_seeds[s], "cpu_fp32_noise_sha256": cpu_hashes[s],
                                  "actual_initial_sha256": init_hash, "png": f"images/s{s}.png" if fn.exists() else None})
    from apsp.data.prompts import _clip_functions
    if not hasattr(runtime, "embed"):
        pe = runtime.registry["prompt_encoder"]
        _, runtime.embed = _clip_functions(pe["repo"], pe["revision"], device=str(runtime.device))
    t0 = time.perf_counter()
    embedding = runtime.embed([task["prompt"]["text"]]).astype(np.float16)
    if embedding.shape != (1, 768) or not np.isfinite(embedding).all():
        raise RuntimeError("common prompt encoder contract failed")
    np.save(stage / "prompt_emb.npy", embedding)
    samples["prompt_encoder_setup_and_encode_s"] = time.perf_counter() - t0
    import pandas as pd
    pd.DataFrame(seed_rows).to_parquet(stage / "seeds.parquet", index=False)
    pd.DataFrame(preview_rows).to_parquet(stage / "previews.parquet", index=False)
    atomic_json(stage / "task.json", task)
    atomic_json(stage / "timings.json", {**samples, "denoiser_calls": calls, "batch_k": 8,
        "decode_chunk": runtime.decode_chunk, "deployment_cost_profile": False,
        "note": "collector timings include both scorers; do not substitute for active-verifier deployment ledger"})
    atomic_json(stage / "provenance.json", {"runtime": fp, "profile": p, "adapter": runtime.adapter_provenance,
        "scorers": runtime.scorer_provenance, "prompt_encoder": runtime.registry["prompt_encoder"],
        "scheduler": dict(runtime.pipe.scheduler.config), "decode_chunk": runtime.decode_chunk,
        "vae_dtype": str(runtime.vae_dtype), "checkpoint_steps": ck})
    valid = sum(r["valid"] for r in seed_rows)
    commit_task(stage, dest, {"schema": "apsp.lite.completion/v1", "contract_hash": contract_hash,
        "task_id": task["task_id"], "n_candidates": 8, "n_previews": 8 * len(ck),
        "n_valid_final": valid, "status": "complete" if valid == 8 and all(r["valid"] for r in preview_rows) else "complete_with_invalid"})
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backup-root", required=True)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-wave", type=int, choices=[250, 1000, 2000], default=2000)
    ap.add_argument("--qualification", required=True)
    a = ap.parse_args(argv)
    primary, backup = Path(a.out).resolve(), Path(a.backup_root).resolve()
    if primary == backup or primary in backup.parents or backup in primary.parents:
        raise ValueError("backup root must be separate from the primary output tree")
    if not 0 <= a.worker_index < a.workers:
        raise ValueError("invalid worker partition")
    registry = load_registry()
    receipt = json.loads(Path(a.qualification).read_text())
    if not receipt.get("pass") or receipt.get("profile_hash") != canonical_hash(registry["profiles"][a.profile]):
        raise ValueError("real B200 profile qualification is required")
    from .runtime import LiteRuntime
    runtime = LiteRuntime(registry["profiles"][a.profile], registry)
    runtime.load_scorers()
    fp = fingerprint()
    if receipt.get("runtime_identity_hash") != canonical_hash(runtime_identity(fp)):
        raise ValueError("qualification runtime/source differs from collector; requalify")
    tasks = [t for t in read_jsonl(a.tasks) if t["profile"] == a.profile and t["wave"] <= a.max_wave]
    tasks = tasks[a.worker_index::a.workers]
    failures = 0
    for task in tasks:
        # Claim contention is recorded as busy, never stolen or counted as completion.
        try:
            with claim(a.out, task["task_id"]):
                for attempt in range(3):
                    try:
                        dest = collect_task(runtime, task, a.out, fp)
                        backup_task(dest, Path(a.backup_root) / "tasks" / task["task_id"])
                        break
                    except (ValueError, FileExistsError):
                        raise  # contract/corruption errors must not be retried as infrastructure
                    except Exception as exc:
                        atomic_json(Path(a.out) / "failures" / f"{task['task_id']}-{attempt}.json",
                            {"task": task, "attempt": attempt + 1, "error_type": type(exc).__name__, "status": "infrastructure_failed"})
                        if attempt == 2:
                            failures += 1
                        runtime.torch.cuda.empty_cache()
        except TaskBusy:
            print(f"TASK_BUSY={task['task_id']}", flush=True)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
