"""Real B200 production-eight-seed qualification, with independent stage receipts."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

from .contracts import canonical_hash, load_registry, uid
from .collector import collect_task, fingerprint, runtime_identity
from .storage import atomic_json, backup_task
from .tasks import make_tasks


def main(argv=None):
    import torch
    from .runtime import LiteRuntime
    from .validate import validate_task
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("requires a real sm_100 B200")
    z = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    if not bool(torch.isfinite(z @ z).all()):
        raise RuntimeError("CUDA BF16 check failed")
    fp, registry = fingerprint(), load_registry()
    cuda_receipt = {"pass": True, "stage": "cuda", "runtime": fp}
    atomic_json(out / "cuda.json", cuda_receipt)
    text = "A red ceramic mug on a wooden table beside a blue book, natural window light."
    prompt = {"prompt_uid": uid(text), "text": text, "text_raw": text, "split": "calibration", "train_rank": None,
              "source": "qualification_authored", "group_id": "qualification-calibration-reserved-v1"}
    tasks = [t for t in make_tasks([prompt]) if t["bank_id"] == 0]
    result = {"cuda": cuda_receipt, "profiles": {}, "qualification_prompt_uid": uid(text),
              "scope": "one calibration prompt x eight seeds per profile; not 50-prompt replay-online qualification"}
    assets_path = out.parent / "assets.json"
    assets = json.loads(assets_path.read_text()) if assets_path.exists() else {}
    common_missing = [s for s in ("imagereward", "pickscore", "prompt_encoder") if not assets.get(s, {}).get("pass")]
    for task in tasks:
        name = task["profile"]
        runtime = None
        stage = "generator_load"
        try:
            if common_missing:
                result["profiles"][name] = {"pass": False, "status": "blocked_by_common_dependency", "dependencies": common_missing}
                continue
            if not assets.get(name, {}).get("pass"):
                result["profiles"][name] = {"pass": False, "status": "blocked_by_generator_assets"}
                continue
            runtime = LiteRuntime(registry["profiles"][name], registry)
            stage = "scorer_load"
            runtime.load_scorers()
            stage = "collector"
            attempt_root = out / "attempts" / canonical_hash(runtime_identity(fp))[:16]
            dest = collect_task(runtime, task, attempt_root / "bank", fp)
            stage = "validator_redecode"
            validation = validate_task(dest, runtime)
            stage = "backup_restore"
            backup_task(dest, attempt_root / "backup" / task["task_id"])
            restored = validate_task(attempt_root / "backup" / task["task_id"])
            receipt = {"pass": validation["all_valid"] and restored["schema_valid"], "profile": name,
                       "profile_hash": task["profile_hash"], "runtime_identity_hash": canonical_hash(runtime_identity(fp)),
                       "validation": validation, "restore": restored, "runtime": fp}
            atomic_json(out / f"{name}.json", receipt)
            result["profiles"][name] = receipt
        except Exception as exc:
            result["profiles"][name] = {"pass": False, "stage": stage, "error_type": type(exc).__name__, "error": str(exc)[:500]}
            traceback.print_exc()
        finally:
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
            result["pass"] = len(result["profiles"]) == 3 and all(r.get("pass") for r in result["profiles"].values())
            atomic_json(out / "summary.json", result)
    print("APSP_B200_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
