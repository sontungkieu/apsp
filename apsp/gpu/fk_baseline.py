"""External baseline: FK-Steering (Singhal et al.) on SD1.5, run ONLINE with the official FKD pipeline code vendored in
the pinned PSP submodule (third_party/psp/Fk-Diffusion-Steering/text_to_image/fkd_diffusers).

Kept from the official method: SMC resampling with ImageReward potentials on x0 previews, eta = 1 stochastic DDIM
sampler (FK needs a stochastic sampler), num_particles k, lambda, potential type, resampling interval, final pick of
the highest-reward particle. Adapted to this study (recorded in every episode): T = 64 steps, CFG 7.5, 512^2, the pinned
SD1.5 weights; the reward function is the project's pinned ImageReward scorer (same model/revision/scoring path) in
place of the package default download; seeds from the project scheme (apsp-v1, candidate 0 of the bank seeds the
generator). Resampling interval scaled from the official 100-step SD setting (every 20 steps over 20..80) to 64 steps
(every 13 over 13..51).

Timing: CUDA-synchronised e2e from before prompt encoding to the delivered PIL image (includes every reward call).
Quality: ImageReward of the delivered image (the inference score of the chosen particle).

  python -m apsp.gpu.fk_baseline --prompt-pool POOL --split train_dev --banks 0 --configs k4_max k4_diff k8_max k8_diff --out RUN/fk_dev
  python -m apsp.gpu.fk_baseline --select RUN/fk_dev --out STORE/fk_frozen.json
  python -m apsp.gpu.fk_baseline --prompt-pool POOL --split val --frozen STORE/fk_frozen.json --out RUN/fk_val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from apsp.gpu.backend import add_backend_args, build_backend
from apsp.gpu.runtime import import_torch

T_STEPS = 64
RESAMPLE = {"resample_frequency": 13, "resampling_t_start": 13, "resampling_t_end": 51}
LAMBDA = 10.0


def parse_config(name: str) -> dict:
    k, pot = name.split("_")
    return {"name": name, "num_particles": int(k[1:]), "potential_type": pot, "lmbda": LAMBDA, **RESAMPLE}


def prompts_for(a) -> pd.DataFrame:
    from apsp.contracts.prereg import require_split_access
    require_split_access(a.split, "external baseline FK-Steering")
    pool = pd.read_parquet(a.prompt_pool)
    sel = pool[pool["split"] == a.split].sort_values("prompt_uid")
    if sel["text"].isna().any():
        raise SystemExit("sealed texts")
    return sel.head(a.limit) if a.limit else sel


def run(a) -> int:
    torch = import_torch()
    from apsp.contracts.identity import derive_seed
    from apsp.gpu import reference_psp
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    ref = reference_psp.load_reference_pipeline(cfg, bundle)
    import fkd_diffusers.fkd_pipeline_sd as fkp

    def pinned_reward(name, images, prompts, metric_to_chase=None):        # project ImageReward (pinned)
        return [float(s) if np.isfinite(s) else -1e3 for s in scorer.score(prompts[0], list(images))]
    fkp.get_reward_function = pinned_reward
    configs = [parse_config(c) for c in a.configs] if a.configs else \
        [parse_config(c) for c in json.loads(Path(a.frozen).read_text())["configs"]]
    inst = prompts_for(a)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ep = out / "episodes.jsonl"
    done = {(json.loads(l)["prompt_uid"], json.loads(l)["bank_id"], json.loads(l)["method"])
            for l in ep.read_text().splitlines()} if ep.exists() else set()
    dev = bundle.device
    ref("a warm-up prompt for timing", num_inference_steps=4, eta=1.0, guidance_scale=cfg["cfg"])   # untimed
    with open(ep, "a", encoding="utf-8") as fh:
        for row in inst.itertuples(index=False):
            for b in a.banks:
                for c in configs:
                    m = f"fk:{c['name']}"
                    if (row.prompt_uid, b, m) in done:
                        continue
                    seed = derive_seed(row.prompt_uid, b, 0)
                    torch.manual_seed(seed)
                    gen = torch.Generator("cpu").manual_seed(seed)
                    fkd_args = {"lmbda": c["lmbda"], "num_particles": c["num_particles"], "use_smc": True,
                                "adaptive_resampling": False, "resample_frequency": c["resample_frequency"],
                                "time_steps": T_STEPS, "resampling_t_start": c["resampling_t_start"],
                                "resampling_t_end": c["resampling_t_end"], "guidance_reward_fn": "ImageReward",
                                "potential_type": c["potential_type"], "resampling": "multinomial",
                                "tempering_schedule": "constant"}
                    torch.cuda.synchronize(dev)
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    ok, best, score, err = False, None, None, None
                    try:
                        imgs = ref([row.text] * c["num_particles"], num_inference_steps=T_STEPS, eta=1.0,
                                   guidance_scale=cfg["cfg"], generator=gen, height=cfg["height"], width=cfg["width"],
                                   fkd_args=fkd_args)[0]
                        sc = scorer.score(row.text, list(imgs))                     # final pick (inference cost)
                        valid = [i for i, s in enumerate(sc) if np.isfinite(s)]
                        if valid:
                            bi = max(valid, key=lambda i: sc[i])
                            ok, best, score = True, imgs[bi], float(sc[bi])
                    except Exception as e:  # noqa: BLE001 - a failed episode is recorded, never retried free
                        err = f"{type(e).__name__}: {e}"[:300]
                    torch.cuda.synchronize(dev)
                    e2e = time.perf_counter() - t0
                    rec = {"prompt_uid": row.prompt_uid, "split": a.split, "bank_id": b, "method": m, "family": "fk",
                           "alpha": None, "config": c, "success": ok, "q_eval": score, "inference_score": score,
                           "q_eval_source": "inference_rm_same_image", "e2e_seconds": e2e, "seed": seed,
                           "n_candidates": c["num_particles"], "forwards": T_STEPS, "n_verify": None,
                           "ledger_spend_us": None, "rm_used_in_inference": True, "error": err,
                           "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved())}
                    if ok:
                        p = out / "images" / f"{row.prompt_uid}_b{b}_{c['name']}.png"
                        best.save(p)
                        rec.update({"image": str(p.relative_to(out)), "image_png_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
    (out / "manifest.json").write_text(json.dumps({
        "label": "external baseline FK-Steering (official FKD code, adapted settings in the module docstring)",
        "split": a.split, "configs": configs, "fkd_code": str(reference_psp.PSP_T2I), "steps": T_STEPS, "eta": 1.0,
        "reward": "project pinned ImageReward"}, indent=1), encoding="utf-8")
    return 0


def select(a) -> int:
    rows = [json.loads(l) for l in (Path(a.select) / "episodes.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if set(df["split"]) != {"train_dev"}:
        raise SystemExit("FK configuration is selected on train_dev only")
    df["Q"] = df["q_eval"].where(df["success"], np.nan)
    by = df.groupby("method").agg(mean_Q=("Q", "mean"), e2e=("e2e_seconds", "mean"), fail=("success", lambda s: 1 - s.mean()))
    picks = {}
    for m, r in by.iterrows():
        k = m.split(":")[1].split("_")[0]
        if k not in picks or r["mean_Q"] > by.loc[picks[k], "mean_Q"]:
            picks[k] = m
    frozen = {"selected_on": "train_dev", "rule": "per particle count, the potential with the higher train_dev mean IR",
              "configs": sorted(p.split(":")[1] for p in picks.values()), "dev_summary": json.loads(by.to_json(orient="index"))}
    Path(a.out).write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(json.dumps(frozen, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--prompt-pool")
    ap.add_argument("--split")
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return select(a) if a.select else run(a)


if __name__ == "__main__":
    sys.exit(main())
