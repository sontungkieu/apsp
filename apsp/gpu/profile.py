"""P0 profiling stages (CC-1..CC-5). Timing: host perf_counter with CUDA synchronisation before/after each
primitive, warm-up 10, n >= 50 (``--smoke`` lowers both and marks the output NOT usable for cost tables).
Primitives are timed through the same engine/scoring functions the online runner uses.

  python -m apsp.gpu.profile --config configs/pilot.yaml --stage model
  python -m apsp.gpu.profile --config configs/pilot.yaml --stage hook
  python -m apsp.gpu.profile --config configs/pilot.yaml --stage validity --prompts-file P.jsonl
  python -m apsp.gpu.profile --config configs/pilot.yaml --stage generator
  python -m apsp.gpu.profile --config configs/pilot.yaml --stage batch-parity
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from apsp.contracts.hashing import package_code_sha, provenance_hash
from apsp.contracts.identity import RowMap, bank_rows, pool_rows, select_top
from apsp.contracts.ledger import ceil_us
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.engine import DenoiseEngine
from apsp.gpu.runtime import import_torch, sync
from apsp.sim.schedules import required_prune_pairs

DEFAULT_PROMPT = "a photograph of an astronaut riding a horse"   # technical prompt outside the research pool


def _timeit(fn, device, warmup: int, n: int, setup=None) -> list[float]:
    out = []
    for i in range(warmup + n):
        if setup:
            setup()
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        dt = (time.perf_counter() - t0) * 1e6
        if i >= warmup:
            out.append(dt)
    return out


def _summ(s: list[float]) -> dict:
    a = np.asarray(s)
    m = float(a.mean())
    return {"n": int(a.size), "mean_us": m, "p50_us": float(np.median(a)), "p95_us": float(np.quantile(a, 0.95)),
            "cv": float(a.std() / m) if m else 0.0, "cv_flag": bool(m and a.std() / m > 0.15)}


def _rows(k: int):
    return bank_rows(0, k) if k <= 8 else pool_rows(2)[:k]


def stage_model(cfg, bundle, scorer, eph, a) -> dict:
    from apsp.gpu.model import write_model_report
    rep = write_model_report(bundle, getattr(scorer, "provenance", {}), Path(a.out_dir) / "M1_model.json")
    return {"M1": "written", "path": str(Path(a.out_dir) / "M1_model.json"), "eph": eph,
            "revision": bundle.provenance.get("model_revision"), "rm_missing_keys": rep["reward_model"].get("missing_keys"),
            "rm_unexpected_keys": rep["reward_model"].get("unexpected_keys")}


def stage_hook(cfg, bundle, scoring, a) -> dict:
    """M0: repeat determinism, no extra forward, hook x0 vs DDIM-transition x0 (diagnostic)."""
    torch = import_torch()
    out = {}
    runs = []
    for mode in ("none", "none", "collect"):
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode=mode)
        eng.prepare_request(a.prompt, "m0-technical")
        eng.start(bank_rows(0))
        x0s, prev, inv_err = {}, None, []
        sched = bundle.scheduler
        for f in range(cfg["steps"]):
            before = eng.x.clone()
            eng._forward()
            if (f + 1) % cfg["delta"] == 0:
                x0s[(f + 1) // cfg["delta"]] = eng.hook.x0.float().cpu()
                # DDIM inversion from (before, after) -> x0, like PSP's estimate_x0_from_ddim_transition (diagnostic)
                t = int(eng.timesteps[f])
                ratio = sched.config.num_train_timesteps // cfg["steps"]
                ac = sched.alphas_cumprod
                a_t = float(ac[t])
                a_p = float(ac[t - ratio]) if t - ratio >= 0 else float(sched.final_alpha_cumprod)
                c1 = math.sqrt(a_p) / math.sqrt(a_t)
                c2 = math.sqrt(1 - a_p) - c1 * math.sqrt(1 - a_t)
                if abs(c2) > 1e-8:
                    eps = (eng.x.float() - c1 * before.float()) / c2
                    x0 = (before.float() - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
                    inv_err.append(float((x0 - eng.hook.x0.float()).abs().max()))
        runs.append({"mode": mode, "unet_calls": eng.n_unet_calls, "x": eng.x.float().cpu(), "x0": x0s,
                     "feature_kernel_calls": eng.hook.feature_kernel_calls, "inv_err": inv_err})
    a0, a1, c = runs
    out["no_extra_forward"] = all(r["unet_calls"] == cfg["steps"] for r in runs)
    out["unet_calls_per_run"] = [r["unet_calls"] for r in runs]
    out["mode_none_feature_kernel_calls"] = a0["feature_kernel_calls"]
    out["repeat_delivered_max_abs"] = float((a0["x"] - a1["x"]).abs().max())
    out["repeat_delivered_exact"] = bool(torch.equal(a0["x"], a1["x"]))
    out["repeat_x0_max_abs_by_j"] = {j: float((a0["x0"][j] - a1["x0"][j]).abs().max()) for j in a0["x0"]}
    out["collect_vs_none_delivered_exact"] = bool(torch.equal(a0["x"], c["x"]))
    out["hook_x0_vs_ddim_inversion_max_abs"] = max(a0["inv_err"]) if a0["inv_err"] else None
    out["note"] = "inversion error is fp16 diagnostic only; exact flags are measured, not assumed"
    return out


def stage_validity(cfg, bundle, scoring, a) -> dict:
    """M3: validity rates per checkpoint and delivered, for the preview precision in use."""
    prompts = _load_prompts(a)
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
    J = cfg["steps"] // cfg["delta"]
    fails = {"latent": np.zeros(J, int), "image": np.zeros(J, int), "score": np.zeros(J, int)}
    dfail = {"latent": 0, "image": 0, "score": 0}
    n = 0
    for uid, text in prompts:
        eng.prepare_request(text, uid)
        eng.start(bank_rows(0))
        for j in range(1, J + 1):
            eng.advance(None, cfg["delta"], False)
            for r in eng.scoring.score_preview(text, eng.hook.x0).records:
                fails["latent"][j - 1] += not r.latent_ok
                fails["image"][j - 1] += r.latent_ok and not r.img_ok
                fails["score"][j - 1] += r.img_ok and not r.score_ok
        for r in eng.scoring.score_delivered(text, eng.x).records:
            dfail["latent"] += not r.latent_ok
            dfail["image"] += r.latent_ok and not r.img_ok
            dfail["score"] += r.img_ok and not r.score_ok
        n += 8
    rates = {k: (v / n).tolist() for k, v in fails.items()}
    worst = max(max(v) for v in rates.values())
    dworst = max(v / n for v in dfail.values())
    return {"preview_dtype": cfg.get("preview_dtype"), "n_prompts": len(prompts), "n_previews": n * J,
            "preview_failure_rate_by_j": rates, "delivered_failures": dfail, "delivered_failure_rate": dworst,
            "escalate_to_research": bool(worst > 1e-3 or dworst > 1e-3),
            "required_n_previews_per_precision": 6400}


def stage_generator(cfg, bundle, scoring, a) -> dict:
    """M2: den(k), ver(k), fin(k), prune(k->k'), feat(k), lfeat(k), fixed(k0) in microseconds."""
    torch = import_torch()
    dev = bundle.device
    W, N = (2, 5) if a.smoke else (10, 50)
    ks = a.ks or list(range(1, 17))
    res = {"den": {}, "ver": {}, "fin": {}, "prune": {}, "feat": {}, "lfeat": {}, "fixed": {}}
    raw = {}
    for k in ks:
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="traj")
        eng.prepare_request(a.prompt, "m2-technical")
        rows = _rows(k)
        # fixed(k0): text encoding + latent prep + scheduler setup for a k0-row request
        raw[f"fixed_{k}"] = _timeit(lambda: (eng.prepare_request(a.prompt, "m2-technical"), eng.start(rows)), dev, W, N)
        eng.start(rows)
        for _ in range(cfg["delta"]):
            eng._forward()
        x_ck, f_ck = eng.x.clone(), eng.forward_idx
        # den(k): one CFG forward + scheduler step at representative timesteps (first, mid, late checkpoints)
        den = []
        eng.hook.mode = "none"          # den excludes trajectory-feature kernels (timed separately as feat)
        for f_idx in (3, 31, 59):
            def setup(fi=f_idx):
                eng.x = x_ck.clone()
                eng.forward_idx = fi
            den += _timeit(eng._forward, dev, W, N, setup=setup)
        raw[f"den_{k}"] = den
        eng.hook.mode = "traj"
        eng.x, eng.forward_idx = x_ck.clone(), f_ck
        eng._forward(); eng._forward(); eng._forward(); eng._forward()          # reach a checkpoint with traj
        x0, (eu, ec), prev = eng.hook.x0, eng.last_eps, eng.hook.prev_x0
        from apsp.gpu.hooks import cfg_gap, channel_stats, drift
        raw[f"feat_{k}"] = _timeit(lambda: (drift(x0, prev).cpu(), cfg_gap(ec, eu).cpu()), dev, W, N)
        raw[f"lfeat_{k}"] = _timeit(lambda: (drift(x0, prev).cpu(), cfg_gap(ec, eu).cpu(), channel_stats(x0)[0].cpu()), dev, W, N)
        if k >= 2:
            raw[f"ver_{k}"] = _timeit(lambda: eng.scoring.score_preview(a.prompt, x0), dev, W, N)
        lat = eng.x
        raw[f"fin_{k}"] = _timeit((lambda: eng.scoring.validate_delivered(a.prompt, lat)) if k == 1
                                  else (lambda: eng.scoring.score_delivered(a.prompt, lat)), dev, W, N)
        for (src, tgt) in required_prune_pairs():
            if src != k:
                continue
            scores = {c: float(i % 5) for i, c in enumerate(rows)}
            snap = (eng.x.clone(), eng.emb.clone(), list(eng.rows))

            def do_prune(t=tgt):
                rm = RowMap(rows)
                idx = rm.retained_indices(select_top(rows, scores, t))
                ii = torch.as_tensor(idx, device=dev)
                xs = snap[0][ii]
                es = snap[1][torch.as_tensor(list(idx) + [i + k for i in idx], device=dev)]
                hx = x0[ii]
                return xs, es, hx
            raw[f"prune_{src}_{tgt}"] = _timeit(do_prune, dev, W, N)
        del eng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summ = {name: _summ(v) for name, v in raw.items()}
    for name, s in summ.items():
        kind, *rest = name.split("_")
        if kind == "prune":
            res["prune"][f"{rest[0]}->{rest[1]}"] = ceil_us(s["mean_us"])
        else:
            res[kind][str(rest[0])] = ceil_us(s["mean_us"])
    costs = {**res, "disp": 0, "once": 0, "pred": {}, "uses_traj": False, "synthetic": False,
             "provisional": bool(a.smoke or not a.measurement), "method_id": "fixed"}
    gch_body = {"eph": a._eph, "summary": {k: [v["mean_us"], v["n"]] for k, v in summ.items()},
                "units": "us", "timer": "perf_counter+cuda.synchronize", "warmup": W, "n": N,
                "profile_code": package_code_sha(["gpu/profile.py", "gpu/engine.py", "gpu/scoring.py"]),
                "smoke": bool(a.smoke)}
    gch = provenance_hash("GCH", gch_body)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "SMOKE_" if a.smoke else ""
    (out_dir / f"{tag}generator_costs.json").write_text(json.dumps({**costs, "gch": gch}, indent=2), encoding="utf-8")
    (out_dir / f"{tag}generator_costs_raw.json").write_text(json.dumps({"raw_us": raw, "summary": summ, "gch_body": gch_body}),
                                                             encoding="utf-8")
    return {"gch": gch, "costs_path": str(out_dir / f"{tag}generator_costs.json"), "provisional": costs["provisional"],
            "cv_flags": [k for k, v in summ.items() if v["cv_flag"]], "summary_mean_us": {k: round(v["mean_us"], 1) for k, v in summ.items()}}


def stage_batch_parity(cfg, bundle, scoring, a) -> dict:
    """M6: same prompt/seeds at batch 1,2,4,8,16 vs the batch-8 reference (16 = pool of two banks)."""
    torch = import_torch()
    J = cfg["steps"] // cfg["delta"]
    sizes = a.batch_sizes or [1, 2, 4, 8, 16]

    def run_group(rows):
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
        eng.prepare_request(a.prompt, "m6-technical")
        eng.start(rows)
        x0, sc = {}, {}
        for j in range(1, J + 1):
            eng.advance(None, cfg["delta"], False)
            if j in (4, 8, 16):
                x0[j] = eng.hook.x0.float().cpu()
                sc[j] = [r.score for r in eng.scoring.score_preview(a.prompt, eng.hook.x0).records]
        d = [r.score for r in eng.scoring.score_delivered(a.prompt, eng.x).records]
        return eng.x.float().cpu(), x0, sc, d

    all16 = pool_rows(2)
    per = {}
    for b in sizes:
        lat, x0, sc, d = {}, {j: {} for j in (4, 8, 16)}, {j: {} for j in (4, 8, 16)}, {}
        for s in range(0, 16, b):
            rows = all16[s:s + b]
            L, X, S, D = run_group(rows)
            for i, c in enumerate(rows):
                lat[c] = L[i]
                d[c] = D[i]
                for j in (4, 8, 16):
                    x0[j][c] = X[j][i]
                    sc[j][c] = S[j][i]
        per[b] = (lat, x0, sc, d)
    ref = per[8]
    rep = {}
    for b, (lat, x0, sc, d) in per.items():
        lat_err = max(float((lat[c] - ref[0][c]).abs().max()) for c in all16)
        sc_err = max(abs(sc[j][c] - ref[2][j][c]) for j in (4, 8, 16) for c in all16
                     if sc[j][c] is not None and ref[2][j][c] is not None)
        d_err = max(abs(d[c] - ref[3][c]) for c in all16 if d[c] is not None and ref[3][c] is not None)
        flips = 0
        for bank in (0, 1):
            rows = bank_rows(bank)
            def winner(scs, dl):
                a4 = sorted(select_top(rows, {c: scs[4][c] for c in rows}, 4))
                a2 = sorted(select_top(a4, {c: scs[8][c] for c in a4}, 2))
                return sorted(select_top(a2, {c: dl[c] for c in a2}, 1))[0]
            flips += winner(sc, d) != winner(ref[2], ref[3])
        rep[str(b)] = {"delivered_latent_max_abs": lat_err, "delivered_bitwise_equal": lat_err == 0.0,
                       "preview_score_max_abs": sc_err, "delivered_score_max_abs": d_err, "B2_winner_flips_of_2": flips}
    return {"reference_batch": 8, "by_batch": rep,
            "note": "eta=0 does not guarantee batch invariance; report deltas, never assume bitwise parity"}


def _load_prompts(a):
    if a.prompts_file:
        out = []
        for l in Path(a.prompts_file).read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                out.append((d.get("prompt_uid") or d["text"][:16], d["text"]))
        return out[: a.limit or None]
    return [("m3-technical", a.prompt)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0 profiling stages")
    add_backend_args(ap)
    ap.add_argument("--stage", choices=["model", "hook", "validity", "generator", "batch-parity"], required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompts-file", default=None, help="jsonl with text (+prompt_uid); calibration prompts for M3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ks", type=int, nargs="*", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny n; output marked SMOKE and provisional")
    ap.add_argument("--measurement", action="store_true", help="this host is the research-designated measurement machine")
    ap.add_argument("--out-dir", default="reports/p0")
    a = ap.parse_args(argv)
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    a._eph = eph
    fn = {"model": lambda: stage_model(cfg, bundle, scorer, eph, a), "hook": lambda: stage_hook(cfg, bundle, scoring, a),
          "validity": lambda: stage_validity(cfg, bundle, scoring, a),
          "generator": lambda: stage_generator(cfg, bundle, scoring, a),
          "batch-parity": lambda: stage_batch_parity(cfg, bundle, scoring, a)}[a.stage]
    t0 = time.perf_counter()
    rep = fn()
    rep.update({"stage": a.stage, "eph": eph, "backend": body["backend"], "smoke": bool(a.smoke),
                "seconds": time.perf_counter() - t0})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{'SMOKE_' if a.smoke else ''}{a.stage}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
