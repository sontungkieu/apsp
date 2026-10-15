"""Batch-drift diagnostic harness (ENGINEERING ONLY; docs/BATCH_DRIFT_AUDIT.md).

Separates the three places where batch size / composition / order can change numbers, always mapping results
back to CandidateId and hashing the inputs, so "same seed integer" is never taken as "same input":

  aa      A/A: the reference plan end to end in THIS process (initial latents, embeddings, every forward,
          checkpoints, decoded tensor and uint8 pixels, preview/delivered scores). Run it in two separate processes
          and compare with --compare-aa; a plan compared with its own tensors is never repeatability evidence
  audit   train/eval flags of every submodule (UNet, VAE, preview VAE, text encoder, ImageReward), parameter
          dtypes, attention processors, autocast / grad state and the numeric flags in effect
  unet    A1: ONE forward at fixed forward indices with bit-identical inputs (latent rows, embeddings,
          timestep) under different batch plans -> eps_uncond / eps_cond / x_prev / x0 per CandidateId
  traj    A2: full 64-forward trajectories per plan: per-candidate first divergent forward and latent error
          (max / RMSE / relative L2) at every checkpoint; optional preview/delivered scores, B2 ranking gaps
          and winner flips (the M6 reproduction with per-candidate detail)
  vae     B: decode the SAME latent tensor (from the reference trajectory, hashed) with different chunks and
          orders; compare the decoded tensor and the uint8 image after postprocess
  rm      C: score the SAME PNG pixels and prompt (pixel hash + preprocessed-tensor hash per image), repeat the
          reference batch, then change batch size / composition / order and map scores back to CandidateId
  method  D: k=8 collector table + online fixed-pair schedules on diagnostic prompts: winner / keep sets /
          delivered Y online vs replay, and method effects (Y(pair) - Y(reference pair)) online vs replay

Engine, decode and scoring code are the production modules, unchanged (the EPH is unaffected). The cost table
is the synthetic engineering-only table (ledger plumbing only). Prompts must carry a technical split label and
are checked against GenEval / the research pool like the technical smoke.

  python -m apsp.gpu.drift_harness --diag-config configs/drift_diagnostic.yaml --part all --out $APSP_STORE/diag1
  python -m apsp.gpu.drift_harness --fake-backend --diag-config configs/drift_diagnostic.yaml --part unet --out /tmp/d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml

from apsp.contracts.identity import CandidateId, bank_rows, pool_rows, select_top
from apsp.contracts.schema import TECHNICAL_SPLITS
from apsp.data.prompts import normalize, prompt_uid
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.decode import decode_delivered, decode_preview
from apsp.gpu.engine import DenoiseEngine
from apsp.gpu.runtime import import_torch, numeric_runtime

PARTS = ("audit", "unet", "traj", "vae", "rm", "method")
EXTRA_PARTS = ("aa",)        # run explicitly (twice, separate processes), not part of "all"
ORDERS = ("fwd", "rev", "interleave")
LABEL = "ENGINEERING ONLY; diagnostic prompts outside research splits; not G1"


class DiagConfigError(ValueError):
    pass


# ------------------------------------------------------------------ config / prompts ---------------------------
def load_diag_config(path: str) -> dict:
    dc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if dc.get("split") not in TECHNICAL_SPLITS:
        raise DiagConfigError(f"diagnostic split must be one of {TECHNICAL_SPLITS}")
    prompts = [normalize(p) for p in dc.get("prompts", [])]
    if not 1 <= len(prompts) <= int(dc.get("max_prompts", 8)) or len(set(prompts)) != len(prompts):
        raise DiagConfigError("diagnostic config needs 1..max_prompts distinct prompts")
    dc["prompts"] = prompts
    rep = dc.setdefault("repro", {})
    rep["prompt"] = normalize(rep.get("prompt", prompts[0]))
    rep.setdefault("uid", prompt_uid(rep["prompt"]))
    dc.setdefault("batch_sizes", [1, 2, 4, 8, 16])
    dc.setdefault("ref_batch", 8)
    dc.setdefault("orders", list(ORDERS))
    dc.setdefault("unet_forward_indices", [0, 16, 32, 63])
    dc.setdefault("vae_chunks", [1, 2, 4, 8, 16])
    dc.setdefault("rm_repeats", 3)
    dc.setdefault("method_pairs", [[4, 8], [2, 9]])
    dc.setdefault("method_banks", 1)
    dc.setdefault("method_prompts", len(prompts))
    dc.setdefault("traj_plans", None)       # None = every plan; trajectories are the expensive part
    if [4, 8] not in [list(p) for p in dc["method_pairs"]]:
        raise DiagConfigError("method_pairs must include the reference pair [4, 8]")
    bad = [o for o in dc["orders"] if o not in ORDERS]
    if bad:
        raise DiagConfigError(f"unknown orders {bad}")
    return dc


def guard_prompts(dc: dict, pool_path: Optional[str] = None) -> dict:
    from apsp.gpu.technical_smoke import check_outside_research
    texts = list(dict.fromkeys(dc["prompts"] + [dc["repro"]["prompt"]]))
    return check_outside_research({"prompts": texts, "prompt_pool_check": dc.get("prompt_pool_check")}, pool_path)


# ------------------------------------------------------------------ numeric helpers ----------------------------
def tensor_sha(t) -> str:
    torch = import_torch()
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update((t.view(torch.int16) if t.dtype in (torch.float16, torch.bfloat16) else t).numpy().tobytes())
    return h.hexdigest()


def compare(a, ref) -> dict:
    """Errors of a vs ref (same shape). Bitwise equality is checked on the original dtype."""
    torch = import_torch()
    eq = bool(a.dtype == ref.dtype and torch.equal(a.cpu(), ref.cpu()))
    x = a.detach().cpu().double().numpy()
    y = ref.detach().cpu().double().numpy()
    d = x - y
    nr = float(np.sqrt((y ** 2).sum()))
    return {"bitwise_equal": eq, "max_abs": float(np.abs(d).max()), "rmse": float(np.sqrt((d ** 2).mean())),
            "rel_l2": float(np.sqrt((d ** 2).sum()) / nr) if nr else (0.0 if not d.any() else float("inf")),
            "ref_rms": float(np.sqrt((y ** 2).mean()))}


def summarize(rows: Sequence[dict], keys=("max_abs", "rmse", "rel_l2")) -> dict:
    if not rows:
        return {}
    out = {f"max_{k}": max(r[k] for r in rows) for k in keys}
    out["n_bitwise_equal"] = sum(bool(r["bitwise_equal"]) for r in rows)
    out["n"] = len(rows)
    return out


def cid_key(c: CandidateId) -> str:
    return f"{c[0]},{c[1]}"


def make_plan(ids: Sequence[CandidateId], batch: int, order: str) -> list[list[CandidateId]]:
    """Groups of `batch` CandidateIds. fwd: pool order; rev: reversed; interleave: by (seed, bank)."""
    ids = list(ids)
    if order == "rev":
        ids = ids[::-1]
    elif order == "interleave":
        ids = sorted(ids, key=lambda c: (c[1], c[0]))
    elif order != "fwd":
        raise ValueError(order)
    return [ids[s:s + batch] for s in range(0, len(ids), batch)]


def plan_list(ids, sizes, orders, ref_batch) -> dict:
    """{name: groups}; the reference plan 'b{ref}_fwd' is always present. Orders only matter for b < len(ids)
    (composition) or b > 1 (order within a group); duplicates are kept for clarity."""
    plans = {f"b{ref_batch}_fwd": make_plan(ids, ref_batch, "fwd")}
    for b in sizes:
        for o in orders:
            if b == 1 and o != "fwd":
                continue      # batch 1: order and composition cannot change anything inside a group
            plans.setdefault(f"b{b}_{o}", make_plan(ids, b, o))
    return plans


def _pil_u8(pil) -> np.ndarray:
    return np.asarray(pil.convert("RGB")).astype(np.int16)


def pixel_compare(a, ref) -> dict:
    x, y = _pil_u8(a), _pil_u8(ref)
    d = np.abs(x - y)
    px = d.max(-1)
    return {"bitwise_equal": bool(not d.any()), "max_abs_u8": int(d.max()), "rmse_u8": float(np.sqrt((d.astype(float) ** 2).mean())),
            "frac_pixels_changed": float((px > 0).mean()), "frac_pixels_gt8": float((px > 8).mean())}


def pixel_sha(pil) -> str:
    a = np.asarray(pil.convert("RGB"))
    return hashlib.sha256(f"{a.shape}|".encode() + a.tobytes()).hexdigest()


# ------------------------------------------------------------------ audit --------------------------------------
def module_audit(bundle, scorer) -> dict:
    torch = import_torch()
    mods = {"unet": getattr(bundle, "unet", None), "vae": getattr(bundle, "vae", None),
            "vae_preview": getattr(bundle, "vae_preview", None), "text_encoder": getattr(bundle, "text_encoder", None),
            "image_reward": getattr(scorer, "model", None)}
    out = {}
    for name, m in mods.items():
        if m is None or not hasattr(m, "named_modules"):
            out[name] = {"present": False, "note": "not a torch module on this backend"}
            continue
        training = [n or "<root>" for n, sm in m.named_modules() if sm.training]
        dtypes = sorted({str(p.dtype) for p in m.parameters()})
        devices = sorted({str(p.device) for p in m.parameters()})
        grads = sum(p.requires_grad for p in m.parameters())
        drop = sorted({type(sm).__name__ for _, sm in m.named_modules() if "Dropout" in type(sm).__name__})
        e = {"present": True, "class": type(m).__name__, "n_submodules": sum(1 for _ in m.modules()),
             "n_training_submodules": len(training), "training_submodules_first10": training[:10],
             "param_dtypes": dtypes, "param_devices": devices, "n_params_requires_grad": int(grads),
             "dropout_types_present": drop}
        procs = getattr(m, "attn_processors", None)
        if isinstance(procs, dict):
            e["attn_processors"] = sorted({type(p).__name__ for p in procs.values()})
        out[name] = e
    if mods["vae"] is not None and mods["vae_preview"] is not None:
        out["vae_preview_is_vae"] = mods["vae"] is mods["vae_preview"]
    rt = numeric_runtime()
    rt.update({"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
               "grad_enabled": bool(torch.is_grad_enabled()), "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
               "autocast_cuda": bool(torch.is_autocast_enabled()),
               "note": "the engine and scorer run under torch.inference_mode() without autocast; flags here are the "
                       "process defaults at audit time"})
    try:
        rt["fp16_reduced_precision_reduction"] = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    except AttributeError:
        pass
    if torch.cuda.is_available():
        rt["device_name"] = torch.cuda.get_device_name(0)
    out["numeric_runtime"] = rt
    out["all_eval"] = all(v.get("n_training_submodules", 0) == 0 for v in out.values() if isinstance(v, dict) and v.get("present"))
    return out


# ------------------------------------------------------------------ A1: single UNet forward -------------------
def _engine(bundle, scoring, cfg, prompt, uid, hook="none") -> DenoiseEngine:
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode=hook)
    eng.prepare_request(prompt, uid)
    return eng


def reference_states(bundle, scoring, cfg, prompt, uid, ids, ref_batch, forward_indices) -> dict:
    """x (before forward f) per CandidateId from the reference plan; also each id's initial-latent sha."""
    want = set(forward_indices)
    states = {f: {} for f in forward_indices}
    init_sha = {}
    for group in make_plan(ids, ref_batch, "fwd"):
        eng = _engine(bundle, scoring, cfg, prompt, uid)
        eng.start(group)
        for i, c in enumerate(group):
            init_sha[c] = tensor_sha(eng.init[i])
        for f in range(cfg["steps"]):
            if f in want:
                for i, c in enumerate(group):
                    states[f][c] = eng.x[i:i + 1].detach().cpu().clone()
            eng._forward()
    return {"states": states, "init_sha": init_sha}


def _capture_x0(eng) -> dict:
    """Record pred_original_sample of every forward (the hook keeps it only at checkpoints); math unchanged."""
    cap, orig = {}, eng.hook.on_forward

    def on_forward(idx, x0, eps_u, eps_c):
        cap["x0"] = x0
        return orig(idx, x0, eps_u, eps_c)
    eng.hook.on_forward = on_forward
    return cap


def unet_forward_invariance(bundle, scoring, cfg, prompt, uid, ids, plans, ref_name, forward_indices) -> dict:
    torch = import_torch()
    ref = reference_states(bundle, scoring, cfg, prompt, uid, ids, int(ref_name[1:].split("_")[0]), forward_indices)
    outputs = {}       # plan -> f -> cid -> dict of tensors
    emb_sha, in_sha = {}, {}
    for pname, groups in plans.items():
        outputs[pname] = {}
        for f in forward_indices:
            outputs[pname][f] = {}
            for group in groups:
                eng = _engine(bundle, scoring, cfg, prompt, uid)
                eng.start(group)
                eng.x = torch.cat([ref["states"][f][c] for c in group], 0).to(eng.b.device)
                eng.forward_idx = f
                for i, c in enumerate(group):
                    in_sha.setdefault((f, c), set()).add(tensor_sha(eng.x[i]))
                emb_sha.setdefault("uncond", set()).add(tensor_sha(eng.uncond))
                emb_sha.setdefault("cond", set()).add(tensor_sha(eng.cond))
                cap = _capture_x0(eng)
                eng._forward()
                eu, ec = eng.last_eps
                for i, c in enumerate(group):
                    outputs[pname][f][c] = {"eps_uncond": eu[i].cpu(), "eps_cond": ec[i].cpu(),
                                            "x_prev": eng.x[i].cpu(), "x0": cap["x0"][i].cpu()}
    rep = {"forward_indices": list(forward_indices), "ref_plan": ref_name,
           "timesteps": {f: int(_engine(bundle, scoring, cfg, prompt, uid).timesteps[f]) for f in forward_indices},
           "inputs_identical_across_plans": all(len(v) == 1 for v in in_sha.values()),
           "embeddings_identical_across_groups": all(len(v) == 1 for v in emb_sha.values()),
           "init_latent_sha": {cid_key(c): s for c, s in ref["init_sha"].items()}, "plans": {}}
    for pname in plans:
        per_f = {}
        for f in forward_indices:
            per = {}
            for c in ids:
                per[cid_key(c)] = {q: compare(outputs[pname][f][c][q], outputs[ref_name][f][c][q])
                                   for q in ("eps_uncond", "eps_cond", "x_prev", "x0")}
            per_f[str(f)] = {"per_candidate": per,
                             "summary": {q: summarize([per[k][q] for k in per]) for q in ("eps_uncond", "eps_cond", "x_prev", "x0")}}
        rep["plans"][pname] = {"groups": [[cid_key(c) for c in g] for g in plans[pname]], "by_forward": per_f}
    return rep


# ------------------------------------------------------------------ A2: trajectories (+ M6 scores) ------------
def run_plan_trajectory(bundle, scoring, cfg, prompt, uid, groups, with_scores: bool, score_js=(4, 8, 16)) -> dict:
    """Per CandidateId: sha of x after every forward, x at every checkpoint, final x, optional scores."""
    J = cfg["steps"] // cfg["delta"]
    out = {"sha": {}, "x_ckpt": {}, "x_final": {}, "preview": {j: {} for j in score_js}, "delivered": {}, "init_sha": {}}
    for group in groups:
        eng = _engine(bundle, scoring, cfg, prompt, uid)
        eng.start(group)
        for i, c in enumerate(group):
            out["init_sha"][c] = tensor_sha(eng.init[i])
            out["sha"][c] = []
            out["x_ckpt"][c] = {}
        for j in range(1, J + 1):
            for _ in range(cfg["delta"]):
                eng._forward()
                for i, c in enumerate(group):
                    out["sha"][c].append(tensor_sha(eng.x[i]))
            for i, c in enumerate(group):
                out["x_ckpt"][c][j] = eng.x[i].detach().cpu().clone()
            if with_scores and j in score_js:
                recs = eng.scoring.score_preview(prompt, eng.hook.x0).records
                for i, c in enumerate(group):
                    out["preview"][j][c] = recs[i].score
        for i, c in enumerate(group):
            out["x_final"][c] = eng.x[i].detach().cpu().clone()
        if with_scores:
            recs = eng.scoring.score_delivered(prompt, eng.x).records
            for i, c in enumerate(group):
                out["delivered"][c] = recs[i].score
    return out


def b2_replay(scores_pv: dict, scores_dl: dict, rows: Sequence[CandidateId], cuts=(4, 8)) -> dict:
    """B2 on a score table: keep 4 at cuts[0], 2 at cuts[1], top-1 delivered; ranking gaps at each cut."""
    def gap(sc: dict, keep: int) -> float:
        o = sorted(sc, key=lambda c: (-sc[c], c))
        return float(sc[o[keep - 1]] - sc[o[keep]])
    s1 = {c: scores_pv[cuts[0]][c] for c in rows}
    a4 = sorted(select_top(rows, s1, 4))
    s2 = {c: scores_pv[cuts[1]][c] for c in a4}
    a2 = sorted(select_top(a4, s2, 2))
    sd = {c: scores_dl[c] for c in a2}
    w = sorted(select_top(a2, sd, 1))[0]
    return {"keep_j1": [cid_key(c) for c in a4], "keep_j2": [cid_key(c) for c in a2], "winner": cid_key(w),
            "y": sd[w], "gap_j1": gap(s1, 4), "gap_j2": gap(s2, 2), "gap_final": gap(sd, 1)}


def trajectory_drift(bundle, scoring, cfg, prompt, uid, ids, plans, ref_name, with_scores=True) -> tuple[dict, dict]:
    runs = {p: run_plan_trajectory(bundle, scoring, cfg, prompt, uid, g, with_scores) for p, g in plans.items()}
    ref = runs[ref_name]
    J = cfg["steps"] // cfg["delta"]
    rep = {"ref_plan": ref_name, "prompt": prompt, "uid": uid, "with_scores": with_scores, "plans": {}}
    banks = sorted({c[0] for c in ids})
    for p, r in runs.items():
        per = {}
        for c in ids:
            first = next((f + 1 for f, (a, b) in enumerate(zip(r["sha"][c], ref["sha"][c])) if a != b), None)
            curve = {str(j): compare(r["x_ckpt"][c][j], ref["x_ckpt"][c][j]) for j in range(1, J + 1)}
            e = {"init_sha_equal_ref": r["init_sha"][c] == ref["init_sha"][c], "first_divergent_forward": first,
                 "latent_by_checkpoint": curve, "final": compare(r["x_final"][c], ref["x_final"][c])}
            if with_scores:
                e["preview_diff"] = {str(j): _sdiff(r["preview"][j].get(c), ref["preview"][j].get(c)) for j in r["preview"]}
                e["delivered_diff"] = _sdiff(r["delivered"].get(c), ref["delivered"].get(c))
            per[cid_key(c)] = e
        summ = {"final": summarize([per[k]["final"] for k in per]),
                "by_checkpoint": {str(j): summarize([per[k]["latent_by_checkpoint"][str(j)] for k in per]) for j in range(1, J + 1)},
                "n_diverged": sum(per[k]["first_divergent_forward"] is not None for k in per),
                "earliest_divergent_forward": min((per[k]["first_divergent_forward"] for k in per
                                                   if per[k]["first_divergent_forward"] is not None), default=None)}
        if with_scores:
            pv = [abs(v) for k in per for v in per[k]["preview_diff"].values() if v is not None]
            dl = [abs(per[k]["delivered_diff"]) for k in per if per[k]["delivered_diff"] is not None]
            summ["preview_score_max_abs"] = max(pv) if pv else None
            summ["delivered_score_max_abs"] = max(dl) if dl else None
            b2 = {}
            for b in banks:
                rows = [c for c in ids if c[0] == b]
                if len(rows) == 8:
                    mine = b2_replay(r["preview"], r["delivered"], rows)
                    theirs = b2_replay(ref["preview"], ref["delivered"], rows)
                    b2[str(b)] = {"this_plan": mine, "ref_plan": theirs, "winner_flip": mine["winner"] != theirs["winner"],
                                  "keep_j1_equal": mine["keep_j1"] == theirs["keep_j1"],
                                  "keep_j2_equal": mine["keep_j2"] == theirs["keep_j2"]}
            summ["b2_by_bank"] = b2
            summ["b2_winner_flips"] = sum(v["winner_flip"] for v in b2.values())
        rep["plans"][p] = {"groups": [[cid_key(c) for c in g] for g in plans[p]], "summary": summ, "per_candidate": per}
    ref_latents = {c: ref["x_final"][c] for c in ids}
    return rep, ref_latents


def _gap(sc: Optional[dict], keep: int) -> Optional[float]:
    if not sc or len(sc) <= keep:
        return None
    o = sorted(sc, key=lambda c: (-sc[c], c))
    return float(sc[o[keep - 1]] - sc[o[keep]])


def _sdiff(a, b):
    return None if a is None or b is None else float(a) - float(b)


# ------------------------------------------------------------------ A/A repeatability ------------------------
def aa_record(bundle, scoring, cfg, prompt, uid, ids, ref_batch) -> tuple[dict, dict]:
    """One A/A run of the reference plan in this process: hashes/scores (JSON) and tensors (saved separately)."""
    torch = import_torch()
    groups = make_plan(ids, ref_batch, "fwd")
    emb = {}
    for _ in groups:      # embeddings as each group's engine computes them
        eng = _engine(bundle, scoring, cfg, prompt, uid)
        emb.setdefault("uncond", []).append(tensor_sha(eng.uncond))
        emb.setdefault("cond", []).append(tensor_sha(eng.cond))
    r = run_plan_trajectory(bundle, scoring, cfg, prompt, uid, groups, with_scores=True)
    with torch.inference_mode():
        img = decode_delivered(bundle, torch.stack([r["x_final"][c] for c in ids], 0).to(bundle.device), ref_batch)
    pils = bundle.to_pil(img)
    J = sorted(next(iter(r["x_ckpt"].values())))
    tensors = {"ids": [cid_key(c) for c in ids],
               "x_ckpt": {cid_key(c): torch.stack([r["x_ckpt"][c][j] for j in J], 0) for c in ids},
               "x_final": {cid_key(c): r["x_final"][c] for c in ids},
               "image": {cid_key(c): img[i].detach().cpu() for i, c in enumerate(ids)},
               "pixels": {cid_key(c): torch.from_numpy(np.asarray(pils[i].convert("RGB")).copy()) for i, c in enumerate(ids)}}
    rec = {"prompt": prompt, "uid": uid, "ref_batch": ref_batch, "ids": tensors["ids"],
           "groups": [[cid_key(c) for c in g] for g in groups], "checkpoints": J, "embedding_sha": emb,
           "init_sha": {cid_key(c): r["init_sha"][c] for c in ids},
           "forward_sha": {cid_key(c): r["sha"][c] for c in ids},
           "final_sha": {cid_key(c): tensor_sha(r["x_final"][c]) for c in ids},
           "image_tensor_sha": {k: tensor_sha(v) for k, v in tensors["image"].items()},
           "pixel_sha": {cid_key(c): pixel_sha(pils[i]) for i, c in enumerate(ids)},
           "preview": {str(j): {cid_key(c): v for c, v in d.items()} for j, d in r["preview"].items()},
           "delivered": {cid_key(c): v for c, v in r["delivered"].items()}}
    return rec, tensors


def _eq_count(a: dict, b: dict) -> dict:
    keys = sorted(set(a) | set(b))
    return {"n": len(keys), "n_equal": sum(a.get(k) == b.get(k) for k in keys)}


def compare_aa(a: dict, b: dict, ta: dict, tb: dict, expect_eph: Optional[str] = None) -> dict:
    """Compare two independent A/A runs (records + tensors). Reports measured differences; no tolerance."""
    ids = a["ids"]
    rep = {"eph": {"run_a": a.get("eph"), "run_b": b.get("eph"), "expected": expect_eph,
                   "a_equals_b": a.get("eph") == b.get("eph"),
                   "equals_expected": None if expect_eph is None else (a.get("eph") == expect_eph == b.get("eph"))},
           "same_inputs_declared": all(a[k] == b[k] for k in ("prompt", "uid", "ref_batch", "groups", "ids")),
           "init_latents": _eq_count(a["init_sha"], b["init_sha"]),
           "embeddings": {k: a["embedding_sha"][k] == b["embedding_sha"][k] for k in a["embedding_sha"]},
           "final_latent_sha": _eq_count(a["final_sha"], b["final_sha"]),
           "image_tensor_sha": _eq_count(a["image_tensor_sha"], b["image_tensor_sha"]),
           "pixel_sha": _eq_count(a["pixel_sha"], b["pixel_sha"])}
    per = {}
    for k in ids:
        fa, fb = a["forward_sha"][k], b["forward_sha"][k]
        first = next((i + 1 for i, (x, y) in enumerate(zip(fa, fb)) if x != y), None)
        per[k] = {"first_divergent_forward": first,
                  "x_ckpt": compare(ta["x_ckpt"][k], tb["x_ckpt"][k]), "x_final": compare(ta["x_final"][k], tb["x_final"][k]),
                  "image": compare(ta["image"][k], tb["image"][k]),
                  "pixels_max_abs_u8": int((ta["pixels"][k].int() - tb["pixels"][k].int()).abs().max()),
                  "delivered_diff": _sdiff(a["delivered"].get(k), b["delivered"].get(k)),
                  "preview_diff": {j: _sdiff(a["preview"][j].get(k), b["preview"][j].get(k)) for j in a["preview"]}}
    pv = [abs(v) for p in per.values() for v in p["preview_diff"].values() if v is not None]
    dl = [abs(p["delivered_diff"]) for p in per.values() if p["delivered_diff"] is not None]
    rep["per_candidate"] = per
    rep["summary"] = {"n_candidates": len(ids),
                      "n_trajectory_identical": sum(p["first_divergent_forward"] is None for p in per.values()),
                      "max_abs_x_final": max(p["x_final"]["max_abs"] for p in per.values()),
                      "max_abs_pixels_u8": max(p["pixels_max_abs_u8"] for p in per.values()),
                      "preview_scores_bitwise_equal": all(v == 0 for v in pv), "preview_max_abs": max(pv) if pv else None,
                      "delivered_scores_bitwise_equal": all(v == 0 for v in dl), "delivered_max_abs": max(dl) if dl else None}
    sm = rep["summary"]
    rep["all_bitwise_identical"] = bool(
        rep["same_inputs_declared"] and rep["init_latents"]["n_equal"] == rep["init_latents"]["n"]
        and all(rep["embeddings"].values()) and sm["n_trajectory_identical"] == sm["n_candidates"]
        and rep["pixel_sha"]["n_equal"] == rep["pixel_sha"]["n"]
        and rep["image_tensor_sha"]["n_equal"] == rep["image_tensor_sha"]["n"]
        and sm["preview_scores_bitwise_equal"] and sm["delivered_scores_bitwise_equal"])
    return rep


# ------------------------------------------------------------------ B: VAE decode -----------------------------
def vae_invariance(bundle, latents: dict, ids, chunks, orders, ref_chunk: int, preview: bool = False) -> tuple[dict, dict]:
    torch = import_torch()
    dec = decode_preview if preview else decode_delivered
    outs = {}
    lat_sha = {cid_key(c): tensor_sha(latents[c]) for c in ids}
    for chunk in chunks:
        for o in orders:
            if chunk == 1 and o != "fwd":
                continue
            order = [c for g in make_plan(ids, len(ids), o) for c in g]
            stack = torch.stack([latents[c] for c in order], 0).to(bundle.device)
            with torch.inference_mode():
                img = dec(bundle, stack, chunk)
            pils = bundle.to_pil(img)
            outs[f"chunk{chunk}_{o}"] = {c: (img[i].detach().cpu(), pils[i]) for i, c in enumerate(order)}
    ref_name = f"chunk{ref_chunk}_fwd"
    if ref_name not in outs:
        raise DiagConfigError(f"ref chunk {ref_chunk} not in vae_chunks")
    ref = outs[ref_name]
    rep = {"path": "preview" if preview else "delivered", "ref": ref_name, "latent_sha": lat_sha, "plans": {}}
    for name, o in outs.items():
        per = {cid_key(c): {"tensor": compare(o[c][0], ref[c][0]), "pixels": pixel_compare(o[c][1], ref[c][1]),
                            "pixel_sha": pixel_sha(o[c][1])} for c in ids}
        rep["plans"][name] = {"per_candidate": per, "summary": {
            "tensor": summarize([v["tensor"] for v in per.values()]),
            "pixels_max_abs_u8": max(v["pixels"]["max_abs_u8"] for v in per.values()),
            "n_pixels_bitwise_equal": sum(v["pixels"]["bitwise_equal"] for v in per.values())}}
    ref_images = {c: ref[c][1] for c in ids}
    return rep, ref_images


# ------------------------------------------------------------------ C: ImageReward -----------------------------
def preprocessed_sha(scorer, pil) -> tuple[str, str]:
    model = getattr(scorer, "model", None)
    if model is not None and hasattr(model, "preprocess"):
        return tensor_sha(model.preprocess(pil)), "ImageReward.preprocess"
    return pixel_sha(pil), "no preprocess on this backend (pixel sha)"


def rm_invariance(scorer, prompt: str, images: dict, ids, plans: dict, ref_name: str, repeats: int,
                  png_dir: Optional[Path] = None) -> dict:
    from PIL import Image
    if png_dir is not None:        # score the exact pixels of the files on disk
        png_dir.mkdir(parents=True, exist_ok=True)
        loaded = {}
        for c in ids:
            p = png_dir / f"cand_{c[0]}_{c[1]}.png"
            images[c].save(p)
            loaded[c] = Image.open(p).convert("RGB")
        images = loaded
    inputs = {}
    for c in ids:
        ps, how = preprocessed_sha(scorer, images[c])
        inputs[cid_key(c)] = {"pixel_sha": pixel_sha(images[c]), "preprocessed_sha": ps, "preprocessed_by": how}

    def score_plan(groups) -> dict:
        out = {}
        for g in groups:
            vals = scorer.score(prompt, [images[c] for c in g])
            for c, v in zip(g, vals):
                out[c] = float(v)
        return out

    ref_runs = [score_plan(plans[ref_name]) for _ in range(max(1, repeats))]
    ref = ref_runs[0]
    rep = {"prompt": prompt, "ref_plan": ref_name, "inputs": inputs,
           "repeat_ref": {"n": len(ref_runs), "bitwise_equal_all": all(r == ref for r in ref_runs),
                          "max_abs": max(abs(r[c] - ref[c]) for r in ref_runs for c in ids)},
           "plans": {}}
    for pname, groups in plans.items():
        sc = score_plan(groups)
        per = {cid_key(c): {"score": sc[c], "ref": ref[c], "diff": sc[c] - ref[c], "bitwise_equal": sc[c] == ref[c]} for c in ids}
        rep["plans"][pname] = {"groups": [[cid_key(c) for c in g] for g in groups], "per_candidate": per,
                               "summary": {"max_abs": max(abs(v["diff"]) for v in per.values()),
                                           "n_bitwise_equal": sum(v["bitwise_equal"] for v in per.values()), "n": len(per)}}
    return rep


# ------------------------------------------------------------------ D: method effects --------------------------
def method_effects(bundle, scoring, cfg, prompts: Sequence[str], banks: int, pairs: Sequence[Sequence[int]]) -> dict:
    from apsp.gpu.collect_bank import collect_instance
    from apsp.gpu.technical_smoke import CheckingEngine, engineering_costs
    from apsp.sim.env import run_schedule
    from apsp.sim.schedules import fixed_pair_spec, pair_outcomes
    mc = engineering_costs()
    pairs = [tuple(p) for p in pairs]
    recs = []
    for text in prompts:
        uid = prompt_uid(text)
        for b in range(banks):
            col = collect_instance(DenoiseEngine(bundle, scoring, cfg, hook_mode="collect"), text, uid, b)
            pv = np.full((8, 16), np.nan)
            for r in col["previews"]:
                pv[r[2], r[3] - 1] = r[6]
            dl = np.array([np.nan if r[3] is None else r[3] for r in sorted(col["delivered"], key=lambda r: r[2])], dtype=float)
            replay = {(o.j1, o.j2): o for o in pair_outcomes(pv, dl, b, mc)}
            inst = {"uid": uid, "prompt": text, "bank_id": b, "init_latent_sha": [s[4] for s in col["seeds"]],
                    "preview_table": pv.tolist(), "delivered_table": dl.tolist(),
                    "delivered_pixel_sha": [None if im is None else pixel_sha(im) for im in col["images"]], "pairs": {}}
            for (j1, j2) in pairs:
                eng = CheckingEngine(bundle, scoring, cfg, hook_mode="none")
                eng.prepare_request(text, uid)
                res = run_schedule(fixed_pair_spec(j1, j2, bank_rows(b)), eng, mc=mc, cap_us=10**12, method_id=f"fixed:{j1},{j2}")
                d = res.delivery
                rp = replay[(j1, j2)]
                prunes = [a for a in eng.audit if a["event"] == "PRUNE"]
                sc1 = {(b, s): pv[s, j1 - 1] for s in range(8)}
                o1 = sorted(sc1, key=lambda c: (-sc1[c], c))
                keep1_replay = sorted(o1[:4])
                sc2 = {c: pv[c[1], j2 - 1] for c in keep1_replay}
                o2 = sorted(sc2, key=lambda c: (-sc2[c], c))
                verify_scores = {a["j"]: dict(zip(map(tuple, a["alive"]), a["scores"])) for a in eng.audit if a["event"] == "VERIFY"}
                w_on, w_rp = d.winner, rp.winner
                inst["pairs"][f"{j1},{j2}"] = {
                    "winner_online": cid_key(d.winner) if d.winner else None,
                    "winner_replay": cid_key(rp.winner) if rp.winner else None,
                    "winner_equal": d.winner == rp.winner, "y_online": d.inference_score, "y_replay": rp.y,
                    "y_diff": (d.inference_score - rp.y) if d.inference_score is not None and rp.y is not None else None,
                    "keep_j1_equal": [tuple(c) for c in prunes[0]["kept"]] == keep1_replay if prunes else None,
                    "keep_j2_equal": sorted(tuple(c) for c in prunes[1]["kept"]) == sorted(o2[:2]) if len(prunes) > 1 else None,
                    "gap_j1_replay": float(sc1[o1[3]] - sc1[o1[4]]), "gap_j2_replay": float(sc2[o2[1]] - sc2[o2[2]]),
                    "verify_j2_max_abs_vs_bank": max(abs(v - pv[c[1], j2 - 1]) for c, v in verify_scores.get(j2, {}).items())
                    if verify_scores.get(j2) else None,
                    "gap_j1_online": _gap(verify_scores.get(j1), 4), "gap_j2_online": _gap(verify_scores.get(j2), 2),
                    # quality of both winners on the same k=8 table (one batch, one process): what a flip costs
                    "bank_y_of_online_winner": float(dl[w_on[1]]) if w_on else None,
                    "bank_y_of_replay_winner": float(dl[w_rp[1]]) if w_rp else None,
                    "bank_y_gap_between_winners": float(dl[w_on[1]] - dl[w_rp[1]]) if w_on and w_rp else None,
                    "identity_checks_ok": all(all(p["checks"].values()) for p in prunes)}
            ref = inst["pairs"]["4,8"]
            for k, v in inst["pairs"].items():
                if v["y_online"] is not None and ref["y_online"] is not None and v["y_replay"] is not None and ref["y_replay"] is not None:
                    eo, er = v["y_online"] - ref["y_online"], v["y_replay"] - ref["y_replay"]
                    v.update({"effect_vs_4_8_online": eo, "effect_vs_4_8_replay": er, "effect_error": eo - er,
                              "effect_sign_agrees": (eo > 0) == (er > 0) or (eo == 0 and er == 0)})
            recs.append(inst)
    flat = [v for r in recs for v in r["pairs"].values()]
    ydiff = [abs(v["y_diff"]) for v in flat if v["y_diff"] is not None]
    eff = [abs(v["effect_error"]) for v in flat if v.get("effect_error") is not None]
    return {"note": "online fixed schedules vs replay on the k=8 table of the same instance; synthetic costs "
                    "(plumbing only); diagnostic prompts; NOT G1",
            "pairs": [list(p) for p in pairs], "instances": recs,
            "summary": {"n_episodes": len(flat), "n_winner_equal": sum(v["winner_equal"] for v in flat),
                        "n_keep_j1_equal": sum(bool(v["keep_j1_equal"]) for v in flat),
                        "n_keep_j2_equal": sum(bool(v["keep_j2_equal"]) for v in flat),
                        "max_abs_y_diff": max(ydiff) if ydiff else None, "mean_abs_y_diff": float(np.mean(ydiff)) if ydiff else None,
                        "max_abs_effect_error": max(eff) if eff else None,
                        "n_effect_sign_disagree": sum(v.get("effect_sign_agrees") is False for v in flat),
                        "identity_checks_ok": all(v["identity_checks_ok"] for v in flat),
                        "winner_changes": [{"uid": r["uid"], "bank_id": r["bank_id"], "pair": k, "winner_online": v["winner_online"],
                                            "winner_replay": v["winner_replay"], "y_diff": v["y_diff"],
                                            "bank_y_gap_between_winners": v["bank_y_gap_between_winners"]}
                                           for r in recs for k, v in r["pairs"].items() if not v["winner_equal"]]}}


# ------------------------------------------------------------------ driver -------------------------------------
def _write(out: Path, name: str, rep: dict) -> None:
    (out / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")


def method_plan(dc: dict) -> dict:
    """The schedules / prompts / banks of the method part, written BEFORE it runs."""
    prompts = dc["prompts"][: dc["method_prompts"]]
    return {"pairs": [list(p) for p in dc["method_pairs"]], "reference_pair": [4, 8],
            "reference_note": "PSP-default cuts j=4, j=8 (8->4->2)",
            "prompts": [{"text": t, "uid": prompt_uid(t)} for t in prompts], "banks": list(range(dc["method_banks"])),
            "n_episodes": len(prompts) * dc["method_banks"] * len(dc["method_pairs"]),
            "costs": "synthetic engineering-only table (plumbing only)", "label": LABEL}


def run_parts(parts: Sequence[str], cfg, bundle, scorer, scoring, dc: dict, out: Path, meta: dict,
              progress: Callable[[str], None] = lambda s: None, aa_tag: str = "run1") -> dict:
    torch = import_torch()
    out.mkdir(parents=True, exist_ok=True)
    ids = list(pool_rows(2))
    ref_name = f"b{dc['ref_batch']}_fwd"
    plans = plan_list(ids, dc["batch_sizes"], dc["orders"], dc["ref_batch"])
    rp, ru = dc["repro"]["prompt"], dc["repro"]["uid"]
    status, ref_latents, ref_images = {}, None, None
    for part in parts:
        t0 = time.perf_counter()
        part_name = part
        if torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake":
            torch.cuda.reset_peak_memory_stats()
        progress(f"[diag] {part} ...")
        if part == "audit":
            rep = module_audit(bundle, scorer)
        elif part == "aa":
            rep, tens = aa_record(bundle, scoring, cfg, rp, ru, ids, dc["ref_batch"])
            torch.save(tens, out / f"aa_{aa_tag}.pt")
            part_name = f"aa_{aa_tag}"
        elif part == "unet":
            rep = unet_forward_invariance(bundle, scoring, cfg, rp, ru, ids, plans, ref_name, dc["unet_forward_indices"])
        elif part == "traj":
            tp = {k: v for k, v in plans.items() if dc["traj_plans"] is None or k in dc["traj_plans"] or k == ref_name}
            missing = sorted(set(dc["traj_plans"] or []) - set(plans))
            if missing:
                raise DiagConfigError(f"traj_plans not among the configured plans: {missing}")
            rep, ref_latents = trajectory_drift(bundle, scoring, cfg, rp, ru, ids, tp, ref_name, with_scores=True)
        elif part == "vae":
            if ref_latents is None:
                ref_latents = {c: x for c, x in run_plan_trajectory(bundle, scoring, cfg, rp, ru, plans[ref_name], False)["x_final"].items()}
            torch.save({cid_key(c): v for c, v in ref_latents.items()}, out / "vae_input_latents.pt")
            rep, ref_images = vae_invariance(bundle, ref_latents, ids, dc["vae_chunks"], dc["orders"], dc["ref_batch"])
        elif part == "rm":
            if ref_images is None:
                if ref_latents is None:
                    ref_latents = run_plan_trajectory(bundle, scoring, cfg, rp, ru, plans[ref_name], False)["x_final"]
                with torch.inference_mode():
                    imgs = decode_delivered(bundle, torch.stack([ref_latents[c] for c in ids], 0).to(bundle.device), dc["ref_batch"])
                ref_images = dict(zip(ids, bundle.to_pil(imgs)))
            rep = rm_invariance(scorer, rp, dict(ref_images), ids, plans, ref_name, dc["rm_repeats"], out / "rm_inputs")
        elif part == "method":
            _write(out, "method_plan", {**method_plan(dc), "eph": meta.get("eph"), "written_before_run": True})
            rep = method_effects(bundle, scoring, cfg, dc["prompts"][: dc["method_prompts"]], dc["method_banks"], dc["method_pairs"])
        else:
            raise ValueError(part)
        res = {}
        if torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake":
            res = {"peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        rep.update({"part": part, "label": LABEL, "seconds": time.perf_counter() - t0, "resources": res, **meta})
        _write(out, part_name, rep)
        status[part_name] = {"seconds": round(rep["seconds"], 2)}
        progress(f"[diag] {part} done in {rep['seconds']:.1f}s")
    return status


def compare_aa_cli(out: Path, tag_a: str, tag_b: str, expect_eph: Optional[str] = None) -> int:
    torch = import_torch()
    if tag_a == tag_b:
        raise DiagConfigError("A/A needs two different runs")
    recs = [json.loads((out / f"aa_{t}.json").read_text(encoding="utf-8")) for t in (tag_a, tag_b)]
    tens = [torch.load(out / f"aa_{t}.pt", map_location="cpu") for t in (tag_a, tag_b)]
    if recs[0].get("pid") is not None and recs[0].get("pid") == recs[1].get("pid"):
        raise DiagConfigError("both A/A runs come from the same process")
    rep = compare_aa(recs[0], recs[1], tens[0], tens[1], expect_eph)
    rep.update({"runs": [tag_a, tag_b], "pids": [r.get("pid") for r in recs], "label": LABEL})
    _write(out, "aa_compare", rep)
    print(json.dumps({"all_bitwise_identical": rep["all_bitwise_identical"], "eph": rep["eph"], "summary": rep["summary"]},
                     default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch-drift diagnostic harness (engineering only)")
    add_backend_args(ap)
    ap.add_argument("--diag-config", default="configs/drift_diagnostic.yaml")
    ap.add_argument("--part", nargs="+", choices=PARTS + EXTRA_PARTS + ("all",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-pool", default=None)
    ap.add_argument("--aa-tag", default="run1", help="name of this A/A run (part aa); one per process")
    ap.add_argument("--compare-aa", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="compare two A/A runs found in --out (no model loaded); writes aa_compare.json")
    ap.add_argument("--expect-eph", default=None, help="EPH the A/A runs are expected to carry (e.g. smoke1's)")
    a = ap.parse_args(argv)
    if a.compare_aa:
        return compare_aa_cli(Path(a.out), *a.compare_aa, expect_eph=a.expect_eph)
    if not a.part:
        ap.error("--part is required unless --compare-aa is given")
    dc = load_diag_config(a.diag_config)
    guard = guard_prompts(dc, a.prompt_pool)
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    parts = list(PARTS) if "all" in a.part else a.part
    meta = {"eph": eph, "pid": os.getpid(), "backend": body["backend"], "prompt_guard": guard, "vae_decode_chunk_cfg": cfg.get("vae_decode_chunk"),
            "diag_config": {k: v for k, v in dc.items()}}
    st = run_parts(parts, cfg, bundle, scorer, scoring, dc, Path(a.out), meta, progress=lambda s: print(s, flush=True),
                   aa_tag=a.aa_tag)
    print(json.dumps({"out": a.out, "parts": st, "eph": eph}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
