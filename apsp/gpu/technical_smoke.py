"""Technical smoke steps for the first GPU session (ENGINEERING ONLY; see configs/technical_smoke.yaml).

Steps (each writes <out>/<step>.json and exits non-zero if a check fails):
  single     one candidate, 64 DDIM steps, delivered decode + ImageReward scoring
  psp        one prompt through PSP-default 8->4->2 (B2 schedule) with identity / CFG-row / fresh-score /
             final-delivery checks after every event
  collector  tiny bank (configured prompts x 1 bank x 8 seeds x 16 checkpoints): validate, resume (all shards
             skipped, no recomputation), hash-mismatch refusal
  parity     online vs replay on the collector bank, same prompts/seeds: traces, survivors, winner, scores.
             A technical comparison on hand-written prompts: NOT G1, NOT statistical parity evidence.

  python -m apsp.gpu.technical_smoke --smoke-config configs/technical_smoke.yaml --step single --out data/tsmoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from apsp.contracts.identity import bank_rows, select_top
from apsp.contracts.schema import TECHNICAL_SPLITS, load_bank, validate_bank
from apsp.data.prompts import normalize, prompt_uid
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.engine import DenoiseEngine
from apsp.gpu.runtime import import_torch

STEPS = ("single", "psp", "collector", "parity")


class SmokeCheckError(AssertionError):
    pass


def load_smoke_config(path: str) -> dict:
    sc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if sc.get("split") not in TECHNICAL_SPLITS:
        raise SmokeCheckError(f"technical smoke split must be one of {TECHNICAL_SPLITS}")
    prompts = [normalize(p) for p in sc["prompts"]]
    if not 1 <= len(prompts) <= int(sc.get("max_prompts", 3)) or len(prompts) > 3:
        raise SmokeCheckError("technical smoke uses 1..3 prompts")
    if len(set(prompts)) != len(prompts):
        raise SmokeCheckError("duplicate technical prompts")
    sc["prompts"] = prompts
    return sc


def check_outside_research(sc: dict, pool_path: str | None = None) -> dict:
    """Refuse technical prompts that belong to a research pool or duplicate GenEval text."""
    from apsp.data.geneval_text import load_geneval_texts
    uids = {prompt_uid(p) for p in sc["prompts"]}
    out = {"pool_checked": False, "geneval_checked": False}
    pool = Path(pool_path or sc.get("prompt_pool_check") or "")
    if pool_path or (pool.name and pool.exists()):
        df = pd.read_parquet(pool)
        clash = uids & set(df["prompt_uid"].astype(str))
        if clash:
            raise SmokeCheckError(f"technical prompts found in the research prompt pool: {sorted(clash)}")
        out["pool_checked"] = True
    try:
        texts, _ = load_geneval_texts()
        g = {normalize(t).lower() for t in texts}
        hit = [p for p in sc["prompts"] if p.lower() in g]
        if hit:
            raise SmokeCheckError(f"technical prompts duplicate GenEval text: {hit}")
        out["geneval_checked"] = True
    except FileNotFoundError:
        pass
    return out


def prompts_frame(sc: dict, n: int | None = None) -> pd.DataFrame:
    ps = sc["prompts"][: n or None]
    return pd.DataFrame({"prompt_uid": [prompt_uid(p) for p in ps], "text": ps, "split": sc["split"],
                         "source": "technical_smoke", "group_id": [prompt_uid(p) for p in ps], "n_clip_tokens": 0})


def engineering_costs():
    """SYNTHETIC cost table, explicitly engineering-only (ledger/mask plumbing). Never a latency claim."""
    from apsp.data.synthetic_bank import synthetic_costs
    return synthetic_costs(method_id="engineering_only")


def _resources(bundle) -> dict:
    out = {}
    torch = import_torch()
    if torch.cuda.is_available() and getattr(bundle, "backend", "") != "fake":
        out["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        out["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    try:
        import resource  # Linux
        out["peak_host_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:
        try:
            import psutil
            out["host_rss_bytes_now"] = int(psutil.Process().memory_info().rss)
        except ImportError:
            pass
    return out


def require(cond: bool, msg: str, failures: list) -> None:
    if not cond:
        failures.append(msg)


# --------------------------------------------------------------------------------------------------
def step_single(cfg, bundle, scoring, sc, out: Path) -> dict:
    fails: list = []
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
    text = sc["prompts"][0]
    uid = prompt_uid(text)
    t0 = time.perf_counter()
    eng.prepare_request(text, uid)
    eng.start([(0, 0)])
    for _ in range(cfg["steps"] // cfg["delta"]):
        eng.advance(None, cfg["delta"], False)
    sb = scoring.score_delivered(text, eng.x)
    dt = time.perf_counter() - t0
    r = sb.records[0]
    require(eng.n_unet_calls == cfg["steps"], f"UNet calls {eng.n_unet_calls} != {cfg['steps']}", fails)
    require(r.latent_ok and r.img_ok and r.score_ok, f"invalid delivery {r}", fails)
    require(r.score is not None and math.isfinite(r.score), "non-finite ImageReward score", fails)
    img = None
    if sb.images[0] is not None:
        img = str(out / "single_s0.png")
        sb.images[0].save(img)
        require(sb.images[0].size == (cfg["width"], cfg["height"]), f"image size {sb.images[0].size}", fails)
    return {"prompt": text, "unet_calls": eng.n_unet_calls, "record": r.__dict__, "image": img,
            "seconds_64_steps_plus_score": dt, "failures": fails}


class CheckingEngine(DenoiseEngine):
    """DenoiseEngine that audits identities after every VERIFY/PRUNE (technical smoke only)."""

    def start(self, rows):
        super().start(rows)
        self.audit = []
        self.last_verify_j = None

    def verify(self, state):
        recs = super().verify(state)
        self.last_verify_j = state.j
        self.audit.append({"event": "VERIFY", "j": state.j, "alive": [list(c) for c in self.rows],
                           "scores": [recs[c].score for c in self.rows]})
        return recs

    def prune(self, state, retained):
        torch = import_torch()
        before_rows = list(self.rows)
        before_x = self.x.clone()
        k = len(before_rows)
        fresh = state.fresh_scores() if state.fresh else None
        super().prune(state, retained)
        kept = [before_rows[i] for i in retained]
        chk = {
            "rows_match_state": tuple(self.rows) == tuple(state.alive) == tuple(kept),
            "old_order_kept": list(retained) == sorted(retained),
            "latents_follow_identity": bool(torch.equal(self.x, before_x[list(retained)])),
            "cfg_uncond_rows": bool(torch.equal(self.emb[: len(kept)], self.uncond.expand(len(kept), -1, -1))),
            "cfg_cond_rows": bool(torch.equal(self.emb[len(kept):], self.cond.expand(len(kept), -1, -1))),
            "fresh_scores_same_j": self.last_verify_j == state.j,
        }
        if fresh is not None:
            chk["keep_is_fixed_topk"] = set(kept) == set(select_top(before_rows, fresh, len(kept)))
        self.audit.append({"event": "PRUNE", "j": state.j, "k_before": k, "kept": [list(c) for c in kept], "checks": chk})


def step_psp(cfg, bundle, scoring, sc, out: Path) -> dict:
    from apsp.sim.env import run_schedule
    from apsp.sim.schedules import fixed_pair_spec
    fails: list = []
    mc = engineering_costs()
    text = sc["prompts"][0]
    uid = prompt_uid(text)
    eng = CheckingEngine(bundle, scoring, cfg, hook_mode="none")
    t0 = time.perf_counter()
    eng.prepare_request(text, uid)
    res = run_schedule(fixed_pair_spec(*cfg["reference_cuts"], bank_rows(0)), eng, mc=mc, cap_us=10**12,
                       method_id="B2-technical")
    dt = time.perf_counter() - t0
    prunes = [a for a in eng.audit if a["event"] == "PRUNE"]
    require([p["j"] for p in prunes] == list(cfg["reference_cuts"]), f"prune js {[p['j'] for p in prunes]}", fails)
    require([p["k_before"] for p in prunes] == [8, 4], "expected 8->4->2", fails)
    for p in prunes:
        for name, ok in p["checks"].items():
            require(bool(ok), f"j={p['j']}: check {name} failed", fails)
    d = res.delivery
    final_alive = [tuple(c) for c in prunes[-1]["kept"]] if prunes else []
    require(not d.failed and d.rm_used, f"final delivery failed / no RM: {d}", fails)
    require(d.winner in final_alive, "winner not among the final survivors", fails)
    if d.delivered_scores:
        best = max((v for v in d.delivered_scores.values() if v is not None), default=None)
        require(d.inference_score == best, "winner is not the top delivered score", fails)
    require(eng.n_unet_calls == cfg["steps"], f"UNet calls {eng.n_unet_calls} != {cfg['steps']}", fails)
    img = None
    if d.winner is not None and eng.last_images.get(d.winner) is not None:
        img = str(out / "psp_winner.png")
        eng.last_images[d.winner].save(img)
    return {"prompt": text, "audit": eng.audit, "winner": list(d.winner) if d.winner else None,
            "inference_score": d.inference_score,
            "delivered_scores": {f"{c[0]},{c[1]}": v for c, v in d.delivered_scores.items()},
            "ledger_spend_us_engineering_only": res.spend_us, "unet_calls": eng.n_unet_calls, "seconds": dt,
            "image": img, "failures": fails}


def step_collector(cfg, bundle, scoring, sc, out: Path, eph: str, body: dict) -> dict:
    from apsp.gpu.collect_bank import CompatibilityError, run_collection
    fails: list = []
    prompts = prompts_frame(sc, sc.get("collector_prompts", 1))
    bank_dir = out / "TSMOKE_BANK"
    bpp = int(sc.get("banks_per_prompt", 1))
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    resume = bank_dir.exists()
    t0 = time.perf_counter()
    s1 = run_collection(eng, bundle, prompts, bank_dir, eph=eph, eph_body=body, cfg=cfg, banks_per_prompt=bpp,
                        resume=resume, machine_id="technical-smoke", commit="see-run-manifest")
    t1 = time.perf_counter()
    require(s1["bank"]["valid"], f"bank invalid: {s1['bank']['errors']}", fails)
    rep = validate_bank(bank_dir)
    require(rep.ok, f"validate_bank: {rep.errors[:5]}", fails)
    require(rep.stats.get("preview_rows") == len(prompts) * bpp * 8 * 16, f"preview rows {rep.stats.get('preview_rows')}", fails)
    calls_before = eng.n_unet_calls
    s2 = run_collection(eng, bundle, prompts, bank_dir, eph=eph, eph_body=body, cfg=cfg, banks_per_prompt=bpp,
                        resume=True, machine_id="technical-smoke", commit="see-run-manifest")
    require(not s2["collected"] and len(s2["skipped_valid"]) == len(prompts) * bpp, "resume recomputed shards", fails)
    require(eng.n_unet_calls == calls_before, "resume ran UNet forwards", fails)
    refused = False
    try:
        run_collection(eng, bundle, prompts, bank_dir, eph="0" * 64, eph_body=body, cfg=cfg, banks_per_prompt=bpp,
                       resume=True, machine_id="technical-smoke", commit="x")
    except CompatibilityError as e:
        refused = "EPH" in str(e)
    require(refused, "EPH mismatch was not refused", fails)
    return {"bank": str(bank_dir), "first_pass": {k: v for k, v in s1.items() if k != "seconds_per_prompt_bank"},
            "seconds_first_pass": t1 - t0, "seconds_per_prompt_bank": s1["seconds_per_prompt_bank"],
            "resume_skipped": s2["skipped_valid"], "eph_mismatch_refused": refused,
            "bank_disk_bytes": sum(p.stat().st_size for p in bank_dir.rglob("*") if p.is_file()), "failures": fails}


def step_parity(cfg, bundle, scoring, sc, out: Path, eph: str) -> dict:
    from apsp.gpu.online import make_method, online_episode
    from apsp.sim.replay import ReplayContext
    from apsp.sim.schedules import fixed_pair_spec
    fails: list = []
    bank_dir = out / "TSMOKE_BANK"
    if not (bank_dir / "manifest.json").exists():
        raise SmokeCheckError("run --step collector first")
    bank = load_bank(bank_dir)
    if set(bank.prompts["split"].astype(str)) - set(TECHNICAL_SPLITS):
        raise SmokeCheckError("parity step only runs on technical-smoke banks")
    if bank.manifest.get("eph") != eph and bank.manifest.get("backend") != "fake":
        raise SmokeCheckError("bank EPH != current EPH; collect again under this EPH")
    base = engineering_costs()
    ctrl = base.with_overrides(disp=150)
    ctx = ReplayContext.build(bank, base, fit_split=sc["split"], guard=False)
    cap = ctx.caps.cap(float(sc.get("alpha", 1.0)))
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
    recs = []
    for key in bank.instances:
        uid, b = key
        for m in sc.get("parity_methods", ["b2"]):
            on, e2e = online_episode(eng, bank.text_of(uid), uid, b, make_method(m, 0), base=base, ctrl=ctrl, cap_us=cap,
                                     C_ref=ctx.caps.C_ref, norm=ctx.norm, prompt_feat=ctx.prompt_feat(uid))
            mr = make_method(m, 0)
            rp = (ctx.run_spec(fixed_pair_spec(*mr[1], bank_rows(b)), key, cap) if mr[0] == "schedule"
                  else ctx.run_policy(mr[1], key, cap, ctrl))
            r = {"uid": uid, "bank_id": b, "method": m, "feasible": [on.feasible, rp.feasible]}
            if on.feasible and rp.feasible:
                r.update({
                    "actions_equal": [e.action for e in on.trace] == [e.action for e in rp.trace],
                    "survivors_equal": [list(e.alive) for e in on.trace] == [list(e.alive) for e in rp.trace],
                    "winner": [list(on.delivery.winner or ()), list(rp.delivery.winner or ())],
                    "winner_equal": on.delivery.winner == rp.delivery.winner,
                    "score_online": on.delivery.inference_score, "score_replay": rp.Y,
                    "score_abs_diff": (abs(on.delivery.inference_score - rp.Y)
                                       if on.delivery.inference_score is not None else None),
                    "spend_equal": on.spend_us == rp.spend_us, "e2e_seconds": e2e})
            recs.append(r)
    agree = sum(bool(r.get("actions_equal") and r.get("survivors_equal") and r.get("winner_equal")) for r in recs)
    require(all(r["feasible"][0] == r["feasible"][1] for r in recs), "feasibility differs online vs replay", fails)
    return {"note": "technical comparison on hand-written prompts; NOT G1 and NOT statistical parity evidence",
            "records": recs, "n": len(recs), "n_trace_survivor_winner_equal": agree,
            "max_score_abs_diff": max((r["score_abs_diff"] for r in recs if r.get("score_abs_diff") is not None), default=None),
            "failures": fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Technical smoke steps (engineering only)")
    add_backend_args(ap)
    ap.add_argument("--smoke-config", default="configs/technical_smoke.yaml")
    ap.add_argument("--step", choices=STEPS + ("all",), required=True)
    ap.add_argument("--out", default="data/tsmoke")
    ap.add_argument("--prompt-pool", default=None, help="research prompt pool to check the technical prompts against")
    a = ap.parse_args(argv)
    sc = load_smoke_config(a.smoke_config)
    guard = check_outside_research(sc, a.prompt_pool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    torch = import_torch()
    rc = 0
    for step in (STEPS if a.step == "all" else (a.step,)):
        if torch.cuda.is_available() and not a.fake_backend:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            if step == "single":
                rep = step_single(cfg, bundle, scoring, sc, out)
            elif step == "psp":
                rep = step_psp(cfg, bundle, scoring, sc, out)
            elif step == "collector":
                rep = step_collector(cfg, bundle, scoring, sc, out, eph, body)
            else:
                rep = step_parity(cfg, bundle, scoring, sc, out, eph)
        except SmokeCheckError as e:
            rep = {"failures": [str(e)]}
        rep.update({"step": step, "eph": eph, "backend": body["backend"], "elapsed_s": time.perf_counter() - t0,
                    "resources": _resources(bundle), "prompt_guard": guard, "label": "ENGINEERING ONLY",
                    "status": "PASS" if not rep.get("failures") else "FAIL"})
        (out / f"{step}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"step": step, "status": rep["status"], "failures": rep.get("failures"),
                          "elapsed_s": round(rep["elapsed_s"], 2)}, default=str))
        if rep["status"] != "PASS":
            rc = 7
    return rc


if __name__ == "__main__":
    sys.exit(main())
