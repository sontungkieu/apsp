"""Online runner (CC-7 / M7 / M10). Uses the SAME runners (apsp.sim.env.run_closed_loop / run_schedule), ledger,
DP mask, observation builder and policy API as replay; only the engine differs (DenoiseEngine on GPU).

Wall-clock of a steady-state request starts before once/C_fixed work and stops when the winner image is on
the CPU (PIL). PNG/log I/O happens after the timer. Model load / warm-up are excluded.

  python -m apsp.gpu.online --task smoke --prompt "a red bicycle on a beach" --synthetic-costs --methods b2 random
  python -m apsp.gpu.online --task replay-parity --bank DIR --costs COSTS.json --split train_dev --methods b2 random
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from apsp.contracts.identity import bank_rows
from apsp.contracts.ledger import MethodCosts, load_method_costs
from apsp.gpu.backend import add_backend_args, build_backend, compute_eph
from apsp.gpu.engine import DenoiseEngine
from apsp.policies.fixed import REFERENCE_CUTS
from apsp.policies.random_policy import RandomFeasiblePolicy
from apsp.policies.threshold import ThresholdParams, ThresholdPolicy
from apsp.sim.budget_dp import G842
from apsp.sim.caps import compute_caps, fixed_method_costs
from apsp.sim.env import run_closed_loop, run_schedule
from apsp.sim.features import NormStats
from apsp.sim.schedules import fixed_pair_spec


def make_method(spec: str, seed: int = 0):
    """'b2' | 'fixed:J1,J2' | 'random' | 'threshold:PATH.json'. Returns ('schedule', (j1,j2)) or ('policy', obj)."""
    if spec == "b2":
        return "schedule", REFERENCE_CUTS
    if spec.startswith("fixed:"):
        j1, j2 = (int(x) for x in spec.split(":", 1)[1].split(","))
        return "schedule", (j1, j2)
    if spec == "random":
        return "policy", RandomFeasiblePolicy(seed)
    if spec.startswith("threshold:"):
        p = json.loads(Path(spec.split(":", 1)[1]).read_text(encoding="utf-8"))
        return "policy", ThresholdPolicy(ThresholdParams(**p.get("params", p)))
    raise ValueError(spec)


def online_episode(engine: DenoiseEngine, prompt: str, uid: str, bank_id: int, method, *, base: MethodCosts,
                   ctrl: MethodCosts, cap_us: int, C_ref: int, norm: NormStats, prompt_feat=None):
    kind, obj = method
    t0 = time.perf_counter()
    engine.prepare_request(prompt, uid)                    # C_fixed: outside cap, inside e2e
    rows = bank_rows(bank_id)
    if kind == "schedule":
        res = run_schedule(fixed_pair_spec(*obj, rows), engine, mc=fixed_method_costs(base), cap_us=cap_us,
                           C_ref=C_ref, method_id=f"fixed{obj}")
    else:
        res = run_closed_loop(obj, engine, rows=rows, grammar=G842(), mc=ctrl, cap_us=cap_us, C_ref=C_ref, norm=norm,
                              prompt_feat=prompt_feat)
    e2e = time.perf_counter() - t0                          # winner PIL already on CPU inside finalize
    return res, e2e


def _y(res) -> Optional[float]:
    if not res.feasible or res.delivery is None or res.delivery.failed:
        return None
    return res.delivery.inference_score


def task_smoke(a, cfg, bundle, scorer, scoring, eph) -> dict:
    from apsp.data.prompts import normalize, prompt_uid
    from apsp.data.synthetic_bank import synthetic_costs
    if not a.prompt:
        raise SystemExit("--prompt is required for --task smoke")
    base = load_method_costs(a.costs) if a.costs else synthetic_costs()
    ctrl = base.with_overrides(disp=a.disp_us, method_id="controller")
    caps = compute_caps(base)
    cap = caps.cap(a.alpha)
    engine = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
    text = normalize(a.prompt)
    uid = prompt_uid(text)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for m in a.methods:
        res, e2e = online_episode(engine, text, uid, 0, make_method(m, a.seed), base=base, ctrl=ctrl, cap_us=cap,
                                  C_ref=caps.C_ref, norm=NormStats.identity())
        img_path = None
        if res.feasible and res.delivery.winner is not None:
            im = engine.last_images.get(res.delivery.winner)
            if im is not None:                                  # I/O after the timer
                img_path = str(out_dir / f"{m.replace(':', '_')}_winner.png")
                im.save(img_path)
        recs.append({"method": m, "feasible": res.feasible, "winner": res.delivery.winner if res.feasible else None,
                     "inference_score": _y(res), "ledger_spend_us": res.spend_us, "cap_us": cap,
                     "prune_js": list(res.prune_js), "n_verify": res.n_verify, "e2e_wall_s": e2e,
                     "unet_calls_total": engine.n_unet_calls, "image": img_path})
    return {"task": "smoke", "prompt": text, "costs": "synthetic (NOT measured)" if not a.costs else a.costs,
            "eph": eph, "records": recs,
            "label": "technical smoke on a prompt outside the research pool; not a quality or speed result"}


def task_replay_parity(a, cfg, bundle, scorer, scoring, eph) -> dict:
    from apsp.contracts.prereg import require_signed, require_split_access
    from apsp.contracts.schema import load_bank
    from apsp.sim.replay import ReplayContext
    bank = load_bank(a.bank)
    research = not (bank.manifest.get("synthetic") or bank.manifest.get("backend") == "fake")
    if research:
        require_signed("B")
        require_split_access(a.split, "G1 replay-parity")
        if bank.manifest.get("eph") != eph:
            raise SystemExit(f"bank EPH {bank.manifest.get('eph', '')[:12]} != online EPH {eph[:12]}: parity claim refused")
    base = load_method_costs(a.costs)
    ctrl = base.with_overrides(disp=a.disp_us, method_id="controller")
    ctx = ReplayContext.build(bank, base, fit_split=a.fit_split, guard=research)
    cap = ctx.caps.cap(a.alpha)
    engine = DenoiseEngine(bundle, scoring, cfg, hook_mode="none")
    keys = bank.instances_of([a.split])[: a.limit or None]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for key in keys:
        uid, b = key
        text = bank.text_of(uid)
        ref_on, _ = online_episode(engine, text, uid, b, make_method("b2"), base=base, ctrl=ctrl, cap_us=cap,
                                   C_ref=ctx.caps.C_ref, norm=ctx.norm, prompt_feat=ctx.prompt_feat(uid))
        ref_rp = ctx.run_spec(fixed_pair_spec(*REFERENCE_CUTS, bank_rows(b)), key, cap)
        for m in a.methods:
            meth = make_method(m, a.seed)
            on, e2e = online_episode(engine, text, uid, b, meth, base=base, ctrl=ctrl, cap_us=cap, C_ref=ctx.caps.C_ref,
                                     norm=ctx.norm, prompt_feat=ctx.prompt_feat(uid))
            meth_r = make_method(m, a.seed)
            rp = (ctx.run_spec(fixed_pair_spec(*meth_r[1], bank_rows(b)), key, cap) if meth_r[0] == "schedule"
                  else ctx.run_policy(meth_r[1], key, cap, ctrl))
            recs.append({"uid": uid, "bank_id": b, "method": m,
                         "y_online": _y(on) if _y(on) is not None else ctx.r_fail,
                         "yref_online": _y(ref_on) if _y(ref_on) is not None else ctx.r_fail,
                         "y_replay": rp.Y, "yref_replay": ref_rp.Y,
                         "winner_online": list(on.delivery.winner) if on.feasible and on.delivery.winner else None,
                         "winner_replay": list(rp.delivery.winner) if rp.feasible and rp.delivery.winner else None,
                         "actions_online": [e.action for e in on.trace], "actions_replay": [e.action for e in rp.trace],
                         "spend_online": on.spend_us, "spend_replay": rp.spend_us, "e2e_wall_s": e2e})
    with open(out / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return {"task": "replay-parity", "n_records": len(recs), "out": str(out / "records.jsonl"), "eph": eph,
            "identical_action_traces": sum(r["actions_online"] == r["actions_replay"] for r in recs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Online runner (shared ledger/mask/obs/policy API)")
    add_backend_args(ap)
    ap.add_argument("--task", choices=["smoke", "replay-parity", "test"], required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--methods", nargs="+", default=["b2", "random"])
    ap.add_argument("--costs", default=None, help="MethodCosts JSON (measured generator costs)")
    ap.add_argument("--synthetic-costs", action="store_true", help="use the SYNTHETIC cost table (smoke only)")
    ap.add_argument("--disp-us", type=int, default=0, help="controller dispatch fee (provisional until M2c)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bank", default=None)
    ap.add_argument("--split", default="train_dev")
    ap.add_argument("--fit-split", default="train_fit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/runs/online")
    a = ap.parse_args(argv)
    if a.task == "test":
        from apsp.contracts.prereg import require_signed
        require_signed("C")
        raise SystemExit("test-split online runs are wired in P4 after PREREG-C freezes the artifacts")
    if a.task == "smoke" and not (a.costs or a.synthetic_costs):
        raise SystemExit("smoke needs --costs or --synthetic-costs")
    cfg, bundle, scorer, scoring, rt = build_backend(a)
    eph, _ = compute_eph(cfg, bundle, scorer, rt)
    rep = task_smoke(a, cfg, bundle, scorer, scoring, eph) if a.task == "smoke" else task_replay_parity(a, cfg, bundle, scorer, scoring, eph)
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
