"""Restore check on a local copy (CPU only): validate the bank, load it, load each RL controller checkpoint and
recompute its replay train_dev score exactly as apsp.policies.reinforce.train_real does; it must equal the best_dev
recorded at training time. Also opens one latent shard and the frozen baselines. Writes a JSON report.

  python scripts/restore_check.py --root D:/Workspace/DMRL_artifacts/run_56353913/full --out restore_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root) / "apsp"
    from apsp.contracts.ledger import load_method_costs
    from apsp.contracts.schema import load_bank, validate_bank
    from apsp.policies.fixed import best_fixed
    from apsp.policies.reinforce import load_policy
    from apsp.sim.replay import ReplayContext
    from apsp.sim.schedules import enumerate_pairs

    rep: dict = {"root": str(root)}
    bank_dir = root / "banks" / "PILOT"
    v = validate_bank(bank_dir, check_files=True)
    rep["bank_validate"] = {"ok": not v.errors, "errors": v.errors[:20]}
    bank = load_bank(bank_dir)
    rep["bank"] = {"bmh": bank.manifest.get("bmh"), "n_train_fit": len(bank.instances_of(["train_fit"])),
                   "n_train_dev": len(bank.instances_of(["train_dev"]))}
    lat = sorted((bank_dir / "latents").rglob("*"))
    lat = [p for p in lat if p.is_file()]
    rep["latents"] = {"n_files": len(lat), "bytes": int(sum(p.stat().st_size for p in lat))}
    if lat:
        p = lat[0]
        try:
            if p.suffix == ".npy":
                arr = np.load(p, mmap_mode="r")
                rep["latents"]["sample"] = {"file": p.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            elif p.suffix == ".npz":
                z = np.load(p)
                rep["latents"]["sample"] = {"file": p.name, "keys": list(z.keys())[:5]}
            else:
                import torch
                t = torch.load(p, map_location="cpu", weights_only=True)
                rep["latents"]["sample"] = {"file": p.name, "type": type(t).__name__}
        except Exception as e:  # noqa: BLE001
            rep["latents"]["sample_error"] = f"{type(e).__name__}: {e}"[:200]
    snap = json.loads((Path(a.root) / "DMRL/docs/PREREG_PROPOSALS/PREREG_B_LOCKED.json").read_text(encoding="utf-8"))
    base = load_method_costs(str(root / "runs/calibA/p0/m2/generator_costs.json"))
    ctx = ReplayContext.build(bank, base, fit_split="train_fit", r_fail=float(snap["B4_r_fail"]))
    fit, dev = bank.instances_of(["train_fit"]), bank.instances_of(["train_dev"])
    mc = base.with_overrides(method_id="M1", uses_traj=False)
    Yf, C = ctx.schedule_table(fit)
    idx = {p: i for i, p in enumerate(enumerate_pairs())}
    Yd, _ = ctx.schedule_table(dev)
    rep["rl"] = {}
    for s in (0, 1, 2):
        art = root / "rl" / f"seed{s}" / "artifact.json"
        meta = json.loads(art.read_text(encoding="utf-8"))
        caps = meta["summary"]["caps"]
        b3 = {a_: best_fixed(Yf, C, ctx.caps.cap(a_)) for a_ in caps}
        pol = load_policy(art)
        sc = float(np.mean([ctx.run_policy(pol, k, ctx.caps.cap(a_), mc).Y - float(Yd[i, idx[(b3[a_].j1, b3[a_].j2)]])
                            for i, k in enumerate(dev) for a_ in caps]))
        n_par = sum(int(np.prod(t.shape)) for t in pol.net.state_dict().values())
        rep["rl"][f"seed{s}"] = {"loaded": True, "params": n_par, "cph": meta.get("cph"),
                                 "replay_dev_recomputed": sc, "replay_dev_recorded": meta["best_dev"],
                                 "match": abs(sc - meta["best_dev"]) < 1e-9,
                                 "optimizer_state_saved": (art.parent / "optimizer.pt").exists()}
    fr = json.loads((root / "tuned" / "methods_frozen.json").read_text(encoding="utf-8"))
    rep["methods_frozen"] = {"n_methods": len(fr["methods"]), "selected_on": fr["selected_on"]}
    rep["ok"] = rep["bank_validate"]["ok"] and all(x["match"] for x in rep["rl"].values())
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
