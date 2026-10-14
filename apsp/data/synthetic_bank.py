"""Synthetic bank in the exact ``apsp-bank-v1.1.1`` schema (fixtures only).

Everything produced here is SYNTHETIC: scores, costs and rewards are generated numbers used to
exercise contracts, DP, replay and training code. They are not evidence about image quality or
real speed. Splits are named ``syn_*`` so prereg guards never confuse them with research splits.

CLI: ``python -m apsp.data.synthetic_bank --out data/banks/SYN --prompts 60``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apsp.contracts.hashing import provenance_hash, sha256_hex
from apsp.contracts.identity import derive_seed
from apsp.contracts.ledger import MethodCosts
from apsp.contracts.schema import J, K0, write_bank

SYN_SPLITS = ("syn_train_fit", "syn_train_dev", "syn_eval")


def synthetic_costs(**overrides) -> MethodCosts:
    """Synthetic, monotone-in-k primitive table (us). Labelled ``synthetic=True``; NOT a measurement."""
    ks = range(1, 17)
    d = dict(
        den={k: 9_000 + 4_000 * k for k in ks},
        ver={k: 30_000 + 12_000 * k for k in ks},
        fin={1: 40_000, **{k: 55_000 + 15_000 * k for k in range(2, 17)}},
        prune={(a, b): 200 + 20 * a for a in (16, 8, 4, 2) for b in (8, 4, 2, 1) if a > b},
        feat={k: 300 + 30 * k for k in ks},
        lfeat={k: 400 + 40 * k for k in ks},
        pred={k: 250 + 10 * k for k in ks},
        fixed={k: 20_000 + 500 * k for k in ks},
        disp=0, once=0, uses_traj=False, synthetic=True, provisional=True, method_id="fixed",
    )
    d.update(overrides)
    return MethodCosts(**d)


def _uid(i: int) -> str:
    return sha256_hex(f"synthetic-prompt-{i}")[:16]


def generate_tables(n_prompts: int = 60, seed: int = 0, n_banks: int = 2, split_frac=(0.6, 0.2, 0.2),
                    p_latent_fail: float = 0.01, p_preview_img_fail: float = 0.004, tie_every: int = 7):
    rng = np.random.default_rng(seed)
    n_fit = int(round(split_frac[0] * n_prompts))
    n_dev = int(round(split_frac[1] * n_prompts))
    splits = (["syn_train_fit"] * n_fit + ["syn_train_dev"] * n_dev + ["syn_eval"] * (n_prompts - n_fit - n_dev))
    uids = [_uid(i) for i in range(n_prompts)]
    prompts = pd.DataFrame({"prompt_uid": uids, "text": [f"synthetic prompt {i}" for i in range(n_prompts)],
                            "split": splits, "source": "synthetic", "group_id": [f"g{i}" for i in range(n_prompts)],
                            "n_clip_tokens": rng.integers(4, 61, n_prompts).astype(np.int16)})
    emb = rng.normal(0, 1, (n_prompts, 768)).astype(np.float16)
    t_in = [946 - 15 * (4 * j - 1) for j in range(1, J + 1)]  # synthetic DDIM-like timestep labels
    seed_rows, pv_rows, dl_rows = [], [], []
    jj = np.arange(1, J + 1)
    for p, uid in enumerate(uids):
        mu_p = rng.normal(0, 0.5)
        speed = rng.uniform(0.3, 2.0)                      # prompt-specific informativeness of previews
        rho = 1.0 - np.exp(-speed * jj / 4.0)
        sig = 0.9 * (1.0 - jj / J) + 0.08
        round_ties = (p % tie_every) == 0
        for b in range(n_banks):
            for s in range(K0):
                sg = derive_seed(uid, b, s)
                seed_rows.append((uid, b, s, sg, sha256_hex(f"syn-latent|{uid}|{b}|{s}")))
                q = rng.normal(0, 1)
                eps = np.zeros(J)
                e = rng.normal()
                for t in range(J):  # AR(1) noise over checkpoints
                    e = 0.6 * e + 0.8 * rng.normal()
                    eps[t] = e
                ir = mu_p + rho * q + sig * eps
                if round_ties:
                    ir = np.round(ir, 1)
                lat_ok = np.ones(J, bool)
                img_ok = np.ones(J, bool)
                if rng.random() < p_latent_fail:
                    fail_from = int(rng.integers(0, J))
                    lat_ok[fail_from:] = False
                img_ok &= lat_ok
                img_ok &= rng.random(J) >= p_preview_img_fail
                sc_ok = img_ok.copy()
                drift = np.concatenate([[np.nan], np.abs(rng.normal(0.3, 0.1, J - 1)) * (1 - jj[1:] / (J + 1))])
                gap = np.abs(rng.normal(0.5, 0.1, J))
                for t in range(J):
                    pv_rows.append((uid, b, s, t + 1, 4 * (t + 1), t_in[t], float(ir[t]) if sc_ok[t] else np.nan,
                                    bool(lat_ok[t]), bool(img_ok[t]), bool(sc_ok[t]),
                                    float(drift[t]) if lat_ok[t] else np.nan, float(gap[t]) if lat_ok[t] else np.nan,
                                    *rng.normal(0, 1, 4).tolist(), *np.abs(rng.normal(1, 0.1, 4)).tolist()))
                d_lat = bool(lat_ok[-1])
                d_img = d_lat and rng.random() >= p_preview_img_fail
                y = mu_p + q + rng.normal(0, 0.15)
                if round_ties:
                    y = round(y, 1)
                dl_rows.append((uid, b, s, float(y) if d_img else np.nan, d_lat, d_img, d_img, None, None))
    seeds = pd.DataFrame(seed_rows, columns=["prompt_uid", "bank_id", "seed_idx", "seed_global", "init_latent_sha"])
    pcols = ["prompt_uid", "bank_id", "seed_idx", "j", "forward_idx", "t_in", "ir", "latent_ok", "img_ok", "score_ok",
             "drift", "cfg_gap"] + [f"mean_c{c}" for c in range(4)] + [f"std_c{c}" for c in range(4)]
    previews = pd.DataFrame(pv_rows, columns=pcols)
    delivered = pd.DataFrame(dl_rows, columns=["prompt_uid", "bank_id", "seed_idx", "ir_deliv", "latent_ok", "img_ok",
                                               "score_ok", "image_path", "image_sha"])
    return prompts, emb, seeds, previews, delivered, t_in


def make_synthetic_bank(out_dir: str | Path, n_prompts: int = 60, seed: int = 0, **kw) -> Path:
    out = Path(out_dir)
    prompts, emb, seeds, previews, delivered, t_in = generate_tables(n_prompts, seed, **kw)
    # synthetic banks have no PNGs: mark delivered images as virtual by keeping img_ok but no file.
    # validate_bank requires image_path iff img_ok, so synthetic rows carry a virtual path + sha.
    vpath = [f"virtual/{u}_{b}_{s}.png" if ok else None
             for u, b, s, ok in delivered[["prompt_uid", "bank_id", "seed_idx", "img_ok"]].itertuples(index=False)]
    delivered["image_path"] = vpath
    delivered["image_sha"] = [sha256_hex(v) if v else None for v in vpath]
    eph_body = {"synthetic": True, "generator": "apsp.data.synthetic_bank", "seed": seed, "n_prompts": n_prompts}
    eph = provenance_hash("EPH", eph_body)
    write_bank(out, prompts=prompts, prompt_emb=emb, seeds=seeds, previews=previews, delivered=delivered,
               eph=eph, eph_body=eph_body, collection={"mode": "synthetic", "seed": seed},
               machine_id="synthetic", commit="synthetic",
               extra_manifest={"synthetic": True, "timesteps_at_checkpoints": t_in,
                               "label": "SYNTHETIC FIXTURE - not image data, not timing data"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a SYNTHETIC bank fixture in the apsp-bank-v1.1.1 schema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--costs-out", default=None, help="also write the synthetic cost table JSON here")
    a = ap.parse_args(argv)
    out = make_synthetic_bank(a.out, a.prompts, a.seed)
    if a.costs_out:
        Path(a.costs_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.costs_out).write_text(json.dumps(synthetic_costs().to_dict(), indent=2), encoding="utf-8")
    print(f"synthetic bank written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
