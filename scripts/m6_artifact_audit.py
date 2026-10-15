"""Read-only audit of batch-drift evidence in a smoke run's artifacts (no model, no GPU).

  python scripts/m6_artifact_audit.py <extracted run dir, e.g. .../smoke1/x/runs/smoke1> --out reports/batch_drift/smoke1_audit.json

Sources and what each can show:
  p0/batch-parity.json (M6)   per-batch MAXIMA only (no per-candidate rows, no RMSE, no ranking gaps)
  tsmoke/TSMOKE_BANK          k=8 collector table: per-candidate preview IR at j=1..16, delivered IR, PNG,
                              init_latent_sha
  tsmoke/psp.json             online B2 on the SAME prompt uid and CandidateIds: j=4 at k=8, j=8 at k=4,
                              delivered at k=2
  tsmoke/single.json + PNG    candidate (0,0) at k=1 for the whole trajectory
  tsmoke/parity.json          delivered score of (0,0) under three online schedules vs the bank
The input directory is never written to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOT_MEASURED = "not measured"


def _png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def pixel_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    px = d.max(-1)
    return {"max_abs_u8": int(d.max()), "mean_abs_u8": float(d.mean()), "rmse_u8": float(np.sqrt((d.astype(float) ** 2).mean())),
            "frac_pixels_changed": float((px > 0).mean()), "frac_pixels_gt8": float((px > 8).mean()),
            "bitwise_equal": bool(not d.any())}


def _f32eq(a, b) -> bool:
    """The bank stores IR as float32; compare at that precision as well as exactly."""
    return bool(np.float32(a) == np.float32(b))


def ranking_gaps(scores: dict, keep: int) -> dict:
    """Gap between the last kept and the first dropped score under the fixed ranking (ties by id)."""
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return {"kept": sorted(order[:keep]), "last_kept": order[keep - 1], "first_dropped": order[keep],
            "gap": float(scores[order[keep - 1]] - scores[order[keep]])}


def audit_m6(run: Path) -> dict:
    m6 = json.loads((run / "p0" / "batch-parity.json").read_text(encoding="utf-8"))
    rows = []
    for b, r in sorted(m6["by_batch"].items(), key=lambda kv: int(kv[0])):
        rows.append({"batch": int(b), "groups": 16 // int(b) if int(b) <= 16 else 1,
                     "delivered_latent_max_abs": r["delivered_latent_max_abs"],
                     "delivered_latent_rmse": NOT_MEASURED, "delivered_latent_rel_l2": NOT_MEASURED,
                     "preview_score_max_abs_over_j4_8_16": r["preview_score_max_abs"],
                     "delivered_score_max_abs": r["delivered_score_max_abs"],
                     "per_candidate_errors": NOT_MEASURED, "ranking_gap_at_cutoff": NOT_MEASURED,
                     "B2_winner_flips_of_2_banks": r["B2_winner_flips_of_2"],
                     "is_reference": int(b) == m6["reference_batch"]})
    return {
        "source": "p0/batch-parity.json", "eph": m6.get("eph"), "seconds": m6.get("seconds"),
        "reference_batch": m6["reference_batch"],
        "metric_definitions": {
            "delivered_latent_max_abs": "max over the 16 CandidateIds (pool rows (0,0)..(1,7)) and all 4x64x64 "
                                        "elements of |x64_b - x64_ref| on the final DDIM latent x^64 (fp16 cast to "
                                        "fp32, scaled latent space, before /vae_scaling_factor)",
            "preview_score_max_abs": "max over j in {4,8,16} and the 16 CandidateIds of |IR_b - IR_ref| on preview "
                                     "x0 scores; decode and ImageReward ran on the same batch b (confounded)",
            "delivered_score_max_abs": "max over CandidateIds of |IR_b - IR_ref| on delivered images (decode + RM "
                                       "on batch b, confounded)",
            "B2_winner_flips": "per bank, B2 (8->4 at j=4, 4->2 at j=8, top-1 delivered) replayed on the batch-b "
                               "scores vs on the batch-8 scores; counts banks whose winner differs (0..2)",
        },
        "rows": rows,
        "limits": [
            "batch 8 is the reference compared with itself: its zeros are trivial, not a repeat test",
            "per-candidate errors, RMSE/relative error, ranking gaps and first divergent forward were not stored",
            "latent error isolates UNet+scheduler; score errors mix UNet, VAE-decode batch and ImageReward batch",
            "flips come from 2 banks of one prompt: no rate can be estimated",
        ],
    }


def identity_audit_m6() -> list:
    """What the M6 code (apsp/gpu/profile.py stage_batch_parity @ 933e9ac) guarantees vs records."""
    return [
        {"item": "CandidateId", "same_across_batches": "yes, by code: every plan slices pool_rows(2); rows of a "
         "group map to tensor rows in order (no prune)", "recorded_in_report": False},
        {"item": "initial latent bytes", "same_across_batches": "by construction: derive_seed('m6-technical', bank, "
         "seed) -> CPU torch.Generator float32 randn -> fp16 cast, per candidate, independent of the batch",
         "recorded_in_report": False, "note": "init_latent_sha is not written by M6 (the collector writes it)"},
        {"item": "prompt embeddings", "same_across_batches": "expected: re-encoded at batch 1 by each group's "
         "prepare_request; tiled to k rows", "recorded_in_report": False,
         "note": "no embedding hash in the report; equality across groups not verified"},
        {"item": "scheduler / timesteps", "same_across_batches": "expected: set_timesteps(64) per engine; the EPH "
         "pins the 64 timesteps", "recorded_in_report": "EPH only"},
        {"item": "CFG row mapping", "same_across_batches": "yes, by code: emb = [uncond x k; cond x k], x_in = "
         "cat([x, x]); eps.chunk(2)", "recorded_in_report": False},
        {"item": "seed integer", "same_across_batches": "yes", "note": "not sufficient on its own; listed for completeness"},
    ]


def audit_cross_batch(run: Path) -> dict:
    ts = run / "tsmoke"
    bank = ts / "TSMOKE_BANK"
    pv = pd.read_parquet(bank / "previews.parquet")
    dl = pd.read_parquet(bank / "delivered.parquet")
    seeds = pd.read_parquet(bank / "seeds.parquet")
    psp = json.loads((ts / "psp.json").read_text(encoding="utf-8"))
    single = json.loads((ts / "single.json").read_text(encoding="utf-8"))
    parity = json.loads((ts / "parity.json").read_text(encoding="utf-8"))
    uid = str(pv["prompt_uid"].iloc[0])
    same_uid = {"bank": uid, "psp_prompt": psp["prompt"], "single_prompt": single["prompt"],
                "note": "technical_smoke derives uid = prompt_uid(text) in single, psp and collector; the bank "
                        "prompt text equals the psp/single prompt"}
    bank_prompt = str(pd.read_parquet(bank / "prompts.parquet")["text"].iloc[0])
    same_uid["prompt_text_equal"] = bank_prompt == psp["prompt"] == single["prompt"]

    def pv_score(j, s):
        return float(pv[(pv.j == j) & (pv.seed_idx == s)].ir.iloc[0])

    def dl_score(s):
        return float(dl[dl.seed_idx == s].ir_deliv.iloc[0])

    per = []
    verify = [a for a in psp["audit"] if a["event"] == "VERIFY"]
    for a in verify:
        k_on = len(a["alive"])
        for c, s in zip(a["alive"], a["scores"]):
            b = pv_score(a["j"], c[1])
            per.append({"cand": c, "quantity": f"preview IR j={a['j']}", "batch_online": k_on, "batch_bank": 8,
                        "online": s, "bank": b, "diff": s - b, "bitwise_equal": s == b, "equal_as_float32": _f32eq(s, b),
                        "batch_history_online": "k=8 from j=0" if a["j"] <= 4 else "k=8 to j=4, then k=4",
                        "init_latent_sha_bank": str(seeds[seeds.seed_idx == c[1]].init_latent_sha.iloc[0])})
    for key, v in psp["delivered_scores"].items():
        s = int(key.split(",")[1])
        per.append({"cand": [0, s], "quantity": "delivered IR", "batch_online": 2, "batch_bank": 8, "online": v,
                    "bank": dl_score(s), "diff": v - dl_score(s), "bitwise_equal": v == dl_score(s),
                    "batch_history_online": "k=8 to j=4, k=4 to j=8, k=2 to j=16; decode+RM at k=2"})
    per.append({"cand": [0, 0], "quantity": "delivered IR", "batch_online": 1, "batch_bank": 8,
                "online": single["record"]["score"], "bank": dl_score(0), "diff": single["record"]["score"] - dl_score(0),
                "bitwise_equal": single["record"]["score"] == dl_score(0),
                "batch_history_online": "k=1 for all 64 forwards; decode+RM at k=1"})
    for r in parity["records"]:
        per.append({"cand": r["winner"][0], "quantity": f"delivered IR, online {r['method']}", "batch_online": "schedule",
                    "batch_bank": 8, "online": r["score_online"], "bank": r["score_replay"],
                    "diff": r["score_online"] - r["score_replay"], "bitwise_equal": r["score_online"] == r["score_replay"],
                    "winner_equal": r["winner_equal"]})

    # ranking gaps on the k=8 bank at the B2 cutoffs
    s4 = {(0, s): pv_score(4, s) for s in range(8)}
    g4 = ranking_gaps(s4, 4)
    s8 = {c: pv_score(8, c[1]) for c in g4["kept"]}
    g8 = ranking_gaps(s8, 2)
    sd = {c: dl_score(c[1]) for c in g8["kept"]}
    gd = ranking_gaps(sd, 1)
    max_online_diff = {"j4_k8": max(abs(p["diff"]) for p in per if p["quantity"] == "preview IR j=4"),
                       "j8_k4": max(abs(p["diff"]) for p in per if p["quantity"] == "preview IR j=8"),
                       "delivered_k2": max(abs(p["diff"]) for p in per if p["quantity"] == "delivered IR" and p["batch_online"] == 2)}

    img0 = _png(bank / "shards" / f"{uid}__b0" / "img_s0.png")
    pix = {"(0,0) k=1 single vs k=8 bank": pixel_diff(_png(ts / "single_s0.png"), img0),
           "(0,0) k=8->4->2 psp winner vs k=8 bank": pixel_diff(_png(ts / "psp_winner.png"), img0)}
    hashes = {p: hashlib.sha256((ts / p).read_bytes()).hexdigest()
              for p in ("single_s0.png", "psp_winner.png")}
    hashes["bank img_s0.png"] = str(dl[dl.seed_idx == 0].image_sha.iloc[0])
    return {"same_prompt": same_uid, "per_candidate": per,
            "b2_ranking_gaps_on_bank_k8": {"j4_keep4": g4, "j8_keep2": g8, "final_keep1": gd},
            "max_abs_diff_online_vs_bank": max_online_diff,
            "j4_k8_all_equal_as_float32": all(p.get("equal_as_float32") for p in per if p["quantity"] == "preview IR j=4"),
            "decisions_stable": {"j4": "same composition -> equal scores, same keep set",
                                 "j8": f"keep set equal online/bank; gap {g8['gap']:.4f} vs max |diff| "
                                       f"{max_online_diff['j8_k4']:.4f}",
                                 "final": f"winner equal; gap {gd['gap']:.4f} vs max |diff| {max_online_diff['delivered_k2']:.4f}"},
            "pixel_diffs": pix, "png_sha256": hashes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="reports/batch_drift/smoke1_audit.json")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    rep = {"run_dir_commit": (run / "commit.txt").read_text().strip() if (run / "commit.txt").exists() else None,
           "label": "ENGINEERING ONLY; technical prompt; not G1",
           "m6": audit_m6(run), "m6_identity_audit": identity_audit_m6(), "cross_batch_from_technical_smoke": audit_cross_batch(run)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), "m6_rows": len(rep["m6"]["rows"]),
                      "per_candidate_rows": len(rep["cross_batch_from_technical_smoke"]["per_candidate"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
