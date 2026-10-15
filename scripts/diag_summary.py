"""Digest of a drift_harness output directory (read-only): the numbers BATCH_DRIFT_AUDIT.md reports.

  python scripts/diag_summary.py <run>/diag --out reports/batch_drift/diag1_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def digest(d: Path) -> dict:
    out = {"dir": str(d)}
    ephs = {}
    for name in ("audit", "aa_run1", "aa_run2", "traj", "unet", "vae", "rm", "method"):
        r = load(d, name)
        if r is None:
            out[name] = "not measured (no report)"
            continue
        ephs[name] = r.get("eph")
        res = {"seconds": round(r.get("seconds", 0), 1), "resources": r.get("resources")}
        if name == "audit":
            res.update({k: r.get(k) for k in ("all_eval", "numeric_runtime")})
            for m in ("unet", "vae", "vae_preview", "text_encoder", "reward_model"):
                if isinstance(r.get(m), dict):
                    res[m] = {k: r[m].get(k) for k in ("present", "n_training_submodules", "param_dtypes", "attn_processors",
                                                       "dropout_types_present") if k in r[m]}
        elif name == "traj":
            res["plans"] = {p: {k: v["summary"].get(k) for k in ("n_diverged", "earliest_divergent_forward",
                                                                  "preview_score_max_abs", "delivered_score_max_abs",
                                                                  "b2_winner_flips")}
                            | {"final_max_abs": v["summary"]["final"].get("max_max_abs"),
                               "final_max_rmse": v["summary"]["final"].get("max_rmse"),
                               "b2": {b: {"flip": x["winner_flip"], "keep_j1_equal": x["keep_j1_equal"],
                                          "keep_j2_equal": x["keep_j2_equal"],
                                          "ref_gaps": [x["ref_plan"]["gap_j1"], x["ref_plan"]["gap_j2"], x["ref_plan"]["gap_final"]]}
                                      for b, x in v["summary"].get("b2_by_bank", {}).items()}}
                            for p, v in r["plans"].items()}
        elif name == "unet":
            res["inputs_identical_across_plans"] = r["inputs_identical_across_plans"]
            res["embeddings_identical_across_groups"] = r["embeddings_identical_across_groups"]
            res["plans"] = {p: {f: {q: (s["summary"][q]["n_bitwise_equal"], s["summary"][q]["max_max_abs"])
                                    for q in ("eps_cond", "eps_uncond", "x_prev")}
                                for f, s in v["by_forward"].items()} for p, v in r["plans"].items()}
        elif name == "vae":
            res["plans"] = {p: {"tensor_n_bitwise": v["summary"]["tensor"]["n_bitwise_equal"],
                                "tensor_max_abs": v["summary"]["tensor"]["max_max_abs"],
                                "pixels_max_abs_u8": v["summary"]["pixels_max_abs_u8"],
                                "n_pixels_bitwise_equal": v["summary"]["n_pixels_bitwise_equal"]} for p, v in r["plans"].items()}
        elif name == "rm":
            res["repeat_ref"] = r["repeat_ref"]
            res["preprocessed_by"] = sorted({v["preprocessed_by"] for v in r["inputs"].values()})
            res["plans"] = {p: v["summary"] for p, v in r["plans"].items()}
        elif name == "method":
            res["summary"] = r["summary"]
            res["per_pair"] = {}
            for inst in r["instances"]:
                for k, v in inst["pairs"].items():
                    res["per_pair"].setdefault(k, []).append(
                        {"uid": inst["uid"][:8], "bank": inst["bank_id"], "winner_equal": v["winner_equal"],
                         "keep_j1_equal": v["keep_j1_equal"], "keep_j2_equal": v["keep_j2_equal"],
                         "y_diff": v["y_diff"], "gap_j1_replay": v["gap_j1_replay"], "gap_j1_online": v.get("gap_j1_online"),
                         "gap_j2_replay": v["gap_j2_replay"], "gap_j2_online": v.get("gap_j2_online"),
                         "effect_error": v.get("effect_error"), "effect_vs_4_8_replay": v.get("effect_vs_4_8_replay"),
                         "bank_y_gap_between_winners": v.get("bank_y_gap_between_winners")})
        out[name] = res
    aa = load(d, "aa_compare")
    out["aa_compare"] = "not measured (no report)" if aa is None else {
        k: aa[k] for k in ("eph", "same_inputs_declared", "init_latents", "embeddings", "final_latent_sha",
                           "image_tensor_sha", "pixel_sha", "summary", "all_bitwise_identical", "pids")}
    out["eph_by_part"] = ephs
    out["eph_single_value"] = len(set(ephs.values())) == 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("diag_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = digest(Path(a.diag_dir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
