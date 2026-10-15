"""Batch-drift harness on the CPU fake backend.

The fake ops are row-independent, so every plan must give bit-identical per-CandidateId results: a wrong
row-to-id mapping would show up as large differences between candidates. Deliberately batch-dependent
fakes check that the harness detects drift and attributes it to the right plans.
"""
import json
import subprocess
import sys
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from apsp.contracts.identity import pool_rows  # noqa: E402
from apsp.gpu import drift_harness as dh  # noqa: E402
from apsp.gpu.backend import build_backend  # noqa: E402
from apsp.gpu.fake_backend import FakeBundle, FakeScorer  # noqa: E402
from apsp.gpu.scoring import ScoringPath  # noqa: E402

IDS = list(pool_rows(2))


def fake(bundle=None, scorer=None):
    cfg, b, s, sp, _ = build_backend(Namespace(config="configs/pilot.yaml", fake_backend=True))
    b = bundle or b
    s = scorer or s
    return cfg, b, s, ScoringPath(b, s, cfg.get("vae_decode_chunk"))


class BatchDependentUNet(FakeBundle):
    def unet_eps(self, x, t, emb):
        return super().unet_eps(x, t, emb) + 1e-3 * x.shape[0]


class ChunkDependentVAE(FakeBundle):
    def decode(self, z_scaled, preview):
        return super().decode(z_scaled, preview) + 1e-2 * z_scaled.shape[0]


class BatchDependentRM(FakeScorer):
    def score(self, prompt, images):
        return [v + 1e-4 * len(images) for v in super().score(prompt, images)]


@pytest.fixture
def dc():
    d = dh.load_diag_config("configs/drift_diagnostic.yaml")
    d.update({"batch_sizes": [1, 4, 8, 16], "unet_forward_indices": [0, 5], "vae_chunks": [1, 8, 16],
              "rm_repeats": 2, "method_pairs": [[4, 8], [2, 9]], "method_prompts": 1, "method_banks": 1})
    return d


def plans(d):
    return dh.plan_list(IDS, d["batch_sizes"], d["orders"], d["ref_batch"])


def test_config_and_guards(tmp_path):
    d = dh.load_diag_config("configs/drift_diagnostic.yaml")
    assert d["split"] == "technical_smoke" and d["repro"]["uid"] == "m6-technical"
    assert dh.guard_prompts(d)["geneval_checked"] is True
    bad = tmp_path / "bad.yaml"
    bad.write_text("split: train_fit\nprompts: [a]\n")
    with pytest.raises(dh.DiagConfigError):
        dh.load_diag_config(str(bad))
    noref = tmp_path / "noref.yaml"
    noref.write_text("split: technical_smoke\nprompts: [a]\nmethod_pairs: [[2, 9]]\n")
    with pytest.raises(dh.DiagConfigError, match="reference pair"):
        dh.load_diag_config(str(noref))


def test_plans_cover_every_candidate_once():
    for b in (1, 2, 4, 8, 16):
        for o in dh.ORDERS:
            groups = dh.make_plan(IDS, b, o)
            flat = [c for g in groups for c in g]
            assert sorted(flat) == sorted(IDS) and all(len(g) == b for g in groups)
    assert dh.make_plan(IDS, 8, "rev")[0][0] == (1, 7)
    assert dh.make_plan(IDS, 2, "interleave")[0] == [(0, 0), (1, 0)]   # mixes banks inside a group


def test_unet_part_maps_rows_to_candidates(dc):
    cfg, b, s, sp = fake()
    rep = dh.unet_forward_invariance(b, sp, cfg, dc["repro"]["prompt"], dc["repro"]["uid"], IDS, plans(dc), "b8_fwd", [0, 5])
    assert rep["inputs_identical_across_plans"] and rep["embeddings_identical_across_groups"]
    assert len(rep["init_latent_sha"]) == 16
    for p in rep["plans"].values():
        for f in p["by_forward"].values():
            for q in ("eps_uncond", "eps_cond", "x_prev", "x0"):
                assert f["summary"][q]["n_bitwise_equal"] == 16, q


def test_unet_part_detects_batch_dependence(dc):
    cfg, b, s, sp = fake(bundle=BatchDependentUNet())
    rep = dh.unet_forward_invariance(b, sp, cfg, dc["repro"]["prompt"], dc["repro"]["uid"], IDS, plans(dc), "b8_fwd", [0])
    for name, p in rep["plans"].items():
        n_eq = p["by_forward"]["0"]["summary"]["eps_cond"]["n_bitwise_equal"]
        assert n_eq == (16 if name.startswith("b8_") else 0), name


def test_traj_part_no_drift_and_m6_tables(dc):
    cfg, b, s, sp = fake()
    rep, lat = dh.trajectory_drift(b, sp, cfg, dc["repro"]["prompt"], dc["repro"]["uid"], IDS, plans(dc), "b8_fwd")
    assert len(lat) == 16
    for p in rep["plans"].values():
        sm = p["summary"]
        assert sm["n_diverged"] == 0 and sm["final"]["n_bitwise_equal"] == 16
        assert sm["b2_winner_flips"] == 0 and sm["delivered_score_max_abs"] == 0.0
        assert set(sm["b2_by_bank"]) == {"0", "1"}
        assert all(v["init_sha_equal_ref"] for v in p["per_candidate"].values())


def test_traj_part_first_divergent_forward(dc):
    cfg, b, s, sp = fake(bundle=BatchDependentUNet())
    rep, _ = dh.trajectory_drift(b, sp, cfg, dc["repro"]["prompt"], dc["repro"]["uid"], IDS,
                                 {k: v for k, v in plans(dc).items() if k in ("b8_fwd", "b8_rev", "b4_fwd")}, "b8_fwd",
                                 with_scores=False)
    assert rep["plans"]["b8_rev"]["summary"]["n_diverged"] == 0
    assert rep["plans"]["b4_fwd"]["summary"]["earliest_divergent_forward"] == 1
    assert rep["plans"]["b4_fwd"]["summary"]["n_diverged"] == 16


def test_vae_part(dc):
    cfg, b, s, sp = fake()
    _, lat = dh.trajectory_drift(b, sp, cfg, dc["repro"]["prompt"], dc["repro"]["uid"], IDS, {"b8_fwd": plans(dc)["b8_fwd"]},
                                 "b8_fwd", with_scores=False)
    rep, imgs = dh.vae_invariance(b, lat, IDS, [1, 8, 16], dc["orders"], 8)
    assert len(imgs) == 16 and len(rep["latent_sha"]) == 16
    for p in rep["plans"].values():
        assert p["summary"]["tensor"]["n_bitwise_equal"] == 16 and p["summary"]["n_pixels_bitwise_equal"] == 16
    cfg2, b2, s2, sp2 = fake(bundle=ChunkDependentVAE())
    rep2, _ = dh.vae_invariance(b2, lat, IDS, [1, 8, 16], dc["orders"], 8)
    assert rep2["plans"]["chunk8_rev"]["summary"]["tensor"]["n_bitwise_equal"] == 16
    assert rep2["plans"]["chunk1_fwd"]["summary"]["tensor"]["n_bitwise_equal"] == 0


def test_rm_part(dc, tmp_path):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(0)
    imgs = {c: Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)) for c in IDS}
    rep = dh.rm_invariance(FakeScorer(), "p", dict(imgs), IDS, plans(dc), "b8_fwd", 2, tmp_path / "png")
    assert rep["repeat_ref"]["bitwise_equal_all"] and len(list((tmp_path / "png").glob("*.png"))) == 16
    assert all(v["pixel_sha"] for v in rep["inputs"].values())
    assert all(p["summary"]["n_bitwise_equal"] == 16 for p in rep["plans"].values())
    rep2 = dh.rm_invariance(BatchDependentRM(), "p", dict(imgs), IDS, plans(dc), "b8_fwd", 2, None)
    for name, p in rep2["plans"].items():
        assert (p["summary"]["max_abs"] == 0.0) == name.startswith("b8_"), name


def test_method_part_fake_online_equals_replay(dc):
    cfg, b, s, sp = fake()
    rep = dh.method_effects(b, sp, cfg, dc["prompts"][:1], 1, [[4, 8], [2, 9]])
    sm = rep["summary"]
    assert sm["n_episodes"] == 2 and sm["n_winner_equal"] == 2 and sm["identity_checks_ok"]
    assert sm["max_abs_y_diff"] == 0.0 and sm["max_abs_effect_error"] == 0.0 and sm["n_effect_sign_disagree"] == 0


def test_module_audit_flags_training_modules():
    cfg, b, s, sp = fake()
    rep = dh.module_audit(b, s)
    assert rep["unet"]["present"] is False and "numeric_runtime" in rep

    class B:
        unet = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
        vae = vae_preview = text_encoder = None
    B.unet.train()
    r = dh.module_audit(B(), s)
    assert r["unet"]["n_training_submodules"] == 3 and r["all_eval"] is False
    assert r["unet"]["dropout_types_present"] == ["Dropout"]
    B.unet.eval()
    assert dh.module_audit(B(), s)["all_eval"] is True


def test_cli_writes_reports(tmp_path, dc):
    cfgp = tmp_path / "dc.yaml"
    import yaml
    small = {k: dc[k] for k in ("split", "prompts", "batch_sizes", "ref_batch", "orders", "unet_forward_indices",
                                "vae_chunks", "rm_repeats", "method_pairs", "method_prompts", "method_banks")}
    small["repro"] = {"prompt": dc["repro"]["prompt"], "uid": "m6-technical"}
    small["batch_sizes"], small["orders"] = [8, 16], ["fwd"]
    cfgp.write_text(yaml.safe_dump(small))
    assert dh.main(["--fake-backend", "--diag-config", str(cfgp), "--part", "all", "--out", str(tmp_path / "o")]) == 0
    for part in dh.PARTS:
        d = json.loads((tmp_path / "o" / f"{part}.json").read_text())
        assert d["part"] == part and d["backend"] == "fake" and "not G1" in d["label"]
    assert (tmp_path / "o" / "vae_input_latents.pt").exists()


def test_m6_artifact_audit_on_fake_run(tmp_path):
    """scripts/m6_artifact_audit.py reads a run dir without writing to it."""
    from apsp.gpu import profile
    from apsp.gpu.technical_smoke import main as smoke_main
    run = tmp_path / "run"
    assert smoke_main(["--fake-backend", "--step", "all", "--out", str(run / "tsmoke")]) == 0
    assert profile.main(["--fake-backend", "--stage", "batch-parity", "--out-dir", str(run / "p0")]) == 0
    before = sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file())
    out = tmp_path / "audit.json"
    r = subprocess.run([sys.executable, "scripts/m6_artifact_audit.py", str(run), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    assert len(rep["m6"]["rows"]) == 5 and rep["m6"]["rows"][0]["delivered_latent_rmse"] == "not measured"
    cb = rep["cross_batch_from_technical_smoke"]
    assert cb["same_prompt"]["prompt_text_equal"] and cb["j4_k8_all_equal_as_float32"]
    after = sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file())
    assert before == after


def test_traj_plans_selection(tmp_path, dc):
    cfg, b, s, sp = fake()
    dc.update({"batch_sizes": [4, 8], "orders": ["fwd", "rev"], "traj_plans": ["b4_fwd"]})
    dh.run_parts(["traj"], cfg, b, s, sp, dc, tmp_path, {})
    rep = json.loads((tmp_path / "traj.json").read_text())
    assert set(rep["plans"]) == {"b8_fwd", "b4_fwd"}           # the reference plan is always included
    dc["traj_plans"] = ["b3_fwd"]
    with pytest.raises(dh.DiagConfigError, match="traj_plans"):
        dh.run_parts(["traj"], cfg, b, s, sp, dc, tmp_path, {})
