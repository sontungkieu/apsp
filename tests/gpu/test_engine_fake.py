"""Engine / scoring / collector / online code paths on the deterministic fake backend (CPU).

These exercise the real GPU-module code without CUDA. They are NOT CUDA tests and say nothing about
numerics, speed or image quality on real hardware.
"""
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from apsp.contracts.identity import bank_rows  # noqa: E402
from apsp.contracts.schema import load_bank, validate_bank  # noqa: E402
from apsp.data.synthetic_bank import synthetic_costs  # noqa: E402
from apsp.gpu.backend import build_backend, compute_eph  # noqa: E402
from apsp.gpu.collect_bank import CompatibilityError, run_collection, shard_ok  # noqa: E402
from apsp.gpu.engine import DenoiseEngine  # noqa: E402
from apsp.gpu.online import make_method, online_episode  # noqa: E402
from apsp.sim.caps import compute_caps  # noqa: E402
from apsp.sim.env import run_schedule  # noqa: E402
from apsp.sim.replay import ReplayContext  # noqa: E402
from apsp.sim.schedules import bon_spec, fixed_pair_spec  # noqa: E402


@pytest.fixture()
def fake():
    args = Namespace(config="configs/pilot.yaml", fake_backend=True, revision=None, preview_dtype=None, local_files_only=False)
    cfg, bundle, scorer, scoring, rt = build_backend(args)
    eph, body = compute_eph(cfg, bundle, scorer, rt)
    return cfg, bundle, scorer, scoring, eph, body


def _prompts(n):
    return pd.DataFrame({"prompt_uid": [f"u{i:03d}" for i in range(n)], "text": [f"technical prompt {i}" for i in range(n)],
                         "split": "calibration", "source": "test", "group_id": [f"g{i}" for i in range(n)],
                         "n_clip_tokens": 5})


def test_engine_schedule_no_extra_forward_and_rows(fake):
    cfg, bundle, scorer, scoring, eph, _ = fake
    eng = DenoiseEngine(bundle, scoring, cfg)
    eng.prepare_request("a technical prompt", "uid-x")
    mc = synthetic_costs()
    res = run_schedule(fixed_pair_spec(4, 8, bank_rows(0)), eng, mc=mc, cap_us=10**9)
    assert eng.n_unet_calls == 64 and bundle.unet_calls == 64
    assert len(eng.rows) == 2 and eng.x.shape[0] == 2 and eng.emb.shape[0] == 4
    assert res.delivery.rm_used and not res.delivery.failed
    assert res.prune_js == (4, 8)


def test_f1_does_not_call_reward_model(fake):
    cfg, bundle, scorer, scoring, eph, _ = fake
    eng = DenoiseEngine(bundle, scoring, cfg)
    eng.prepare_request("a technical prompt", "uid-y")
    before = scorer.n_calls
    res = run_schedule(bon_spec(1, bank_rows(0)), eng, mc=synthetic_costs(), cap_us=10**9)
    assert scorer.n_calls == before
    assert not res.delivery.rm_used and res.delivery.winner == (0, 0) and res.delivery.inference_score is None
    assert eng.last_images[(0, 0)] is not None


def test_validity_before_pil(fake):
    cfg, bundle, scorer, scoring, eph, _ = fake
    lat = torch.randn(3, 4, 8, 8)
    lat[1, 0, 0, 0] = float("nan")
    sb = scoring.score_preview("p", lat)
    assert [r.latent_ok for r in sb.records] == [True, False, True]
    assert sb.records[1].score is None and sb.images[1] is None
    orig = bundle.decode

    def bad_decode(z, preview):
        img = orig(z, preview)
        img[0, 0, 0, 0] = float("inf")       # image invalid although latent finite
        return img
    bundle.decode = bad_decode
    sb = scoring.score_delivered("p", torch.randn(2, 4, 8, 8))
    assert sb.records[0].latent_ok and not sb.records[0].img_ok and not sb.records[0].score_ok
    assert sb.records[1].score_ok


def test_hook_modes(fake):
    cfg, bundle, scorer, scoring, eph, _ = fake
    for mode, expect_calls in (("none", 0), ("traj", 16)):
        eng = DenoiseEngine(bundle, scoring, cfg, hook_mode=mode)
        eng.prepare_request("p", "u")
        eng.start(bank_rows(0))
        for _ in range(16):
            eng.advance(None, 4, False)
        assert eng.hook.feature_kernel_calls == expect_calls
    assert np.isfinite(eng.hook.last["drift"].numpy()).all()


@pytest.fixture()
def fake_bank(fake, tmp_path):
    cfg, bundle, scorer, scoring, eph, body = fake
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    bank_dir = tmp_path / "FAKEBANK"
    summ = run_collection(eng, bundle, _prompts(4), bank_dir, eph=eph, eph_body=body, cfg=cfg, banks_per_prompt=2,
                          resume=False, machine_id="test", commit="test")
    return bank_dir, summ


def test_collector_writes_valid_bank_and_resumes(fake, fake_bank):
    cfg, bundle, scorer, scoring, eph, body = fake
    bank_dir, summ = fake_bank
    assert summ["bank"]["valid"], summ["bank"]["errors"]
    assert len(summ["collected"]) == 8
    rep = validate_bank(bank_dir)
    assert rep.ok and rep.stats["instances"] == 8
    # partial shard from an interrupted run is quarantined, valid shards are skipped
    (bank_dir / "shards" / ".tmp_u000__b0_999").mkdir()
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    s2 = run_collection(eng, bundle, _prompts(5), bank_dir, eph=eph, eph_body=body, cfg=cfg, banks_per_prompt=2,
                        resume=True, machine_id="test", commit="test")
    assert len(s2["skipped_valid"]) == 8 and len(s2["collected"]) == 2 and s2["quarantined"]
    assert (bank_dir / "failed").exists()
    # resumed bank identity: shards collected earlier are byte-identical to a fresh collection of the same keys
    assert s2["bank"]["valid"]


def test_collector_rejects_mismatch(fake, fake_bank):
    cfg, bundle, scorer, scoring, eph, body = fake
    bank_dir, _ = fake_bank
    eng = DenoiseEngine(bundle, scoring, cfg, hook_mode="collect")
    with pytest.raises(CompatibilityError):
        run_collection(eng, bundle, _prompts(4), bank_dir, eph="0" * 64, eph_body=body, cfg=cfg, banks_per_prompt=2,
                       resume=True, machine_id="t", commit="t")
    with pytest.raises(CompatibilityError):
        run_collection(eng, bundle, _prompts(4), bank_dir, eph=eph, eph_body=body, cfg=cfg, banks_per_prompt=2,
                       resume=False, machine_id="t", commit="t")
    shard = bank_dir / "shards" / "u001__b1"
    with open(shard / "previews.parquet", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(CompatibilityError, match="checksum"):
        shard_ok(shard, eph, json.loads((bank_dir / "collector.json").read_text())["config_hash"])


def test_online_equals_replay_on_collected_bank(fake, fake_bank):
    """Replay of the collected bank and online execution through the same runners agree exactly on the
    row-independent fake backend (Mệnh đề 2 code-path check; real CUDA parity is measured by M6/G1)."""
    cfg, bundle, scorer, scoring, eph, _ = fake
    bank_dir, _ = fake_bank
    bank = load_bank(bank_dir)
    base = synthetic_costs()
    ctrl = base.with_overrides(disp=120)
    ctx = ReplayContext.build(bank, base, fit_split="calibration", guard=False)
    eng = DenoiseEngine(bundle, scoring, cfg)
    n_cmp = 0
    for alpha in (0.5, 1.0):
        cap = ctx.caps.cap(alpha)
        for key in bank.instances:
            uid, b = key
            for m in ("b2", "random", "fixed:2,9"):
                on, _ = online_episode(eng, bank.text_of(uid), uid, b, make_method(m, 7), base=base, ctrl=ctrl,
                                       cap_us=cap, C_ref=ctx.caps.C_ref, norm=ctx.norm, prompt_feat=ctx.prompt_feat(uid))
                mr = make_method(m, 7)
                rp = (ctx.run_spec(fixed_pair_spec(*mr[1], bank_rows(b)), key, cap) if mr[0] == "schedule"
                      else ctx.run_policy(mr[1], key, cap, ctrl))
                assert on.feasible == rp.feasible
                if not on.feasible:
                    continue
                assert [e.action for e in on.trace] == [e.action for e in rp.trace]
                assert on.spend_us == rp.spend_us and on.delivery.winner == rp.delivery.winner
                assert on.delivery.inference_score == pytest.approx(rp.Y, abs=1e-6)
                n_cmp += 1
    assert n_cmp > 20


def test_gpu_cli_refuses_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("host has CUDA")
    from apsp.gpu import collect_bank, online, profile
    from apsp.gpu.runtime import CudaUnavailableError
    for mod, argv in ((collect_bank, ["--smoke-prompt", "x"]), (profile, ["--stage", "model"]),
                      (online, ["--task", "smoke", "--prompt", "x", "--synthetic-costs"])):
        with pytest.raises(CudaUnavailableError, match="CUDA is not available"):
            mod.main(argv)


def test_online_cli_fake_smoke(tmp_path, capsys):
    from apsp.gpu import online
    rc = online.main(["--fake-backend", "--task", "smoke", "--prompt", "a technical prompt", "--synthetic-costs",
                      "--methods", "b2", "random", "--out", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert all(r["feasible"] and r["ledger_spend_us"] <= r["cap_us"] for r in out["records"])
