import hashlib
import json
from collections import Counter

import pytest

from apsp.lite.contracts import checkpoints, noise_seed, uid, load_registry, canonical_hash
from apsp.lite.grammar import Grammar, State, fixed_schedules
from apsp.lite.accessor import PaidAccessor
from apsp.lite.tasks import make_tasks
from apsp.lite.prompts import group_rows, build
from apsp.lite.contracts import QUOTAS
from apsp.lite.storage import TaskBusy, atomic_json, backup_task, claim, commit_task, verify_complete


def test_checkpoint_grid_and_offline_identity():
    assert checkpoints(8, 8) == list(range(1, 9))
    assert checkpoints(50, 16) == [4, 7, 10, 13, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 47, 50]
    assert uid(" e\u0301  cat\n") == uid("é cat")
    seed = int.from_bytes(hashlib.sha256(f"apsp-v2|offline|abc|1|7".encode()).digest()[:8], "little") & ((1 << 63) - 1)
    assert noise_seed("abc", 1, 7) == seed
    assert len({noise_seed("abc", b, s) for b in (0, 1) for s in range(8)}) == 16


def test_task_counts_and_test_lock():
    prompts = [{"prompt_uid": str(i), "split": "train" if i < 2000 else "dev", "train_rank": i + 1 if i < 2000 else None} for i in range(2300)]
    tasks = make_tasks(prompts)
    assert len(tasks) == 13800
    assert len({t["task_id"] for t in tasks}) == 13800
    assert Counter(t["profile"] for t in tasks) == {p: 4600 for p in load_registry()["profiles"]}
    train = [t for t in tasks if t["prompt"]["split"] == "train"]
    assert sum(t["wave"] == 250 for t in train) * 8 == 12000
    registry = load_registry()
    assert sum(registry["profiles"][t["profile"]]["checkpoints"] * 8 for t in tasks) == 1177600
    with pytest.raises(ValueError, match="test sampling"):
        make_tasks([{"prompt_uid": "test", "split": "test_id"}])


def grammar():
    return Grammar(checkpoints(8, 8), {2: 2, 4: 4, 8: 8}, {2: 20, 4: 40, 8: 80},
                   {2: 10, 4: 20, 8: 40}, {(8, 4): 1, (4, 2): 1}, setup=3, dispatch=1)


def test_optional_no_prune_and_final_all_k():
    g = grammar()
    assert len(fixed_schedules(8)) == 29
    assert len(fixed_schedules(16)) == 121
    assert g.lower_bound(State(8, 8)) == 40
    assert g.lower_bound(State(8, 4)) == 20
    assert g.lower_bound(State(8, 2)) == 10
    q, spent = State(), g.setup
    for _ in range(8):
        q, fee = g.transitions(q)["CONTINUE"]
        spent += fee
    assert q.k == 8
    assert spent + g.fin[8] == 115
    assert not g.feasible(g.setup + g.lower_bound() - 1)


def test_budget_mask_reserves_terminal_cost_and_no_same_checkpoint_double_prune():
    g = grammar()
    q = State(2, 8, True)
    nxt, fee = g.transitions(q)["PRUNE"]
    assert not g.mask(q, fee + g.lower_bound(nxt) - 1)["PRUNE"]
    nxt, _ = g.transitions(nxt)["VERIFY"]
    assert "PRUNE" not in g.transitions(nxt)
    nxt, _ = g.transitions(nxt)["CONTINUE"]
    nxt, _ = g.transitions(nxt)["VERIFY"]
    assert "PRUNE" in g.transitions(nxt)


def test_actor_cannot_read_future_inactive_or_terminal_scores():
    rows = [{"j": j, "bank_id": 0, "seed_idx": s, "ir": 10 * j + s, "pickscore_raw": 9000 + s} for j in (1, 2) for s in range(8)]
    final = [{"bank_id": 0, "seed_idx": s, "ir": 100 + s} for s in range(8)]
    a = PaidAccessor(rows, final, 3)
    assert not a.view().bought_scores
    with pytest.raises(ValueError): a.buy_final()
    a.continue_to_next_checkpoint()
    a.buy_verify()
    assert max(a.view().bought_scores.values()) == 17
    with pytest.raises(TypeError): a.view().bought_scores[(0, 0)] = 999
    a.prune([(0, s) for s in range(4)])
    with pytest.raises(ValueError): a.prune([(0, 0), (0, 1)])
    assert all(cid[1] < 4 for _, history in a.view().history for cid, score in history)
    a.continue_to_next_checkpoint()
    assert not a.view().bought_scores
    a.continue_to_next_checkpoint()
    assert len(a.buy_final()) == 4


def test_durable_resume_rejects_corruption_and_conflicting_contract(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    stage.mkdir()
    atomic_json(stage / "record.json", {"score": -999.0})
    receipt = {"contract_hash": "abc", "status": "complete"}
    commit_task(stage, dest, receipt)
    assert verify_complete(dest, "abc")
    backup_task(dest, tmp_path / "backup")
    assert verify_complete(tmp_path / "backup")
    with pytest.raises(ValueError, match="different contract"): verify_complete(dest, "other")
    (dest / "record.json").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"): verify_complete(dest, "abc")


def test_duplicate_alias_preserves_all_parent_groups():
    def r(text, row_id, parent):
        return {"text_raw": text, "source": "diffusiondb", "source_row_id": row_id, "source_revision": "abc", "parent_group": parent}
    rows = [r("A red bird", "1", "one"), r("A red bird", "2", "two"), r("Unrelated landscape", "3", "two")]
    groups, report = group_rows(rows)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2
    assert report["unique_rows"] == 2


def test_manifest_exact_quotas_groups_and_nested_sizes():
    # Artificial text-only fixtures validate splitting; these never enter a release.
    rows = []
    for source in ("diffusiondb", "coco_captions", "structured", "parti"):
        count = sum(q.get(source, 0) for q in QUOTAS.values()) + 20
        for i in range(count):
            row = {"source": source, "text_raw": hashlib.sha256(f"fixture-{source}-{i}".encode()).hexdigest(),
                   "parent_group": f"{source}-{i}", "source_row_id": str(i), "source_revision": "fixture"}
            if source == "coco_captions": row["image_id"] = i
            if source == "structured": row.update(template_family_id=f"fixture-{i}", scene_spec={"fixture": i})
            rows.append(row)
    assigned, _ = build(rows)
    assert Counter((r["split"], r["source"]) for r in assigned) == {(sp, src): n for sp, quota in QUOTAS.items() for src, n in quota.items()}
    groups = {}
    for row in assigned:
        assert groups.setdefault(row["group_id"], row["split"]) == row["split"]
    for size in (250, 1000, 2000):
        subset = [r for r in assigned if r["split"] == "train" and r["train_rank"] <= size]
        assert len(subset) == size
        assert Counter(r["source"] for r in subset) == {"diffusiondb": size // 2, "coco_captions": size // 5, "structured": size * 3 // 10}


def test_exclusive_claims_are_never_stolen(tmp_path):
    with claim(tmp_path, "task"):
        with pytest.raises(TaskBusy):
            with claim(tmp_path, "task"): pass
    with claim(tmp_path, "task"): pass


def test_task_cli_writes_freeze_hash(tmp_path):
    from apsp.lite.tasks import main
    prompt_file = tmp_path / "train.jsonl"
    prompt_file.write_text(json.dumps({"prompt_uid": "abc", "split": "train", "train_rank": 1}) + "\n")
    output = tmp_path / "tasks.jsonl"
    main(["--prompts", str(prompt_file), "--out", str(output)])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["n_tasks"] == 6
    assert manifest["n_candidates"] == 48


def test_portable_archive_paths_and_symlinks():
    import importlib.util
    import tarfile
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "env/lite/ensure_runtime.py"
    spec = importlib.util.spec_from_file_location("ensure_apsp_lite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    link = tarfile.TarInfo("opt/apsp-runtime/bin/python")
    link.type = tarfile.SYMTYPE
    link.linkname = "/opt/apsp-python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
    module.validate_members([link])
    with pytest.raises(ValueError): module.validate_members([tarfile.TarInfo("../../etc/passwd")])
    with pytest.raises(ValueError): module.validate_members([tarfile.TarInfo("opt/other/file")])
    link.linkname = "/etc/passwd"
    with pytest.raises(ValueError): module.validate_members([link])
    link.name = "opt/apsp-runtime/lib"
    link.linkname = "/opt/apsp-python/lib"
    with pytest.raises(ValueError): module.validate_members([link, tarfile.TarInfo("opt/apsp-runtime/lib/evil")])
