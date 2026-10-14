import math

import numpy as np
import pytest

from apsp.contracts.hashing import (HashBodyError, canonical_bytes, code_sha, provenance_hash, sha256_hex)
from apsp.contracts.identity import (RowMap, cfg_row_indices, derive_seed, pool_rows, rank_candidates, select_top)


def test_type_separation_and_determinism():
    hs = {sha256_hex(v) for v in (1, 1.0, "1", True, [1], {"a": 1})}
    assert len(hs) == 6
    assert sha256_hex({"b": 1, "a": [1, 2]}) == sha256_hex({"a": [1, 2], "b": 1})
    assert sha256_hex([1, 2]) != sha256_hex([2, 1])


def test_golden_bytes_pinned():
    body = {"k": [1, 1.5, "x", None, True], "n": {"z": 0}}
    assert canonical_bytes(body) == (
        b'["d",[["k",["l",[["i","1"],["f","1.5"],["s","x"],["n"],["b",true]]]],["n",["d",[["z",["i","0"]]]]]]]')
    assert sha256_hex(body) == sha256_hex({"n": {"z": 0}, "k": [1, 1.5, "x", None, True]})


@pytest.mark.parametrize("bad", [math.nan, math.inf, {1, 2}, {1: "x"}, np.zeros(2), object()])
def test_rejects_unhashable(bad):
    with pytest.raises(HashBodyError):
        sha256_hex({"v": bad})


def test_nfc_key_collision_rejected():
    with pytest.raises(HashBodyError):
        sha256_hex({"é": 1, "é": 2})
    # values are never normalised: distinct strings hash differently
    assert sha256_hex("é") != sha256_hex("é")


def test_provenance_body_rules():
    with pytest.raises(HashBodyError):
        provenance_hash("EPH", {"eph": "x"})              # self reference
    with pytest.raises(HashBodyError):
        provenance_hash("BMH", {"created_at": "2026"})    # timestamp
    with pytest.raises(HashBodyError):
        provenance_hash("GCH", {"path": "/abs/file"})     # absolute path
    assert provenance_hash("EPH", {"a": 1}) != provenance_hash("BMH", {"a": 1})


def test_code_sha_source_change_and_crlf(tmp_path):
    f = tmp_path / "m.py"
    f.write_bytes(b"x = 1\n")
    h1 = code_sha([f], tmp_path)
    f.write_bytes(b"x = 1\r\n")
    assert code_sha([f], tmp_path) == h1
    f.write_bytes(b"x = 2\n")
    assert code_sha([f], tmp_path) != h1


def test_ranking_ties_and_invalid_last():
    ids = [(0, i) for i in range(6)]
    s = {(0, 0): 1.0, (0, 1): 2.0, (0, 2): 2.0, (0, 3): None, (0, 4): math.nan, (0, 5): -1.0}
    assert rank_candidates(ids, s) == [(0, 1), (0, 2), (0, 0), (0, 5), (0, 3), (0, 4)]
    assert select_top(ids, s, 3) == {(0, 1), (0, 2), (0, 0)}


def test_rowmap_keeps_old_order_and_pool16():
    rows = pool_rows(2)
    assert rows[:2] == ((0, 0), (0, 1)) and rows[8] == (1, 0) and len(rows) == 16
    rm = RowMap(rows)
    idx = rm.retained_indices({(1, 3), (0, 7), (0, 2)})
    assert idx == (2, 7, 11)
    rm.apply_keep(idx)
    assert rm.rows == ((0, 2), (0, 7), (1, 3))
    with pytest.raises(ValueError):
        RowMap(rows).apply_keep((3, 1))
    assert cfg_row_indices([0, 2], 4) == [0, 2, 4, 6]


def test_seed_derivation_golden():
    import hashlib
    d = hashlib.sha256(b"apsp-v1|abc|1|7").digest()
    assert derive_seed("abc", 1, 7) == int.from_bytes(d[:4], "little") & 0x7FFFFFFF
    assert 0 <= derive_seed("abc", 0, 0) < 2**31
    assert derive_seed("abc", 0, 1) != derive_seed("abc", 1, 0)
