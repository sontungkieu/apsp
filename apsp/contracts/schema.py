"""Bank schema ``apsp-bank-v1.1.1`` (Guide §9): writer, loader, in-memory arrays and ``validate_bank``.

CLI: ``python -m apsp.contracts.schema --bank DIR --validate``
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .hashing import file_sha256, provenance_hash
from .identity import derive_seed

SCHEMA_VERSION = "apsp-bank-v1.1.1"
J = 16
DELTA = 4
K0 = 8
N_BANKS = 2
SPLITS = ("calibration", "train_fit", "train_dev", "val", "val_ext", "test", "smoke",
          "syn_train_fit", "syn_train_dev", "syn_eval")  # syn_*: synthetic fixtures only

PROMPT_COLS = {"prompt_uid": "string", "text": "string", "split": "string", "source": "string",
               "group_id": "string", "n_clip_tokens": "int16"}
SEED_COLS = {"prompt_uid": "string", "bank_id": "int8", "seed_idx": "int8", "seed_global": "int64",
             "init_latent_sha": "string"}
PREVIEW_COLS = {"prompt_uid": "string", "bank_id": "int8", "seed_idx": "int8", "j": "int8", "forward_idx": "int16",
                "t_in": "int16", "ir": "float32", "latent_ok": "bool", "img_ok": "bool", "score_ok": "bool",
                "drift": "float32", "cfg_gap": "float32",
                **{f"mean_c{c}": "float32" for c in range(4)}, **{f"std_c{c}": "float32" for c in range(4)}}
DELIVERED_COLS = {"prompt_uid": "string", "bank_id": "int8", "seed_idx": "int8", "ir_deliv": "float32",
                  "latent_ok": "bool", "img_ok": "bool", "score_ok": "bool", "image_path": "string",
                  "image_sha": "string"}
GENERATION_FILES = ("prompts.parquet", "prompt_emb.npy", "prompt_emb_index.parquet", "seeds.parquet",
                    "previews.parquet", "delivered.parquet")


def _coerce(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    out = df[list(cols)].copy()
    for c, t in cols.items():
        out[c] = out[c].astype(t)
    return out


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".tmp_" + path.name)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def bmh_body(bank_dir: Path, eph: str, collection: dict) -> dict:
    """BMH body: EPH, collection settings and checksums of generation files (eval/ and temp files excluded)."""
    checksums = {}
    for name in GENERATION_FILES:
        p = bank_dir / name
        if p.exists():
            checksums[name] = file_sha256(p)
    return {"eph": eph, "schema_version": SCHEMA_VERSION, "collection": collection, "files": checksums}


def write_bank(bank_dir: str | Path, *, prompts: pd.DataFrame, prompt_emb: np.ndarray, seeds: pd.DataFrame,
               previews: pd.DataFrame, delivered: pd.DataFrame, eph: str, eph_body: dict, collection: dict,
               machine_id: str, commit: str, extra_manifest: Optional[dict] = None) -> dict:
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    prompts = _coerce(prompts, PROMPT_COLS)
    _write_parquet(bank_dir / "prompts.parquet", prompts)
    emb = np.asarray(prompt_emb, dtype=np.float16)
    if emb.shape != (len(prompts), 768):
        raise ValueError(f"prompt_emb must be float16[N,768], got {emb.shape}")
    tmp = bank_dir / ".tmp_prompt_emb.npy"
    np.save(tmp, emb)
    os.replace(tmp, bank_dir / "prompt_emb.npy")
    _write_parquet(bank_dir / "prompt_emb_index.parquet",
                   pd.DataFrame({"prompt_uid": prompts["prompt_uid"].astype("string"), "row": np.arange(len(prompts), dtype=np.int32)}))
    _write_parquet(bank_dir / "seeds.parquet", _coerce(seeds, SEED_COLS))
    _write_parquet(bank_dir / "previews.parquet", _coerce(previews, PREVIEW_COLS))
    _write_parquet(bank_dir / "delivered.parquet", _coerce(delivered, DELIVERED_COLS))
    body = bmh_body(bank_dir, eph, collection)
    manifest = {
        "schema_version": SCHEMA_VERSION, "bank_name": bank_dir.name, "eph": eph, "eph_body": eph_body,
        "bmh": provenance_hash("BMH", body), "bmh_body": body, "machine_id": machine_id, "commit": commit,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    atomic_write_json(bank_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------------------
@dataclass
class BankArrays:
    """Privileged in-memory view of a bank. Only env/oracle/training-label/reward code may hold it."""

    manifest: dict
    prompts: pd.DataFrame
    prompt_emb: np.ndarray
    emb_row: dict
    instances: list                    # list of (uid, bank_id)
    preview_ir: dict                   # key -> float32[8,16] (NaN where invalid)
    preview_flags: dict                # key -> bool[8,16,3]
    drift: dict
    cfg_gap: dict
    chan: dict                         # key -> float32[8,16,8] (mean_c0..3, std_c0..3)
    delivered_ir: dict                 # key -> float32[8]
    delivered_flags: dict              # key -> bool[8,3]
    t_in: dict = field(default_factory=dict)  # j -> timestep

    def split_of(self, uid: str) -> str:
        return self._split[uid]

    def __post_init__(self):
        self._split = dict(zip(self.prompts["prompt_uid"].astype(str), self.prompts["split"].astype(str)))
        self._text = dict(zip(self.prompts["prompt_uid"].astype(str), self.prompts["text"].astype(str)))

    def text_of(self, uid: str) -> str:
        return self._text[uid]

    def uids(self, split: Optional[str] = None) -> list[str]:
        u = self.prompts["prompt_uid"].astype(str)
        if split is not None:
            u = u[self.prompts["split"].astype(str) == split]
        return list(u)

    def instances_of(self, splits: Iterable[str]) -> list:
        s = set(splits)
        return [key for key in self.instances if self._split[key[0]] in s]


def load_bank(bank_dir: str | Path) -> BankArrays:
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    prompts = pd.read_parquet(bank_dir / "prompts.parquet")
    emb = np.load(bank_dir / "prompt_emb.npy")
    idx = pd.read_parquet(bank_dir / "prompt_emb_index.parquet")
    emb_row = dict(zip(idx["prompt_uid"].astype(str), idx["row"].astype(int)))
    pv = pd.read_parquet(bank_dir / "previews.parquet")
    dl = pd.read_parquet(bank_dir / "delivered.parquet")
    pv = pv.sort_values(["prompt_uid", "bank_id", "seed_idx", "j"]).reset_index(drop=True)
    dl = dl.sort_values(["prompt_uid", "bank_id", "seed_idx"]).reset_index(drop=True)
    n_inst = len(pv) // (K0 * J)
    if n_inst * K0 * J != len(pv) or len(dl) != n_inst * K0:
        raise ValueError("bank tables do not have 8x16 previews / 8 delivered rows per instance")

    def blk(col, dtype):
        return pv[col].to_numpy(dtype=dtype).reshape(n_inst, K0, J)

    keys_arr = pv[["prompt_uid", "bank_id"]].iloc[:: K0 * J]
    instances = [(str(u), int(b)) for u, b in zip(keys_arr["prompt_uid"], keys_arr["bank_id"])]
    ir = blk("ir", np.float32)
    fl = np.stack([blk("latent_ok", bool), blk("img_ok", bool), blk("score_ok", bool)], axis=-1)
    ir = np.where(fl[..., 2], ir, np.nan).astype(np.float32)
    drift, gap = blk("drift", np.float32), blk("cfg_gap", np.float32)
    chan = np.stack([blk(f"mean_c{c}", np.float32) for c in range(4)] + [blk(f"std_c{c}", np.float32) for c in range(4)], -1)
    dir_ = dl["ir_deliv"].to_numpy(np.float32).reshape(n_inst, K0)
    dfl = np.stack([dl[c].to_numpy(bool).reshape(n_inst, K0) for c in ("latent_ok", "img_ok", "score_ok")], -1)
    dir_ = np.where(dfl[..., 2], dir_, np.nan).astype(np.float32)
    t_in = {int(j): int(t) for j, t in pv.drop_duplicates("j")[["j", "t_in"]].itertuples(index=False)}
    return BankArrays(
        manifest=manifest, prompts=prompts, prompt_emb=emb, emb_row=emb_row, instances=instances,
        preview_ir={k: ir[i] for i, k in enumerate(instances)}, preview_flags={k: fl[i] for i, k in enumerate(instances)},
        drift={k: drift[i] for i, k in enumerate(instances)}, cfg_gap={k: gap[i] for i, k in enumerate(instances)},
        chan={k: chan[i] for i, k in enumerate(instances)},
        delivered_ir={k: dir_[i] for i, k in enumerate(instances)}, delivered_flags={k: dfl[i] for i, k in enumerate(instances)},
        t_in=t_in,
    )


# --------------------------------------------------------------------------------------
@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_bank(bank_dir: str | Path, check_files: bool = True) -> ValidationReport:
    bank_dir = Path(bank_dir)
    rep = ValidationReport()
    E = rep.errors.append
    try:
        manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        E(f"manifest unreadable: {e}")
        return rep
    if manifest.get("schema_version") != SCHEMA_VERSION:
        E(f"schema_version {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    for name in GENERATION_FILES:
        if not (bank_dir / name).exists():
            E(f"missing {name}")
    if rep.errors:
        return rep
    prompts = pd.read_parquet(bank_dir / "prompts.parquet")
    seeds = pd.read_parquet(bank_dir / "seeds.parquet")
    pv = pd.read_parquet(bank_dir / "previews.parquet")
    dl = pd.read_parquet(bank_dir / "delivered.parquet")
    idx = pd.read_parquet(bank_dir / "prompt_emb_index.parquet")
    emb = np.load(bank_dir / "prompt_emb.npy")
    for name, df, cols in (("prompts", prompts, PROMPT_COLS), ("seeds", seeds, SEED_COLS),
                           ("previews", pv, PREVIEW_COLS), ("delivered", dl, DELIVERED_COLS)):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            E(f"{name} missing columns {miss}")
    if rep.errors:
        return rep
    # uniqueness / splits
    if prompts["prompt_uid"].duplicated().any():
        E("duplicate prompt_uid in prompts")
    bad_split = set(prompts["split"].astype(str)) - set(SPLITS)
    if bad_split:
        E(f"unknown splits {sorted(bad_split)}")
    # embeddings
    if emb.dtype != np.float16 or emb.ndim != 2 or emb.shape[1] != 768:
        E(f"prompt_emb must be float16[N,768], got {emb.dtype}{emb.shape}")
    if idx["prompt_uid"].duplicated().any() or idx["row"].duplicated().any():
        E("prompt_emb_index not unique")
    if set(idx["prompt_uid"].astype(str)) != set(prompts["prompt_uid"].astype(str)):
        E("prompt_emb_index does not map exactly the prompts")
    if len(idx) and (idx["row"].max() >= emb.shape[0] or idx["row"].min() < 0):
        E("prompt_emb_index row out of range")
    if not np.isfinite(emb.astype(np.float32)).all():
        E("non-finite prompt embedding")
    # seeds
    key3 = ["prompt_uid", "bank_id", "seed_idx"]
    if seeds.duplicated(key3).any():
        E("duplicate seed rows")
    exp_seed = [derive_seed(u, b, s) for u, b, s in seeds[key3].itertuples(index=False)]
    bad = int((np.asarray(exp_seed) != seeds["seed_global"].to_numpy()).sum())
    if bad:
        E(f"{bad} seed_global values do not follow apsp-v1 derivation")
    dup_int = seeds["seed_global"].duplicated().sum()
    if dup_int:
        rep.warnings.append(f"{int(dup_int)} equal integer seeds for distinct CandidateIds (logged, not an identity collision)")
    # previews: 8x16 per prompt-bank, contiguous j, forward_idx = 4j
    if pv.duplicated(key3 + ["j"]).any():
        E("duplicate preview rows")
    g = pv.groupby(["prompt_uid", "bank_id"])
    sizes = g.size()
    if (sizes != K0 * J).any():
        E(f"{int((sizes != K0 * J).sum())} prompt-banks without 8x16 preview rows")
    js = sorted(pv["j"].unique().tolist())
    if js != list(range(1, J + 1)):
        E(f"checkpoints {js} not contiguous 1..16")
    if (pv["forward_idx"].astype(int) != DELTA * pv["j"].astype(int)).any():
        E("forward_idx != 4j")
    t_map = pv.groupby("j")["t_in"].nunique()
    if (t_map > 1).any():
        E("t_in not a single timestep per checkpoint")
    if "timesteps_at_checkpoints" in manifest:
        tj = pv.drop_duplicates("j").set_index("j")["t_in"].astype(int).to_dict()
        if [tj.get(j) for j in range(1, J + 1)] != list(manifest["timesteps_at_checkpoints"]):
            E("t_in does not match manifest timesteps")
    seeds_keys = set(map(tuple, seeds[key3].astype(str).to_numpy().tolist()))
    pv_keys = set(map(tuple, pv[key3].drop_duplicates().astype(str).to_numpy().tolist()))
    if seeds_keys != pv_keys:
        E("seed rows and preview candidates differ")
    if set(pv["prompt_uid"].astype(str)) - set(prompts["prompt_uid"].astype(str)):
        E("previews reference unknown prompts")

    def chain(df, where):
        sok, iok, lok = df["score_ok"].to_numpy(bool), df["img_ok"].to_numpy(bool), df["latent_ok"].to_numpy(bool)
        if (sok & ~iok).any() or (iok & ~lok).any():
            E(f"{where}: validity chain violated (score_ok=>img_ok=>latent_ok)")
        s = df["ir" if "ir" in df else "ir_deliv"].to_numpy(np.float64)
        if (np.isfinite(s) != sok).any():
            E(f"{where}: score finite iff score_ok violated")

    chain(pv, "previews")
    chain(dl, "delivered")
    # latent NaN cannot recover along a trajectory
    lat = pv.sort_values(key3 + ["j"])
    lok = lat["latent_ok"].to_numpy(bool).reshape(-1, J) if len(lat) % J == 0 else None
    if lok is not None and (lok[:, 1:] & ~lok[:, :-1]).any():
        E("latent_ok recovers after a latent failure")
    # delivered
    if dl.duplicated(key3).any():
        E("duplicate delivered rows")
    if set(map(tuple, dl[key3].astype(str).to_numpy().tolist())) != seeds_keys:
        E("delivered rows do not cover the seed table")
    for row in dl.itertuples(index=False):
        has_img = row.image_path is not None and not (isinstance(row.image_path, float) and math.isnan(row.image_path)) and not pd.isna(row.image_path)
        if bool(row.img_ok) != has_img:
            E(f"delivered {row.prompt_uid}/{row.bank_id}/{row.seed_idx}: image_path presence != img_ok")
            continue
        if has_img and check_files and not (manifest.get("synthetic") and str(row.image_path).startswith("virtual/")):
            p = bank_dir / str(row.image_path)
            if not p.exists():
                E(f"missing image {row.image_path}")
            elif file_sha256(p) != row.image_sha:
                E(f"image checksum mismatch {row.image_path}")
    # BMH
    body = bmh_body(bank_dir, manifest.get("eph", ""), manifest.get("bmh_body", {}).get("collection", {}))
    if provenance_hash("BMH", body) != manifest.get("bmh"):
        E("BMH mismatch: generation files changed after manifest was written")
    rep.stats = {"prompts": int(len(prompts)), "instances": int(len(sizes)), "preview_rows": int(len(pv)),
                 "preview_invalid": int((~pv["score_ok"]).sum()), "delivered_invalid": int((~dl["score_ok"]).sum()),
                 "synthetic": bool(manifest.get("synthetic", False))}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate an apsp-bank-v1.1.1 directory")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument("--no-files", action="store_true", help="skip PNG checksum verification")
    a = ap.parse_args(argv)
    rep = validate_bank(a.bank, check_files=not a.no_files)
    print(json.dumps({"ok": rep.ok, "errors": rep.errors[:50], "warnings": rep.warnings, "stats": rep.stats}, indent=2))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
