"""Core-only validation never opens latent or PNG; optional native/round-trip GPU checks."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors import safe_open

from apsp.lite.storage import verify_complete
from .contracts import ENCODER, FEATURES, SCHEMA, canonical_hash, check_task, registry

KEY = ['profile_id', 'prompt_uid', 'bank_id', 'seed_idx']


def validate_task(path, *, sealed=True, check_latents=False, runtime=None):
    path = Path(path)
    if sealed and not verify_complete(path):
        raise ValueError('Task is not sealed')
    t = json.loads((path / 'task.json').read_text())
    check_task(t)
    prov = json.loads((path / 'provenance.json').read_text())
    if (prov['schema'] != SCHEMA or prov['registry_hash'] != canonical_hash(registry()) or
            prov['profile'] != registry()['profiles'][t['profile']] or prov['storage_cast'] != 'none' or
            prov['feature_contract'] != FEATURES or any(prov['encoder'][k] != v for k, v in ENCODER.items())):
        raise ValueError('Provenance/encoder/native contract differs')
    if not prov['scorers'].get('imagereward') or not prov['scorers'].get('pickscore'):
        raise ValueError('Missing scorer provenance')
    tables = {n: pd.read_parquet(path / (n + '.parquet')) for n in
              ('prompts', 'candidates', 'previews', 'delivered', 'tasks')}
    if [len(tables[n]) for n in tables] != [1, 8, 64, 8, 1]:
        raise ValueError('Core row cardinality differs')
    prompt = tables['prompts'].iloc[0]
    if prompt.prompt_uid != t['prompt']['prompt_uid'] or prompt['split'] != t['prompt']['split']:
        raise ValueError('Prompt foreign key/split differs')
    for name in ('candidates', 'previews', 'delivered'):
        df = tables[name]
        keys = KEY + (['j'] if name == 'previews' else [])
        if df.duplicated(keys).any() or set(df.seed_idx) != set(range(8)):
            raise ValueError('Duplicate/missing candidate key')
        if not (df.profile_id.eq(t['profile']).all() and df.prompt_uid.eq(prompt.prompt_uid).all() and
                df.bank_id.eq(t['bank_id']).all()):
            raise ValueError('Candidate foreign key differs')
    if list(tables['candidates'].sort_values('seed_idx').seed_global) != t['seeds']:
        raise ValueError('Seed identity differs')
    for s in range(8):
        frame = tables['previews'][tables['previews'].seed_idx == s].sort_values('j')
        if list(frame.j) != list(range(1, 9)) or list(frame.forward_idx) != list(range(1, 9)):
            raise ValueError('Checkpoint order differs')
    for name in ('previews', 'delivered'):
        for r in tables[name].itertuples():
            if r.valid and (not r.img_ok or not r.latent_ok or pd.isna(r.ir) or pd.isna(r.pickscore_raw) or not np.isfinite(r.ir) or not np.isfinite(r.pickscore_raw) or not r.pixel_sha256):
                raise ValueError('Invalid/nonfinite score called valid')
            if not r.valid and not r.failure_code:
                raise ValueError('Failure lacks reason')
    for r in tables['previews'].itertuples():
        if r.progress != r.j / 8 or (r.j == 1 and r.drift_available):
            raise ValueError('Progress/first drift contract differs')
        for name in ('drift', 'cfg_gap'):
            value, available = getattr(r, name), getattr(r, name + '_available')
            if available and (pd.isna(value) or not np.isfinite(value)):
                raise ValueError('Nonfinite feature marked available')
            if not available and not pd.isna(value):
                raise ValueError('Unavailable feature must be null')
        if not prov['profile']['cfg_branches_available'] and r.cfg_gap_available:
            raise ValueError('Fake CFG feature')
    with safe_open(str(path / 'prompt_features.safetensors'), framework='np') as f:
        emb = f.get_tensor('embedding')
    if emb.shape != (1, 384) or emb.dtype != np.float32 or not np.isfinite(emb).all() or not np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5):
        raise ValueError('Core384 embedding shape/dtype/norm')
    native_checked = False
    if check_latents:
        import torch
        from .collector import tensor_hash
        from safetensors.torch import load_file
        z = load_file(str(path / 'latents.safetensors'))
        if set(z) != set(prov['native_tensors']):
            raise ValueError('Native tensor keys differ')
        for name, meta in prov['native_tensors'].items():
            value = z[name]
            if str(value.dtype).removeprefix('torch.') != meta['dtype'] or list(value.shape) != meta['shape'] or tensor_hash(value) != meta['sha256']:
                raise ValueError('Native tensor changed at serialization: ' + name)
            if value.shape != (8, 4, prov['profile']['height'] // 8, prov['profile']['width'] // 8):
                raise ValueError('Native tensor shape differs')
        native_checked = True
    redecode = None
    if runtime is not None:
        from .collector import frames_at, pixel_hash
        results = runtime.decode_score(t['prompt']['text'], frames_at(path)[-1])
        expected = tables['delivered'].sort_values('seed_idx')
        for i, row in enumerate(expected.itertuples()):
            actual = None if results[0][i] is None else pixel_hash(results[0][i])
            if actual != row.pixel_sha256:
                raise ValueError('Delivered pixels changed at re-decode')
        redecode = True
    return {'pass': True, 'schema': SCHEMA, 'n_candidates': 8, 'n_previews': 64,
            'all_valid': bool(tables['delivered'].valid.all() and tables['previews'].valid.all()),
            'native_checked': native_checked, 'redecode_pixels_identical': redecode}
