"""Verified Core-first export with optional native latent and preselected QA extensions."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.numpy import save_file
from safetensors import safe_open

from apsp.lite.storage import atomic_json, files_manifest, fsync_dir, verify_complete
from .contracts import ENCODER, SCHEMA, canonical_hash, file_hash
from .validate import validate_task


def export_release(tasks, out, *, mode='core_latent_qa', max_bytes=80_000_000_000,
                   include_calibration=False, expected_tasks=None, planned_manifests=(), failures_roots=(), qualification_root=None):
    if mode not in ('core_latent_qa', 'core_qa') or not 0 < max_bytes <= 80_000_000_000:
        raise ValueError('Invalid mode or cap; do not increase the 80GB contract')
    tasks = list(tasks)
    out = Path(out)
    if any(p.name in ('release-v1', 'backup-v1') for p in (out, *out.parents)):
        raise ValueError('Never write exports into old v1 trees')
    if out.exists():
        raise FileExistsError('Export must be immutable')
    stage = out.with_name(out.name + '.staging-' + str(os.getpid()))
    stage.mkdir(parents=True, exist_ok=False)
    (stage / 'core').mkdir()
    tables = {n: [] for n in ('prompts', 'candidates', 'previews', 'delivered', 'tasks')}
    embeddings, prompt_rows, seen, retention = {}, {}, set(), []
    main_tasks = 0
    copied_bytes = 0
    completed = {}
    for task_dir in sorted(map(Path, tasks)):
        validate_task(task_dir)
        task = json.loads((task_dir / 'task.json').read_text())
        if task['prompt']['split'] == 'calibration' and not include_calibration:
            continue
        key = (task['profile'], task['prompt']['prompt_uid'], task['bank_id'])
        if key in seen:
            raise ValueError('Duplicate logical task across reuse/new roots')
        seen.add(key)
        task_id = task['task_id']
        completed[task_id] = task
        main_tasks += int(task['prompt']['split'] != 'calibration')
        provenance_dir = stage / 'provenance' / task_id
        provenance_dir.mkdir(parents=True)
        for name in ('provenance.json', 'task.json', 'timings.json', 'COMPLETE.json'):
            shutil.copyfile(task_dir / name, provenance_dir / ('source-complete.json' if name == 'COMPLETE.json' else name))
            copied_bytes += (task_dir / name).stat().st_size
        with safe_open(str(task_dir / 'prompt_features.safetensors'), framework='np') as f:
            emb = f.get_tensor('embedding')[0]
        prompt_uid = task['prompt']['prompt_uid']
        if prompt_uid in embeddings and not np.allclose(embeddings[prompt_uid], emb, atol=1e-6):
            raise ValueError('Common embedding inconsistent across profiles/banks')
        embeddings.setdefault(prompt_uid, emb)
        row = pd.read_parquet(task_dir / 'prompts.parquet').iloc[0].to_dict()
        if prompt_uid in prompt_rows and prompt_rows[prompt_uid] != row:
            # NaN fields are normalized below; compare semantic split/text separately.
            if any(prompt_rows[prompt_uid][k] != row[k] for k in ('text', 'split', 'group_id', 'source')):
                raise ValueError('Prompt metadata/split conflict')
        prompt_rows.setdefault(prompt_uid, row)
        latent_uri = None
        if mode == 'core_latent_qa':
            latent = task_dir / 'latents.safetensors'
            if not latent.is_file():
                raise ValueError('Full export requires native latent; use explicit core_qa fallback')
            validate_task(task_dir, check_latents=True)
            target = stage / 'latent' / task_id / 'latents.safetensors'
            target.parent.mkdir(parents=True)
            shutil.copyfile(latent, target)
            copied_bytes += latent.stat().st_size
            latent_uri = f'latent/{task_id}/latents.safetensors#final'
        for name in ('candidates', 'previews', 'delivered', 'tasks'):
            df = pd.read_parquet(task_dir / (name + '.parquet'))
            if 'image_uri' in df:
                uris = []
                for uri in df.image_uri:
                    if uri is None or pd.isna(uri):
                        uris.append(None)
                        continue
                    path = Path(uri)
                    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'qa':
                        raise ValueError('Unsafe QA reference')
                    target = stage / 'qa' / task_id / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(task_dir / path, target)
                    copied_bytes += (task_dir / path).stat().st_size
                    uris.append(f'qa/{task_id}/{path.name}')
                df['image_uri'] = uris
            if name == 'delivered':
                df['final_latent_uri'] = latent_uri
            if name == 'tasks':
                df['source_task_uri'] = str(task_dir.resolve())
                df['source_complete_sha256'] = file_hash(task_dir / 'COMPLETE.json')
            tables[name].append(df)
        retention.append({'task_id': task_id, 'source_uri': str(task_dir.resolve()),
                          'native_exported': latent_uri is not None,
                          'retention_expiry': None, 'durability': 'operator_must_confirm_storage_retention'})
        if copied_bytes > max_bytes:
            raise ValueError('Export cap exceeded; staging retained; choose core_qa explicitly')
    if not seen:
        raise ValueError('No completed tasks selected')
    pd.DataFrame([prompt_rows[k] for k in sorted(prompt_rows)]).to_parquet(stage / 'core/prompts.parquet', index=False)
    for name in ('candidates', 'previews', 'delivered', 'tasks'):
        pd.concat(tables[name], ignore_index=True).to_parquet(stage / 'core' / (name + '.parquet'), index=False)
    order = sorted(embeddings)
    save_file({'embedding': np.stack([embeddings[k] for k in order])}, str(stage / 'core/prompt_features.safetensors'))
    atomic_json(stage / 'core/prompt_features.index.json', {k: i for i, k in enumerate(order)})
    atomic_json(stage / 'retention.json', retention)
    planned = {}
    for filename in planned_manifests:
        from apsp.lite.prompts import read_jsonl
        for t in read_jsonl(filename):
            if t['task_id'] in planned and planned[t['task_id']] != t:
                raise ValueError('Conflicting planned task IDs')
            planned[t['task_id']] = t
    if planned and not set(completed) <= set(planned):
        raise ValueError('Completed task is outside the supplied release plan')
    planned = planned or completed
    failure_attempts = {}
    for root in failures_roots:
        for path in (Path(root) / 'failures').glob('*.json'):
            failure = json.loads(path.read_text())
            failure_attempts[failure['task_id']] = max(failure_attempts.get(failure['task_id'], 0), failure['attempt'])
    plan_rows = []
    roots_by_id = {json.loads((Path(path) / 'task.json').read_text())['task_id']: Path(path) for path in tasks
                   if (Path(path) / 'task.json').is_file()}
    for task_id, t in sorted(planned.items()):
        attempts = failure_attempts.get(task_id, 0)
        status = 'failed_infra' if attempts >= 3 else 'pending'
        durable = False
        if task_id in completed:
            path = roots_by_id[task_id]
            status = pd.read_parquet(path / 'tasks.parquet').iloc[0]['status']
            ack = path.parent.parent / 'acks' / (task_id + '.json')
            if ack.exists():
                data = json.loads(ack.read_text())
                durable = data['complete_sha256'] == file_hash(path / 'COMPLETE.json') and verify_complete(data['backup'])
            attempts += 1
        plan_rows.append({'task_id': task_id, 'profile_id': t['profile'], 'prompt_uid': t['prompt']['prompt_uid'],
                          'bank_id': t['bank_id'], 'split': t['prompt']['split'], 'status': status,
                          'attempts': attempts, 'durable_ack_verified': bool(durable)})
    pd.DataFrame(plan_rows).to_parquet(stage / 'core/task_plan.parquet', index=False)
    status_counts = pd.DataFrame(plan_rows).groupby(['profile_id','split','status'], dropna=False).size()
    status_report = [{'profile_id': k[0], 'split': k[1], 'status': k[2], 'tasks': int(v)} for k,v in status_counts.items()]
    atomic_json(stage / 'coverage.json', {'planned_tasks': len(planned), 'collected_tasks': len(completed),
                'status_counts': status_report, 'expected_main_tasks': expected_tasks,
                'durable_ack_tasks': sum(r['durable_ack_verified'] for r in plan_rows),
                'backup_note': 'ACK verifies a copy, not independent storage durability; operator must attest retention/failure domain.'})
    if qualification_root is not None:
        for profile in {t['profile'] for t in completed.values()}:
            path = Path(qualification_root) / (profile + '.json')
            target = stage / 'qualification' / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
    (stage / 'DATASET_CARD_LITE2.md').write_text(
        '# APSP Lite2 Core384\n\nCore: prompts, embedding index, candidates, all preview/final raw scores, masks, '
        'features and per-task provenance. task_plan.parquet maps all planned IDs to terminal/pending status; '
        'tasks.parquet describes collected tasks. Core replay does not open PNG/latent.\n\n'
        'Source text/model/scorer licenses and revisions are inherited from original provenance and pinned '
        'model cards; this release grants no additional upstream rights. No test generation or controller '
        'efficacy claim. Labels are finite/validity checks, not aesthetic QA. Latent extension availability '
        'and unconfirmed storage retention are explicit in RELEASE_MANIFEST.json and retention.json.\n', encoding='utf-8')
    (stage / 'COLLECTION_REPORT_LITE2.md').write_text(
        '# Collection report\n\n' + json.dumps({'main_tasks': main_tasks, 'calibration_tasks': len(seen)-main_tasks,
            'planned_status_counts': status_report, 'mode': mode}, indent=2) +
        '\n\nRaw timings/model/runtime/source/feature contracts are in provenance/. Qualification and '
        'K=2/4/8 cost samples are in qualification/ when supplied; missing cost samples are not zero. '
        'Partial coverage, model failure and infrastructure failure remain distinct.\n', encoding='utf-8')
    from .replay import baselines
    replay = baselines(stage / 'core')
    atomic_json(stage / 'replay-check.json', {k: v for k, v in replay.items() if k != 'results'})
    files = files_manifest(stage)
    payload_bytes = sum((stage / p).stat().st_size for p in files)
    report = {'schema': SCHEMA, 'mode': mode, 'tasks': len(seen), 'candidates': len(seen) * 8,
              'preview_rows': len(seen) * 64, 'expected_tasks': expected_tasks,
              'main_tasks': main_tasks, 'calibration_tasks': len(seen) - main_tasks,
              'coverage': 'complete' if expected_tasks == main_tasks else 'partial_or_unspecified',
              'native_latent_status': 'exported' if mode == 'core_latent_qa' else 'not_received_optional_extension',
              'encoder': ENCODER, 'payload_bytes': payload_bytes, 'cap_bytes': max_bytes,
              'replay_pass': replay['pass'], 'files_sha256': files}
    atomic_json(stage / 'RELEASE_MANIFEST.json', report)
    with (stage / 'SHA256SUMS').open('x', encoding='utf-8', newline='\n') as f:
        for name, sha in sorted({**files, 'RELEASE_MANIFEST.json': file_hash(stage / 'RELEASE_MANIFEST.json')}.items()):
            f.write(f'{sha}  {name}\n')
    total = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
    if total > max_bytes:
        raise ValueError('Export including overhead exceeds cap; staging retained')
    validate_export(stage)
    fsync_dir(stage)
    os.rename(stage, out)
    fsync_dir(out.parent)
    return {**report, 'total_bytes': total, 'path': str(out)}


def validate_export(root):
    root = Path(root)
    report = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    if report['schema'] != SCHEMA:
        raise ValueError('Wrong export schema')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Export must be materialized, without symlinks')
    expected = report['files_sha256']
    actual = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob('*')
              if p.is_file() and p.name not in ('RELEASE_MANIFEST.json', 'SHA256SUMS')}
    if expected != actual:
        raise ValueError('Export checksum/extra/missing files')
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(root / relative) != sha:
            raise ValueError('SHA256SUMS mismatch')
    counts = {'candidates': report['tasks'] * 8, 'previews': report['tasks'] * 64,
              'delivered': report['tasks'] * 8, 'tasks': report['tasks']}
    for name, count in counts.items():
        frame = pd.read_parquet(root / 'core' / (name + '.parquet'))
        keys = ['profile_id', 'prompt_uid', 'bank_id'] + ([] if name == 'tasks' else ['seed_idx']) + (['j'] if name == 'previews' else [])
        if len(frame) != count or frame.duplicated(keys).any():
            raise ValueError('Export cardinality/keys differ')
    prompts = pd.read_parquet(root / 'core/prompts.parquet')
    index = json.loads((root / 'core/prompt_features.index.json').read_text())
    with safe_open(str(root / 'core/prompt_features.safetensors'), framework='np') as f:
        embedding = f.get_tensor('embedding')
    if (embedding.shape != (len(index), 384) or embedding.dtype != np.float32 or
            not np.isfinite(embedding).all() or not np.allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-5) or
            set(index.values()) != set(range(len(index))) or prompts.prompt_uid.duplicated().any() or
            set(index) != set(prompts.prompt_uid)):
        raise ValueError('Export Core384 index/embedding/UID differs')
    candidates = pd.read_parquet(root / 'core/candidates.parquet')
    foreign = set(map(tuple, candidates[['profile_id','prompt_uid','bank_id','seed_idx']].to_numpy()))
    for name in ('previews','delivered'):
        df = pd.read_parquet(root / 'core' / (name + '.parquet'))
        if set(map(tuple, df[['profile_id','prompt_uid','bank_id','seed_idx']].to_numpy())) != foreign:
            raise ValueError('Export candidate foreign keys differ')
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if total > report['cap_bytes']:
        raise ValueError('Export cap exceeded')
    return {'pass': True, 'tasks': report['tasks'], 'total_bytes': total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', required=True, help='Lite2 release root; repeat for compatible reuse roots')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['core_latent_qa', 'core_qa'], default='core_latent_qa')
    p.add_argument('--expected-tasks', type=int, default=4800)
    p.add_argument('--include-calibration', action='store_true')
    p.add_argument('--reuse-audit', action='append', default=[])
    p.add_argument('--planned-tasks', action='append', default=[])
    p.add_argument('--failures-root', action='append', default=[])
    p.add_argument('--qualification-root')
    a = p.parse_args()
    paths = [p for r in a.root for p in (Path(r) / 'tasks').iterdir()]
    for file in a.reuse_audit:
        report = json.loads(Path(file).read_text())
        for row in report['records']:
            if row['status'] == 'accepted_full':
                if file_hash(Path(row['path']) / 'COMPLETE.json') != row['complete_sha256']:
                    raise ValueError('Reused source changed after audit')
                paths.append(Path(row['path']))
    print(json.dumps(export_release(paths, a.out,
                                    mode=a.mode, expected_tasks=a.expected_tasks,
                                    include_calibration=a.include_calibration, planned_manifests=a.planned_tasks,
                                    failures_roots=a.failures_root, qualification_root=a.qualification_root)))


if __name__ == '__main__':
    main()
