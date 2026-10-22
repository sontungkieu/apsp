"""Freeze Lite2 tasks from the existing provenance-bearing, disjoint v1 prompt pool."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from apsp.lite.prompts import read_jsonl, write_jsonl
from apsp.lite.storage import atomic_json
from .contracts import PROFILES, QUOTAS, SCHEMA, canonical_hash, file_hash, task, uid


def build(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError('Use a new immutable manifest directory')
    pools = {s: read_jsonl(source / f'{s}.jsonl') for s in
             ('train', 'dev', 'calibration', 'test_id', 'test_shift')}
    uids, groups = {}, {}
    for split, rows in pools.items():
        for r in rows:
            if r['prompt_uid'] != uid(r['text']) or r['split'] != split:
                raise ValueError('Prompt identity/split mismatch')
            if r['prompt_uid'] in uids:
                raise ValueError('Duplicate UID across input manifests')
            uids[r['prompt_uid']] = split
            if not r.get('group_id') or not r.get('source') or not r.get('source_revision'):
                raise ValueError('Missing split/group/source provenance')
            if r['group_id'] in groups and groups[r['group_id']] != split:
                raise ValueError('Parent/near-duplicate group crosses splits')
            groups[r['group_id']] = split
    train = pools['train']
    if len(train) != 2000 or sorted(r['train_rank'] for r in train) != list(range(1, 2001)):
        raise ValueError('Need exactly ranked train2000; do not fabricate missing prompts')
    if dict(Counter(r['source'] for r in train)) != QUOTAS['train']:
        raise ValueError('Train source quotas differ')
    dev = []
    for name, count in QUOTAS['dev'].items():
        candidates = sorted((r for r in pools['dev'] if r['source'] == name), key=lambda r: r['prompt_uid'])
        if len(candidates) < count:
            raise ValueError('Insufficient disjoint dev source: ' + name)
        dev.extend(candidates[:count])
    calibration = sorted(pools['calibration'], key=lambda r: r['prompt_uid'])
    if len(calibration) != 50 or len(pools['test_id']) != 300 or len(pools['test_shift']) != 200:
        raise ValueError('Need calibration50/test500; test prompts are never collected')
    main = [task(r, p, b) for r in train + dev for p in PROFILES
            for b in ((0,) if r['split'] == 'train' else (0, 1))]
    qual_uids = {r['prompt_uid'] for r in calibration[:10]}
    calib = [task(r, p, b, qa_final=b == 0, qa_preview=b == 0 and r['prompt_uid'] in qual_uids)
             for r in calibration for p in PROFILES
             for b in ((0, 1) if r['prompt_uid'] in qual_uids else (0,))]
    main.sort(key=lambda t: (t['phase'], t['prompt']['split'] != 'dev', t['task_id']))
    out.mkdir(parents=True)
    write_jsonl(out / 'main.jsonl', main)
    write_jsonl(out / 'calibration.jsonl', calib)
    write_jsonl(out / 'qualification.jsonl', [t for t in calib if t['prompt']['prompt_uid'] in qual_uids])
    write_jsonl(out / 'test-locked.jsonl', pools['test_id'] + pools['test_shift'])
    for name, rows in [('train', train), ('dev', dev), ('calibration-prompts', calibration)]:
        write_jsonl(out / (name + '.jsonl'), rows)
    report = {'schema': SCHEMA, 'main_tasks': len(main), 'main_candidates': len(main) * 8,
              'main_preview_rows': len(main) * 64, 'calibration_union_tasks': len(calib),
              'qualification_tasks': 40, 'qa_final_png': 800, 'qa_preview_png': 480,
              'test_status': 'locked_manifest_only_no_generation',
              'selection': 'v1 disjoint source groups; dev UID order before reward; v1 train ranks retained',
              'dedup_coverage': 'inherits supplied v1 group/exclusion audit; does not claim semantic coverage',
              'input_sha256': {s: file_hash(source / f'{s}.jsonl') for s in pools},
              'files_sha256': {p.name: file_hash(p) for p in sorted(out.glob('*.jsonl'))}}
    atomic_json(out / 'manifest.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-manifests', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(build(a.v1_manifests, a.out))


if __name__ == '__main__':
    main()
