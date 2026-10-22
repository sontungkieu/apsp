"""Single-GPU resident worker. Frozen partitions, node qualification, atomic claims and backup."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from apsp.lite.prompts import read_jsonl
from apsp.lite.storage import TaskBusy, atomic_json, backup_task, claim
from apsp.lite.performance import ExecutionPolicy
from .contracts import PROFILES, SCHEMA, canonical_hash, check_task, file_hash, identity, registry, source_hash


def initialize(root, manifest):
    root = Path(root).resolve()
    if root.name in ('release-v1', 'backup-v1'):
        raise ValueError('Use a new Lite2 namespace')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tasks').mkdir(exist_ok=True)
    expected = {'schema': SCHEMA, 'manifest_sha256': file_hash(manifest),
                'registry_hash': canonical_hash(registry()), 'source_hash': source_hash()}
    path = root / 'LITE2_RELEASE.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Release source/manifest contract differs')
    else:
        # Initialization is a separate command, before concurrent workers start.
        with path.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(expected, f, sort_keys=True, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init', action='store_true')
    p.add_argument('--backup-root')
    p.add_argument('--profile', choices=PROFILES)
    p.add_argument('--qualification')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--phase', type=int, choices=[1, 2], default=2)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--blocked-tasks', help='Frozen JSONL old-job reservations, created by the company queue agent')
    p.add_argument('--reuse-audit', help='Frozen reuse.json; only accepted full native Lite2 records are skipped')
    a = p.parse_args()
    if a.init:
        print(json.dumps(initialize(a.out, a.tasks)))
        return
    if not all((a.backup_root, a.profile, a.qualification)) or not 0 <= a.worker_index < a.workers:
        p.error('Worker needs profile, qualification, backup and valid partition')
    primary, backup = Path(a.out).resolve(), Path(a.backup_root).resolve()
    if primary == backup or primary in backup.parents or backup in primary.parents:
        raise ValueError('Primary and backup trees must be separate')
    release = json.loads((primary / 'LITE2_RELEASE.json').read_text())
    if release != {'schema': SCHEMA, 'manifest_sha256': file_hash(a.tasks),
                   'registry_hash': canonical_hash(registry()), 'source_hash': source_hash()}:
        raise ValueError('Run --init once on a new release; source/manifest must remain frozen')
    blocked = set()
    aliases = {'sdxl_lightning8': 'sdxl_lightning8_1024'}
    if a.blocked_tasks:
        for t in read_jsonl(a.blocked_tasks):
            name = aliases.get(t.get('profile'), t.get('profile'))
            prompt_uid = t.get('prompt_uid') or t.get('prompt', {}).get('prompt_uid')
            blocked.add((name, prompt_uid, t['bank_id']))
    accepted = set()
    if a.reuse_audit:
        audit = json.loads(Path(a.reuse_audit).read_text())
        if audit['manifest_sha256'] != file_hash(a.tasks) or audit['registry_hash'] != canonical_hash(registry()):
            raise ValueError('Reuse audit from another manifest/registry')
        for row in audit['records']:
            if row['status'] == 'accepted_full':
                # Never trust an audit without verifying its referenced immutable artifact now.
                from .validate import validate_task
                validate_task(row['path'], check_latents=True)
                if file_hash(Path(row['path']) / 'COMPLETE.json') != row['complete_sha256']:
                    raise ValueError('Reused artifact changed after audit')
                accepted.add(row['target_task_id'])
    tasks = []
    all_tasks = read_jsonl(a.tasks)
    seen = set()
    for t in all_tasks:
        check_task(t)
        if t['task_id'] in seen:
            raise ValueError('Duplicate task ID in manifest')
        seen.add(t['task_id'])
        if t['profile'] == a.profile and t['phase'] <= a.phase:
            tasks.append(t)
    # Partition before exclusions: accepted/busy lists cannot shift task ownership.
    tasks = tasks[a.worker_index::a.workers]
    tasks = [t for t in tasks if t['task_id'] not in accepted and
             (t['profile'], t['prompt']['prompt_uid'], t['bank_id']) not in blocked]
    if a.max_tasks is not None:
        if a.max_tasks < 1:
            raise ValueError('max-tasks must be positive')
        tasks = tasks[:a.max_tasks]
    receipt = json.loads(Path(a.qualification).read_text())
    execution = ExecutionPolicy(**receipt['execution'])
    fp, identity_hash = identity(execution)
    if (not receipt['pass'] or not receipt.get('performance_enabled') or receipt.get('schema') != SCHEMA or receipt['profile'] != a.profile or
            receipt['runtime_identity_hash'] != identity_hash or receipt['registry_hash'] != canonical_hash(registry())):
        raise ValueError('Fresh Lite2 qualification for this source/runtime/policy is required')
    from apsp.lite.runtime import LiteRuntime
    from .encoder import CoreEncoder
    from .collector import collect_task
    runtime = LiteRuntime(registry()['profiles'][a.profile], registry(), execution=execution)
    runtime.load_scorers()
    encoder = CoreEncoder()
    worker_id = f'{socket.gethostname()}-{a.profile}-{a.worker_index}-pid{os.getpid()}'
    heartbeat = primary / 'workers' / (worker_id + '.json')
    base = {'worker_id': worker_id, 'manifest_sha256': file_hash(a.tasks), 'runtime': fp,
            'profile': a.profile, 'worker_index': a.worker_index, 'workers': a.workers,
            'qualification_sha256': file_hash(a.qualification), 'selected_tasks': len(tasks)}
    failures, done = 0, 0
    for t in tasks:
        atomic_json(heartbeat, {**base, 'state': 'running', 'task_id': t['task_id'], 'done': done, 'time': time.time()})
        try:
            with claim(primary, t['task_id']):
                for attempt in range(3):
                    try:
                        dest = collect_task(runtime, encoder, t, primary, fp)
                        backup_task(dest, backup / 'tasks' / t['task_id'])
                        atomic_json(primary / 'acks' / (t['task_id'] + '.json'),
                                    {'task_id': t['task_id'], 'backup': str(backup / 'tasks' / t['task_id']),
                                     'complete_sha256': file_hash(dest / 'COMPLETE.json'), 'time': time.time()})
                        done += 1
                        break
                    except (ValueError, FileExistsError):
                        raise
                    except Exception as exc:
                        atomic_json(primary / 'failures' / f'{t["task_id"]}-{attempt}.json',
                                    {'task_id': t['task_id'], 'attempt': attempt + 1,
                                     'status': 'failed_infra', 'error_type': type(exc).__name__})
                        if attempt == 2:
                            failures += 1
        except TaskBusy:
            failures += 1
    atomic_json(heartbeat, {**base, 'state': 'failed' if failures else 'completed',
                           'done': done, 'failures': failures, 'time': time.time()})
    print('LITE2_WORKER=' + json.dumps({'done': done, 'failures': failures, 'worker_id': worker_id}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
