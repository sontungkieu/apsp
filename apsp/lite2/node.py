"""Node-local manager for the eight assigned B200s. Never modifies old jobs or shared schedulers."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from apsp.lite.prompts import read_jsonl
from apsp.lite.storage import atomic_json
from .contracts import PROFILES, canonical_hash, file_hash, registry, source_hash
from .worker import initialize


def gpu_inventory(uuids):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    cards = {r.split(',', 1)[0].strip(): r.split(',', 1)[1].strip() for r in text.splitlines() if ',' in r}
    if len(uuids) != 8 or len(set(uuids)) != 8 or any('B200' not in cards.get(u, '') for u in uuids):
        raise ValueError('Exactly eight unique, assigned B200 UUIDs required; do not infer allocation')
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    occupied = {r.strip() for r in text.splitlines()}
    if occupied.intersection(uuids):
        raise ValueError('Assigned GPU is occupied; wait, never stop another job')
    return {u: cards[u] for u in uuids}


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-manifests', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--backup-root', required=True)
    p.add_argument('--gpu-uuids', required=True, help='Eight live verified allocated UUIDs, comma separated')
    p.add_argument('--old-reservations', required=True, help='Frozen old running/queued/reserved task JSONL; explicit empty file if none')
    p.add_argument('--old-root', action='append', default=[])
    p.add_argument('--hyper-workers', type=int, default=2)
    p.add_argument('--stage', choices=['all', 'qualify', 'collect'], default='all')
    p.add_argument('--no-export', action='store_true')
    a = p.parse_args()
    out, backup = Path(a.out).resolve(), Path(a.backup_root).resolve()
    if out == backup or out in backup.parents or backup in out.parents or any(p.name in ('release-v1', 'backup-v1') for p in (out, *out.parents, backup, *backup.parents)):
        raise ValueError('Fresh, separate Lite2 output/backup roots required')
    if not 1 <= a.hyper_workers <= 7 or not Path(a.old_reservations).is_file():
        raise ValueError('Need 1..7 Hyper workers and explicit reservation snapshot')
    uuids = [s.strip() for s in a.gpu_uuids.split(',')]
    cards = gpu_inventory(uuids)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'NODE_MANAGER.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = str(time.time_ns())
        run = out / 'manager-runs' / attempt
        run.mkdir(parents=True)
        atomic_json(run / 'inventory.json', {'hostname': socket.gethostname(), 'cards': cards,
                    'source_hash': source_hash(), 'registry_hash': canonical_hash(registry()),
                    'old_reservations_sha256': file_hash(a.old_reservations), 'pid': os.getpid()})
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}

        def command(module, *args, gpu=None, label=None):
            child_env = dict(env)
            if gpu: child_env['CUDA_VISIBLE_DEVICES'] = gpu
            args = [sys.executable, '-m', module, *map(str, args)]
            atomic_json(run / ((label or module.rsplit('.', 1)[-1]) + '-command.json'),
                        {'argv': args, 'gpu_uuid': gpu, 'cwd': os.getcwd()})
            subprocess.run(args, env=child_env, check=True)

        manifests = out / 'manifests'
        if not manifests.exists():
            command('apsp.lite2.manifests', '--v1-manifests', a.v1_manifests, '--out', manifests)
        else:
            report = json.loads((manifests / 'manifest.json').read_text())
            for name, sha in report['input_sha256'].items():
                if file_hash(Path(a.v1_manifests) / (name + '.jsonl')) != sha:
                    raise ValueError('Input pool changed; use new namespace')
            for name, sha in report['files_sha256'].items():
                if file_hash(manifests / name) != sha: raise ValueError('Frozen manifest changed')
        qualification = out / 'qualification'
        if a.stage in ('all', 'qualify'):
            if not qualification.exists():
                command('apsp.lite2.qualify', '--tasks', manifests / 'qualification.jsonl',
                        '--out', qualification, gpu=uuids[0])
            if a.stage == 'qualify': return
        for name in PROFILES:
            receipt = json.loads((qualification / (name + '.json')).read_text())
            if not receipt['pass'] or not receipt['performance_enabled'] or receipt['runtime']['source_hash'] != source_hash() or receipt['runtime']['hostname'] != socket.gethostname():
                raise ValueError('Fresh passing node-specific Lite2 optimized qualification required')
        from .reuse import audit
        reuse_path = run / 'reuse-main.json'
        atomic_json(reuse_path, audit(a.old_root, manifests / 'main.jsonl'))
        main_root = out / 'main'
        initialize(main_root, manifests / 'main.jsonl')

        def wave(tasks_file, root, backup_root, reuse, *, phase=2, slots=None, max_tasks=None, label='wave'):
            slots = slots or [a.hyper_workers, 8 - a.hyper_workers]
            # All children from the previous wave have finished before a partition changes.
            gpu_inventory(uuids)
            children = []
            offset = 0
            with contextlib.ExitStack() as stack:
                for profile, count in zip(PROFILES, slots):
                    for index in range(count):
                        gpu = uuids[offset]; offset += 1
                        args = [sys.executable, '-m', 'apsp.lite2.worker', '--tasks', str(tasks_file),
                                '--out', str(root), '--backup-root', str(backup_root), '--profile', profile,
                                '--qualification', str(qualification / (profile + '.json')),
                                '--worker-index', str(index), '--workers', str(count), '--phase', str(phase),
                                '--blocked-tasks', a.old_reservations, '--reuse-audit', str(reuse)]
                        if max_tasks: args.extend(['--max-tasks', str(max_tasks)])
                        log = stack.enter_context((run / f'{label}-{profile}-{index}.log').open('x'))
                        child = subprocess.Popen(args, env={**env, 'CUDA_VISIBLE_DEVICES': gpu},
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append((child, profile, index))
                        atomic_json(run / f'{label}-{profile}-{index}-job.json',
                                    {'pid': child.pid, 'gpu_uuid': gpu, 'argv': args, 'profile': profile})
                failures = [(profile, index, child.wait()) for child, profile, index in children]
            if any(rc for _, _, rc in failures):
                raise RuntimeError('Lite2 wave failed; logs retained; existing jobs untouched: ' + str(failures))

        wave(manifests / 'main.jsonl', main_root, backup / 'main', reuse_path,
             slots=[1, 1], max_tasks=10, phase=1, label='canary')
        from .validate import validate_task
        for path in (main_root / 'tasks').iterdir():
            if not validate_task(path, check_latents=True)['all_valid']:
                raise ValueError('Canary has model failures; do not expand')
        wave(manifests / 'main.jsonl', main_root, backup / 'main', reuse_path,
             slots=[1, 3], max_tasks=10, phase=1, label='scale4')
        wave(manifests / 'main.jsonl', main_root, backup / 'main', reuse_path, phase=1, label='phase1')
        for path in (main_root / 'tasks').iterdir(): validate_task(path)
        wave(manifests / 'main.jsonl', main_root, backup / 'main', reuse_path, phase=2, label='phase2')
        qa_root = out / 'calibration'
        initialize(qa_root, manifests / 'calibration.jsonl')
        qa_reuse = run / 'reuse-calibration.json'
        old_qa = [qualification / p / 'fast' for p in PROFILES]
        atomic_json(qa_reuse, audit(old_qa, manifests / 'calibration.jsonl'))
        wave(manifests / 'calibration.jsonl', qa_root, backup / 'calibration', qa_reuse, label='qa')
        result = {'main_root': str(main_root), 'qa_root': str(qa_root),
                  'reuse_main': str(reuse_path), 'reuse_calibration': str(qa_reuse), 'attempt': attempt}
        if not a.no_export:
            from .export import export_release
            refs = [r['path'] for file in (reuse_path, qa_reuse) for r in json.loads(file.read_text())['records']
                    if r['status'] == 'accepted_full']
            paths = list((main_root / 'tasks').iterdir()) + list((qa_root / 'tasks').iterdir()) + refs
            report = export_release(paths, out / 'exports' / attempt, include_calibration=True, expected_tasks=4800)
            result.update(export=report['path'], coverage=report['coverage'], main_tasks=report['main_tasks'])
        atomic_json(run / 'result.json', result)
        print('LITE2_NODE=' + json.dumps(result), flush=True)


if __name__ == '__main__': main()
