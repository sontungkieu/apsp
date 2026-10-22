"""Conservative old-bank audit: checksum success alone is not native Lite2 compatibility."""
import argparse
import json
from collections import Counter
from pathlib import Path

from apsp.lite.prompts import read_jsonl
from apsp.lite.storage import atomic_json, verify_complete
from .contracts import SCHEMA, canonical_hash, file_hash, registry


def audit(roots, tasks):
    target = {(t['profile'], t['prompt']['prompt_uid'], t['bank_id']): t for t in read_jsonl(tasks)}
    aliases = {'sdxl_lightning8': 'sdxl_lightning8_1024'}
    records, seen = [], set()
    for root in roots:
        for path in sorted((Path(root) / 'tasks').iterdir()):
            if not path.is_dir() or not (path / 'COMPLETE.json').is_file():
                continue  # never inspect or modify staging or live claims
            row = {'path': str(path.resolve())}
            try:
                if not verify_complete(path):
                    continue
                t = json.loads((path / 'task.json').read_text())
                key = (aliases.get(t['profile'], t['profile']), t['prompt']['prompt_uid'], t['bank_id'])
                row.update(profile=key[0], prompt_uid=key[1], bank_id=key[2],
                           complete_sha256=file_hash(path / 'COMPLETE.json'))
                if key not in target:
                    row['status'] = 'reserve_or_out_of_scope'
                elif t.get('schema') != SCHEMA:
                    row.update(target_task_id=target[key]['task_id'], status='blocked_native_dtype',
                               reason='v1 casts preview/final storage; native/feature/decode compatibility unproved; preserve v1')
                else:
                    from .validate import validate_task
                    validate_task(path, check_latents=True)
                    if t != target[key]:
                        raise ValueError('Task/retention/manifest recipe differs')
                    if key in seen:
                        row['status'] = 'duplicate_compatible_reserve'
                    else:
                        row.update(status='accepted_full', target_task_id=target[key]['task_id'])
                        seen.add(key)
            except (ValueError, KeyError, OSError) as exc:
                row.update(status='invalid_or_incompatible', error_type=type(exc).__name__)
            records.append(row)
    return {'schema': SCHEMA, 'manifest_sha256': file_hash(tasks), 'registry_hash': canonical_hash(registry()),
            'counts': dict(Counter(r['status'] for r in records)), 'records': records,
            'scope': 'Conservative native-compatible reuse only; old text/UID and reserve artifacts remain available'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root', action='append', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    report = audit(a.old_root, a.tasks)
    atomic_json(a.out, report)
    print(json.dumps(report['counts']))
