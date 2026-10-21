"""Offline source/policy binding check; does not allocate a GPU or qualify a company node."""
import json
from pathlib import Path
from apsp.lite.contracts import canonical_hash,file_hash
from apsp.lite.collector import runtime_identity
from apsp.lite.performance import ExecutionPolicy


def main():
    root=Path(__file__).resolve().parents[2]
    release=json.loads((root/'reports/performance/RELEASE.json').read_text())
    code={p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'apsp').rglob('*.py'))}
    if canonical_hash(code)!=release['qualified_core_source_sha256']:raise ValueError('Collector source differs from measured source')
    if file_hash(root/'env/lite/uv.lock')!=release['lock_sha256']:raise ValueError('Runtime lock differs')
    for rel,expected in release['receipts_sha256'].items():
        if file_hash(root/rel)!=expected:raise ValueError('Receipt hash mismatch: '+rel)
    for name in release['profiles']:
        receipt=json.loads((root/f'reports/performance/{name}-qualification.json').read_text())
        policy=ExecutionPolicy.load(root/f'env/lite/performance/{name}.json')
        if not receipt['pass'] or receipt['runtime']['execution']!=policy.to_dict():raise ValueError('Execution policy not qualified')
        if receipt['runtime_identity_hash']!=canonical_hash(runtime_identity(receipt['runtime'])):raise ValueError('Qualification identity mismatch')
        if receipt['runtime']['source_hash']!=canonical_hash(code):raise ValueError('Qualification from another source')
    print('PERFORMANCE_RELEASE_PASS '+canonical_hash(code))


if __name__=='__main__':main()
