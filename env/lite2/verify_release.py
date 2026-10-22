"""Offline binding of the shipped collector to real Modal receipts; not node GPU qualification."""
import json
from pathlib import Path
from apsp.lite2.contracts import source_hash,file_hash,canonical_hash,registry,policy,SCHEMA


def main():
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'reports/lite2-v2/RELEASE.json').read_text())
    if manifest['core_source_sha256']!=source_hash():raise ValueError('Collector differs from measured source')
    if manifest['lock_sha256']!=file_hash(root/'env/lite/uv.lock'):raise ValueError('Runtime lock differs')
    for name,sha in manifest['files_sha256'].items():
        if file_hash(root/name)!=sha:raise ValueError('Receipt/code hash differs: '+name)
    for name in registry()['profiles']:
        receipt=json.loads((root/'reports/lite2-v2'/(name+'.json')).read_text())
        if (not receipt['pass'] or not receipt['performance_enabled'] or receipt['schema']!=SCHEMA or
                receipt['runtime']['source_hash']!=source_hash() or receipt['execution']!=policy().to_dict() or
                receipt['registry_hash']!=canonical_hash(registry()) or len(receipt['tasks'])!=20 or
                not receipt['core_only_replay_pass'] or not receipt['restore_independent_process']):
            raise ValueError('Incomplete or incompatible qualification: '+name)
        for t in receipt['tasks']:
            if not t['validation']['pass'] or not t['validation']['native_checked'] or not t['validation']['redecode_pixels_identical']:
                raise ValueError('Native/round-trip task failed')
            if t['parity'] is not None and not t['parity']['pass']:raise ValueError('Parity failed')
    print('LITE2_RELEASE_BINDING_PASS '+source_hash()+'; company node qualification still required')


if __name__=='__main__':main()
