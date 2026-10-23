"""Read-only Lite2 EDA: CPU, exported Core, no model import or dependency install.

This module deliberately lives outside apsp/: adding analysis must not invalidate
the qualified collector's core-source hash. Input is one immutable V2 export.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import unicodedata

import numpy as np
import pandas as pd
from safetensors import safe_open

KEY = ['profile_id', 'prompt_uid', 'bank_id']
SEED = KEY + ['seed_idx']
PROFILES = ['hyper_sd15_cfg8_512', 'sdxl_lightning8_1024']
TABLES = ['prompts', 'tasks', 'task_plan', 'candidates', 'previews', 'delivered']
SCHEMA = 'apsp-core-lite2-v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def child(root, name):
    """All release references must remain inside the supplied, materialized export."""
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe input reference: ' + str(name))
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError('Symlink input reference: ' + str(name))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Input reference escapes export')
    return path


def bootstrap_mean(values, *, seed=20261023, repeats=1000):
    """Percentile CI over independent prompt/group aggregates, never seed rows."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                    lo=None, hi=None, status='insufficient_clusters')
    rng = np.random.default_rng(seed)
    # Bounded memory even for full collection; one bootstrap replicate at a time.
    draws = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(repeats)]
    return dict(n=len(x), mean=float(x.mean()), lo=float(np.quantile(draws, .025)),
                hi=float(np.quantile(draws, .975)), status='exploratory_percentile_95ci')


def describe(frame, by, columns):
    rows = []
    for key, group in frame.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for column in columns:
            values = pd.to_numeric(group[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
            row = dict(zip(by, key))
            row.update(metric=column, rows=len(values), finite=int(values.notna().sum()),
                       mean=values.mean(), std=values.std(), min=values.min(),
                       p01=values.quantile(.01), p10=values.quantile(.1), p50=values.quantile(.5),
                       p90=values.quantile(.9), p99=values.quantile(.99), max=values.max())
            rows.append(row)
    return pd.DataFrame(rows)


def tensor_header(path, *, values=False):
    """Read headers only by default; optional byte hashes + finite stats one tensor at a time."""
    path = Path(path)
    with path.open('rb') as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError('Truncated safetensors header')
        length = struct.unpack('<Q', length_bytes)[0]
        if length > 16 << 20 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(f.read(length))
        rows = []
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            lo, hi = meta['data_offsets']
            if not 0 <= lo <= hi <= path.stat().st_size - 8 - length:
                raise ValueError('Invalid tensor offsets')
            row = dict(tensor=name, dtype=meta['dtype'], shape=meta['shape'], bytes=hi-lo)
            if values:
                f.seek(8 + length + lo)
                raw = f.read(hi-lo)
                row['value_sha256'] = hashlib.sha256(raw).hexdigest()
                dtype = {'F16': '<f2', 'F32': '<f4', 'F64': '<f8'}.get(meta['dtype'])
                if dtype:
                    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
                    finite = np.isfinite(x)
                    row.update(finite_fraction=float(finite.mean()),
                               rms=float(np.sqrt(np.mean(x[finite]**2))) if finite.any() else None,
                               min=float(x[finite].min()) if finite.any() else None,
                               max=float(x[finite].max()) if finite.any() else None)
            rows.append(row)
    return header.get('__metadata__', {}), rows


class Audit:
    def __init__(self, export, out, *, full_hash=False, native_values=0, seed=20261023,
                 bootstrap=1000, expected_main=4800, test_manifests=(),
                 collection=None, manifests=None, backup=None, backup_full_hash=False):
        self.root, self.out = Path(export).resolve(), Path(out).resolve()
        self.collection=Path(collection).resolve() if collection else None
        self.manifests=Path(manifests).resolve() if manifests else None
        self.backup=Path(backup).resolve() if backup else None
        self.backup_full_hash=backup_full_hash
        for protected in [self.root,self.collection,self.manifests,self.backup]:
            if protected and (self.out.is_relative_to(protected) or protected.is_relative_to(self.out)):
                raise ValueError('EDA output must be outside export/collection/manifests/backup and their ancestors')
        if self.out.exists():
            raise FileExistsError('Use a fresh EDA output directory (immutable analysis run)')
        if not (self.root/'RELEASE_MANIFEST.json').is_file():
            raise ValueError('Supply an exported V2 root containing RELEASE_MANIFEST.json and core/')
        self.manifest = read_json(self.root/'RELEASE_MANIFEST.json')
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError('Wrong schema; V1 and raw task roots are not V2 exports')
        self.out.mkdir(parents=True)
        self.full_hash, self.native_values = full_hash, native_values
        self.seed, self.bootstrap, self.expected_main = seed, bootstrap, expected_main
        self.test_manifests = list(map(Path, test_manifests))
        self.checks, self.tables, self.sections, self.figures = [], {}, {}, []
        self.data = {}
        self.record('release_identity', {'root': str(self.root), 'schema': SCHEMA,
                    'manifest_sha256': sha256(self.root/'RELEASE_MANIFEST.json'),
                    'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                    'numpy': np.__version__, 'pandas': pd.__version__, 'full_hash': full_hash,
                    'native_values_per_profile': native_values, 'rng_seed': seed,
                    'bootstrap_repeats': bootstrap, 'expected_main_tasks': expected_main,
                    'scope': 'CPU read-only exploratory audit; no model/scorer calls; no online efficacy claim'})

    def record(self, name, data):
        self.sections[name] = data
        write_json(self.out/(name+'.json'), data)
        return data

    def table(self, name, frame):
        self.tables[name] = frame
        frame.to_csv(self.out/(name+'.csv'), index=False)
        return frame

    def check(self, name, ok, detail='', *, severity='error'):
        self.checks.append(dict(check=name, status='PASS' if ok else ('WARN' if severity=='warning' else 'FAIL'),
                                severity=severity, detail=str(detail)))

    def integrity(self):
        expected = self.manifest['files_sha256']
        rows = []
        for name, digest in sorted(expected.items()):
            path = child(self.root, name)
            selected = self.full_hash or not name.startswith(('latent/', 'qa/'))
            status = 'NOT_HASHED_EXTENSION'
            if not path.is_file():
                status = 'MISSING'
            elif selected:
                status = 'PASS' if sha256(path) == digest else 'HASH_MISMATCH'
            rows.append(dict(file=name, bytes=path.stat().st_size if path.is_file() else 0,
                             status=status))
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
        self.check('manifest_file_set', actual == set(expected) | {'RELEASE_MANIFEST.json','SHA256SUMS'},
                   sorted(actual ^ (set(expected) | {'RELEASE_MANIFEST.json','SHA256SUMS'})))
        self.check('no_symlinks', not any(p.is_symlink() for p in self.root.rglob('*')))
        # Bind the manifest itself against SHA256SUMS, and require the exact listed set.
        listed = {}
        for line in (self.root/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            child(self.root, name)
            if name in listed:
                raise ValueError('Duplicate SHA256SUMS entry')
            listed[name] = digest
        self.check('checksum_index', listed == {**expected, 'RELEASE_MANIFEST.json':
                                               sha256(self.root/'RELEASE_MANIFEST.json')})
        self.check('selected_payload_hashes', all(r['status'] in ('PASS','NOT_HASHED_EXTENSION') for r in rows))
        self.table('file_integrity', pd.DataFrame(rows))
        self.record('integrity_scope', {'full_release_bytes_hashed': self.full_hash,
                    'extension_files_not_hashed': sum(r['status']=='NOT_HASHED_EXTENSION' for r in rows),
                    'note': 'Headers are checked separately. Partial hashing is not a full-release PASS.'})
        if any(c['status']=='FAIL' for c in self.checks):
            self.finish()
            raise ValueError('Integrity failed; see checks.csv. Statistics deliberately stopped.')
        return self.tables['file_integrity']

    def load(self):
        for name in TABLES:
            self.data[name] = pd.read_parquet(self.root/'core'/(name+'.parquet'))
        # Arrow all-null columns (notably Lightning CFG) arrive as object/None.
        # Preserve missingness while making numerical checks work on this valid case.
        for name in ['previews','delivered']:
            frame=self.data[name]
            for column in ['ir','pickscore_raw','progress','drift','cfg_gap']:
                if column in frame:frame[column]=pd.to_numeric(frame[column],errors='coerce')
        self.index = read_json(self.root/'core/prompt_features.index.json')
        with safe_open(str(self.root/'core/prompt_features.safetensors'), framework='np') as f:
            self.embedding = f.get_tensor('embedding')
        p, t, plan, c, v, d = [self.data[n] for n in TABLES]
        self.check('prompt_uid_unique', not p.prompt_uid.duplicated().any())
        self.check('logical_tasks_unique', not t.duplicated(KEY).any() and not t.task_id.duplicated().any())
        self.check('plan_task_unique', not plan.task_id.duplicated().any() and not plan.duplicated(KEY).any())
        for name, keys, n in [('candidates',SEED,8),('delivered',SEED,8),('previews',SEED+['j'],64)]:
            frame = self.data[name]
            counts = frame.groupby(KEY, dropna=False).size()
            self.check(name+'_cardinality', len(frame)==len(t)*n and counts.eq(n).all()
                       and not frame.duplicated(keys).any())
        foreign = set(map(tuple,c[SEED].to_numpy()))
        self.check('candidate_task_foreign_keys', set(map(tuple,c[KEY].to_numpy()))==set(map(tuple,t[KEY].to_numpy())))
        for name in ['previews','delivered']:
            self.check(name+'_foreign_keys', set(map(tuple,self.data[name][SEED].to_numpy()))==foreign)
        self.check('exact_seed_indices', c.groupby(KEY).seed_idx.apply(lambda x: sorted(x)==list(range(8))).all())
        self.check('exact_checkpoint_indices', v.groupby(SEED).j.apply(lambda x: sorted(x)==list(range(1,9))).all())
        self.check('prompt_foreign_keys', set(t.prompt_uid)<=set(p.prompt_uid))
        self.check('prompt_source_groups_present',p[['group_id','source','source_revision']].notna().all().all()
                   and p.group_id.astype(str).str.len().gt(0).all())
        self.check('collected_subset_of_plan', set(t.task_id)<=set(plan.task_id))
        status_join = t.merge(plan,on='task_id',suffixes=('_actual','_plan'),validate='one_to_one')
        for col in KEY+['split','status']:
            self.check('plan_matches_'+col, status_join[col+'_actual'].eq(status_join[col+'_plan']).all())
        self.check('no_generated_test', not t.split.astype(str).str.startswith('test').any())
        self.check('profiles_allowed', set(t.profile_id)<=set(PROFILES))
        self.check('bank_contract', t.bank_id.isin([0,1]).all() and t.loc[t.split=='train','bank_id'].eq(0).all())
        self.check('core384', self.embedding.shape==(len(self.index),384) and self.embedding.dtype==np.float32
                   and np.isfinite(self.embedding).all() and set(self.index)==set(p.prompt_uid)
                   and set(self.index.values())==set(range(len(self.index)))
                   and np.allclose(np.linalg.norm(self.embedding,axis=1),1,atol=1e-5))
        self.table('checks',pd.DataFrame(self.checks))
        if any(c['status']=='FAIL' and c['severity']=='error' for c in self.checks):
            self.finish()
            raise ValueError('Core structural checks failed; see checks.csv')
        prompt_meta = p.drop(columns=['split'], errors='ignore')
        meta = t.merge(prompt_meta,on='prompt_uid',validate='many_to_one')
        self.tasks = meta
        self.final = d.merge(meta,on=KEY,validate='many_to_one',suffixes=('','_task'))
        self.preview = v.merge(meta,on=KEY,validate='many_to_one',suffixes=('','_task'))
        return self.table('row_counts',pd.DataFrame([{'table':n,'rows':len(f)} for n,f in self.data.items()]))

    def coverage(self):
        t,p = self.tasks,self.data['task_plan']
        self.table('coverage',p.groupby(['profile_id','split','bank_id','status'],dropna=False)
                   .agg(tasks=('task_id','size'),prompt_uids=('prompt_uid','nunique'),
                        acked=('durable_ack_verified','sum'),attempts=('attempts','sum')).reset_index())
        self.table('source_coverage',t.groupby(['profile_id','split','bank_id','source'],dropna=False)
                   .agg(tasks=('task_id','size'),prompt_uids=('prompt_uid','nunique')).reset_index())
        main=t[t.split.isin(['train','dev'])]
        self.check('main_4800_or_configured',len(main)==self.expected_main,f'{len(main)}/{self.expected_main}')
        expected={(profile,'train',0):2000 for profile in PROFILES}
        expected.update({(profile,'dev',bank):200 for profile in PROFILES for bank in [0,1]})
        rows=[]
        for (profile,split,bank),n in expected.items():
            found=main[(main.profile_id==profile)&(main.split==split)&(main.bank_id==bank)]
            rows.append(dict(profile_id=profile,split=split,bank_id=bank,expected=n,actual=len(found),
                             delta=len(found)-n))
        self.table('canonical_plan_coverage',pd.DataFrame(rows))
        calib=t[t.split=='calibration']
        calibration_rows=[]
        for profile in PROFILES:
            g=calib[calib.profile_id==profile]
            calibration_rows.append(dict(profile_id=profile,expected_tasks=60,tasks=len(g),
                expected_uids=50,prompt_uids=g.prompt_uid.nunique(),bank0=int(g.bank_id.eq(0).sum()),
                bank1=int(g.bank_id.eq(1).sum())))
        self.table('canonical_calibration_coverage',pd.DataFrame(calibration_rows))
        quota_rows=[]
        for profile in PROFILES:
            for split,quotas in [('train',{'diffusiondb':1000,'coco_captions':400,'structured':600}),
                                 ('dev',{'diffusiondb':100,'coco_captions':40,'structured':60})]:
                for source,n in quotas.items():
                    g=main[(main.profile_id==profile)&(main.split==split)&(main.source==source)]
                    quota_rows.append(dict(profile_id=profile,split=split,source=source,
                                           expected_uids=n,prompt_uids=g.prompt_uid.nunique()))
        self.table('canonical_source_quotas',pd.DataFrame(quota_rows))
        good=all(r['delta']==0 for r in rows)
        good=good and all(r['tasks']==60 and r['prompt_uids']==50 and r['bank0']==50 and r['bank1']==10 for r in calibration_rows)
        good=good and all(r['expected_uids']==r['prompt_uids'] for r in quota_rows)
        self.check('canonical_collection_coverage',good,'4800 main +120 calibration +source quotas; subsets fail intentionally')
        return self.tables['coverage']

    def source_and_backup(self):
        planned={}; manifest_checks=[];records=[]
        if self.manifests:
            frozen=read_json(self.manifests/'manifest.json')
            for name,digest in frozen['files_sha256'].items():
                path=child(self.manifests,name)
                ok=path.is_file() and sha256(path)==digest
                self.check('frozen_manifest_'+name,ok)
                manifest_checks.append(dict(file=name,status='PASS' if ok else 'FAIL'))
            for name in ['main.jsonl','calibration.jsonl']:
                for line in child(self.manifests,name).read_text(encoding='utf-8').splitlines():
                    if not line.strip():continue
                    r=json.loads(line)
                    self.check('planned_id_unique_'+r['task_id'],r['task_id'] not in planned)
                    planned[r['task_id']]=r
            self.check('frozen_plan_matches_export_plan',set(planned)==set(self.data['task_plan'].task_id))
            for t in self.tasks.itertuples():
                if t.task_id in planned:
                    exported=read_json(child(self.root,'provenance/'+t.task_id+'/task.json'))
                    self.check('frozen_task_matches_'+t.task_id,planned[t.task_id]==exported)
        for t in self.tasks.itertuples():
            group='calibration' if t.split=='calibration' else 'main'
            row=dict(task_id=t.task_id,profile_id=t.profile_id,split=t.split,
                     source_marker='UNAVAILABLE',backup_marker='UNAVAILABLE',ack='UNAVAILABLE',
                     source_kind='worker',backup_kind='designated_backup_root')
            receipt=child(self.root,'provenance/'+t.task_id+'/source-complete.json')
            expected=sha256(receipt)
            if self.collection:
                source=child(self.collection,group+'/tasks/'+t.task_id)
                # The shipped manager reuses 40 qualification fast tasks for calibration.
                # They have a local qualification backup but no worker ACK. Follow only
                # this exact documented path inside the explicitly supplied collection.
                qsource=child(self.collection,'qualification/'+t.profile_id+'/fast/tasks/'+t.task_id)
                recorded=Path(getattr(t,'source_task_uri','')).resolve()
                reused=t.split=='calibration' and recorded==qsource.resolve() and not source.exists()
                if reused:
                    source=qsource
                    row['source_kind']='qualification_reuse'
                marker=source/'COMPLETE.json'
                row['source_marker']='PASS' if marker.is_file() and sha256(marker)==expected else 'MISSING_OR_MISMATCH'
                ack=child(self.collection,group+'/acks/'+t.task_id+'.json')
                if ack.is_file():
                    a=read_json(ack)
                    wanted=child(self.backup,group+'/tasks/'+t.task_id) if self.backup else None
                    row['ack']='PASS' if (a.get('task_id')==t.task_id and a.get('complete_sha256')==expected
                        and wanted and Path(a.get('backup','')).resolve()==wanted) else 'MISMATCH_OR_NO_BACKUP_ROOT'
                elif reused:row['ack']='NOT_EMITTED_QUALIFICATION_REUSE'
                else:row['ack']='MISSING'
                self.check('source_marker_'+t.task_id,row['source_marker']=='PASS')
                if not reused:self.check('live_ack_'+t.task_id,row['ack']=='PASS')
            if self.backup:
                backup=child(self.backup,group+'/tasks/'+t.task_id)
                if self.collection and row['source_kind']=='qualification_reuse' and not backup.exists():
                    backup=child(self.collection,'qualification/'+t.profile_id+'/backup/'+t.task_id)
                    row['backup_kind']='qualification_local_copy'
                marker=backup/'COMPLETE.json'
                row['backup_marker']='PASS' if marker.is_file() and sha256(marker)==expected else 'MISSING_OR_MISMATCH'
                self.check('backup_marker_'+t.task_id,row['backup_marker']=='PASS')
                if row['backup_marker']=='PASS' and self.backup_full_hash:
                    expected_files=read_json(marker)['files_sha256']
                    actual={p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
                    ok=actual==set(expected_files)
                    for name,digest in expected_files.items():
                        path=child(backup,name)
                        ok=ok and path.is_file() and sha256(path)==digest
                    row['backup_payload']='PASS' if ok else 'FAIL'
                    self.check('backup_payload_'+t.task_id,ok)
            records.append(row)
        self.table('frozen_manifest_integrity',pd.DataFrame(manifest_checks))
        self.table('live_source_backup',pd.DataFrame(records))
        self.record('live_backup_scope',{'collection':self.collection,'manifests':self.manifests,'backup':self.backup,
                    'backup_payload_full_hash':self.backup_full_hash,
                    'note':'Marker/ACK checks bind receipts only. Full backup payload hash is a separate opt-in. '
                           'Qualification-reused calibration tasks have a documented local backup and no worker ACK; '
                           'do not relabel this as a durable worker ACK. Neither proves independent failure domains, retention or future availability.'})
        return self.tables['live_source_backup']

    def leakage(self):
        p=self.data['prompts'].copy()
        p['normalized_text']=p.text.map(lambda x:re.sub(r'\s+',' ',unicodedata.normalize('NFC',x)).strip())
        self.check('prompt_text_uid', all(hashlib.sha256(r.normalized_text.encode()).hexdigest()==r.prompt_uid
                                       for r in p.itertuples()))
        rows=[]
        for field in ['prompt_uid','normalized_text','group_id','parent_group','image_id','template_family_id']:
            if field not in p: continue
            valid=p.dropna(subset=[field]); valid=valid[valid[field].astype(str).str.len()>0]
            overlap=valid.groupby(field).split.nunique()
            bad=set(overlap[overlap>1].index)
            for r in valid[valid[field].isin(bad)].itertuples():
                rows.append({'field':field,'value':str(getattr(r,field)),'prompt_uid':r.prompt_uid,'split':r.split})
            if field in ['prompt_uid','normalized_text','group_id']:
                self.check('split_disjoint_'+field,not bad,f'{len(bad)} overlapping identities')
        self.table('split_overlap',pd.DataFrame(rows,columns=['field','value','prompt_uid','split']))
        external=[]
        paths=list(self.test_manifests)
        if self.manifests and not paths:paths=[self.manifests/'test-locked.jsonl']
        for path in paths:
            # Manifest-only inspection: no test score/image/model access.
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                r=json.loads(line);r=r.get('prompt',r)
                external.append(r)
        if external:
            test=pd.DataFrame(external)
            for field in ['prompt_uid','group_id']:
                if field in test:
                    shared=set(p[field].dropna())&set(test[field].dropna())
                    self.check('locked_test_overlap_'+field,not shared,f'{len(shared)} overlaps')
        self.record('test_leakage_scope', {'test_manifest_rows':len(external),
                    'status':'CHECKED_MANIFEST_ONLY' if external else 'UNAVAILABLE_NO_EXTERNAL_TEST_MANIFEST',
                    'note':'No external test manifest => cannot assert disjointness from held-out test.'})
        return self.tables['split_overlap']

    def validity(self):
        rows=[]
        for name in ['previews','delivered']:
            f=self.data[name]
            valid=f.valid.fillna(False)
            finite=np.isfinite(f.ir)&np.isfinite(f.pickscore_raw)
            self.check(name+'_valid_flags',not (valid&(~f.img_ok|~f.latent_ok|~finite|f.pixel_sha256.isna())).any())
            self.check(name+'_invalid_reasons',not ((~valid)&f.failure_code.isna()).any())
            for key,g in f.groupby(KEY[:1],dropna=False):
                rows.append(dict(table=name,profile_id=key[0],rows=len(g),valid=int(g.valid.sum()),
                                 invalid=int((~g.valid).sum()),finite_scores=int((np.isfinite(g.ir)&np.isfinite(g.pickscore_raw)).sum())))
            self.table(name+'_failure_reasons',f.loc[~valid].groupby(['profile_id','failure_code'],dropna=False).size().rename('rows').reset_index())
        self.table('validity',pd.DataFrame(rows))
        task_valid=self.preview.groupby(KEY).valid.all() & self.final.groupby(KEY).valid.all()
        statuses=self.tasks.set_index(KEY).status
        expected=task_valid.map({True:'complete_valid',False:'complete_with_model_failure'})
        self.check('task_status_matches_rows',statuses.reindex(expected.index).eq(expected).all())
        f=self.final
        self.table('score_distribution',describe(f[f.valid],['profile_id','split','source'],['ir','pickscore_raw']))
        self.table('score_tails',f.sort_values('ir').groupby('profile_id',group_keys=False).head(25))
        return self.tables['validity']

    def diversity(self):
        f=self.final
        rows=[]; curves=[]
        for key,g in f.groupby(KEY,sort=True):
            g=g.sort_values('seed_idx')
            row=dict(zip(KEY,key));row.update(split=g.split.iloc[0],source=g.source.iloc[0],
                        group_id=g.group_id.iloc[0],valid_seeds=int(g.valid.sum()),
                        unique_pixels=g.pixel_sha256.nunique())
            for metric in ['ir','pickscore_raw']:
                row.update({metric+'_std':np.nan,metric+'_range':np.nan,
                            metric+'_best':np.nan,metric+'_mean':np.nan})
                a=g.loc[g.valid,metric].to_numpy(float);a=a[np.isfinite(a)]
                if not len(a):continue
                row.update({metric+'_std':float(np.std(a,ddof=1)) if len(a)>1 else np.nan,
                            metric+'_range':float(np.ptp(a)),metric+'_best':float(a.max()),
                            metric+'_mean':float(a.mean())})
                for k in range(1,len(a)+1):
                    # Exact expectation over all C(8,k) subsets, independent of seed ordering.
                    best=float(np.mean([max(a[list(s)]) for s in itertools.combinations(range(len(a)),k)]))
                    curves.append({**dict(zip(KEY,key)),'metric':metric,'k':k,'expected_best':best,
                                   'gain_vs_random_one':best-float(a.mean()),'all_eight_valid':len(a)==8})
            rows.append(row)
        self.task_scores=pd.DataFrame(rows)
        self.table('task_diversity',self.task_scores)
        self.table('seed_budget_exact',pd.DataFrame(curves,columns=KEY+[
            'metric','k','expected_best','gain_vs_random_one','all_eight_valid']))
        self.table('seed_position',describe(f[f.valid],['profile_id','split','seed_idx'],['ir','pickscore_raw']))
        # Pixel repetition is a collapse signal, never a semantic-diversity measurement.
        self.table('duplicate_final_pixels',f[f.pixel_sha256.notna()&f.pixel_sha256.duplicated(keep=False)]
                   .sort_values('pixel_sha256'))
        return self.task_scores

    def trajectories(self):
        v=self.preview.merge(self.final[SEED+['ir','pickscore_raw','valid']],on=SEED,
                             suffixes=('','_final'),validate='many_to_one')
        rows=[]; selection=[]
        for (profile,split,j),g in v.groupby(['profile_id','split','j']):
            for metric in ['ir','pickscore_raw']:
                good=g[g.valid&g.valid_final].copy()
                good=good[np.isfinite(good[metric])&np.isfinite(good[metric+'_final'])]
                # Mean per-task rank agreement; pooled correlations confound prompt difficulty.
                correlations=[]
                for _,a in good.groupby(KEY):
                    x,y=a[metric].rank(),a[metric+'_final'].rank()
                    if x.nunique()>1 and y.nunique()>1:correlations.append(x.corr(y))
                rows.append(dict(profile_id=profile,split=split,j=j,metric=metric,rows=len(g),
                                 valid_pairs=len(good),mean_preview=good[metric].mean(),
                                 mean_final=good[metric+'_final'].mean(),
                                 mean_within_task_spearman=np.mean(correlations) if correlations else None,
                                 defined_rank_tasks=len(correlations)))
        self.table('trajectory_ranking',pd.DataFrame(rows))
        for key,g in v.groupby(KEY):
            if not (len(g)==64 and g.valid.all() and g.valid_final.all()):continue
            for j,a in g.groupby('j'):
                for metric in ['ir','pickscore_raw']:
                    ordered=a.sort_values([metric,'seed_idx'],ascending=[False,True])
                    oracle=float(a[metric+'_final'].max())
                    best_seed=int(a.sort_values([metric+'_final','seed_idx'],ascending=[False,True]).iloc[0].seed_idx)
                    for k in [1,2,4]:
                        kept=ordered.head(k)
                        selection.append({**dict(zip(KEY,key)),'j':int(j),'metric':metric,'k':k,
                            'oracle_survives':best_seed in set(kept.seed_idx),
                            'final_regret':oracle-float(kept[metric+'_final'].max())})
        self.table('pruning_diagnostics',pd.DataFrame(selection,columns=KEY+[
            'j','metric','k','oracle_survives','final_regret']))
        # Non-monotonic trajectories are measured, never assumed to be collector failures.
        sorted_v=self.preview.sort_values(SEED+['j']).copy()
        sorted_v['delta_ir']=sorted_v.groupby(SEED).ir.diff()
        self.table('trajectory_increments',describe(sorted_v,['profile_id','split','j'],['delta_ir']))
        f=self.preview
        self.check('progress_contract',np.allclose(f.progress,f.j/8))
        for feature in ['drift','cfg_gap']:
            available=f[feature+'_available'].fillna(False)
            self.check(feature+'_availability',not (available&~np.isfinite(f[feature])).any()
                       and not ((~available)&f[feature].notna()).any())
        self.check('first_drift_missing',not f.loc[f.j==1,'drift_available'].any())
        self.check('lightning_cfg_missing',not f.loc[f.profile_id==PROFILES[1],'cfg_gap_available'].any())
        self.table('feature_distribution',describe(f,['profile_id','split','j'],['drift','cfg_gap']))
        return self.tables['trajectory_ranking']

    def paired_analysis(self):
        s=self.task_scores
        rows=[]
        for profile,g in s[s.split=='dev'].groupby('profile_id'):
            for metric in ['ir_best','ir_mean','pickscore_raw_best']:
                pivot=g.pivot(index=['prompt_uid','group_id'],columns='bank_id',values=metric)
                if not {0,1}<=set(pivot):continue
                pivot=pivot[[0,1]].dropna()
                diff=(pivot[1]-pivot[0]).rename('delta').reset_index()
                clusters=diff.groupby('group_id',dropna=False).delta.mean()
                rows.append(dict(profile_id=profile,contrast='bank1_minus_bank0',metric=metric,
                                 paired_prompts=len(diff),**bootstrap_mean(clusters,seed=self.seed,repeats=self.bootstrap)))
        for split,g in s.groupby('split'):
            for metric in ['ir_best','ir_mean','pickscore_raw_best']:
                pivot=g.pivot(index=['prompt_uid','bank_id','group_id'],columns='profile_id',values=metric)
                if not set(PROFILES)<=set(pivot):continue
                pivot=pivot[PROFILES].dropna()
                diff=(pivot[PROFILES[1]]-pivot[PROFILES[0]]).rename('delta').reset_index()
                clusters=diff.groupby('group_id',dropna=False).delta.mean()
                rows.append(dict(split=split,contrast='lightning_minus_hyper',metric=metric,
                                 paired_prompts=diff.prompt_uid.nunique(),
                                 **bootstrap_mean(clusters,seed=self.seed,repeats=self.bootstrap)))
        self.table('paired_cluster_intervals',pd.DataFrame(rows))
        self.record('statistical_scope',{'unit':'source group_id (averaged prompts/banks within group)',
                    'estimand':'equally weighted group mean paired difference, not seed-weighted mean',
                    'ci':'95% percentile cluster bootstrap; exploratory, no p-values or significance labels',
                    'multiplicity':'many descriptive contrasts; no confirmatory claim',
                    'cross_profile':'paired prompts reduce mix confounding; different resolution/model remains confounded',
                    'unknown':'QA selection bias, out-of-domain generalization and online causal effects remain untested'})
        return self.tables['paired_cluster_intervals']

    def scorer_and_feature_diagnostics(self):
        rows=[]; associations=[]
        for key,g in self.final.groupby(KEY):
            a=g[g.valid].dropna(subset=['ir','pickscore_raw'])
            if len(a)<2:continue
            ir=a.sort_values(['ir','seed_idx'],ascending=[False,True]).iloc[0]
            ps=a.sort_values(['pickscore_raw','seed_idx'],ascending=[False,True]).iloc[0]
            rows.append({**dict(zip(KEY,key)),'split':a.split.iloc[0],'source':a.source.iloc[0],
                         'valid_seeds':len(a),'winner_agrees':ir.seed_idx==ps.seed_idx,
                         'ir_regret_of_ps_winner':float(ir.ir-ps.ir),
                         'ps_regret_of_ir_winner':float(ps.pickscore_raw-ir.pickscore_raw),
                         'within_task_spearman':a.ir.rank().corr(a.pickscore_raw.rank())})
        self.table('scorer_agreement',pd.DataFrame(rows,columns=KEY+['split','source','valid_seeds',
            'winner_agrees','ir_regret_of_ps_winner','ps_regret_of_ir_winner','within_task_spearman']))
        v=self.preview.merge(self.final[SEED+['ir','valid']],on=SEED,suffixes=('','_final'),validate='many_to_one')
        for key,g in v.groupby(KEY+['j']):
            for feature in ['drift','cfg_gap']:
                a=g[g.valid&g.valid_final&g[feature+'_available']].dropna(subset=[feature,'ir_final'])
                defined=len(a)>2 and a[feature].nunique()>1 and a.ir_final.nunique()>1
                associations.append({**dict(zip(KEY+['j'],key)),'feature':feature,'available_seeds':len(a),
                    'within_task_spearman_to_final_ir':a[feature].rank().corr(a.ir_final.rank()) if defined else None})
        self.table('feature_rank_associations',pd.DataFrame(associations))
        return self.tables['scorer_agreement']

    def offline_policies(self):
        """Fixed legal two-stage masks. Terminal oracle is evaluator-only, never actor input.

        Operation counts are logical workload units, NOT measured GPU latency or money.
        Selection uses only current active-scoring preview and alive candidate IDs.
        """
        rows=[];rng=np.random.default_rng(self.seed)
        finals={key:g.set_index('seed_idx') for key,g in self.final.groupby(KEY)}
        for key,g in self.preview.groupby(KEY):
            end=finals[key]
            # Explicit complete-case diagnostic, with exclusions reported rather than imputed.
            if not g.valid.all() or not end.valid.all():continue
            stage={j:a.set_index('seed_idx') for j,a in g.groupby('j')}
            for active in ['ir','pickscore_raw']:
                oracle=float(end[active].max())
                for policy in ['full8','fixed_2_5','fixed_3_6','fixed_4_7','random_3_6']:
                    alive=list(range(8));forward_units=0;preview_score_units=0
                    schedule={'full8':(), 'fixed_2_5':(2,5), 'fixed_3_6':(3,6),
                              'fixed_4_7':(4,7),'random_3_6':(3,6)}[policy]
                    for j in range(1,9):
                        forward_units+=len(alive)
                        if j not in schedule:continue
                        preview_score_units+=len(alive)
                        if policy.startswith('random'):
                            alive=sorted(rng.choice(alive,len(alive)//2,replace=False).tolist())
                        else:
                            current=stage[j].loc[alive].reset_index()
                            alive=current.sort_values([active,'seed_idx'],ascending=[False,True]).head(len(alive)//2).seed_idx.tolist()
                    # Paid terminal choice among survivors only. Inactive scorer is diagnostic.
                    chosen=end.loc[alive].reset_index().sort_values([active,'seed_idx'],ascending=[False,True]).iloc[0]
                    rows.append({**dict(zip(KEY,key)),'split':end.split.iloc[0],'active':active,'policy':policy,
                                 'chosen_seed':int(chosen.seed_idx),'survivors':len(alive),
                                 'final_active_score':float(chosen[active]),'oracle_regret':oracle-float(chosen[active]),
                                 'delivered_ir':float(chosen.ir),'delivered_ps':float(chosen.pickscore_raw),
                                 'candidate_forward_units':forward_units,'preview_verify_units':preview_score_units,
                                 'final_verify_units':len(alive)})
        frame=pd.DataFrame(rows);self.table('offline_policy_diagnostics',frame)
        self.record('offline_policy_scope',{'included_tasks':len(frame)//10,'total_tasks':len(self.tasks),
                    'selection':'all-preview and delivered valid tasks only',
                    'random':'one fixed-seed masked trajectory per task/scorer; not an estimated random-policy expectation',
                    'cost':'logical candidate forward/score units, no GPU latency/savings prediction',
                    'equivalence':'stored full-batch trajectories approximate pruned execution; online equivalence unverified',
                    'split':'train/dev/calibration shown separately; no tuning or selection on test'})
        return frame

    def embeddings(self):
        p=self.data['prompts'].copy()
        order=[self.index[u] for u in p.prompt_uid]
        x=self.embedding[order].astype(np.float64)
        p['embedding_norm']=np.linalg.norm(x,axis=1)
        self.table('prompt_tokens',describe(p,['split','source'],
                   [n for n in ['original_tokens','used_tokens','embedding_norm'] if n in p]))
        if 'truncated' in p:
            self.table('truncation',p.groupby(['split','source'],dropna=False).truncated.agg(['size','sum','mean']).reset_index())
        # Exact nearest train neighbor for each dev/calibration prompt, bounded 256-row blocks.
        train=np.flatnonzero(p.split.eq('train').to_numpy());other=np.flatnonzero(~p.split.eq('train').to_numpy())
        rows=[]
        if len(train):
            for begin in range(0,len(other),256):
                ids=other[begin:begin+256];similarity=x[ids]@x[train].T
                top=similarity.argmax(axis=1)
                for i,k in enumerate(ids):
                    near=train[top[i]]
                    rows.append(dict(prompt_uid=p.iloc[k].prompt_uid,split=p.iloc[k].split,
                                     train_uid=p.iloc[near].prompt_uid,cosine=float(similarity[i,top[i]]),
                                     text=p.iloc[k].text,train_text=p.iloc[near].text))
        self.table('near_duplicate_train_neighbors',pd.DataFrame(rows,columns=[
            'prompt_uid','split','train_uid','cosine','text','train_text']))
        centered=x-x.mean(axis=0)
        _,singular,axes=np.linalg.svd(centered,full_matrices=False)
        projection=centered@axes[:2].T
        for i in range(projection.shape[1]):p['pc'+str(i+1)]=projection[:,i]
        self.table('embedding_pca',p)
        self.record('embedding_spectrum',{'variance':(singular**2/max(len(x)-1,1)).tolist(),
                    'note':'Cosine proximity is an investigation signal; it does not prove semantic leakage.'})
        return self.tables['prompt_tokens']

    def native_and_provenance(self):
        rows=[];timings=[];provenance=[];qa=[]
        sampled={profile:0 for profile in PROFILES}
        for t in self.tasks.sort_values('task_id').itertuples():
            base=child(self.root,'provenance/'+t.task_id)
            task,prov=read_json(base/'task.json'),read_json(base/'provenance.json')
            actual=read_json(base/'timings.json')
            timings.append(dict(task_id=t.task_id,profile_id=t.profile_id,split=t.split,**actual))
            runtime=prov.get('runtime',{})
            provenance.append(dict(task_id=t.task_id,profile_id=t.profile_id,
                registry_hash=prov.get('registry_hash'),storage_cast=prov.get('storage_cast'),
                runtime_identity=json.dumps(runtime,sort_keys=True),
                encoder_identity=json.dumps({k:v for k,v in prov.get('encoder',{}).items()
                                             if k not in ['original_tokens','used_tokens','truncated']},sort_keys=True),
                scorer_identity=json.dumps(prov.get('scorers',{}),sort_keys=True)))
            self.check('task_metadata_'+t.task_id,task['task_id']==t.task_id and task['profile']==t.profile_id
                       and task['prompt']['prompt_uid']==t.prompt_uid and task['bank_id']==t.bank_id)
            candidates=self.data['candidates']
            selected=candidates[(candidates.profile_id==t.profile_id)&(candidates.prompt_uid==t.prompt_uid)&
                                (candidates.bank_id==t.bank_id)].sort_values('seed_idx')
            self.check('seed_identity_'+t.task_id,list(selected.seed_global)==task['seeds'])
            for s in selected.itertuples():
                msg=f'apsp-v2|offline|{s.prompt_uid}|{s.bank_id}|{s.seed_idx}'.encode()
                expected=int.from_bytes(hashlib.sha256(msg).digest()[:8],'little')&((1<<63)-1)
                self.check('seed_scheme_'+t.task_id+'_'+str(s.seed_idx),s.seed_global==expected)
            self.check('no_storage_cast_'+t.task_id,prov.get('storage_cast')=='none')
            for flag in ['qa_final','qa_preview']:
                qa.append(dict(task_id=t.task_id,profile_id=t.profile_id,split=t.split,kind=flag,
                               preselected=bool(task.get(flag,False))))
            latent=child(self.root,'latent/'+t.task_id+'/latents.safetensors')
            if not latent.exists():continue
            values=sampled[t.profile_id]<self.native_values
            meta,headers=tensor_header(latent,values=values)
            if values:sampled[t.profile_id]+=1
            native=prov.get('native_tensors',{})
            self.check('native_keys_'+t.task_id,{h['tensor'] for h in headers}==set(native))
            self.check('native_metadata_'+t.task_id,meta.get('schema')==SCHEMA and meta.get('storage_cast')=='none')
            for h in headers:
                expected=native.get(h['tensor'],{})
                dtype={'float16':'F16','float32':'F32','float64':'F64'}.get(expected.get('dtype'))
                h.update(task_id=t.task_id,profile_id=t.profile_id,values_checked=values)
                self.check('native_header_'+t.task_id+'_'+h['tensor'],h['dtype']==dtype and h['shape']==expected.get('shape'))
                if values:
                    self.check('native_value_hash_'+t.task_id+'_'+h['tensor'],h['value_sha256']==expected.get('sha256'))
                    self.check('native_finite_'+t.task_id+'_'+h['tensor'],h.get('finite_fraction')==1.)
                rows.append(h)
            required={'final',*[f'x0_{j:02d}' for j in range(1,9)]}
            self.check('required_native_tensors_'+t.task_id,required<=set(native))
            profile_spec=prov.get('profile',{})
            if 'height' in profile_spec and 'width' in profile_spec:
                shape=[8,4,profile_spec['height']//8,profile_spec['width']//8]
                self.check('native_profile_shape_'+t.task_id,all(h['shape']==shape for h in headers))
            expected_x0='F16' if t.profile_id==PROFILES[0] else 'F32'
            self.check('measured_native_dtype_recipe_'+t.task_id,
                       all(h['dtype']==(expected_x0 if h['tensor'].startswith('x0_') else 'F16') for h in headers))
        self.table('native_tensors',pd.DataFrame(rows))
        self.table('task_timings',pd.DataFrame(timings))
        self.table('provenance_identities',pd.DataFrame(provenance))
        self.table('qa_selection',pd.DataFrame(qa))
        if self.manifest['mode']=='core_latent_qa':
            self.check('native_task_coverage',len({r['task_id'] for r in rows})==len(self.tasks))
        identities=pd.DataFrame(provenance)
        for profile,g in identities.groupby('profile_id'):
            self.check('single_registry_'+profile,g.registry_hash.nunique(dropna=False)==1)
            runtimes=[json.loads(s) for s in g.runtime_identity]
            for field in ['source_hash','lock_sha256','seed_scheme']:
                observed={str(r.get(field)) for r in runtimes}
                self.check('single_runtime_'+profile+'_'+field,len(observed)==1,str(observed))
        self.record('native_scope',{'header_tensors':len(rows),'value_sampled_tasks':sampled,
                    'native_values_requested_per_profile':self.native_values,
                    'status':'HEADERS_AND_SAMPLED_VALUES' if sum(sampled.values()) else 'HEADERS_ONLY',
                    'note':'No GPU re-decode, no raw-source/backup revalidation, no semantic image QA.'})
        return self.tables['native_tensors']

    def qa_pixel_audit(self):
        """Audit all saved QA PNG pixels on CPU; does not regenerate missing images."""
        from PIL import Image
        rows=[]
        for name,frame in [('previews',self.preview),('delivered',self.final)]:
            for row in frame[frame.image_uri.notna()].itertuples():
                path=child(self.root,row.image_uri)
                with Image.open(path) as im:
                    rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
                actual=hashlib.sha256(rgb.tobytes()).hexdigest()
                ok=actual==row.pixel_sha256
                rows.append(dict(table=name,task_id=row.task_id,profile_id=row.profile_id,seed_idx=row.seed_idx,
                                 j=getattr(row,'j',None),image_uri=row.image_uri,width=rgb.shape[1],height=rgb.shape[0],
                                 pixel_hash_matches=ok,channel_std=float(rgb.std()),
                                 black_fraction=float((rgb<3).all(axis=-1).mean()),
                                 white_fraction=float((rgb>252).all(axis=-1).mean())))
                self.check('qa_pixels_'+row.image_uri,ok)
        frame=pd.DataFrame(rows)
        self.table('qa_pixel_integrity',frame)
        self.record('qa_pixel_scope',{'saved_images_checked':len(rows),
                    'note':'Pixel hashes and blankness heuristics only; no image regeneration, no aesthetic guarantee.'})
        return frame

    def costs_storage(self):
        t=self.tables['task_timings']
        columns=[c for c in t if c.endswith('_s')]
        self.table('timing_distribution',describe(t,['profile_id','split'],columns))
        costs=[]
        for path in sorted((self.root/'qualification').glob('*.json')):
            q=read_json(path)
            for r in q.get('cost_profile',{}).get('samples',[]):
                costs.append(dict(profile_id=path.stem,**r))
        self.table('qualification_cost_samples',pd.DataFrame(costs))
        files=self.tables['file_integrity'].copy();files['component']=files.file.str.split('/').str[0]
        self.table('storage_components',files.groupby('component').agg(files=('file','size'),bytes=('bytes','sum')).reset_index())
        actual=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.check('storage_80GB',actual<=80_000_000_000,f'{actual} decimal bytes')
        self.record('storage_and_cost_scope',{'export_bytes':actual,'GB_decimal':actual/1e9,'GiB':actual/2**30,
                    'qualification_cost_samples':len(costs),
                    'note':'Task wall sums are work, not fleet makespan. No GPU utilization/idle/ETA claim without scheduler telemetry. '
                           'Decode/IR/PickScore totals may overlap; do not sum blindly. ACK is not independent durability.'})
        return self.tables['timing_distribution']

    def plots(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return self.record('plot_status',{'status':'UNAVAILABLE_MATPLOTLIB','note':'Tables/HTML still complete; no automatic pip install.'})
        plt.rcParams.update({'figure.dpi':110,'savefig.dpi':160,'axes.grid':True,'grid.alpha':.2})
        def save(name,fig):
            fig.tight_layout();fig.savefig(self.out/(name+'.png'));plt.close(fig);self.figures.append(name+'.png')
        f=self.final
        for metric in ['ir','pickscore_raw']:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for profile,g in f[f.valid].groupby('profile_id'):
                a=np.sort(g[metric].dropna().to_numpy())
                axes[0].hist(a,bins=45,density=True,histtype='step',label=profile)
                axes[1].plot(a,np.arange(1,len(a)+1)/max(len(a),1),label=profile)
            axes[0].set(title=metric+' final density',xlabel=metric,ylabel='density')
            axes[1].set(title='Empirical CDF (seed rows, descriptive)',xlabel=metric,ylabel='ECDF')
            axes[0].legend(fontsize=7);save('scores_'+metric,fig)
        frame=self.tables['trajectory_ranking']
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for profile,g in frame[(frame.metric=='ir')&(frame.split=='dev')].groupby('profile_id'):
            axes[0].plot(g.j,g.mean_preview,marker='o',label=profile)
            axes[1].plot(g.j,g.mean_within_task_spearman,marker='o',label=profile)
        axes[0].set(title='Dev preview IR',xlabel='checkpoint j',ylabel='mean IR')
        axes[1].set(title='Within-task rank vs delivered',xlabel='checkpoint j',ylabel='mean Spearman')
        axes[0].legend(fontsize=7);save('trajectories',fig)
        b=self.tables['seed_budget_exact'];fig,ax=plt.subplots(figsize=(7,4))
        for profile,g in b[(b.metric=='ir')&b.all_eight_valid].groupby('profile_id'):
            a=g.groupby('k').gain_vs_random_one.mean();ax.plot(a.index,a.values,marker='o',label=profile)
        ax.set(title='Exact subset best-of-k headroom (all-valid tasks)',xlabel='k',ylabel='IR gain vs random one')
        ax.legend(fontsize=7);save('seed_budget',fig)
        p=self.tables['embedding_pca'];fig,ax=plt.subplots(figsize=(7,5))
        if {'pc1','pc2'}<=set(p):
            for split,g in p.groupby('split'):ax.scatter(g.pc1,g.pc2,s=8,alpha=.5,label=split)
            ax.legend();ax.set(title='Core384 PCA; descriptive only',xlabel='PC1',ylabel='PC2')
        save('embedding_pca',fig)
        prune=self.tables['pruning_diagnostics'];fig,axes=plt.subplots(1,2,figsize=(12,4))
        if not prune.empty:
            for (profile,k),g in prune[prune.metric=='ir'].groupby(['profile_id','k']):
                a=g.groupby('j')[['oracle_survives','final_regret']].mean()
                label=profile+' k='+str(k)
                axes[0].plot(a.index,a.oracle_survives,label=label)
                axes[1].plot(a.index,a.final_regret,label=label)
        axes[0].set(title='One-shot pruning diagnostic',xlabel='j',ylabel='oracle seed survival')
        axes[1].set(title='Privileged offline diagnostic',xlabel='j',ylabel='delivered IR regret')
        axes[0].legend(fontsize=6);save('pruning_headroom',fig)
        # Profile/split views accompany pooled distributions and privileged diagnostics.
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        s=self.task_scores
        for profile,g in s.groupby('profile_id'):
            axes[0].hist(g.ir_std.dropna(),bins=30,histtype='step',density=True,label=profile)
            axes[1].hist(g.ir_range.dropna(),bins=30,histtype='step',density=True,label=profile)
        axes[0].set(title='Within-task seed diversity',xlabel='IR std',ylabel='density')
        axes[1].set(title='Within-task seed spread',xlabel='IR max-min',ylabel='density')
        axes[0].legend(fontsize=7);save('seed_diversity',fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,feature in zip(axes,['drift','cfg_gap']):
            for profile,g in self.preview.groupby('profile_id'):
                a=g[g[feature+'_available']].groupby('j')[feature].median()
                if len(a):ax.plot(a.index,a.values,marker='o',label=profile)
            ax.set(title=feature+' (available only)',xlabel='j',ylabel='median');ax.legend(fontsize=7)
        save('feature_dynamics',fig)
        fig,ax=plt.subplots(figsize=(8,4))
        pairs=self.tables['paired_cluster_intervals']
        if len(pairs):
            for i,r in enumerate(pairs.itertuples()):
                if pd.notna(r.mean) and pd.notna(r.lo) and pd.notna(r.hi):
                    ax.plot([r.lo,r.hi],[i,i],color='steelblue');ax.scatter([r.mean],[i],s=14)
            ax.set_yticks(range(len(pairs)),[str(r.contrast)+' '+str(r.metric) for r in pairs.itertuples()],fontsize=6)
        ax.axvline(0,color='gray',lw=1);ax.set(title='Exploratory paired group CIs (different reward units)',xlabel='paired difference')
        save('paired_cluster_intervals',fig)
        timing=self.tables['task_timings'];cols=[c for c in timing if c.endswith('_s')]
        if cols:
            fig,ax=plt.subplots(figsize=(10,4))
            timing.groupby('profile_id')[cols].median().T.plot.bar(ax=ax)
            ax.set(title='Median task stages; may overlap, do not sum blindly',ylabel='seconds')
            save('timing_stages',fig)
        agreement=self.tables['scorer_agreement'];fig,ax=plt.subplots(figsize=(8,4))
        if len(agreement):
            agreement.groupby(['profile_id','split']).winner_agrees.mean().plot.bar(ax=ax)
        ax.set(title='IR vs PickScore winner agreement',ylabel='fraction');save('scorer_winners',fig)
        return self.record('plot_status',{'status':'RENDERED','figures':self.figures})

    def qa_gallery(self, max_images=48):
        """Only existing preselected QA images; no cherry-picked full-data quality estimate."""
        rows=self.final[self.final.image_uri.notna()].sort_values(['profile_id','ir','task_id','seed_idx'])
        selected=[]
        for _,g in rows.groupby('profile_id'):
            n=max(1,max_images//(max(rows.profile_id.nunique(),1)*3))
            middle=g.iloc[max(0,len(g)//2-n//2):len(g)//2+(n+1)//2]
            chosen=pd.concat([g.head(n),middle,g.tail(n)]).drop_duplicates(SEED)
            selected.extend(chosen.to_dict('records'))
        import base64
        cards=[]
        for row in selected[:max_images]:
            path=child(self.root,row['image_uri'])
            if not path.is_file():continue
            from PIL import Image
            import io
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((320,320))
                buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=80)
            data=base64.b64encode(buffer.getvalue()).decode()
            caption=f"{row['profile_id']} | seed {row['seed_idx']} | IR {row['ir']:.4f} | PS {row['pickscore_raw']:.4f}"
            cards.append('<figure><img src="data:image/jpeg;base64,'+data+'"><figcaption>'+html.escape(caption)+
                         '<p>'+html.escape(str(row['text']))+'</p></figcaption></figure>')
        (self.out/'qa_gallery.html').write_text('<meta charset="utf-8"><h1>Preselected QA subset only</h1>'
             '<p>Low/middle/high scores within saved QA; no population quality estimate. '
             'Manual labels: alignment, counting, composition, artifacts, scorer mismatch.</p>'+''.join(cards),encoding='utf-8')
        self.table('qa_manual_review',pd.DataFrame([{**{k:r[k] for k in SEED+['ir','pickscore_raw','image_uri','text']},
                    'alignment':'','counting':'','artifacts':'','scorer_mismatch':'','reviewer_note':''} for r in selected[:max_images]]))
        return {'gallery_images':len(cards),'path':str(self.out/'qa_gallery.html')}

    def finish(self):
        self.table('checks',pd.DataFrame(self.checks))
        errors=[c for c in self.checks if c['status']=='FAIL' and c['severity']=='error']
        summary={'verification_status':'ANALYZED' if not errors else 'ANALYZED_WITH_FAILURES',
                 'failed_checks':len(errors),'check_count':len(self.checks),'input':str(self.root),
                 'warning_count':sum(c['status']=='WARN' for c in self.checks),
                 'full_file_hash':self.full_hash,'native_value_sample_per_profile':self.native_values,
                 'execution_host':platform.node(),
                 'note':'Results describe this supplied export only. '
                        'No full-release PASS under partial hash; no controller-training/online efficacy conclusion.'}
        if 'tasks' in self.data:
            summary['row_counts']={name:len(frame) for name,frame in self.data.items()}
            summary['split_task_counts']=self.data['tasks'].split.value_counts().to_dict()
        self.record('summary',summary)
        passport={'stage':'exploratory post-collection EDA','material':'immutable exported Lite2 Core/native/QA metadata',
                  'raw_data_access':'read-only supplied root','analysis_output':str(self.out),
                  'external_services':'none','status':summary['verification_status'],
                  'scope_limits':['finite validity is not aesthetic QA','correlation is not causation',
                    'paired groups, not independent seed rows','test manifest only; no test scores',
                    'preselected QA is not a representative quality survey','approximate replay is not online speedup']}
        self.record('material_passport',passport)
        notes=['## Material Passport',json.dumps(passport,ensure_ascii=False,indent=2),
               '## Audit outcome',json.dumps(summary,ensure_ascii=False,indent=2),
               '## Reading the evidence',
               '1. Start with checks.csv, coverage.csv and canonical_plan_coverage.csv. Missing tasks are not zero scores.',
               '2. Compare rewards within profile/source/split, then paired prompt/group contrasts. IR and PickScore have different scales.',
               '3. Seed budget is exact subset headroom, not a deployable controller. j=8 is terminal diagnostic, not a legal preview VERIFY.',
               '4. Native headers alone do not prove finite tensors or lossless provenance. Enable sampled native values/full hash explicitly.',
               '5. QA gallery only covers saved preselected calibration images. Review manual CSV before quality claims.',
               '6. ACK confirms a historical copy, not current independent durability. No raw/backup paths are opened implicitly.',
               '7. Bring summary.json, checks.csv, coverage.csv, paired_cluster_intervals.csv, trajectory_ranking.csv, '
               'seed_budget_exact.csv and qualification_cost_samples.csv back for interpretation.',
               '## Statistical fallacy review (11 considerations)',
               '- Simpson: stratified source/split tables accompany aggregates; inspect reversals manually.',
               '- Ecological: group-level contrasts do not imply per-image effects.',
               '- Berkson/selection: all-valid pruning diagnostics and QA selection are explicitly conditional.',
               '- Collider: no adjusted causal model or causal interpretation is fitted.',
               '- Base rates: validity and failure tables retain numerator/denominator.',
               '- Multiple comparisons: intervals are exploratory; no significance screening.',
               '- P-hacking: fixed tables/checkpoints; no outcome-selected threshold tuning.',
               '- Regression to mean: tail gallery is descriptive; no before/after improvement claim.',
               '- Survivorship: planned tasks, failures and missing rows remain visible.',
               '- Pseudoreplication: paired CI resamples source groups, never seed rows.',
               '- Correlation/causation: feature/reward and embedding proximity do not establish causality.',
               'These are analysis design safeguards, not a claim that all statistical pathologies were ruled out.']
        (self.out/'REPORT.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')
        blocks=['<meta charset="utf-8"><title>APSP Lite2 EDA</title>',
                '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto}table{border-collapse:collapse;font-size:12px}'
                'td,th{border:1px solid #ddd;padding:5px}pre{white-space:pre-wrap}img{max-width:100%}</style>',
                '<h1>APSP Lite2 EDA</h1><pre>'+html.escape('\n\n'.join(notes))+'</pre>']
        for name in self.figures:blocks.append('<h2>'+html.escape(name)+'</h2><img src="'+name+'">')
        for name,frame in self.tables.items():
            blocks.extend(['<h2>'+html.escape(name)+'</h2>',
                           '<p>'+str(len(frame))+' rows; full table: '+name+'.csv</p>',
                           frame.head(60).to_html(index=False,escape=True)])
        blocks.append('<p><a href="qa_gallery.html">QA gallery (preselected subset)</a></p>')
        (self.out/'report.html').write_text('\n'.join(blocks),encoding='utf-8')
        hashes={p.name:sha256(p) for p in self.out.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
        write_json(self.out/'ARTIFACT_SHA256.json',hashes)
        return summary

    def run(self):
        for name in ['integrity','load','coverage','source_and_backup','leakage','validity','diversity','trajectories',
                     'paired_analysis','scorer_and_feature_diagnostics','offline_policies',
                     'embeddings','native_and_provenance','qa_pixel_audit','costs_storage','plots','qa_gallery']:
            print('EDA:',name,flush=True)
            getattr(self,name)()
        return self.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',required=True);p.add_argument('--out',required=True)
    p.add_argument('--full-hash',action='store_true')
    p.add_argument('--native-values',type=int,default=0,help='Tasks per profile; 0=headers only')
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--test-manifest',action='append',default=[])
    p.add_argument('--collection');p.add_argument('--manifests');p.add_argument('--backup')
    p.add_argument('--backup-full-hash',action='store_true')
    a=p.parse_args()
    if a.native_values<0 or a.bootstrap<100:p.error('native-values>=0, bootstrap>=100 required')
    audit=Audit(a.export,a.out,full_hash=a.full_hash,native_values=a.native_values,
                bootstrap=a.bootstrap,seed=a.seed,test_manifests=a.test_manifest,
                collection=a.collection,manifests=a.manifests,backup=a.backup,
                backup_full_hash=a.backup_full_hash)
    try:result=audit.run()
    except Exception:
        audit.finish()
        raise
    print(json.dumps(result,ensure_ascii=False))
    if result['failed_checks']:raise SystemExit(2)


if __name__=='__main__':main()
