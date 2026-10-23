"""EDA regression: execute all notebook cells on an actual synthetic V2 export.

These fixtures test analysis correctness, never claim company-data results.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from test_core import core_task  # Reuse the sealed, public V2 contract fixture.
from apsp.lite2.export import export_release

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('eda_v2',ROOT/'env/lite2/eda_v2.py')
eda=importlib.util.module_from_spec(spec)
sys.modules['eda_v2']=eda
spec.loader.exec_module(eda)


@pytest.fixture
def eda_export(core_task,tmp_path):
    export=tmp_path/'export'
    export_release([core_task],export,mode='core_qa',expected_tasks=1)
    return export


def reseal(export):
    manifest=eda.read_json(export/'RELEASE_MANIFEST.json')
    manifest['files_sha256']={p.relative_to(export).as_posix():eda.sha256(p)
        for p in export.rglob('*') if p.is_file() and p.name not in ['RELEASE_MANIFEST.json','SHA256SUMS']}
    eda.write_json(export/'RELEASE_MANIFEST.json',manifest)
    files={**manifest['files_sha256'],'RELEASE_MANIFEST.json':eda.sha256(export/'RELEASE_MANIFEST.json')}
    (export/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(files.items())))


def company_layout(core_task,tmp_path):
    task=eda.read_json(core_task/'task.json');tid=task['task_id']
    collection=tmp_path/'collection';backup=tmp_path/'backup';manifests=collection/'manifests'
    manifests.mkdir(parents=True)
    source=collection/'main/tasks'/tid;mirror=backup/'main/tasks'/tid
    shutil.copytree(core_task,source);shutil.copytree(core_task,mirror)
    ack=collection/'main/acks'/(tid+'.json');ack.parent.mkdir(parents=True)
    eda.write_json(ack,dict(task_id=tid,backup=str(mirror.resolve()),complete_sha256=eda.sha256(source/'COMPLETE.json')))
    (manifests/'main.jsonl').write_text(json.dumps(task)+'\n')
    (manifests/'calibration.jsonl').write_text('')
    (manifests/'test-locked.jsonl').write_text('')
    eda.write_json(manifests/'manifest.json',{'files_sha256':{p.name:eda.sha256(p) for p in manifests.glob('*.jsonl')}})
    return collection,backup,manifests


def test_end_to_end_core_only_exports_all_reports(eda_export,tmp_path):
    a=eda.Audit(eda_export,tmp_path/'eda',bootstrap=100,expected_main=1)
    result=a.run()
    # The one-task fixture intentionally fails canonical full collection coverage.
    assert result['failed_checks']==1
    assert result['verification_status']=='ANALYZED_WITH_FAILURES'
    assert a.tables['seed_budget_exact'].query("metric=='ir' and k==8").iloc[0].expected_best==pytest.approx(.7)
    assert len(a.tables['offline_policy_diagnostics'])==10
    assert a.sections['native_scope']['header_tensors']==0
    assert a.sections['test_leakage_scope']['test_manifest_rows']==0
    assert (a.out/'report.html').is_file() and (a.out/'REPORT.md').is_file()
    assert not (eda_export/'report.html').exists()
    for name,digest in eda.read_json(a.out/'ARTIFACT_SHA256.json').items():
        assert eda.sha256(a.out/name)==digest


def test_hash_tamper_stops_before_loading(eda_export,tmp_path):
    with (eda_export/'core/tasks.parquet').open('ab') as f:f.write(b'tamper')
    a=eda.Audit(eda_export,tmp_path/'eda')
    with pytest.raises(ValueError,match='Integrity failed'):a.integrity()
    assert not a.data
    assert eda.read_json(a.out/'summary.json')['failed_checks']>0


def test_per_task_checkpoint_corruption_not_hidden_by_global_row_count(eda_export,tmp_path):
    path=eda_export/'core/previews.parquet';v=pd.read_parquet(path)
    v.loc[0,'j']=2;v.to_parquet(path,index=False);reseal(eda_export)
    a=eda.Audit(eda_export,tmp_path/'eda');a.integrity()
    with pytest.raises(ValueError,match='structural'):a.load()
    assert any(r['check']=='exact_checkpoint_indices' and r['status']=='FAIL' for r in a.checks)


def test_output_never_inside_release_or_backup(eda_export,tmp_path):
    with pytest.raises(ValueError,match='outside'):eda.Audit(eda_export,eda_export/'analysis')
    with pytest.raises(ValueError,match='outside'):eda.Audit(eda_export,tmp_path,backup=tmp_path/'backup')


def test_safe_references_reject_parent_and_absolute(tmp_path):
    for name in ['../secret',str((tmp_path/'secret').resolve())]:
        with pytest.raises(ValueError):eda.child(tmp_path,name)


def test_native_header_and_byte_hash_without_torch(tmp_path):
    x=np.arange(128,dtype=np.float32).reshape(2,4,4,4)
    path=tmp_path/'native.safetensors';save_file({'x0_01':x},str(path),metadata={'schema':eda.SCHEMA})
    meta,rows=eda.tensor_header(path,values=True)
    assert meta['schema']==eda.SCHEMA and rows[0]['dtype']=='F32'
    assert rows[0]['shape']==[2,4,4,4]
    import hashlib
    assert rows[0]['value_sha256']==hashlib.sha256(x.tobytes()).hexdigest()
    assert rows[0]['finite_fraction']==1


def test_native_nonfinite_is_visible(tmp_path):
    path=tmp_path/'native.safetensors';save_file({'final':np.array([np.nan,1],dtype=np.float16)},str(path))
    assert eda.tensor_header(path,values=True)[1][0]['finite_fraction']==.5


def test_cluster_ci_insufficient_and_deterministic():
    assert eda.bootstrap_mean([1.])['lo'] is None
    a=eda.bootstrap_mean([1,2,3],repeats=100)
    assert a==eda.bootstrap_mean([1,2,3],repeats=100)
    assert a['n']==3 and a['lo']<=a['mean']<=a['hi']


def test_live_ack_backup_and_frozen_plan(core_task,eda_export,tmp_path):
    collection,backup,manifests=company_layout(core_task,tmp_path)
    a=eda.Audit(eda_export,tmp_path/'eda',collection=collection,backup=backup,
                manifests=manifests,backup_full_hash=True)
    a.integrity();a.load();f=a.source_and_backup()
    assert f.iloc[0].ack=='PASS' and f.iloc[0].backup_payload=='PASS'
    assert all(r['status']=='PASS' for r in a.checks)
    backup_file=next((backup/'main/tasks').glob('*/task.json'));backup_file.write_text('tampered')
    b=eda.Audit(eda_export,tmp_path/'eda-again',collection=collection,backup=backup,
                manifests=manifests,backup_full_hash=True)
    b.integrity();b.load();b.source_and_backup()
    assert any(r['check'].startswith('backup_payload_') and r['status']=='FAIL' for r in b.checks)


def test_all_notebook_code_cells_execute_on_fixture(core_task,eda_export,tmp_path,monkeypatch):
    collection,backup,manifests=company_layout(core_task,tmp_path)
    env={'APSP_EDA_EXPORT':eda_export,'APSP_EDA_OUT':tmp_path/'notebook-out',
         'APSP_EDA_SOURCE':ROOT,'APSP_EDA_COLLECTION':collection,'APSP_EDA_BACKUP':backup,
         'APSP_EDA_MANIFESTS':manifests,'APSP_EDA_FULL_HASH':'1','APSP_EDA_NATIVE_VALUES':'0'}
    for k,v in env.items():monkeypatch.setenv(k,str(v))
    notebook=eda.read_json(ROOT/'notebooks/APSP_LITE2_EDA_COMPANY.ipynb')
    namespace={'__name__':'__notebook__'}
    count=0
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            exec(compile(''.join(cell['source']),f'<notebook-cell-{i}>','exec'),namespace)
            count+=1
    assert count>=20
    assert (tmp_path/'notebook-out/report.html').is_file()
    assert namespace['A'].sections['summary']['failed_checks']==2  # 4800 target + canonical profile counts


def test_exact_subset_curve_and_mask_policy(eda_export,tmp_path):
    a=eda.Audit(eda_export,tmp_path/'eda');a.integrity();a.load();a.diversity()
    f=a.tables['seed_budget_exact'].query("metric=='ir'").sort_values('k')
    assert f.iloc[0].expected_best==pytest.approx(.35)
    assert f.iloc[-1].expected_best==pytest.approx(.7)
    assert (f.expected_best.diff().dropna()>=0).all()
    p=a.offline_policies()
    assert p.query("policy=='full8'").candidate_forward_units.eq(64).all()
    assert p.query("policy=='fixed_3_6'").candidate_forward_units.eq(40).all()
    assert p.query("policy=='fixed_3_6'").preview_verify_units.eq(12).all()
    assert p.query("policy=='fixed_3_6'").survivors.eq(2).all()


def test_two_profiles_dev_banks_are_paired_at_group_level(core_task,tmp_path):
    from apsp.lite2.contracts import task,registry
    from apsp.lite.contracts import uid
    from apsp.lite.storage import commit_task
    paths=[]
    for split,text in [('train','Train mug.'),('dev','Dev mug one.'),('dev','Dev mug two.')]:
        prompt=dict(text=text,text_raw=text,prompt_uid=uid(text),split=split,source='structured',
                    source_revision='synthetic-only',group_id=uid(text),train_rank=1 if split=='train' else None)
        for profile in eda.PROFILES:
            for bank in ([0] if split=='train' else [0,1]):
                t=task(prompt,profile,bank);dest=tmp_path/'multi/tasks'/t['task_id']
                stage=tmp_path/('stage-'+t['task_id']);shutil.copytree(core_task,stage)
                (stage/'COMPLETE.json').unlink()
                eda.write_json(stage/'task.json',t)
                prov=eda.read_json(stage/'provenance.json');prov['profile']=registry()['profiles'][profile]
                eda.write_json(stage/'provenance.json',prov)
                pd.DataFrame([prompt]).to_parquet(stage/'prompts.parquet',index=False)
                offset=(.1 if profile==eda.PROFILES[1] else 0)+bank*.2
                for name in ['candidates','previews','delivered','tasks']:
                    f=pd.read_parquet(stage/(name+'.parquet'))
                    f['profile_id']=profile;f['prompt_uid']=prompt['prompt_uid'];f['bank_id']=bank
                    if name=='candidates':f['seed_global']=t['seeds']
                    if name in ['previews','delivered']:f['ir']=f.ir+offset
                    if name=='tasks':f['task_id']=t['task_id'];f['split']=split
                    f.to_parquet(stage/(name+'.parquet'),index=False)
                dest.parent.mkdir(parents=True,exist_ok=True)
                commit_task(stage,dest,{'schema':eda.SCHEMA,'task_id':t['task_id'],'contract_hash':'fixture'})
                paths.append(dest)
    export=tmp_path/'paired-export';export_release(paths,export,mode='core_qa',expected_tasks=10)
    a=eda.Audit(export,tmp_path/'paired-eda',bootstrap=100)
    a.integrity();a.load();a.diversity();a.paired_analysis();a.embeddings()
    paired=a.tables['paired_cluster_intervals']
    bank=paired[paired.contrast=='bank1_minus_bank0']
    assert bank.query("metric=='ir_best'")['mean'].to_numpy()==pytest.approx([.2,.2])
    assert bank.query("metric=='ir_best'").n.eq(2).all()  # 2 groups, never 32 seeds.
    model=paired[(paired.contrast=='lightning_minus_hyper')&(paired.split=='dev')&(paired.metric=='ir_mean')]
    assert model.iloc[0]['mean']==pytest.approx(.1) and model.iloc[0].n==2
    assert len(a.tables['near_duplicate_train_neighbors'])==2


def test_every_saved_qa_pixel_hash_checked(core_task,tmp_path):
    from PIL import Image
    from apsp.lite.storage import files_manifest
    import hashlib
    path=core_task/'qa/final-s0.png';path.parent.mkdir()
    pixels=np.full((8,8,3),128,dtype=np.uint8);Image.fromarray(pixels).save(path)
    f=pd.read_parquet(core_task/'delivered.parquet');f.loc[0,'image_uri']='qa/final-s0.png'
    f.loc[0,'pixel_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
    f.to_parquet(core_task/'delivered.parquet',index=False)
    marker=eda.read_json(core_task/'COMPLETE.json');marker['files_sha256']=files_manifest(core_task)
    eda.write_json(core_task/'COMPLETE.json',marker)
    export=tmp_path/'qa-export';export_release([core_task],export,mode='core_qa',expected_tasks=1)
    a=eda.Audit(export,tmp_path/'qa-eda');a.integrity();a.load()
    assert a.qa_pixel_audit().iloc[0].pixel_hash_matches
    Image.fromarray(np.zeros_like(pixels)).save(export/f.loc[0,'image_uri'].replace('qa/','qa/'+core_task.name+'/'))
    assert not a.qa_pixel_audit().iloc[0].pixel_hash_matches


def test_qualification_reuse_is_not_a_missing_worker_ack(core_task,tmp_path):
    from apsp.lite2.contracts import task
    from apsp.lite.storage import files_manifest
    original=eda.read_json(core_task/'task.json')
    prompt={**original['prompt'],'split':'calibration','train_rank':None}
    t=task(prompt,original['profile'],0)
    eda.write_json(core_task/'task.json',t)
    pd.DataFrame([prompt]).to_parquet(core_task/'prompts.parquet',index=False)
    f=pd.read_parquet(core_task/'tasks.parquet');f['split']='calibration'
    f.to_parquet(core_task/'tasks.parquet',index=False)
    marker=eda.read_json(core_task/'COMPLETE.json');marker['files_sha256']=files_manifest(core_task)
    eda.write_json(core_task/'COMPLETE.json',marker)
    collection=tmp_path/'collection';backup=tmp_path/'backup';backup.mkdir()
    base=collection/'qualification'/original['profile']
    qsource=base/'fast/tasks'/t['task_id'];qbackup=base/'backup'/t['task_id']
    shutil.copytree(core_task,qsource);shutil.copytree(core_task,qbackup)
    export=tmp_path/'reuse-export'
    export_release([qsource],export,mode='core_qa',include_calibration=True,expected_tasks=0)
    a=eda.Audit(export,tmp_path/'reuse-eda',collection=collection,backup=backup,backup_full_hash=True)
    a.integrity();a.load();row=a.source_and_backup().iloc[0]
    assert row.source_marker=='PASS' and row.backup_marker=='PASS' and row.backup_payload=='PASS'
    assert row.source_kind=='qualification_reuse' and row.backup_kind=='qualification_local_copy'
    assert row.ack=='NOT_EMITTED_QUALIFICATION_REUSE'
    assert not any(c['status']=='FAIL' for c in a.checks)
