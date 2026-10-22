import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from apsp.lite.contracts import uid
from apsp.lite.storage import commit_task
from apsp.lite2.contracts import SCHEMA, ENCODER, FEATURES, canonical_hash, registry, task
from apsp.lite2.validate import validate_task
from apsp.lite2.replay import CoreEpisode, baselines
from apsp.lite2.export import export_release, validate_export
from apsp.lite2.pack import pack, restore


@pytest.fixture
def core_task(tmp_path):
    prompt = dict(text='A red mug.', text_raw='A red mug.', prompt_uid=uid('A red mug.'),
                  split='train', source='structured', source_revision='fixture', group_id='fixture', train_rank=1)
    t = task(prompt, 'sdxl_lightning8_1024', 0)
    stage, dest = tmp_path/'stage', tmp_path/'tasks'/t['task_id']
    stage.mkdir(); dest.parent.mkdir()
    (stage/'task.json').write_text(json.dumps(t))
    prov = {'schema': SCHEMA, 'registry_hash': canonical_hash(registry()),
            'profile': registry()['profiles'][t['profile']], 'storage_cast': 'none',
            'feature_contract': FEATURES, 'encoder': ENCODER,
            'scorers': {'imagereward': {'fixture': True}, 'pickscore': {'fixture': True}}}
    (stage/'provenance.json').write_text(json.dumps(prov)); (stage/'timings.json').write_text('{}')
    pd.DataFrame([prompt]).to_parquet(stage/'prompts.parquet', index=False)
    key = dict(profile_id=t['profile'], prompt_uid=prompt['prompt_uid'], bank_id=0)
    candidates = [{**key, 'seed_idx': s, 'seed_global': t['seeds'][s]} for s in range(8)]
    previews = [{**key, 'seed_idx': s, 'j': j, 'forward_idx': j, 'progress': j/8,
                 'drift': None if j==1 else .1, 'drift_available': j!=1,
                 'cfg_gap': None, 'cfg_gap_available': False, 'ir': s*.1, 'pickscore_raw': s+.2,
                 'img_ok': True, 'latent_ok': True, 'valid': True, 'failure_code': None,
                 'pixel_sha256': 'a'*64, 'image_uri': None} for j in range(1,9) for s in range(8)]
    delivered = [{**key,'seed_idx':s,'ir':s*.1,'pickscore_raw':s+.2,'img_ok':True,
                  'latent_ok':True,'valid':True,'failure_code':None,'pixel_sha256':'b'*64,
                  'image_uri':None,'final_latent_uri':None} for s in range(8)]
    rows = [{**key,'task_id':t['task_id'],'split':'train','status':'complete_valid'}]
    for name,data in [('candidates',candidates),('previews',previews),('delivered',delivered),('tasks',rows)]:
        pd.DataFrame(data).to_parquet(stage/(name+'.parquet'),index=False)
    emb = np.zeros((1,384),dtype=np.float32); emb[0,0]=1
    save_file({'embedding':emb},str(stage/'prompt_features.safetensors'))
    commit_task(stage,dest,{'schema':SCHEMA,'contract_hash':'fixture','task_id':t['task_id']})
    return dest


def test_core_export_replay_and_streaming_restore(core_task,tmp_path):
    assert validate_task(core_task)['pass']
    out=tmp_path/'export'
    report=export_release([core_task],out,mode='core_qa',expected_tasks=1)
    assert report['coverage']=='complete' and report['native_latent_status'].startswith('not_received')
    assert baselines(out/'core')['episodes']==3
    assert not list(out.rglob('*.png')) and not (out/'latent').exists()
    archive=pack(out,tmp_path/'parts',part_bytes=2048)
    assert len(archive['parts'])>1
    assert restore(tmp_path/'parts',tmp_path/'restored')['pass']
    assert baselines(tmp_path/'restored/core')['episodes']==3


def test_export_rejects_duplicate_logical_tasks(core_task,tmp_path):
    with pytest.raises(ValueError,match='Duplicate'):
        export_release([core_task,core_task],tmp_path/'export',mode='core_qa')


def test_restore_detects_corrupt_part(core_task,tmp_path):
    export_release([core_task],tmp_path/'export',mode='core_qa')
    report=pack(tmp_path/'export',tmp_path/'parts',part_bytes=2048)
    path=tmp_path/'parts'/report['parts'][0]['file'];path.write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):
        restore(tmp_path/'parts',tmp_path/'restored')


def test_actor_no_future_inactive_or_pruned_score():
    rows=[dict(j=j,seed_idx=s,ir=s+j,pickscore_raw=999999,progress=j/8,
               drift=None,drift_available=False,cfg_gap=None,cfg_gap_available=False,
               future_sentinel='MUST_NOT_LEAK') for j in range(1,9) for s in range(8)]
    final=[dict(seed_idx=s,ir=-999999) for s in range(8)]
    env=CoreEpisode(rows,final,[0.]*384)
    assert not env.view().paid and not env.view().features
    with pytest.raises(ValueError,match='hidden'):env.final()
    env.advance();view=env.view()
    assert not view.paid and all('ir' not in row and 'pickscore_raw' not in row and 'future_sentinel' not in row for row in view.features.values())
    env.verify();env.prune([0,1,2,3]);env.advance()
    assert set(env.view().features)=={0,1,2,3} and not env.view().paid
    with pytest.raises(ValueError):env.prune([0,1])


def test_test_sampling_and_train_bank1_are_forbidden():
    p=dict(text='x',prompt_uid=uid('x'),split='test_id',train_rank=None)
    with pytest.raises(ValueError):task(p,'hyper_sd15_cfg8_512',0)
    p.update(split='train',train_rank=1)
    with pytest.raises(ValueError):task(p,'hyper_sd15_cfg8_512',1)


def test_unsealed_or_modified_core_fails(core_task):
    with (core_task/'task.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='checksum'):validate_task(core_task)


def test_planned_missing_and_failed_infra_are_not_counted_complete(core_task,tmp_path):
    from apsp.lite.prompts import write_jsonl
    original=json.loads((core_task/'task.json').read_text())
    extra=task({**original['prompt'],'text':'Another mug.','text_raw':'Another mug.',
                'prompt_uid':uid('Another mug.'),'train_rank':2},original['profile'],0)
    planned=tmp_path/'planned.jsonl';write_jsonl(planned,[original,extra])
    failure_root=tmp_path/'worker';(failure_root/'failures').mkdir(parents=True)
    (failure_root/'failures/attempt.json').write_text(json.dumps({'task_id':extra['task_id'],'attempt':3}))
    out=tmp_path/'export'
    report=export_release([core_task],out,mode='core_qa',expected_tasks=2,
                          planned_manifests=[planned],failures_roots=[failure_root])
    status=pd.read_parquet(out/'core/task_plan.parquet').set_index('task_id')
    assert status.loc[extra['task_id'],'status']=='failed_infra'
    assert status.loc[original['task_id'],'status']=='complete_valid'
    assert not status.durable_ack_verified.any()
    assert report['coverage']=='partial_or_unspecified' and report['main_tasks']==1
    assert (out/'DATASET_CARD_LITE2.md').is_file() and (out/'COLLECTION_REPORT_LITE2.md').is_file()
