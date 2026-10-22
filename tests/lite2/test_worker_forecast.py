import importlib.util
import json
import sys
from pathlib import Path
import pytest

from apsp.lite.prompts import write_jsonl
from apsp.lite.contracts import uid
from apsp.lite2.contracts import task,registry,canonical_hash,policy,SCHEMA
from apsp.lite2.worker import initialize,main


def test_bad_qualification_rejected_before_model_constructor(tmp_path,monkeypatch):
    prompt=dict(text='A mug.',text_raw='A mug.',prompt_uid=uid('A mug.'),split='train',train_rank=1)
    manifest=tmp_path/'tasks.jsonl';write_jsonl(manifest,[task(prompt,'sdxl_lightning8_1024',0)])
    initialize(tmp_path/'out',manifest)
    receipt={'pass':True,'performance_enabled':True,'schema':SCHEMA,'profile':'sdxl_lightning8_1024',
             'execution':policy().to_dict(),'runtime_identity_hash':'wrong','registry_hash':canonical_hash(registry())}
    q=tmp_path/'q.json';q.write_text(json.dumps(receipt))
    monkeypatch.setattr('apsp.lite2.worker.identity',lambda p:({},'correct'))
    monkeypatch.setattr(sys,'argv',['worker','--tasks',str(manifest),'--out',str(tmp_path/'out'),
        '--backup-root',str(tmp_path/'backup'),'--profile','sdxl_lightning8_1024','--qualification',str(q)])
    with pytest.raises(ValueError,match='qualification'):main()
    assert not list((tmp_path/'out/tasks').iterdir())


def load_forecast():
    path=Path(__file__).resolve().parents[2]/'env/lite2/forecast.py'
    spec=importlib.util.spec_from_file_location('lite2_forecast_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.forecast


def test_forecast_is_native_bytes_not_compression_or_legacy_ratio(tmp_path):
    for name in registry()['profiles']:
        rows=[{'bytes_by_file':{'latents.safetensors':12_000_000,'prompts.parquet':2000,
                               'final-s0.png':10000,'j1-s0.png':10000},
               'native_tensors':{}} for _ in range(20)]
        (tmp_path/(name+'.json')).write_text(json.dumps({'pass':True,'performance_enabled':True,
                                                       'tasks':rows,'cost_profile':{}}))
    report=load_forecast()(tmp_path)
    assert report['native_upper_bytes']==4920*12_000_000
    assert report['recommended_mode']=='core_latent_qa'
    for path in tmp_path.glob('*.json'):
        data=json.loads(path.read_text())
        for row in data['tasks']:row['bytes_by_file']['latents.safetensors']=25_000_000
        path.write_text(json.dumps(data))
    report=load_forecast()(tmp_path)
    assert report['recommended_mode']=='core_qa' and report['native_upper_bytes']>80_000_000_000
