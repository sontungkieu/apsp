from pathlib import Path
from collections import Counter
import pytest
from apsp.lite.contracts import uid
from apsp.lite.prompts import write_jsonl,read_jsonl
from apsp.lite2.manifests import build


def pool(root):
    counts={'train':{'diffusiondb':1000,'coco_captions':400,'structured':600},
            'dev':{'diffusiondb':150,'coco_captions':60,'structured':90},
            'calibration':{'structured':50},'test_id':{'structured':300},'test_shift':{'parti':200}}
    root.mkdir()
    for split,sources in counts.items():
        rows=[]
        for source,count in sources.items():
            for i in range(count):
                text=f'{split} {source} item {i}'
                rows.append(dict(text=text,text_raw=text,prompt_uid=uid(text),group_id=uid(text),
                                 source=source,source_revision='fixture-rev',split=split,
                                 train_rank=len(rows)+1 if split=='train' else None))
        write_jsonl(root/f'{split}.jsonl',rows)


def test_v1_pool_maps_exact_lite2_counts_and_qa_union(tmp_path):
    pool(tmp_path/'v1')
    report=build(tmp_path/'v1',tmp_path/'v2')
    assert report['main_tasks']==4800 and report['main_preview_rows']==307200
    main=read_jsonl(tmp_path/'v2/main.jsonl')
    assert Counter((t['prompt']['split'],t['profile']) for t in main)=={
        ('train','hyper_sd15_cfg8_512'):2000,('train','sdxl_lightning8_1024'):2000,
        ('dev','hyper_sd15_cfg8_512'):400,('dev','sdxl_lightning8_1024'):400}
    assert sum(t['prompt']['split']=='train' for t in main)==4000
    assert all(t['bank_id']==0 for t in main if t['prompt']['split']=='train')
    qual=read_jsonl(tmp_path/'v2/qualification.jsonl')
    calib=read_jsonl(tmp_path/'v2/calibration.jsonl')
    assert len(qual)==40 and len(calib)==120 and len({t['task_id'] for t in calib})==120
    assert sum(t['qa_final'] for t in calib)*8==800
    assert sum(t['qa_preview'] for t in calib)*8*3==480
    assert not any(t['prompt']['split'].startswith('test') for t in main+calib)


def test_cross_split_group_rejected(tmp_path):
    pool(tmp_path/'v1')
    import json
    path=tmp_path/'v1/dev.jsonl'
    rows=read_jsonl(path)
    rows[0]['group_id']=read_jsonl(tmp_path/'v1/train.jsonl')[0]['group_id']
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match='crosses splits'):build(tmp_path/'v1',tmp_path/'v2')
