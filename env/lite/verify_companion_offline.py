"""CPU loader/readiness check against only the restored companion cache; network blocked.

This validates tokenizer/config/shard resolution and the common prompt encoder.
It does not repeat generator/scorer GPU qualification or prove cross-host parity.
"""
import argparse
import json
import os
import socket
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    os.environ.update(HF_HOME=str(root/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
    sys.path.insert(0, str(root/'source'))
    def blocked(*args, **kwargs): raise RuntimeError('network disabled for companion offline check')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoConfig, AutoProcessor, CLIPTokenizer, T5TokenizerFast
    from diffusers import DiffusionPipeline
    from safetensors import safe_open
    from apsp.gpu.scoring import pinned_bert_dir
    from apsp.data.prompts import _clip_functions
    from apsp.lite.contracts import canonical_hash, file_hash
    registry = json.loads((root/'source/env/lite/profiles.json').read_text())
    meta = json.loads((root/'COMPANION_MANIFEST.json').read_text())
    source_hash = canonical_hash({p.relative_to(root/'source').as_posix(): file_hash(p) for p in sorted((root/'source/apsp').rglob('*.py'))})
    assert source_hash == meta['qualified_core_source_sha256']
    resolved = {}
    for name, p in registry['profiles'].items():
        cache = Path(snapshot_download(p['repo'], revision=p['revision'], local_files_only=True))
        assert cache.is_relative_to(root/'hf')
        config = DiffusionPipeline.load_config(p['repo'], revision=p['revision'], local_files_only=True)
        assert config['_class_name'] == p['pipeline']
        tokenizers = []
        for sub in ('tokenizer','tokenizer_2'):
            if (cache/sub).is_dir():
                cls = T5TokenizerFast if (cache/sub/'spiece.model').exists() else CLIPTokenizer
                tokenizer = cls.from_pretrained(str(cache/sub), local_files_only=True)
                assert len(tokenizer('A ceramic mug')['input_ids']) > 0
                tokenizers.append(sub)
        adapter = p.get('adapter')
        if adapter:
            assert Path(hf_hub_download(adapter['repo'], adapter['filename'], revision=adapter['revision'], local_files_only=True)).is_relative_to(root/'hf')
        resolved[name] = {'config': True, 'tokenizers': tokenizers, 'adapter': bool(adapter)}
    ps = registry['scorers']['pickscore']
    AutoConfig.from_pretrained(ps['model_repo'], revision=ps['model_revision'], local_files_only=True)
    processor = AutoProcessor.from_pretrained(ps['processor_repo'], revision=ps['processor_revision'], local_files_only=True)
    assert len(processor(text=['A ceramic mug'], return_tensors='pt')['input_ids']) == 1
    ir = registry['scorers']['imagereward']
    pinned_bert_dir(ir['tokenizer_revision'], local_files_only=True)
    for name in ('ImageReward.pt','med_config.json'):
        hf_hub_download('THUDM/ImageReward', name, revision=ir['revision'], local_files_only=True)
    safetensors_files, tensor_keys, shard_indices = 0, 0, 0
    for p in (root/'hf/hub').rglob('*.safetensors'):
        with safe_open(str(p), framework='pt', device='cpu') as f: tensor_keys += len(list(f.keys()))
        safetensors_files += 1
    for p in (root/'hf/hub').rglob('*.safetensors.index.json'):
        index = json.loads(p.read_text())
        assert all((p.parent/name).is_file() for name in set(index['weight_map'].values()))
        shard_indices += 1
    pe = registry['prompt_encoder']
    count, embed = _clip_functions(pe['repo'], pe['revision'], device='cpu')
    embeddings = embed(['A ceramic mug'], bs=1)
    assert embeddings.shape == (1,768)
    import numpy as np
    assert np.isfinite(embeddings).all()
    prepared = root/'data/lite-v1/prepared/manifests'
    assert sum(1 for _ in (prepared/'tasks.jsonl').open()) == 13800
    result = {'pass': True, 'stage': 'companion_cpu_offline_loader_readiness', 'network': 'blocked_by_socket_patch_and_hf_offline', 'cache_root': str(root/'hf'), 'source_commit': meta['source_commit'], 'qualified_core_source_sha256': source_hash, 'profiles': resolved, 'scorers': 'config_processor_pinned_paths_resolved', 'safetensors_files': safetensors_files, 'safetensors_tensor_keys': tensor_keys, 'shard_indices': shard_indices, 'common_encoder_forward_shape': list(embeddings.shape), 'tasks': 13800, 'full_generator_scorer_gpu_rerun': False, 'sdxl50_cross_container_parity': 'not_claimed'}
    (root/'OFFLINE_LOADER_RECEIPT.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPANION_OFFLINE_JSON='+json.dumps(result), flush=True)


if __name__ == '__main__': main()
