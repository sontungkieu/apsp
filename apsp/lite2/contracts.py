from __future__ import annotations

import copy
import json
from pathlib import Path

from apsp.lite.contracts import canonical_hash, file_hash, noise_seed, normalize, uid
from apsp.lite.performance import ExecutionPolicy

SCHEMA = 'apsp-core-lite2-v1'
PROFILES = ('hyper_sd15_cfg8_512', 'sdxl_lightning8_1024')
ROOT = Path(__file__).resolve().parents[2]
QUOTAS = {'train': {'diffusiondb': 1000, 'coco_captions': 400, 'structured': 600},
          'dev': {'diffusiondb': 100, 'coco_captions': 40, 'structured': 60}}
ENCODER = {'repo': 'sentence-transformers/all-MiniLM-L6-v2',
           'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
           'width': 384, 'max_length': 256, 'pooling': 'attention_mask_mean',
           'normalization': 'l2', 'storage_dtype': 'float32'}
FEATURES = {'progress': 'j/8',
            'drift': 'rms(x0[j]-x0[j-1])/max(rms(x0[j-1]),1e-12); first unavailable',
            'cfg_gap': 'rms(eps_text-eps_uncond) before CFG; unavailable when guidance<=1',
            'availability': 'missing/nonfinite values are null with *_available=false'}


def registry():
    old = json.loads((ROOT / 'env/lite/profiles.json').read_text(encoding='utf-8'))
    lightning = copy.deepcopy(old['profiles']['sdxl_lightning8'])
    hyper = {'repo': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
             'revision': '451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
             'pipeline': 'StableDiffusionPipeline', 'scheduler': 'DDIMScheduler',
             'timestep_spacing': 'trailing', 'dtype': 'float16', 'height': 512, 'width': 512,
             'steps': 8, 'checkpoints': 8, 'guidance': 7.5, 'cfg_branches_available': True,
             'disable_safety_checker': True,
             'adapter': {'kind': 'lora', 'repo': 'ByteDance/Hyper-SD',
                         'revision': 'bc08d970a87c74c71209491d64e3525845698863',
                         'filename': 'Hyper-SD15-8steps-CFG-lora.safetensors', 'fuse_scale': 1.0}}
    return {'schema': SCHEMA, 'profiles': dict(zip(PROFILES, (hyper, lightning))),
            'scorers': old['scorers'], 'prompt_encoder': ENCODER, 'features': FEATURES,
            'negative_prompt': '', 'eta': 0.0, 's_churn': 0.0,
            'native_storage': 'one tensor per checkpoint, original dtype; no storage cast'}


def policy(optimized=True):
    return ExecutionPolicy(decode_streams=4 if optimized else 1, score_batch=8 if optimized else 1,
                           fuse_checkpoints=optimized, png_compress_level=1,
                           cache_prompt_embedding=True)


def task(prompt, profile, bank, *, qa_final=False, qa_preview=False):
    if profile not in PROFILES or prompt['split'] not in ('train', 'dev', 'calibration'):
        raise ValueError('Unknown profile or forbidden test split')
    if uid(prompt['text']) != prompt['prompt_uid'] or normalize(prompt['text']) != prompt['text']:
        raise ValueError('Prompt identity differs from normalized text')
    if bank not in (0, 1) or (prompt['split'] == 'train' and bank != 0):
        raise ValueError('Lite2 train has bank0 only')
    spec = registry()['profiles'][profile]
    return {'schema': SCHEMA, 'task_id': canonical_hash([SCHEMA, profile, prompt['prompt_uid'], bank]),
            'profile': profile, 'profile_hash': canonical_hash(spec), 'prompt': prompt, 'bank_id': bank,
            'seeds': [noise_seed(prompt['prompt_uid'], bank, s) for s in range(8)],
            'phase': 1 if prompt['split'] != 'train' or prompt['train_rank'] <= 1000 else 2,
            'qa_final': qa_final, 'qa_preview': qa_preview}


def check_task(t):
    expected = task(t['prompt'], t['profile'], t['bank_id'],
                    qa_final=t.get('qa_final', False), qa_preview=t.get('qa_preview', False))
    if t != expected:
        raise ValueError('Task differs from frozen Lite2 manifest contract')


def identity(execution):
    from apsp.lite.collector import fingerprint, runtime_identity
    fp = fingerprint(execution)
    fp.update(schema=SCHEMA, registry_sha256=canonical_hash(registry()))
    return fp, canonical_hash(runtime_identity(fp))


def source_hash():
    return canonical_hash({p.relative_to(ROOT).as_posix(): file_hash(p)
                           for p in sorted((ROOT / 'apsp').rglob('*.py'))})
