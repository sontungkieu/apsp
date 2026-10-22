"""Native-dtype eight-seed collector. All output belongs to a new Lite2 namespace."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from apsp.lite.storage import atomic_json, commit_task, verify_complete
from .contracts import ENCODER, FEATURES, SCHEMA, canonical_hash, check_task, registry


def pixel_hash(image):
    import numpy as np
    return hashlib.sha256(np.asarray(image.convert('RGB'), dtype=np.uint8).tobytes()).hexdigest()


def tensor_hash(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def frames_at(path):
    from safetensors.torch import load_file
    z = load_file(str(Path(path) / 'latents.safetensors'))
    return [z[f'x0_{j:02d}'] for j in range(1, 9)] + [z['final']]


def score_frames(runtime, text, frames):
    """Fusion changes decode input batching only; stored tensors retain individual dtypes."""
    if len(frames) != 9 or any(z.shape[0] != 8 for z in frames):
        raise ValueError('Need exactly eight preview frames and one final, each K=8')
    if runtime.execution.fuse_checkpoints:
        fused = runtime.decode_score(text, runtime.torch.cat(frames, 0))
        return [tuple([v[n * 8:(n + 1) * 8] for v in fused[:4]] +
                      [fused[4] if n == 0 else {k: 0. for k in fused[4]}]) for n in range(9)]
    return [runtime.decode_score(text, frame) for frame in frames]


def collect_task(runtime, encoder, t, root, fp, *, keep_latents=True, keep_initial=False, capture=None):
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from apsp.lite.runtime import CleanTap

    check_task(t)
    if runtime.spec != registry()['profiles'][t['profile']]:
        raise ValueError('Runtime profile differs from Lite2 manifest')
    root = Path(root)
    if root.name in ('release-v1', 'backup-v1'):
        raise ValueError('Never append Lite2 into legacy v1 roots')
    (root / 'tasks').mkdir(parents=True, exist_ok=True)
    contract = canonical_hash({'task': t, 'runtime': {k: v for k, v in fp.items()
                                                   if k not in ('hostname', 'source_commit')},
                               'keep_latents': keep_latents, 'keep_initial': keep_initial})
    dest = root / 'tasks' / t['task_id']
    if verify_complete(dest, contract):
        return dest
    stage = root / 'staging' / (t['task_id'] + '-' + str(time.time_ns()))
    stage.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    noise = [torch.randn((1, runtime.C, runtime.h, runtime.w), dtype=torch.float32,
                         generator=torch.Generator('cpu').manual_seed(s)) for s in t['seeds']]
    initial = torch.cat(noise).to(runtime.dtype).to(runtime.device)
    tap = CleanTap(runtime.pipe.scheduler, range(1, 9), capture_on_gpu=runtime.execution.capture_on_gpu)
    gap = []

    def cfg_hook(module, args, output):
        z = output.sample if hasattr(output, 'sample') else output[0]
        if z.shape[0] != 16:
            raise ValueError('CFG uncond/text ordering requires 2*K model outputs')
        uncond, text = z.detach().float().chunk(2)
        gap.append((text - uncond).square().flatten(1).mean(1).sqrt())

    hook = runtime.pipe.unet.register_forward_hook(cfg_hook) if runtime.spec['cfg_branches_available'] else None
    before = runtime.denoiser_calls
    args = dict(prompt=t['prompt']['text'], negative_prompt='', num_images_per_prompt=8,
                num_inference_steps=8, guidance_scale=runtime.spec['guidance'],
                height=runtime.H, width=runtime.W, latents=initial, output_type='latent')
    if runtime.spec['scheduler'] == 'DDIMScheduler':
        args['eta'] = 0.0
    runtime.sync()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            final = runtime.pipe(**args).images
    finally:
        runtime.pipe.scheduler.step = tap.original
        if hook is not None:
            hook.remove()
    runtime.sync()
    generation_s = time.perf_counter() - started
    if runtime.denoiser_calls - before != 8 or tap.n != 8 or sorted(tap.x0) != list(range(1, 9)):
        raise ValueError('Eight-forward/checkpoint contract failed')
    if hook is not None and len(gap) != 8:
        raise ValueError('CFG feature hook missed a forward')
    tensors = {f'x0_{j:02d}': z.detach().contiguous().cpu() for j, z in tap.x0.items()}
    tensors['final'] = final.detach().contiguous().cpu()
    actual_initial = tap.initial.detach().contiguous().cpu()
    if keep_initial or t['qa_final']:
        tensors['initial'] = actual_initial
    native = {name: {'dtype': str(z.dtype).removeprefix('torch.'), 'shape': list(z.shape),
                     'sha256': tensor_hash(z)} for name, z in tensors.items()}
    if keep_latents:
        save_file(tensors, str(stage / 'latents.safetensors'),
                  metadata={'schema': SCHEMA, 'native_tensors': json.dumps(native, sort_keys=True),
                            'storage_cast': 'none', 'layout': 'unpacked_NCHW',
                            'vae_scaling_factor': str(runtime.pipe.vae.config.scaling_factor),
                            'vae_shift_factor': str(getattr(runtime.pipe.vae.config, 'shift_factor', None))})
    frames = [tensors[f'x0_{j:02d}'] for j in range(1, 9)] + [tensors['final']]
    results = score_frames(runtime, t['prompt']['text'], frames)
    if capture is not None:
        capture.extend(results)
    embedding, encoder_meta = encoder.encode(t['prompt']['text'])
    save_file({'embedding': embedding}, str(stage / 'prompt_features.safetensors'))
    prompt = t['prompt']
    prompt_row = {k: prompt.get(k) for k in ('prompt_uid', 'text', 'text_raw', 'source', 'source_revision',
                                             'source_row_id', 'parent_group', 'group_id', 'split',
                                             'train_rank', 'primary_category', 'template_family_id', 'image_id')}
    prompt_row.update(truncated=encoder_meta['truncated'], original_tokens=encoder_meta['original_tokens'],
                      used_tokens=encoder_meta['used_tokens'], metadata_json=json.dumps(prompt, ensure_ascii=False, sort_keys=True))
    pd.DataFrame([prompt_row]).to_parquet(stage / 'prompts.parquet', index=False)
    candidates, previews, delivered = [], [], []
    gaps = [z.cpu().tolist() for z in gap]
    for s in range(8):
        candidates.append({'profile_id': t['profile'], 'prompt_uid': prompt['prompt_uid'],
                           'bank_id': t['bank_id'], 'seed_idx': s, 'seed_global': t['seeds'][s],
                           'initial_sha256': tensor_hash(actual_initial[s]), 'cpu_noise_sha256': tensor_hash(noise[s]),
                           'rng': 'apsp-v2-sha256-cpu'})
    for i, result in enumerate(results):
        images, flags, irs, pss, _ = result
        if any(len(v) != 8 for v in result[:4]):
            raise ValueError('Scoring lost candidate rows')
        delta = None
        if 0 < i < 8:
            prev, now = frames[i - 1].float(), frames[i].float()
            delta = ((now - prev).square().flatten(1).mean(1).sqrt() /
                     prev.square().flatten(1).mean(1).sqrt().clamp_min(1e-12)).tolist()
        for s in range(8):
            row = {k: candidates[s][k] for k in ('profile_id', 'prompt_uid', 'bank_id', 'seed_idx')}
            row.update(ir=irs[s], pickscore_raw=pss[s], **flags[s],
                       pixel_sha256=None if images[s] is None else pixel_hash(images[s]))
            save_png = (i == 8 and t['qa_final']) or (i in (0, 3, 7) and t['qa_preview'])
            image_uri = None
            if save_png and images[s] is not None:
                relative = Path('qa') / (f'final-s{s}.png' if i == 8 else f'j{i+1}-s{s}.png')
                (stage / 'qa').mkdir(exist_ok=True)
                images[s].save(stage / relative, compress_level=runtime.execution.png_compress_level)
                image_uri = relative.as_posix()
            if i == 8:
                delivered.append({**row, 'image_uri': image_uri,
                                  'final_latent_uri': 'latents.safetensors#final' if keep_latents else None})
            else:
                d = None if delta is None or not math.isfinite(delta[s]) else delta[s]
                g = gaps[i][s] if gaps else None
                g = g if g is not None and math.isfinite(g) else None
                previews.append({**row, 'j': i + 1, 'progress': (i + 1) / 8,
                                 'drift': d, 'drift_available': d is not None,
                                 'cfg_gap': g, 'cfg_gap_available': g is not None,
                                 'image_uri': image_uri, 'native_dtype': native[f'x0_{i+1:02d}']['dtype'],
                                 **{k: v for k, v in tap.meta[i + 1].items() if k != 'preview_origin'}})
    for name, rows in [('candidates', candidates), ('previews', previews), ('delivered', delivered)]:
        pd.DataFrame(rows).to_parquet(stage / (name + '.parquet'), index=False)
    valid = all(row['valid'] for row in previews + delivered)
    status = 'complete_valid' if valid else 'complete_with_model_failure'
    pd.DataFrame([{'task_id': t['task_id'], 'profile_id': t['profile'], 'prompt_uid': prompt['prompt_uid'],
                   'bank_id': t['bank_id'], 'split': prompt['split'], 'status': status,
                   'n_candidates': 8, 'n_previews': 64}]).to_parquet(stage / 'tasks.parquet', index=False)
    atomic_json(stage / 'task.json', t)
    atomic_json(stage / 'provenance.json', {'schema': SCHEMA, 'runtime': fp,
                'profile': runtime.spec, 'adapter': runtime.adapter_provenance,
                'scorers': runtime.scorer_provenance, 'encoder': encoder_meta, 'feature_contract': FEATURES,
                'registry_hash': canonical_hash(registry()), 'native_tensors': native, 'storage_cast': 'none',
                'scheduler': dict(runtime.pipe.scheduler.config), 'checkpoint_meta': tap.meta,
                'vae_dtype': str(runtime.vae_dtype), 'decode_contract': 'native -> float32 scale/shift -> VAE dtype',
                'latent_retained': keep_latents, 'initial_retained': 'initial' in tensors})
    atomic_json(stage / 'timings.json', {'generation_s': generation_s,
                'wall_before_commit_s': time.perf_counter() - begin,
                'denoiser_calls': runtime.denoiser_calls - before,
                **{k: sum(r[4].get(k, 0.) for r in results) for k in results[0][4]}})
    # Validate before making anything visible as COMPLETE. Temporary staging is preserved on failure.
    from .validate import validate_task
    validate_task(stage, sealed=False, check_latents=keep_latents)
    commit_task(stage, dest, {'schema': SCHEMA, 'contract_hash': contract, 'task_id': t['task_id'],
                             'status': status, 'n_candidates': 8, 'n_previews': 64})
    return dest
