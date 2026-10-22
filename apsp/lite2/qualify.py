"""Real 20-task/profile qualification of native storage, Core384, parity, export and restore."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from apsp.lite.prompts import read_jsonl
from apsp.lite.storage import atomic_json, backup_task
from apsp.lite.performance import ExecutionPolicy
from .contracts import PROFILES, SCHEMA, canonical_hash, file_hash, identity, policy, registry


def cost_samples(runtime, text):
    """Native batch K=2/4/8 timing. The active IR verifier excludes diagnostic PickScore cost."""
    import torch
    samples = []
    for k in (2, 4, 8):
        for repeat in range(2):
            noise = torch.randn((k, runtime.C, runtime.h, runtime.w),
                                generator=torch.Generator('cpu').manual_seed(12345)).to(runtime.dtype).to(runtime.device)
            args = dict(prompt=text, negative_prompt='', num_images_per_prompt=k, num_inference_steps=8,
                        guidance_scale=runtime.spec['guidance'], height=runtime.H, width=runtime.W,
                        latents=noise, output_type='latent')
            if runtime.spec['scheduler'] == 'DDIMScheduler': args['eta'] = 0.0
            runtime.sync(); started = time.perf_counter()
            with torch.inference_mode(): final = runtime.pipe(**args).images
            runtime.sync(); generation = time.perf_counter() - started
            _, _, _, _, timing = runtime.decode_score(text, final.detach().cpu())
            samples.append({'k': k, 'repeat': repeat, 'generation_8_forwards_s': generation,
                            'L_per_forward_s': generation / 8, 'decode_s': timing['decode_s'],
                            'ir_s': timing['imagereward_s'], 'pickscore_diagnostic_s': timing['pickscore_s'],
                            'V_IR_s': timing['decode_s'] + timing['imagereward_s']})
    return {'samples': samples, 'method': 'native-batch real GPU wall times, synchronization included',
            'limitation': 'Cost measurements do not establish replay-online trajectory equivalence'}


def qualify(tasks, out, profiles=PROFILES):
    import torch
    from apsp.lite.runtime import LiteRuntime
    from .collector import collect_task, frames_at, score_frames
    from .encoder import CoreEncoder
    from .validate import validate_task
    from .export import export_release
    from .pack import pack
    # Primary implementation reused solely as a numerical comparator, with cardinality checked below.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'env/lite'))
    from benchmark_performance import compare
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0):
        raise ValueError('Real sm_100 B200 required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(tasks)
    summary = {'schema': SCHEMA, 'pass': False, 'tasks_sha256': file_hash(tasks), 'profiles': {}}
    for name in profiles:
        selected = sorted((t for t in rows if t['profile'] == name),
                          key=lambda t: (t['prompt']['prompt_uid'], t['bank_id']))
        if (len(selected) != 20 or len({t['prompt']['prompt_uid'] for t in selected}) != 10 or
                any(t['prompt']['split'] != 'calibration' for t in selected) or
                any({t['bank_id'] for t in selected if t['prompt']['prompt_uid'] == uid} != {0, 1}
                    for uid in {t['prompt']['prompt_uid'] for t in selected})):
            raise ValueError('Qualification requires exactly ten calibration prompts x two banks/profile')
        execution = policy(True)
        fp, fingerprint_hash = identity(execution)
        runtime = LiteRuntime(registry()['profiles'][name], registry(), execution=execution)
        runtime.load_scorers()
        encoder = CoreEncoder()
        # Warm up encoder outside task timings. Denoiser/VAE/scorers warm up on first calibration task.
        encoder.encode(selected[0]['prompt']['text'])
        root = out / name
        checks, pairs = [], []
        for i, t in enumerate(selected):
            captured = []
            runtime.sync(); start = time.perf_counter()
            dest = collect_task(runtime, encoder, t, root / 'fast', fp, keep_initial=True, capture=captured)
            backup_task(dest, root / 'backup' / t['task_id'])
            runtime.sync(); fast_wall = time.perf_counter() - start
            validate = validate_task(dest, check_latents=True, runtime=runtime)
            saved = frames_at(dest)
            parity = None
            if t['bank_id'] == 0:
                runtime.execution = policy(False)
                reference = score_frames(runtime, t['prompt']['text'], saved)
                if len(reference) != len(captured) or len(reference) != 9:
                    raise ValueError('Preview/final cardinality lost')
                parity = compare(reference, captured)
                runtime.execution = execution
            # Two warm paired tasks, excluding the first model/scorer cold task.
            if i in (2, 4):
                runtime.execution = policy(False)
                old_fp, _ = identity(runtime.execution)
                runtime.sync(); start = time.perf_counter()
                old = collect_task(runtime, encoder, t, root / 'canonical', old_fp, keep_initial=True)
                backup_task(old, root / 'canonical-backup' / t['task_id'])
                runtime.sync(); legacy_wall = time.perf_counter() - start
                from safetensors.torch import load_file
                a = load_file(str(dest / 'latents.safetensors')); b = load_file(str(old / 'latents.safetensors'))
                same = set(a) == set(b) and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)
                pairs.append({'task_id': t['task_id'], 'fast_wall_s': fast_wall, 'canonical_wall_s': legacy_wall,
                              'speedup': legacy_wall / fast_wall, 'native_tensors_identical': same})
                runtime.execution = execution
            checks.append({'task_id': t['task_id'], 'validation': validate, 'parity': parity,
                           'bytes_by_file': {p.name: p.stat().st_size for p in dest.rglob('*') if p.is_file()},
                           'native_tensors': json.loads((dest / 'provenance.json').read_text())['native_tensors']})
            atomic_json(root / 'progress.json', {'tasks': checks, 'pairs': pairs})
            print('LITE2_TASK=' + json.dumps({'profile': name, 'done': i + 1, 'valid': validate['all_valid'],
                                            'native': validate['native_checked'], 'parity': None if parity is None else parity['pass']}), flush=True)
        gain = statistics.mean(p['speedup'] for p in pairs)
        passed = (all(c['validation']['pass'] and c['validation']['all_valid'] and
                      (c['parity'] is None or c['parity']['pass']) for c in checks) and
                  len(pairs) == 2 and all(p['native_tensors_identical'] for p in pairs))
        export = export_release(list((root / 'fast/tasks').iterdir()), root / 'export',
                                include_calibration=True, expected_tasks=0)
        parts = pack(root / 'export', root / 'parts', part_bytes=20_000_000)
        # Independent process verifies packed restore, checksum and Core-only replay, no model load.
        subprocess.run([sys.executable, '-m', 'apsp.lite2.pack', 'restore', '--parts', str(root / 'parts'),
                        '--out', str(root / 'restored')], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, '-m', 'apsp.lite2.replay', '--core', str(root / 'restored/core')],
                       check=True, stdout=subprocess.DEVNULL)
        # No PNG/latent access is possible through a physically isolated Core-only copy.
        import shutil
        shutil.copytree(root / 'restored/core', root / 'core-only')
        subprocess.run([sys.executable, '-m', 'apsp.lite2.replay', '--core', str(root / 'core-only')],
                       check=True, stdout=subprocess.DEVNULL)
        costs = cost_samples(runtime, selected[0]['prompt']['text'])
        atomic_json(root / 'costs.json', costs)
        receipt = {'pass': passed, 'schema': SCHEMA, 'profile': name, 'runtime': fp,
                   'runtime_identity_hash': fingerprint_hash, 'registry_hash': canonical_hash(registry()),
                   'execution': execution.to_dict(), 'tasks': checks, 'pairs': pairs,
                   'mean_paired_speedup': gain, 'export_bytes': export['total_bytes'],
                   'parts_bytes': parts['total_archive_bytes'], 'restore_independent_process': True,
                   'core_only_replay_pass': True, 'performance_enabled': passed and gain >= 1.10,
                   'cost_profile': costs,
                   'scope': '20 calibration tasks/profile, native dtype, Core384; not full company bulk collection'}
        atomic_json(out / (name + '.json'), receipt)
        summary['profiles'][name] = {k: receipt[k] for k in ('pass', 'performance_enabled', 'mean_paired_speedup', 'export_bytes')}
        atomic_json(out / 'summary.json', summary)
        print('LITE2_PROFILE=' + json.dumps({'profile': name, **summary['profiles'][name]}), flush=True)
        del runtime, encoder, saved, captured, reference, a, b
        gc.collect(); torch.cuda.empty_cache()
    summary['pass'] = all(r['pass'] for r in summary['profiles'].values())
    atomic_json(out / 'summary.json', summary)
    print('LITE2_QUALIFICATION=' + json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    a = p.parse_args()
    if not qualify(a.tasks, a.out, a.profiles)['pass']:
        raise SystemExit(1)
