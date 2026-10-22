#!/usr/bin/env python3
"""Four-cell Linux/NVIDIA A/B, fresh subprocess per cell/repeat.

synthetic: compiled checkpoint/DMA stress workload, NOT MOSS throughput.
moss: actual trainer with pre-existing local weights and verified feature cache.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CELLS = [('adamw', False), ('adamw_fused', False), ('adamw', True), ('adamw_fused', True)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class MemorySampler:
    """RSS/VMS/USS of worker plus DataLoader children, sampled every 50 ms.

    Summed RSS can double-count shared pages. USS is preferred for private RAM.
    Pinned memory is included in RSS, not measured separately on older PyTorch.
    """
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peaks = dict(rss_bytes=0, vms_bytes=0, uss_bytes=None)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        import psutil
        totals = dict(rss_bytes=0, vms_bytes=0, uss_bytes=0)
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            processes = [self.process]
        for process in processes:
            try:
                info = process.memory_info()
                totals['rss_bytes'] += info.rss
                totals['vms_bytes'] += info.vms
                try:
                    uss = process.memory_full_info().uss
                    if totals['uss_bytes'] is not None:
                        totals['uss_bytes'] += uss
                except (psutil.Error, AttributeError):
                    totals['uss_bytes'] = None
            except psutil.Error:
                continue
        with self.lock:
            for key, value in totals.items():
                if value is not None:
                    self.peaks[key] = max(self.peaks[key] or 0, value)

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.05)

    def reset(self):
        with self.lock:
            self.peaks = dict(rss_bytes=0, vms_bytes=0, uss_bytes=None)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()


def assert_tree_close(a, b, *, rtol, atol):
    import torch
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol, equal_nan=False)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), 'state keys differ'
        for key in a: assert_tree_close(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), 'state lengths differ'
        for x, y in zip(a, b): assert_tree_close(x, y, rtol=rtol, atol=atol)
    elif isinstance(a, float):
        assert abs(a - b) <= atol + rtol * abs(b), (a, b)
    else:
        assert a == b, (a, b)


def metadata(args):
    import torch
    import importlib.metadata
    versions = {}
    for package in ['triton', 'lycoris-lora', 'diffusers', 'transformers', 'safetensors', 'psutil']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        smi = None
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), nvidia_smi=smi,
                dependencies=versions,
                arguments=vars(args), source_sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT / 'src/moss_trainer').glob('*.py'))},
                environment={k: os.environ.get(k) for k in
                             ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF',
                              'LYCORIS_KERNEL_BACKEND', 'CUBLAS_WORKSPACE_CONFIG']})


def synthetic(args, directory):
    import torch
    from torch.utils.checkpoint import checkpoint
    from moss_trainer.compilation import _offload_checkpoint_input
    from moss_trainer.config import Training
    from moss_trainer.scheduling import build_optimizer, build_scheduler, training_plan
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    blocks = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(args.width, args.width * 2), torch.nn.SiLU(),
                            torch.nn.Linear(args.width * 2, args.width)).cuda()
        for _ in range(args.depth)])
    training = Training(optimizer=args.optimizer, max_steps=args.warmup + args.steps,
                        gradient_accumulation_steps=args.accumulation, warmup_ratio=0)
    parameters = list(blocks.parameters())
    opt = build_optimizer(parameters, training)
    scheduler = build_scheduler(opt, training, training_plan(training, training.max_steps))
    forwards = []
    for block in blocks:
        def forward(x, block=block): return checkpoint(block, x, use_reentrant=False)
        compiled = torch.compile(forward, fullgraph=True, dynamic=True)
        forwards.append(_offload_checkpoint_input(compiled, block, pool=args.pool))
    # A private fixed workload generator does not contaminate the training RNG.
    generator = torch.Generator(device='cuda').manual_seed(args.seed + 1)
    x = torch.randn(args.batch, args.tokens, args.width, generator=generator, device='cuda')
    target = torch.randn(x.shape, generator=generator, device='cuda')

    def update():
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.accumulation):
            value = x.detach().requires_grad_()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for forward in forwards: value = value + forward(value)
            loss = (value.float() - target).square().flatten(1).mean(1).sum()
            if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
            loss.backward()
            losses.append(loss.detach())
        for p in parameters:
            if p.grad is not None: p.grad.div_(args.batch * args.accumulation)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        return float(torch.stack(losses).sum() / (args.batch * args.accumulation)), float(norm)

    losses, durations, norms = [], [], []
    for _ in range(args.warmup): update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resume_path = directory / 'midpoint.pt'
    with MemorySampler() as memory:
        for step in range(args.steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, norm = update()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            losses.append(loss)
            norms.append(norm)
            if step + 1 == args.steps // 2:
                # Excluded from timed updates. Restore through a real disk roundtrip.
                torch.save(dict(model=blocks.state_dict(), optimizer=opt.state_dict(),
                                scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate()), resume_path)
    peaks = gpu_peaks()
    final = copy.deepcopy(blocks.state_dict())
    final_opt, final_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict())
    final_rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                     python_rng=random.getstate())
    state = torch.load(resume_path, map_location='cpu', weights_only=True)
    blocks.load_state_dict(state['model'])
    # Fresh optimizer/scheduler proves state is not inherited from the first run.
    opt = build_optimizer(parameters, training)
    scheduler = build_scheduler(opt, training, training_plan(training, training.max_steps))
    opt.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    random.setstate(state['python_rng'])
    replay_losses = [update()[0] for _ in range(args.steps - args.steps // 2)]
    assert_tree_close(final, blocks.state_dict(), rtol=args.rtol, atol=args.atol)
    assert_tree_close(final_opt, opt.state_dict(), rtol=args.rtol, atol=args.atol)
    assert_tree_close(final_sched, scheduler.state_dict(), rtol=0, atol=0)
    assert_tree_close(losses[args.steps // 2:], replay_losses, rtol=args.rtol, atol=args.atol)
    assert_tree_close(final_rng, dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                     python_rng=random.getstate()), rtol=0, atol=0)
    torch.save({k: v.cpu() for k, v in final.items()}, directory / 'final.pt')
    return dict(durations=durations, losses=losses, grad_norms=norms,
                images=[args.batch * args.accumulation] * len(durations),
                tokens_per_image=args.tokens, memory=memory.peaks, **peaks,
                resume={'passed': True, 'scope': 'model/optimizer/scheduler/RNG disk roundtrip; not MOSS adapter'})


def gpu_peaks():
    import torch
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def moss(args, directory):
    import torch
    from safetensors.torch import load_file
    from moss_trainer import training as trainer
    from moss_trainer.config import load_config, validate
    from moss_trainer.checkpoints import load_checkpoint
    from moss_trainer.data import verify_cache
    from moss_trainer.buckets import BucketBatchSampler
    original = load_config(args.config)
    if not (original.training.compile_blocks and original.training.checkpoint_offload):
        raise ValueError('MOSS A/B requires compile_blocks=true and checkpoint_offload=true in source config')
    total = args.warmup + args.steps
    split = total // 2
    config = dataclasses.replace(original, training=dataclasses.replace(original.training,
        optimizer=args.optimizer, checkpoint_offload_pool=args.pool,
        output_dir=str(directory / 'continuous'), max_epochs=0, max_steps=total,
        save_every_epochs=0, save_every=split, tensorboard=False, seed=args.seed),
        sample=dataclasses.replace(original.sample, every_epochs=0))
    validate(config)
    write_json(directory / 'effective-config.json', config.as_dict())
    spec, _ = verify_cache(config)
    batches = BucketBatchSampler(spec['records'], config.dataset.batch_size,
                                 config.dataset.num_repeats, args.seed).total_batches
    records, resumed_records = [], []
    active = records
    memory = None
    class Observer:
        def __init__(self, *a, **kw): self.previous = None
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def image(self, *a): pass
        def scalars(self, step, values):
            torch.cuda.synchronize()
            now = time.perf_counter()
            reported = values['performance/update_seconds']
            count = round(values['performance/images_per_second'] * reported)
            active.append(dict(step=step, loss=values['train/loss'],
                seconds=now - self.previous if self.previous is not None else None, images=count))
            self.previous = now
            if active is records and step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
                memory.reset()
    # Only the benchmark process substitutes the optional telemetry sink.
    trainer.TensorBoardLogger = Observer
    with MemorySampler() as memory:
        final_checkpoint = trainer.train(config)
    peaks = gpu_peaks()
    # A subsequent update's interval includes the previous checkpoint write.
    measured = [r for r in records if r['step'] > args.warmup and r['seconds'] is not None
                and (r['step'] - 1) % split != 0]
    # train() leaves an installed adapter attached to its model; GC between runs.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    active = resumed_records
    resume_config = dataclasses.replace(config, training=dataclasses.replace(
        config.training, output_dir=str(directory / 'resumed')))
    midpoint = Path(config.training.output_dir) / f'step-{split:08d}'
    resumed_checkpoint = trainer.train(resume_config, str(midpoint))
    expected = load_file(str(final_checkpoint / 'adapter.safetensors'))
    actual = load_file(str(resumed_checkpoint / 'adapter.safetensors'))
    assert_tree_close(expected, actual, rtol=args.rtol, atol=args.atol)
    a = torch.load(final_checkpoint / 'trainer-state.pt', map_location='cpu', weights_only=True)
    b = torch.load(resumed_checkpoint / 'trainer-state.pt', map_location='cpu', weights_only=True)
    for key in ['optimizer', 'scheduler', 'step', 'epoch', 'next_batch', 'torch_rng', 'cuda_rng', 'python_rng']:
        assert_tree_close(a[key], b[key], rtol=args.rtol if key == 'optimizer' else 0,
                          atol=args.atol if key == 'optimizer' else 0)
    assert_tree_close([r['loss'] for r in records if r['step'] > split],
                      [r['loss'] for r in resumed_records], rtol=args.rtol, atol=args.atol)
    rejected = []
    for new_stage in [False, True]:
        for field, value in [('optimizer', 'adamw' if args.optimizer == 'adamw_fused' else 'adamw_fused'),
                             ('checkpoint_offload_pool', not args.pool)]:
            changed = dataclasses.replace(config, training=dataclasses.replace(config.training, **{field: value}))
            try:
                load_checkpoint(final_checkpoint if new_stage else midpoint, changed, spec,
                                batch_count=batches, new_stage=new_stage)
            except ValueError as exc:
                if 'settings differ' not in str(exc): raise
                rejected.append(dict(field=field, new_stage=new_stage))
            else:
                raise AssertionError('incompatible checkpoint accepted')
    torch.save(expected, directory / 'final.pt')
    return dict(durations=[r['seconds'] for r in measured], images=[r['images'] for r in measured],
                losses=[r['loss'] for r in records], all_updates=records,
                memory=memory.peaks, **peaks, cache_fingerprint=spec['fingerprint'],
                resume={'passed': True, 'scope': 'actual MOSS adapter/optimizer/scheduler/RNG/cursor/loss replay',
                        'incompatible_settings_rejected': rejected})


def summarize(result):
    durations = result['durations']
    if not durations: raise ValueError('no measured updates; increase steps/warmup')
    ordered = sorted(durations)
    return dict(images_per_second=sum(result['images']) / sum(durations),
                step_mean_seconds=statistics.mean(durations),
                step_median_seconds=statistics.median(durations),
                step_p95_seconds=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                measured_updates=len(durations))


def worker(args):
    import torch
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('Requires Linux and CUDA-enabled PyTorch; no CPU fallback or simulated timings')
    if args.device >= torch.cuda.device_count(): raise ValueError('CUDA device index out of range')
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'environment.json', metadata(args))
    with (directory / 'training.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        result = (synthetic if args.mode == 'synthetic' else moss)(args, directory)
    result.update(optimizer=args.optimizer, pool=args.pool, mode=args.mode, **summarize(result))
    write_json(directory / 'result.json', result)


def matrix(args):
    import torch
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = [(repeat, opt, pool) for repeat in range(args.repeats) for opt, pool in CELLS]
    random.Random(args.seed).shuffle(cells)
    write_json(output / 'plan.json', dict(arguments=vars(args), order=cells))
    results, failures = [], []
    for repeat, optimizer, pool in cells:
        name = f'r{repeat}-{optimizer}-pool{int(pool)}'
        directory = output / name
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
               '--output', str(directory), '--optimizer', optimizer,
               '--steps', str(args.steps), '--warmup', str(args.warmup), '--seed', str(args.seed),
               '--width', str(args.width), '--depth', str(args.depth), '--batch', str(args.batch),
               '--tokens', str(args.tokens), '--accumulation', str(args.accumulation),
               '--rtol', str(args.rtol), '--atol', str(args.atol), '--device', str(args.device)]
        if pool: cmd.append('--pool')
        if args.config: cmd.extend(['--config', str(Path(args.config).resolve())])
        print(f'Running {name}', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            failures.append(dict(cell=name, returncode=completed.returncode))
        else:
            value = json.loads((directory / 'result.json').read_text())
            results.append(dict(cell=name, repeat=repeat, directory=str(directory), **value))
    numeric = []
    for repeat in range(args.repeats):
        reference = output / f'r{repeat}-adamw-pool0'
        if not (reference / 'final.pt').exists(): continue
        baseline = torch.load(reference / 'final.pt', map_location='cpu', weights_only=True)
        baseline_losses = json.loads((reference / 'result.json').read_text())['losses']
        for row in [r for r in results if r['repeat'] == repeat]:
            try:
                candidate = torch.load(Path(row['directory']) / 'final.pt', map_location='cpu', weights_only=True)
                # Pool changes should preserve the calculation exactly. Fused AdamW
                # permits configurable numerical tolerance, never silent acceptance.
                rtol, atol = (0, 0) if row['optimizer'] == 'adamw' else (args.rtol, args.atol)
                assert_tree_close(baseline, candidate, rtol=rtol, atol=atol)
                assert_tree_close(baseline_losses, row['losses'], rtol=rtol, atol=atol)
                numeric.append(dict(cell=row['cell'], passed=True, rtol=rtol, atol=atol))
            except AssertionError as exc:
                numeric.append(dict(cell=row['cell'], passed=False, error=str(exc)))
        # Isolate the pool effect within fused AdamW too; comparing both fused
        # cells only to ordinary AdamW would mask pool errors inside its tolerance.
        fused_rows = [r for r in results if r['repeat'] == repeat and r['optimizer'] == 'adamw_fused']
        if len(fused_rows) == 2:
            pair = sorted(fused_rows, key=lambda r: r['pool'])
            try:
                states = [torch.load(Path(r['directory']) / 'final.pt', map_location='cpu', weights_only=True)
                          for r in pair]
                assert_tree_close(*states, rtol=0, atol=0)
                assert_tree_close(pair[0]['losses'], pair[1]['losses'], rtol=0, atol=0)
                numeric.append(dict(cell=f'r{repeat}-fused-pool-pair', passed=True, rtol=0, atol=0))
            except AssertionError as exc:
                numeric.append(dict(cell=f'r{repeat}-fused-pool-pair', passed=False, error=str(exc)))
    groups = []
    for optimizer, pool in CELLS:
        rows = [r for r in results if r['optimizer'] == optimizer and r['pool'] == pool]
        if rows:
            throughput = [r['images_per_second'] for r in rows]
            groups.append(dict(optimizer=optimizer, pool=pool, completed_repeats=len(rows),
                images_per_second_median=statistics.median(throughput),
                images_per_second_min=min(throughput), images_per_second_max=max(throughput),
                peak_allocated_bytes_max=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes_max=max(r['peak_reserved_bytes'] for r in rows),
                rss_bytes_max=max(r['memory']['rss_bytes'] for r in rows)))
    write_json(output / 'summary.json', dict(mode=args.mode, results=results, groups=groups,
               numeric=numeric, failures=failures, warning='Synthetic results are not full MOSS performance'))
    print(str(output / 'summary.json'))
    if failures or any(not x['passed'] for x in numeric) or len(results) != len(cells):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['synthetic', 'moss'], default='synthetic')
    p.add_argument('--config', help='MOSS config; weights and feature cache must already exist')
    p.add_argument('--output', required=True, help='new output directory; refuses overwrite')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--optimizer', choices=['adamw', 'adamw_fused'], default='adamw', help=argparse.SUPPRESS)
    p.add_argument('--pool', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.warmup < 2 or args.steps < 4 or args.repeats < 1:
        p.error('warmup >= 2, steps >= 4, repeats >= 1 required')
    if min(args.width, args.depth, args.batch, args.tokens, args.accumulation) < 1 or args.device < 0:
        p.error('workload dimensions must be positive; device must be nonnegative')
    import math
    if any(not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol)):
        p.error('numeric tolerances must be finite and nonnegative')
    if args.mode == 'moss' and not args.config: p.error('--mode moss requires --config')
    (worker if args.worker else matrix)(args)


if __name__ == '__main__':
    main()
