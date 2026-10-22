import copy
import dataclasses
import gc
import json

import pytest
import torch

from _isolated import ROOT, checkpoint_helpers
from moss_trainer.config import Config, Training, load_config, validate
from moss_trainer.scheduling import build_optimizer, build_scheduler, training_plan


@pytest.mark.parametrize('optimizer', ['adamw', 'adamw_fused'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_accumulation_clipping_numeric_and_optimizer_roundtrip(optimizer, device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA PyTorch required')
    torch.manual_seed(42)
    net = torch.nn.Linear(7, 3).to(device)
    training = Training(optimizer=optimizer, max_steps=6, warmup_ratio=0,
                        gradient_accumulation_steps=2)
    opt = build_optimizer(net.parameters(), training)
    sched = build_scheduler(opt, training, training_plan(training, 6))
    assert opt.defaults['foreach'] is False
    assert bool(opt.defaults.get('fused')) == (optimizer == 'adamw_fused')
    x, y = torch.randn(3, 7, device=device), torch.randn(3, 3, device=device)
    def step(net, opt, sched):
        opt.zero_grad(set_to_none=True)
        # A partial accumulation group: 2 + 1 samples, actual count normalization.
        for begin, end in [(0, 2), (2, 3)]:
            (net(x[begin:end]) - y[begin:end]).square().mean(1).sum().backward()
        for p in net.parameters(): p.grad.div_(3)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2, error_if_nonfinite=True)
        opt.step()
        sched.step()
    step(net, opt, sched)
    path = tmp_path / 'state.pt'
    torch.save({'model': net.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict()}, path)
    step(net, opt, sched)
    resumed = torch.nn.Linear(7, 3).to(device)
    fresh_opt = build_optimizer(resumed.parameters(), training)
    fresh_sched = build_scheduler(fresh_opt, training, training_plan(training, 6))
    state = torch.load(path, weights_only=True, map_location='cpu')
    resumed.load_state_dict(state['model'])
    fresh_opt.load_state_dict(state['optimizer'])
    fresh_sched.load_state_dict(state['scheduler'])
    step(resumed, fresh_opt, fresh_sched)
    for a, b in zip(net.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sched.state_dict() == fresh_sched.state_dict()
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_fused_matches_regular_with_same_gradients():
    torch.manual_seed(17)
    a = torch.nn.Linear(7, 3)
    b = copy.deepcopy(a)
    normal = build_optimizer(a.parameters(), Training())
    fused = build_optimizer(b.parameters(), Training(optimizer='adamw_fused'))
    for _ in range(8):
        for pa, pb in zip(a.parameters(), b.parameters()):
            gradient = torch.randn_like(pa)
            pa.grad, pb.grad = gradient.clone(), gradient.clone()
        normal.step()
        fused.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-7)


def test_config_defaults_aliases_and_invalid_combinations(tmp_path):
    assert Config().training.optimizer == 'adamw'
    assert Config().training.checkpoint_offload_pool is False
    for sample in ('lora', 'lokr'):
        config = load_config(ROOT / f'configs/{sample}.toml')
        validate(config)
        for compile_blocks, checkpoint_offload in [(False, False), (False, True), (True, False)]:
            t = dataclasses.replace(config.training, checkpoint_offload_pool=True,
                                    compile_blocks=compile_blocks, checkpoint_offload=checkpoint_offload)
            with pytest.raises(ValueError, match='checkpoint_offload_pool'):
                validate(dataclasses.replace(config, training=t))
    text = (ROOT / 'configs/lora.toml').read_text(encoding='utf-8').replace(
        'checkpoint_offload_pool = false', 'checkpoint_offload_pool = true').replace(
        'optimizer = "adamw"', 'optimizer = "adamw_fused"')
    path = tmp_path / 'config.toml'
    path.write_text(text, encoding='utf-8')
    assert load_config(path).training.checkpoint_offload_pool
    assert load_config(path).training.optimizer == 'adamw_fused'


@pytest.mark.parametrize('optimizer', ['adamw', 'adamw_fused'])
@pytest.mark.parametrize('pool', [False, True])
def test_signature_rejects_optimizer_and_pool_switch(optimizer, pool):
    helpers = checkpoint_helpers()
    t = Training(optimizer=optimizer, checkpoint_offload_pool=pool)
    c = Config(training=t)
    baseline = helpers._resume_signature(c, 'cache')
    for field, value in [('optimizer', 'adamw_fused' if optimizer == 'adamw' else 'adamw'),
                         ('checkpoint_offload_pool', not pool)]:
        other = dataclasses.replace(c, training=dataclasses.replace(t, **{field: value}))
        assert helpers._resume_signature(other, 'cache') != baseline


def test_legacy_default_signature_is_unchanged():
    # Compare against the exact pre-PR algorithm by removing only its new field.
    helpers = checkpoint_helpers()
    c = Config(training=Training(scheduler='constant', warmup_steps=0))
    old = c.as_dict()
    del old['training']['checkpoint_offload_pool']
    class Legacy:
        training = c.training
        def as_dict(self): return copy.deepcopy(old)
    # Old algorithm's offload-loop must not mention the field that did not exist.
    source = (ROOT / 'src/moss_trainer/checkpoints.py').read_text(encoding='utf-8')
    source = source.replace(', "checkpoint_offload_pool")', ')')
    import ast
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if getattr(n, 'name', '') == '_resume_signature']
    scope = {'digest': helpers.digest}
    exec(compile(tree, '<legacy-signature>', 'exec'), scope)
    assert scope['_resume_signature'](Legacy(), 'cache') == helpers._resume_signature(c, 'cache')


@pytest.mark.parametrize('optimizer', ['adamw', 'adamw_fused'])
@pytest.mark.parametrize('pool', [False, True])
def test_checkpoint_roundtrip_rejects_switch_and_new_stage(optimizer, pool, tmp_path):
    helpers = checkpoint_helpers()
    if not hasattr(__import__('os'), 'O_DIRECTORY'):
        # Windows smoke coverage exercises serialization/validation/restore only.
        # Linux executes actual file and directory fsync implementations too.
        import types
        helpers.save_checkpoint.__globals__['os'] = types.SimpleNamespace(fsync=lambda fd: None)
        helpers.save_checkpoint.__globals__['_sync_directory'] = lambda path: None
    t = Training(optimizer=optimizer, checkpoint_offload_pool=pool, max_steps=2,
                 gradient_accumulation_steps=1, scheduler='constant', warmup_ratio=0,
                 compile_blocks=True, checkpoint_offload=True, fused_situ=True, fused_rmsnorm=True)
    c = Config(training=t)
    validate(c)
    net = torch.nn.Linear(3, 2)
    opt = build_optimizer(net.parameters(), t)
    sched = build_scheduler(opt, t, training_plan(t, 2))
    def update(net, opt, sched):
        opt.zero_grad(set_to_none=True)
        net(torch.ones(1, 3)).square().sum().backward()
        opt.step()
        sched.step()
    update(net, opt, sched)
    spec = {'fingerprint': 'cache', 'model_fingerprint': 'model'}
    path = helpers.save_checkpoint(tmp_path, net, opt, sched, c, spec, 1, 0, 1)
    state = helpers.load_checkpoint(path, c, spec, batch_count=2)
    expected = copy.deepcopy(net)
    update(net, opt, sched)
    resumed_opt = build_optimizer(expected.parameters(), t)
    resumed_sched = build_scheduler(resumed_opt, t, training_plan(t, 2))
    helpers.restore_checkpoint(path, state, expected, resumed_opt, resumed_sched)
    update(expected, resumed_opt, resumed_sched)
    for a, b in zip(net.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    completed = helpers.save_checkpoint(tmp_path, net, opt, sched, c, spec, 2, 1, 0)
    for new_stage in (False, True):
        for key, val in [('optimizer', 'adamw_fused' if optimizer == 'adamw' else 'adamw'),
                         ('checkpoint_offload_pool', not pool)]:
            changed = dataclasses.replace(c, training=dataclasses.replace(t, **{key: val}))
            with pytest.raises(ValueError, match='settings differ'):
                helpers.load_checkpoint(completed if new_stage else path, changed, spec,
                                        batch_count=2, new_stage=new_stage)
    stage = dataclasses.replace(c, training=dataclasses.replace(t, learning_rate=2e-4))
    saved = helpers.load_checkpoint(completed, stage, spec, batch_count=2, new_stage=True)
    stage_opt = build_optimizer(net.parameters(), stage.training)
    stage_sched = build_scheduler(stage_opt, stage.training, training_plan(stage.training, 2))
    helpers.restore_checkpoint(completed, saved, net, stage_opt, stage_sched, new_stage=True)
    assert stage_opt.param_groups[0]['lr'] == 2e-4
    assert stage_sched.last_epoch == 0
    assert all(s['step'].item() == 2 for s in stage_opt.state.values())
