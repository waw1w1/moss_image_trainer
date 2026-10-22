"""Real DMA/autograd tests: automatically skipped on CPU-only PyTorch."""
import gc
import threading
import types

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from _isolated import offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA PyTorch required')


@pytest.mark.parametrize('pool', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_real_checkpoint_gradients_repeated_backward_and_streams(pool, compiled):
    torch.manual_seed(23)
    block = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(),
                                torch.nn.Linear(32, 16)).cuda().train()
    # Match production placement: offload hooks outside compile(checkpoint()).
    def forward(x): return checkpoint(block, x, use_reentrant=False)
    forward = torch.compile(forward, fullgraph=True) if compiled else forward
    wrapped = offload()._offload_checkpoint_input(forward, block, pool=pool)
    main = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for index in range(12):
        count = (3, 5, 3)[index % 3]
        reference = torch.randn(count, 16, device='cuda', requires_grad=True)
        actual = reference.detach().clone().requires_grad_()
        expected_loss = block(reference).square().mean()
        expected = torch.autograd.grad(expected_loss, (reference, *block.parameters()))
        streams[0].wait_stream(main)
        with torch.cuda.stream(streams[0]):
            loss = wrapped(actual).square().mean()
        streams[1].wait_stream(streams[0])
        for repeat in range(2):
            with torch.cuda.stream(streams[repeat]):
                found = torch.autograd.grad(loss, (actual, *block.parameters()), retain_graph=repeat == 0)
            main.wait_stream(streams[repeat])
            for a, b in zip(found, expected):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        del loss, found
        gc.collect()
    torch.cuda.synchronize()


def test_actual_dma_aliases_noncontiguous_and_multiple_live_graphs():
    class SaveViews(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x, x.t(), x[:, :8])
            return x.square().sum()
        @staticmethod
        def backward(ctx, g):
            x, transposed, sliced = ctx.saved_tensors
            torch.testing.assert_close(x.t(), transposed)
            torch.testing.assert_close(x[:, :8], sliced)
            return 2 * x * g
    block = types.SimpleNamespace(training=True)
    wrapped = offload()._offload_checkpoint_input(SaveViews.apply, block, pool=True)
    live = []
    for k in range(8):
        base = torch.randn(16, 16, device='cuda')
        x = (base if k % 2 else base.t()).detach().requires_grad_()
        live.append((x, wrapped(x)))
    for x, loss in reversed(live):
        torch.testing.assert_close(torch.autograd.grad(loss, x)[0], 2 * x)
    torch.cuda.synchronize()


def test_concurrent_unpacks_and_pool_reuse():
    # Concurrent graphs share one per-block cache; each has its own saved lease.
    wrapped = offload()._offload_checkpoint_input(lambda x: x.square().sum(),
                                                 types.SimpleNamespace(training=True), pool=True)
    errors = []
    def run(seed):
        try:
            with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream()):
                for k in range(20):
                    x = torch.full((128, 128), seed + k, device='cuda', requires_grad=True)
                    grad, = torch.autograd.grad(wrapped(x), x)
                    torch.testing.assert_close(grad, 2 * x)
                torch.cuda.current_stream().synchronize()
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors, errors
