"""CPU protocol/fault tests. These do not claim to validate CUDA DMA."""
import gc
import types
import weakref

import pytest

from _isolated import offload


class Tensor:
    layout = 'strided'
    dtype = 'float32'
    device = 'cuda:0'
    is_cuda = True

    def __init__(self, runtime, shape=(4, 4), strides=(4, 1), pointer=None, host=False):
        self.runtime, self.shape, self.strides = runtime, shape, strides
        self.pointer = pointer if pointer is not None else id(self)
        self.host = host
        if host:
            self.device, self.is_cuda = 'cpu', False

    def size(self): return self.shape
    def stride(self): return self.strides
    def numel(self): return 16
    def is_contiguous(self): return self.strides == (self.shape[1], 1)
    def untyped_storage(self): return types.SimpleNamespace(data_ptr=lambda: self.pointer)
    def detach(self): return self
    def record_stream(self, stream): self.runtime.log.append(('source_live', stream.name))
    def copy_(self, other, non_blocking):
        self.runtime.log.append(('d2h' if self.host else 'h2d', self.runtime.stream.name, id(self), id(other)))
        if not self.host and self.runtime.fail_copy:
            raise RuntimeError('injected copy failure after H2D enqueue')


class Stream:
    def __init__(self, runtime, name): self.runtime, self.name = runtime, name
    def wait_event(self, event): self.runtime.log.append(('wait', self.name, event))
    def record_event(self):
        if self.runtime.fail_record:
            raise RuntimeError('injected record failure after H2D enqueue')
        event = Event(self.runtime)
        event.record(self)
        return event


class Event:
    def __init__(self, runtime): self.runtime = runtime
    def record(self, stream): self.runtime.log.append(('record', stream.name, self))


class Runtime:
    def __init__(self):
        self.log, self.allocations, self.fail_record = [], [], False
        self.fail_copy = False
        self.stream = Stream(self, 'forward')
        self.torch = types.SimpleNamespace(
            Tensor=Tensor, strided='strided', is_grad_enabled=lambda: True,
            empty_strided=self.allocate,
            cuda=types.SimpleNamespace(current_stream=lambda device: self.stream,
                                       Event=lambda: Event(self)),
            autograd=types.SimpleNamespace(graph=types.SimpleNamespace(saved_tensors_hooks=self.hooks)))

    def allocate(self, shape, strides, *, dtype, device, pin_memory=False):
        tensor = Tensor(self, shape, strides, host=device == 'cpu')
        self.allocations.append(weakref.ref(tensor))
        return tensor

    def hooks(self, pack, unpack):
        self.pack, self.unpack = pack, unpack
        return Hooks()


class Hooks:
    def __enter__(self): return self
    def __exit__(self, *args): pass


def setup_runtime():
    runtime = Runtime()
    # Override the global torch name in the real helper's execution namespace.
    module = offload()
    module._offload_checkpoint_input.__globals__['torch'] = runtime.torch
    wrapped = module._offload_checkpoint_input(lambda x: runtime.pack(x),
                                               types.SimpleNamespace(training=True), pool=True)
    return runtime, wrapped


def test_retained_graph_multiple_readers_and_reuse_waits():
    r, wrapped = setup_runtime()
    x = Tensor(r)
    packed = wrapped(x)
    host_id = id(packed.host)
    ready = packed.ready
    readers = []
    for name in ('backward-1', 'backward-2', 'backward-3'):
        r.stream = Stream(r, name)
        r.unpack(packed)
        readers.append(packed.events[-1])
    # Retained saved object must not be reused even after repeated unpacks.
    other = wrapped(Tensor(r))
    assert id(other.host) != host_id
    del other
    del packed
    gc.collect()
    r.stream = Stream(r, 'next-forward')
    r.log.clear()
    replacement = wrapped(x)
    assert id(replacement.host) == host_id
    waits = [entry[2] for entry in r.log if entry[0] == 'wait']
    assert waits == [ready, *readers]
    assert max(i for i, e in enumerate(r.log) if e[0] == 'wait') < next(
        i for i, e in enumerate(r.log) if e[0] == 'd2h')


@pytest.mark.parametrize('failure', ['fail_record', 'fail_copy'])
def test_failed_h2d_event_is_never_recycled(failure):
    r, wrapped = setup_runtime()
    packed = wrapped(Tensor(r))
    old_host = packed.host  # Retain identity so Python cannot recycle its id.
    setattr(r, failure, True)
    with pytest.raises(RuntimeError, match='injected'):
        r.unpack(packed)
    del packed
    gc.collect()
    setattr(r, failure, False)
    replacement = wrapped(Tensor(r))
    assert replacement.host is not old_host


def test_cache_consumption_does_not_check_then_pop():
    r, wrapped = setup_runtime()
    class RacingCache(list):
        def __bool__(self):
            # Deterministically emulate another thread consuming the slot after
            # its availability check, before pop. Atomic pop avoids this gap.
            self.clear()
            return True
    cache_cell = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__))['cache']
    packed = wrapped(Tensor(r))
    del packed
    gc.collect()
    cache_cell.cell_contents = RacingCache(cache_cell.cell_contents)
    assert wrapped(Tensor(r)).host is not None


@pytest.mark.parametrize('shape,strides', [((8, 4), (4, 1)), ((4, 4), (1, 4))])
def test_layout_change_and_noncontiguous_fallback(shape, strides):
    r, wrapped = setup_runtime()
    packed = wrapped(Tensor(r))
    host = packed.host
    del packed
    gc.collect()
    replacement = wrapped(Tensor(r, shape, strides))
    if strides == (shape[1], 1):
        assert replacement.host is not host
    else:
        assert isinstance(replacement, tuple)


def test_pack_discard_without_backward_waits_for_d2h():
    r, wrapped = setup_runtime()
    packed = wrapped(Tensor(r))
    host, ready = packed.host, packed.ready
    del packed
    gc.collect()
    r.log.clear()
    assert wrapped(Tensor(r)).host is host
    assert r.log[0] == ('wait', 'forward', ready)


def test_forward_exception_releases_saved_lease():
    r, _ = setup_runtime()
    module = offload()
    module._offload_checkpoint_input.__globals__['torch'] = r.torch
    hosts = []
    def fail(x):
        packed = r.pack(x)
        hosts.append(packed.host)
        raise RuntimeError('forward failed')
    wrapped = module._offload_checkpoint_input(fail, types.SimpleNamespace(training=True), pool=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='forward failed'):
            wrapped(Tensor(r))
        gc.collect()
    assert hosts[0] is hosts[1]
