"""Bounded exact-kernel CUDA graphs for SAM2 memory encoder and mask decoder.

These modules are launch-heavy and have small, stable signature sets. Capture
owns parameter conversions (autocast cache disabled), copies dynamic inputs and
clones every returned tensor. Expanded read-only inputs preserve zero strides.
Only official eager predictor/module types and CUDA inference are eligible.
"""
from __future__ import annotations

import atexit
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import threading
import time
import warnings

import torch


_CAPTURE_ALLOWED = ContextVar('sam2_small_graph_capture_allowed', default=True)
_SERIAL_PRODUCER = ContextVar('sam2_small_graph_serial_producer', default=False)


def _coordinate_cpu_overlap():
    """Capture only before the known driver starts CPU mask work in a chunk.

    Its first yielded frame repeats the previous boundary and is skipped. The
    first two results are buffered before either reaches the consumer, so all
    capture APIs finish before caller CUDA work. Later calls only replay existing
    graphs. Context is set in the producer thread around source next(),
    since ContextVars do not automatically cross a newly created thread.
    """
    from experiments.sam2_profile.patches import cpu_overlap
    previous = cpu_overlap._prefetch
    if getattr(previous, '_small_graph_capture_scope', False):
        return

    @wraps(previous)
    def prefetch(source_factory, *args, **kwargs):
        def scoped_source():
            source = iter(source_factory())
            buffered = []
            item = None

            def advance(allow_capture):
                capture_token = _CAPTURE_ALLOWED.set(allow_capture)
                serial_token = _SERIAL_PRODUCER.set(True)
                try:
                    return next(source)
                finally:
                    _SERIAL_PRODUCER.reset(serial_token)
                    _CAPTURE_ALLOWED.reset(capture_token)

            try:
                exhausted, pending_error = False, None
                # Finish both capture windows before _prefetch can expose even
                # the skipped boundary's wait_event/record_stream to its caller.
                for _ in range(2):
                    try:
                        buffered.append(advance(True))
                    except StopIteration:
                        exhausted = True
                        break
                    except BaseException as error:
                        # Preserve an earlier buffered result before forwarding
                        # the next source error, as synchronous iteration does.
                        pending_error = error
                        break
                while buffered:
                    item = buffered.pop(0)
                    yield item
                    item = None
                if pending_error is not None:
                    raise pending_error
                if exhausted:
                    return
                while True:
                    try:
                        item = advance(False)
                    except StopIteration:
                        return
                    yield item
                    item = None
            finally:
                buffered.clear()
                item = None
                close = getattr(source, 'close', None)
                if close is not None:
                    close()
        return previous(scoped_source, *args, **kwargs)

    prefetch._small_graph_capture_scope = True
    cpu_overlap._prefetch = prefetch


def _map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if type(value) is dict:
        return {k: _map(v, fn) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_map(v, fn) for v in value)
    if value is None or type(value) in (bool, int, str):
        return value
    raise TypeError(f'Unsupported graph value: {type(value)}')


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise TypeError('Graph dictionaries require string keys')
        return ('dict', tuple((k, _signature(v)) for k, v in value.items()))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_signature(v) for v in value))
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    raise TypeError(f'Unsupported graph signature: {type(value)}')


def _base_slice(tensor):
    # Expanded inputs are read-only in both supported modules. Copy one element
    # along each broadcast dimension, preserving the original strides exactly.
    return tuple(slice(0, 1) if size > 1 and stride == 0 else slice(None)
                 for size, stride in zip(tensor.shape, tensor.stride()))


def _copy_inputs(target, source):
    if isinstance(target, torch.Tensor):
        reduced = _base_slice(source)
        target[reduced].copy_(source[reduced])
    elif type(target) is dict:
        for key in target:
            _copy_inputs(target[key], source[key])
    elif type(target) in (tuple, list):
        for x, y in zip(target, source):
            _copy_inputs(x, y)


class SmallModuleGraphs:
    def __init__(self, module, *, min_calls=2, max_entries=4, max_bytes=512 << 20):
        self.module = module
        self.original = module.forward
        self.signature = inspect.signature(self.original)
        self.min_calls = min_calls
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = {}
        self.seen = Counter()
        self.failed = set()
        self.counts = Counter()
        self.capture_seconds = 0.0
        self.stream = None

    def stats(self):
        return {**dict(self.counts), 'entries': len(self.entries),
                'capture_seconds': self.capture_seconds,
                'active_bytes': sum(x[3] for x in self.entries.values()),
                'reserved_delta_bytes': sum(x[4] for x in self.entries.values())}

    def __call__(self, *args, **kwargs):
        def eager():
            self.counts['eager'] += 1
            return self.original(*args, **kwargs)
        if (self.module.training or torch.is_grad_enabled()
                or not torch.is_inference_mode_enabled()):
            return eager()
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        tensors = []
        try:
            _map(values, lambda t: tensors.append(t))
            if not tensors or any(not t.is_cuda or t.layout != torch.strided
                                  or not t.is_floating_point()
                                  or any(x < 0 for x in t.stride())
                                  or torch._debug_has_internal_overlap(t[_base_slice(t)]) != 0
                                  for t in tensors):
                return eager()
            device = tensors[0].device
            if any(t.device != device for t in tensors) or torch.cuda.is_current_stream_capturing():
                return eager()
            current = torch.cuda.current_stream(device)
            key = (_signature(values), torch.is_autocast_enabled('cuda'),
                   torch.get_autocast_dtype('cuda'), current.cuda_stream,
                   # cpu_overlap reuses one stream and joins before checkpoints.
                   # Distinct chunk threads cannot concurrently access this pool.
                   'serialized_cpu_overlap' if _SERIAL_PRODUCER.get() else threading.get_ident(),
                   torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                   torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                   torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
        except TypeError:
            return eager()
        entry = self.entries.get(key)
        if entry is None:
            if not _CAPTURE_ALLOWED.get():
                self.counts['capture_deferred'] += 1
                return eager()
            if key in self.failed or len(self.entries) >= self.max_entries:
                return eager()
            if key not in self.seen and len(self.seen) >= 64:
                return eager()
            self.seen[key] += 1
            if self.seen[key] < self.min_calls:
                return eager()
            try:
                with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                                    dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                    entry = self._capture(values, device, current)
            except RuntimeError as error:
                self.failed.add(key)
                self.counts['capture_failed'] += 1
                warnings.warn(f'SAM2 small-module graph signature fell back: {error}', RuntimeWarning)
                return eager()
            if sum(x[3] for x in self.entries.values()) + entry[3] > self.max_bytes:
                current.synchronize()
                entry[1].reset()
                self.failed.add(key)
                self.counts['budget_rejected'] += 1
                return eager()
            self.entries[key] = entry
            self.counts['captured'] += 1
        else:
            _copy_inputs(entry[0], values)
        entry[1].replay()
        self.counts['replays'] += 1
        return _map(entry[2], lambda t: t.clone())

    def _capture(self, values, device, current):
        began = time.perf_counter()
        before = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        static = _map(values, lambda t: torch.empty_strided(t.shape, t.stride(),
                                                          dtype=t.dtype, device=t.device))
        _copy_inputs(static, values)
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                warm = self.original(**static)
            del warm
        current.wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            output = self.original(**static)
        current.wait_stream(self.stream)
        # Verify the output schema without launching more GPU work.
        _signature(output)
        active = max(0, torch.cuda.memory_allocated(device) - before)
        extra_reserved = max(0, torch.cuda.memory_reserved(device) - reserved)
        self.capture_seconds += time.perf_counter() - began
        return static, graph, output, active, extra_reserved


def install(predictor):
    """ab.py entry point, separately graph both small supported modules."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.modeling.memory_encoder import MemoryEncoder
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    if type(predictor) is not SAM2VideoPredictor or hasattr(predictor, '_small_graphs'):
        return
    # The helper is idle until CPU overlap is enabled. Wrap it now so either
    # patch installation order has the same safe capture windows.
    _coordinate_cpu_overlap()
    wrappers = {}
    for name, expected in [('memory_encoder', MemoryEncoder), ('sam_mask_decoder', MaskDecoder)]:
        module = getattr(predictor, name)
        if type(module) is expected and not hasattr(module.forward, '_torchdynamo_orig_callable'):
            wrapper = SmallModuleGraphs(module)
            module.forward = wrapper
            wrappers[name] = wrapper
    predictor._small_graphs = wrappers
    atexit.register(lambda: print('SAM2_SMALL_GRAPHS_STATS ' + json.dumps(
        {name: wrapper.stats() for name, wrapper in wrappers.items()}), flush=True))
