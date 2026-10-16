"""Exact-kernel CUDA graphs for the tensor-only SAM2 memory attention module.

Install after fused_rope. This does not graph predictor state or object lifecycle.
Each graph owns its input buffers and clones its output before returning, so a
later object/frame cannot overwrite a tensor retained by the predictor. Unseen
or unsupported signatures use eager. Cache admission is delayed past transient
memory-bank/pointer sizes; the cache stops admitting at four entries rather
than evicting graphs while asynchronous output clones may still reference them.

The estimated active-allocation budget is supplementary: CUDA graph pools can
reserve more than their live tensors. Both allocated and reserved deltas are
reported by stats(). No allocator/artifact caches are cleared.
"""
from __future__ import annotations

import atexit
from collections import Counter
import json
from dataclasses import dataclass
import time
import threading
import warnings

import torch


@dataclass
class _Entry:
    inputs: tuple
    graph: object
    output: torch.Tensor
    active_bytes: int
    reserved_bytes: int


class MemoryAttentionGraphs:
    """Single-host-thread inference wrapper; unsupported execution stays eager."""

    def __init__(self, module, *, min_calls=64, max_entries=4, max_bytes=1536 << 20):
        self.module = module
        self.original = module.forward
        self.min_calls = min_calls
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = {}
        self.seen = Counter()
        self.failed = set()
        self.counts = Counter()
        self.capture_seconds = 0.0
        self._stream = None

    @staticmethod
    def _meta(tensor):
        if tensor is None:
            return None
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
                tensor.device, tensor.layout)

    def stats(self):
        return {**dict(self.counts), 'entries': len(self.entries),
                'min_calls': self.min_calls, 'max_entries': self.max_entries,
                'max_bytes': self.max_bytes,
                'cached_memory_tokens': [int(x.inputs[1].shape[0]) for x in self.entries.values()],
                'cached_pointer_tokens': [key[2] for key in self.entries],
                'tracked_signatures': len(self.seen), 'failed_signatures': len(self.failed),
                'capture_seconds': self.capture_seconds,
                'active_bytes': sum(x.active_bytes for x in self.entries.values()),
                'reserved_delta_bytes': sum(x.reserved_bytes for x in self.entries.values())}

    def __call__(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        def eager():
            self.counts['eager'] += 1
            return self.original(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens)

        as_list = isinstance(curr, list)
        if as_list:
            if not isinstance(curr_pos, list) or len(curr) != 1 or len(curr_pos) != 1:
                return eager()
            values = (curr[0], memory, curr_pos[0], memory_pos)
        else:
            values = (curr, memory, curr_pos, memory_pos)
        tensors = [v for v in values if v is not None]
        if (self.module.training or torch.is_grad_enabled() or not tensors
                or type(num_obj_ptr_tokens) is not int
                or any(not isinstance(v, torch.Tensor) or not v.is_cuda
                       or v.layout != torch.strided or not v.is_floating_point()
                       or torch._debug_has_internal_overlap(v) != 0 for v in tensors)
                or any(v.device != tensors[0].device for v in tensors)
                or torch.cuda.is_current_stream_capturing()):
            return eager()
        # A cache entry is never replayed from a second CUDA stream/host thread.
        stream = torch.cuda.current_stream(tensors[0].device)
        key = (as_list, tuple(self._meta(v) for v in values), num_obj_ptr_tokens,
               torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
               torch.is_inference_mode_enabled(), stream.cuda_stream, threading.get_ident())

        def invoke(flat):
            x, bank, x_pos, bank_pos = flat
            return self.original([x] if as_list else x, bank,
                                 [x_pos] if as_list else x_pos, bank_pos,
                                 num_obj_ptr_tokens)

        entry = self.entries.get(key)
        if entry is None:
            if key in self.failed or len(self.entries) >= self.max_entries:
                return eager()
            # Bound even the lightweight metadata: long clips can have many
            # conditioning-frame and object-pointer signatures.
            if key not in self.seen and len(self.seen) >= 128:
                return eager()
            self.seen[key] += 1
            if self.seen[key] < self.min_calls:
                return eager()
            try:
                # Chunk-level autocast caches are cleared before later replays.
                # Record casts inside the graph so all cast buffers belong to
                # its pool instead of dangling after a chunk/checkpoint.
                with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                                    dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                    entry = self._capture(values, invoke, stream)
            except RuntimeError as error:
                self.failed.add(key)
                self.counts['capture_failed'] += 1
                warnings.warn(f'SAM2 memory CUDA graph disabled for one signature: {error}', RuntimeWarning)
                return eager()
            if sum(x.active_bytes for x in self.entries.values()) + entry.active_bytes > self.max_bytes:
                stream.synchronize()
                entry.graph.reset()
                self.failed.add(key)
                self.counts['budget_rejected'] += 1
                return eager()
            self.entries[key] = entry
            self.counts['captured'] += 1
        else:
            for target, source in zip(entry.inputs, values):
                if target is not None:
                    target.copy_(source)
        entry.graph.replay()
        self.counts['replays'] += 1
        # This allocation is intentionally outside the graph: downstream keeps
        # outputs alive across objects, chunk restarts and checkpoint prompts.
        return entry.output.clone()

    def _capture(self, values, invoke, current_stream):
        started = time.perf_counter()
        device = next(v.device for v in values if v is not None)
        before_active = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        static = tuple(None if v is None else torch.empty_strided(
            v.shape, v.stride(), dtype=v.dtype, device=v.device) for v in values)
        for target, source in zip(static, values):
            if target is not None:
                target.copy_(source)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        self._stream.wait_stream(current_stream)
        with torch.cuda.stream(self._stream):
            # Prepare cuBLAS/SDPA and fused-RoPE tables before capture.
            for _ in range(3):
                warm_output = invoke(static)
            del warm_output
        current_stream.wait_stream(self._stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._stream):
            output = invoke(static)
        current_stream.wait_stream(self._stream)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError('MemoryAttention returned a non-tensor output')
        active = max(0, torch.cuda.memory_allocated(device) - before_active)
        reserved = max(0, torch.cuda.memory_reserved(device) - before_reserved)
        self.capture_seconds += time.perf_counter() - started
        return _Entry(static, graph, output, active, reserved)


def install(predictor):
    """ab.py patch entry point; attach statistics as predictor._memory_graphs."""
    if hasattr(predictor, '_memory_graphs'):
        return
    wrapper = MemoryAttentionGraphs(predictor.memory_attention)
    predictor.memory_attention.forward = wrapper
    predictor._memory_graphs = wrapper
    atexit.register(lambda: print('SAM2_CUDA_GRAPHS_STATS ' + json.dumps(wrapper.stats()), flush=True))
