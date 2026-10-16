"""Output-identical accelerations for SAM2 propagation on CUDA.

Two changes, each measured bit-identical (ids, boxes, masks on every frame of
Han-Ber4, FelixClaar and the BHC-FAG window) and about 1.12x and 1.04x faster
end to end on the GB10. Evidence and the rejected alternatives:
`docs/sam2-gpu-bottleneck.md` and the 2026-10-16 section of
`docs/sam2-speed-research.md`.

- `install_fused_rope` replaces upstream `apply_rotary_enc`, which casts BF16
  q/k to FP32, multiplies as complex64 and casts back (about 13 GB/frame of
  memory traffic in cross-attention), with one Triton kernel that reads BF16
  once and writes BF16 once. It is bit-identical only with the FMA order eager
  complex64 multiply uses on this stack, real = fma(a, c, -(b*d)) and
  imag = fma(b, c, a*d); other orders differ by one BF16 ulp. Because that is a
  property of the compiler stack rather than a guarantee, the first call of
  every input shape is checked against the previous implementation with
  `torch.equal`, and any mismatch or Triton failure disables the kernel for the
  rest of the process.
- `install_small_module_graphs` replays the memory encoder and mask decoder as
  CUDA graphs. Both run once per object per frame as hundreds of tiny kernels,
  so launch overhead dominates them. Same kernels, so same output; inputs are
  copied into owned static buffers and every output is cloned, because
  TrackManager keeps masks alive across frames.
"""
from __future__ import annotations

from collections import Counter
import inspect
import os
from pathlib import Path
import threading
import warnings

import torch


def _fusable(x: torch.Tensor, freqs_cis: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-2] > 0
            and x.shape[-2] % freqs_cis.shape[0] == 0 and x.shape[-1] == 2 * freqs_cis.shape[1])


def install_fused_rope() -> bool:
    """Replace sam2's `apply_rotary_enc` with the fused Triton kernel where it applies.

    Install after `install_rope_repeat_cache`; anything the kernel does not
    handle exactly goes to that previous function. Returns whether the fused
    path is installed (False without Triton or upstream SAM2). Idempotent.
    """
    try:
        import sam2.modeling.sam.transformer as transformer
        import triton
        import triton.language as tl
    except ImportError:
        return False
    if getattr(transformer.apply_rotary_enc, "_fused", False):
        return True
    # Triton's bundled ptxas predates sm_121 (GB10); torch ships one that knows it.
    ptxas = Path(torch.__file__).parent / "bin" / "ptxas"
    if "TRITON_PTXAS_PATH" not in os.environ and ptxas.is_file():
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)

    @triton.jit
    def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_pairs, period, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_pairs
        a = tl.load(x_ptr + 2 * offs, mask=mask).to(tl.float32)
        b = tl.load(x_ptr + 2 * offs + 1, mask=mask).to(tl.float32)
        f = offs % period  # the frequency table tiles the sequence; never repeat it
        c = tl.load(cos_ptr + f, mask=mask)
        d = tl.load(sin_ptr + f, mask=mask)
        tl.store(out_ptr + 2 * offs, tl.fma(a, c, -(b * d)).to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + 2 * offs + 1, tl.fma(b, c, a * d).to(tl.bfloat16), mask=mask)

    def rotate(x, cos, sin):
        out = torch.empty_like(x)
        n = x.numel() // 2
        rope_kernel[(triton.cdiv(n, 1024),)](x, cos, sin, out, n, cos.numel(), BLOCK=1024)
        return out

    def fusable_pair(xq, xk, freqs_cis, repeat_freqs_k):
        return (_fusable(xq, freqs_cis) and xq.shape[-2] == freqs_cis.shape[0]
                and (xk.shape[-2] == 0 or _fusable(xk, freqs_cis))
                and (repeat_freqs_k or xk.shape[-2] in (0, xq.shape[-2])))

    previous = transformer.apply_rotary_enc
    tables: dict = {}
    verified: set = set()
    disabled = False

    def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
        nonlocal disabled
        if (disabled or torch.cuda.is_current_stream_capturing()
                or not (freqs_cis.is_cuda and freqs_cis.ndim == 2 and fusable_pair(xq, xk, freqs_cis, repeat_freqs_k))):
            return previous(xq, xk, freqs_cis, repeat_freqs_k)
        key = (id(freqs_cis), freqs_cis.data_ptr())
        if key not in tables:
            if len(tables) >= 16:  # one entry per attention layer is all a model needs
                tables.clear()
            # The entry holds freqs_cis itself, so its id cannot be reused while cached.
            tables[key] = (freqs_cis, freqs_cis.real.contiguous(), freqs_cis.imag.contiguous())
        _, cos, sin = tables[key]
        try:
            out = rotate(xq, cos, sin), (xk if xk.shape[-2] == 0 else rotate(xk, cos, sin))
        except Exception as error:  # Triton compile or launch failure
            disabled = True
            warnings.warn(f"fused RoPE disabled, using upstream: {error}", RuntimeWarning)
            return previous(xq, xk, freqs_cis, repeat_freqs_k)
        signature = (tuple(xq.shape), tuple(xk.shape), bool(repeat_freqs_k))
        if signature not in verified:
            reference = previous(xq, xk, freqs_cis, repeat_freqs_k)
            if not (torch.equal(out[0], reference[0]) and torch.equal(out[1], reference[1])):
                disabled = True
                warnings.warn("fused RoPE is not bit-identical to upstream on this stack; "
                              "disabled", RuntimeWarning)
                return reference
            verified.add(signature)
        return out

    apply_rotary_enc._fused = True
    transformer.apply_rotary_enc = apply_rotary_enc
    return True


def _map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if type(value) is dict:
        return {k: _map(v, fn) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_map(v, fn) for v in value)
    if value is None or type(value) in (bool, int, str):
        return value
    raise TypeError(f"unsupported graph value: {type(value)}")


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise TypeError("graph dictionaries need string keys")
        return ("dict", tuple((k, _signature(v)) for k, v in value.items()))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_signature(v) for v in value))
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    raise TypeError(f"unsupported graph signature: {type(value)}")


def _base_slice(tensor: torch.Tensor) -> tuple:
    # Expanded (stride-0) inputs are read-only in both modules: copy one element
    # along each broadcast dimension, keeping the original strides exactly.
    return tuple(slice(0, 1) if size > 1 and stride == 0 else slice(None)
                 for size, stride in zip(tensor.shape, tensor.stride()))


def _copy_inputs(target, source) -> None:
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
    """A module's `forward` replayed as CUDA graphs, one per input signature.

    A signature is captured on its second call and replayed afterwards. Any
    call this cannot replay exactly runs eagerly: grad or training mode, a
    non-CUDA / non-floating / overlapping input, a nested capture, a capture
    failure, or a full cache (`max_entries` graphs or `max_bytes` of captured
    allocations). Data-dependent Python control flow cannot be baked in: it
    needs a host sync, which fails the capture and falls back to eager.
    """

    def __init__(self, module, *, min_calls: int = 2, max_entries: int = 4, max_bytes: int = 512 << 20):
        self.module = module
        self.original = module.forward
        self.signature = inspect.signature(self.original)
        self.min_calls = min_calls
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: dict = {}
        self.seen: Counter = Counter()
        self.failed: set = set()
        self.counts: Counter = Counter()
        self.stream = None

    def __call__(self, *args, **kwargs):
        if (self.module.training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled()
                or torch.cuda.is_current_stream_capturing()):
            return self._eager(args, kwargs)
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        tensors: list = []
        try:
            _map(values, tensors.append)
            if not tensors or any(not t.is_cuda or t.layout != torch.strided or not t.is_floating_point()
                                  or any(s < 0 for s in t.stride())
                                  or torch._debug_has_internal_overlap(t[_base_slice(t)]) != 0
                                  for t in tensors):
                return self._eager(args, kwargs)
            device = tensors[0].device
            if any(t.device != device for t in tensors):
                return self._eager(args, kwargs)
            current = torch.cuda.current_stream(device)
            key = (_signature(values), torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
                   current.cuda_stream, threading.get_ident(),
                   torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                   torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                   torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
        except TypeError:
            return self._eager(args, kwargs)
        entry = self.entries.get(key)
        if entry is None:
            if key in self.failed or len(self.entries) >= self.max_entries:
                return self._eager(args, kwargs)
            if key not in self.seen and len(self.seen) >= 64:
                return self._eager(args, kwargs)
            self.seen[key] += 1
            if self.seen[key] < self.min_calls:
                return self._eager(args, kwargs)
            try:
                # Parameter casts belong inside the graph, not in autocast's cache.
                with torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"),
                                    dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False):
                    entry = self._capture(values, device, current)
            except RuntimeError as error:
                self.failed.add(key)
                self.counts["capture_failed"] += 1
                warnings.warn(f"SAM2 CUDA graph fell back to eager: {error}", RuntimeWarning)
                return self._eager(args, kwargs)
            if sum(e[3] for e in self.entries.values()) + entry[3] > self.max_bytes:
                current.synchronize()
                entry[1].reset()
                self.failed.add(key)
                self.counts["budget_rejected"] += 1
                return self._eager(args, kwargs)
            self.entries[key] = entry
            self.counts["captured"] += 1
        else:
            _copy_inputs(entry[0], values)
        entry[1].replay()
        self.counts["replays"] += 1
        return _map(entry[2], lambda t: t.clone())

    def _eager(self, args, kwargs):
        self.counts["eager"] += 1
        return self.original(*args, **kwargs)

    def _capture(self, values, device, current):
        before = torch.cuda.memory_allocated(device)
        static = _map(values, lambda t: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device))
        _copy_inputs(static, values)
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for _ in range(3):  # warm cuBLAS/cuDNN workspaces outside the capture
                self.original(**static)
        current.wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: a CUDA call from another thread (none today) must not
        # invalidate this capture or fail in that thread.
        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
            output = self.original(**static)
        current.wait_stream(self.stream)
        _signature(output)  # reject unsupported output types before first use
        return static, graph, output, max(0, torch.cuda.memory_allocated(device) - before)


def install_small_module_graphs(predictor) -> dict:
    """Wrap upstream SAM2VideoPredictor's memory encoder and mask decoder. Idempotent.

    Returns the wrappers by module name; empty for any other predictor type.
    """
    try:
        from sam2.modeling.memory_encoder import MemoryEncoder
        from sam2.modeling.sam.mask_decoder import MaskDecoder
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError:
        return {}
    if type(predictor) is not SAM2VideoPredictor:
        return {}
    if hasattr(predictor, "_small_graphs"):
        return predictor._small_graphs
    wrappers = {}
    for name, expected in (("memory_encoder", MemoryEncoder), ("sam_mask_decoder", MaskDecoder)):
        module = getattr(predictor, name)
        if type(module) is expected and not hasattr(module.forward, "_torchdynamo_orig_callable"):
            wrappers[name] = module.forward = SmallModuleGraphs(module)
    predictor._small_graphs = wrappers
    return wrappers
