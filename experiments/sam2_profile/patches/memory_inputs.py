"""Share identical cross-attention memory additions/casts within one forward.

SAM2's four MemoryAttention layers receive the same memory and position tensors.
Each repeats memory+pos, and CUDA BF16 autocast recasts identical FP32 key/value
inputs for each Linear. This patch performs the original addition once and the
same BF16 casts once per MemoryAttention invocation. It preserves eager query,
normalization, projection, attention, dropout and residual operation order.

The ContextVar scope lasts only one forward, resets on exceptions and nests
safely. Nothing is cached across frames, objects, resets or prompt corrections.
Unsupported modules, direct layer calls, training, autograd, non-inference,
non-CUDA, non-BF16-autocast and unsupported input dtypes use the prior methods.

GPU microtest (serialize with all other GPU work):
    python -m experiments.sam2_profile.patches.memory_inputs --microtest \
        --output runs/sam2_gpu_bottleneck/ab/20261016_followup/memory_inputs_micro.json
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch

_SCOPE = ContextVar('sam2_shared_memory_inputs', default=None)
_ALLOWED_DTYPES = (torch.float32, torch.bfloat16)


class _Scope:
    __slots__ = ('owner', 'memory', 'pos', 'value', 'keys')

    def __init__(self, owner):
        self.owner = owner
        self.memory = None
        self.pos = None
        self.value = None
        self.keys = {}


def _has_observing_hooks(module):
    from torch.nn.modules import module as module_impl
    return (bool(module_impl._global_forward_pre_hooks or module_impl._global_forward_hooks)
            or any(child._forward_pre_hooks or child._forward_hooks for child in module.modules()))


def _context_eligible(module):
    return (not module.training and torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled() and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == torch.bfloat16)


def _inputs_eligible(layer, memory, pos, tgt):
    from sam2.modeling.sam.transformer import RoPEAttention

    attention = layer.cross_attn_image
    if (not _context_eligible(layer) or type(attention) is not RoPEAttention
            or type(attention.k_proj) is not torch.nn.Linear
            or type(attention.v_proj) is not torch.nn.Linear
            or not memory.is_cuda or memory.dtype not in _ALLOWED_DTYPES
            or memory.ndim != 3 or memory.device != tgt.device
            or attention.training):
        return False
    if layer.pos_enc_at_cross_attn_keys:
        if (not isinstance(pos, torch.Tensor) or pos.device != memory.device
                or pos.dtype not in _ALLOWED_DTYPES or pos.shape != memory.shape):
            return False
    # Hooks can inspect the pre-autocast dtype or mutate inputs; preserve their
    # original observations instead of passing explicitly cast cached tensors.
    for module in (attention, attention.k_proj, attention.v_proj):
        if module._forward_pre_hooks or module._forward_hooks:
            return False
    return True


def _make_ca(previous, owner):
    @wraps(previous)
    def forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        from sam2.modeling.sam.transformer import RoPEAttention

        scope = _SCOPE.get()
        if (scope is None or scope.owner is not owner
                or not _inputs_eligible(self, memory, pos, tgt)
                or (scope.memory is not None and (scope.memory is not memory or scope.pos is not pos))):
            return previous(tgt, memory, query_pos, pos, num_k_exclude_rope)

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {'num_k_exclude_rope': num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        query = tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2
        if scope.memory is None:
            scope.memory, scope.pos = memory, pos
        use_position = bool(self.pos_enc_at_cross_attn_keys)
        if use_position not in scope.keys:
            # Do the same addition before the same dtype conversion that the
            # original Linear autocast wrapper would perform. Preserve format:
            # Tensor.to(dtype) is also the operation used by cached_cast.
            key = memory + pos if use_position else memory
            scope.keys[use_position] = key.to(dtype=torch.bfloat16)
        if scope.value is None:
            scope.value = memory.to(dtype=torch.bfloat16)
        tgt2 = self.cross_attn_image(q=query, k=scope.keys[use_position], v=scope.value, **kwds)
        return tgt + self.dropout2(tgt2)
    return forward_ca


def install(predictor) -> None:
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer

    owner = predictor if isinstance(predictor, MemoryAttention) else getattr(predictor, 'memory_attention', None)
    if type(owner) is not MemoryAttention or getattr(owner, '_shared_memory_inputs_patch', False):
        return
    # Avoid replacing behavior from an unknown MemoryAttentionLayer subclass or
    # an unrelated earlier _forward_ca override.
    if any(type(layer) is not MemoryAttentionLayer
           or getattr(layer._forward_ca, '__func__', None) is not MemoryAttentionLayer._forward_ca
           for layer in owner.layers):
        return
    previous_forward = owner.forward

    @wraps(previous_forward)
    def forward(self, *args, **kwargs):
        if not _context_eligible(self) or _has_observing_hooks(self):
            return previous_forward(*args, **kwargs)
        token = _SCOPE.set(_Scope(self))
        try:
            return previous_forward(*args, **kwargs)
        finally:
            _SCOPE.reset(token)

    for layer in owner.layers:
        layer._forward_ca = MethodType(_make_ca(layer._forward_ca, owner), layer)
    owner.forward = MethodType(forward, owner)
    owner._shared_memory_inputs_patch = True


def _microtest(args):
    import json
    from pathlib import Path
    import time

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from experiments.sam2_profile.fused_rope import install_fused_rope

    assets = Path('/home/valentinweyer/projects/handball-computer-vision')
    config_path = assets / 'sam2-upstream/sam2/configs/sam2.1/sam2.1_hiera_l.yaml'
    checkpoint = assets / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'
    config = OmegaConf.load(config_path)
    module = instantiate(config.model.memory_attention).eval().cuda()
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    prefix = 'memory_attention.'
    module.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
    del state
    torch.manual_seed(31)
    install_fused_rope()
    original = module.forward
    install(module)
    assert getattr(module, '_shared_memory_inputs_patch', False), 'patch did not install'
    rows = []
    retained = []
    report = dict(checkpoint=str(checkpoint), trained_weights=True, torch=torch.__version__,
                  device=torch.cuda.get_device_name(), baseline='fused_rope', results=rows)

    def save():
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')

    def bench(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        return dict(cuda_ms=start.elapsed_time(end) / args.iterations,
                    wall_ms=(time.perf_counter() - wall) * 1000 / args.iterations)

    def check(name, memories, pointers, memory_dtype, pos_dtype, tokens=4096,
              timed=False, list_curr=False):
        memory_tokens = tokens * memories + pointers
        curr = torch.randn(tokens, 1, 256, device='cuda')
        curr_pos = torch.randn_like(curr)
        memory = torch.randn(memory_tokens, 1, 64, device='cuda', dtype=memory_dtype)
        # Expanded zero-stride position embeddings exercise preserve-format casts.
        memory_pos = torch.randn(1, 1, 64, device='cuda', dtype=pos_dtype).expand(memory_tokens, 1, 64)
        saved = [value.clone() for value in (curr, memory, curr_pos, memory_pos)]
        kwargs = dict(curr=[curr] if list_curr else curr, memory=memory,
                      curr_pos=[curr_pos] if list_curr else curr_pos,
                      memory_pos=memory_pos, num_obj_ptr_tokens=pointers)
        expected = original(**kwargs)
        actual = module(**kwargs)
        changed = int((expected != actual).sum())
        row = dict(name=name, memory_tokens=memory_tokens, pointer_tokens=pointers,
                   memory_dtype=str(memory_dtype), position_dtype=str(pos_dtype),
                   autocast=torch.is_autocast_enabled('cuda'),
                   grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled(),
                   equal=torch.equal(expected, actual), changed=changed,
                   max_abs=float((expected - actual).abs().max()),
                   inputs_unchanged=all(torch.equal(value, clone)
                                        for value, clone in zip((curr, memory, curr_pos, memory_pos), saved)),
                   scope_reset=_SCOPE.get() is None)
        if timed and row['equal']:
            row['base'] = bench(lambda: original(**kwargs))
            row['candidate'] = bench(lambda: module(**kwargs))
            row['wall_speedup'] = row['base']['wall_ms'] / row['candidate']['wall_ms']
        rows.append(row)
        retained.append((actual, actual.detach().clone()))
        report['all_equal_so_far'] = all(item['equal'] for item in rows)
        save()
        print(json.dumps(row), flush=True)
        if not row['equal'] or not row['inputs_unchanged'] or not row['scope_reset']:
            raise SystemExit('Shared memory inputs failed exactness or isolation')

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for memories, pointers in ((1, 0), (1, 4), (3, 16), (7, 64)):
            check(f'bank_{memories}_pointers_{pointers}', memories, pointers,
                  torch.float32, torch.float32, timed=True, list_curr=memories == 3)
        check('bf16_memory_fp32_position', 3, 16, torch.bfloat16, torch.float32)
        check('bf16_memory_bf16_position', 3, 16, torch.bfloat16, torch.bfloat16)
        # Unsupported FP16 inputs must preserve the original path exactly.
        check('fp16_input_fallback', 1, 4, torch.float16, torch.float16, tokens=64)
        old_flags = [(layer.pos_enc_at_cross_attn_keys, layer.pos_enc_at_cross_attn_queries)
                     for layer in module.layers]
        for index, layer in enumerate(module.layers):
            layer.pos_enc_at_cross_attn_keys = bool(index % 2)
            layer.pos_enc_at_cross_attn_queries = not bool(index % 2)
        try:
            check('mixed_positional_flags', 1, 4, torch.float32, torch.float32, tokens=64)
        finally:
            for layer, flags in zip(module.layers, old_flags):
                layer.pos_enc_at_cross_attn_keys, layer.pos_enc_at_cross_attn_queries = flags
        with torch.autocast('cuda', enabled=False):
            check('no_autocast_fallback', 1, 4, torch.float32, torch.float32, tokens=64)
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        check('grad_enabled_fallback', 1, 4, torch.float32, torch.float32, tokens=64)
    report['retained_outputs_equal'] = all(torch.equal(value, clone) for value, clone in retained)
    report['all_equal'] = all(row['equal'] and row['inputs_unchanged'] and row['scope_reset'] for row in rows)
    save()
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}), flush=True)
    if not report['retained_outputs_equal']:
        raise SystemExit('Previously retained outputs were overwritten')


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--microtest', action='store_true', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    _microtest(args)
