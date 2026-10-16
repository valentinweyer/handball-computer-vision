"""Rotate newly projected SAM2 q/k buffers in place during inference.

Only patches predictor-owned RoPEAttention instances. The key suffix containing
object pointers is untouched. Each invocation owns its projection buffers, so
outputs retained by the predictor are never overwritten by another invocation.
Unsupported strides, dtypes, frequency tables and autograd use the installed
rotary implementation with the upstream assignment semantics.

GPU microtest (run serially with all other GPU measurements):
    python -m experiments.sam2_profile.patches.rope_inplace --microtest
"""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
import triton

from experiments.sam2_profile.fused_rope import _rope_kernel


def _rotate_inplace(x, cos, sin):
    n = x.numel() // 2
    if n:
        _rope_kernel[(triton.cdiv(n, 1024),)](
            x, cos, sin, x, n, cos.numel(), BLOCK=1024)
    return x


def install(predictor) -> None:
    import sam2.modeling.sam.transformer as transformer

    tables = {}

    def fusable(x, freqs):
        return (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.shape[-2] > 0 and x.shape[-2] % freqs.shape[0] == 0
                and x.shape[-1] == 2 * freqs.shape[1])

    def forward(self, q, k, v, num_k_exclude_rope=0):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        num_k_rope = k.size(-2) - num_k_exclude_rope
        key_prefix = k[:, :, :num_k_rope]
        freqs = self.freqs_cis
        if (not torch.is_grad_enabled() and freqs.is_cuda and freqs.ndim == 2
                and freqs.dtype == torch.complex64 and freqs.shape[0] > 0
                and q.shape[-2] == freqs.shape[0] and fusable(q, freqs)
                and (key_prefix.shape[-2] == 0 or fusable(key_prefix, freqs))
                and (self.rope_k_repeat or key_prefix.shape[-2] in (0, q.shape[-2]))):
            # Retain the source table so Python object and allocator address reuse
            # cannot cause a stale cache hit; bound retention across shape changes.
            cache_key = (id(freqs), freqs.data_ptr(), tuple(freqs.shape))
            # Inference tensors have no version counter; query without raising
            # an exception on every object/layer invocation.
            version = None if torch.is_inference(freqs) else freqs._version
            cache_key += (version,)
            if cache_key not in tables:
                if len(tables) >= 16:
                    tables.clear()
                tables[cache_key] = (freqs, freqs.real.contiguous(), freqs.imag.contiguous())
            _, cos, sin = tables[cache_key]
            _rotate_inplace(q, cos, sin)
            _rotate_inplace(key_prefix, cos, sin)
        else:
            q, k[:, :, :num_k_rope] = transformer.apply_rotary_enc(
                q, key_prefix, freqs_cis=freqs, repeat_freqs_k=self.rope_k_repeat)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return self.out_proj(self._recombine_heads(out))

    for module in predictor.modules():
        if isinstance(module, transformer.RoPEAttention) and not getattr(module, '_rope_inplace_patch', False):
            module.forward = MethodType(forward, module)
            module._rope_inplace_patch = True


def _microtest():
    import json
    from sam2.modeling.sam.transformer import RoPEAttention
    from experiments.sam2_profile.fused_rope import install_fused_rope

    def bench(fn, iterations=15):
        for _ in range(3):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations

    torch.manual_seed(7)
    install_fused_rope()
    module = RoPEAttention(embedding_dim=256, num_heads=1, kv_in_dim=64,
                           rope_k_repeat=True).eval().cuda()
    original = module.forward
    install(module)
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for memories, pointers in [(1, 0), (1, 4), (3, 16), (7, 64)]:
            q = torch.randn(1, 4096, 256, device='cuda')
            k = torch.randn(1, 4096 * memories + pointers, 64, device='cuda')
            q_saved, k_saved = q.clone(), k.clone()
            expected = original(q, k, k, num_k_exclude_rope=pointers)
            actual = module(q, k, k, num_k_exclude_rope=pointers)
            torch.cuda.synchronize()
            row = dict(memories=memories, pointers=pointers,
                       equal=torch.equal(expected, actual),
                       changed=int((expected != actual).sum()),
                       max_abs=float((expected - actual).abs().max()),
                       inputs_unchanged=torch.equal(q, q_saved) and torch.equal(k, k_saved))
            if row['equal'] and row['inputs_unchanged']:
                row['base_ms'] = bench(lambda: original(q, k, k, num_k_exclude_rope=pointers))
                row['candidate_ms'] = bench(lambda: module(q, k, k, num_k_exclude_rope=pointers))
                row['speedup'] = row['base_ms'] / row['candidate_ms']
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not row['equal'] or not row['inputs_unchanged']:
                raise SystemExit('In-place RoPE module equality failed')
        # Non-contiguous projected heads and FP32 projections take the fallback.
        with torch.autocast('cuda', enabled=False):
            q = torch.randn(2, 64, 256, device='cuda')
            k = torch.randn(2, 64, 64, device='cuda')
            expected = original(q, k, k)
            actual = module(q, k, k)
            assert torch.equal(expected, actual), 'FP32 fallback differs'
    print(json.dumps({'all_equal': True, 'cases': len(rows) + 1}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--microtest', action='store_true', required=True)
    parser.parse_args()
    _microtest()
