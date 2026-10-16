"""Single-kernel RoPE for SAM2 memory attention, bit-identical to upstream.

Upstream `apply_rotary_enc` casts BF16 q/k to FP32, multiplies as complex64
against a (repeated) complex frequency table and casts back: ~6 full passes
over a 28672x256 key tensor per object per layer. This kernel reads BF16 once,
rotates in FP32 registers and writes BF16 once, indexing the frequency table
modulo its length instead of repeating it.

Bit-identity depends on the FMA pattern: eager complex64 multiply on this
stack equals real = fma(a, c, -(b*d)), imag = fma(b, c, a*d). Other orderings
(and torch.compile of a real-arithmetic rewrite) differ by one BF16 ulp in a
few dozen of 7.3M values.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_pairs, period, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    a = tl.load(x_ptr + 2 * offs, mask=mask).to(tl.float32)
    b = tl.load(x_ptr + 2 * offs + 1, mask=mask).to(tl.float32)
    f = offs % period
    c = tl.load(cos_ptr + f, mask=mask)
    d = tl.load(sin_ptr + f, mask=mask)
    re = tl.fma(a, c, -(b * d))
    im = tl.fma(b, c, a * d)
    tl.store(out_ptr + 2 * offs, re.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + 2 * offs + 1, im.to(tl.bfloat16), mask=mask)


def _rotate(x, cos, sin):
    out = torch.empty_like(x)
    n = x.numel() // 2
    _rope_kernel[(triton.cdiv(n, 1024),)](x, cos, sin, out, n, cos.numel(), BLOCK=1024)
    return out


def install_fused_rope() -> None:
    """Replace sam2's apply_rotary_enc with the fused kernel where it applies.

    Falls back to the previously installed function for anything that is not
    contiguous BF16 on CUDA with a (tokens, pairs) frequency table that tiles
    the sequence. Install after `install_rope_repeat_cache`. Idempotent.
    """
    import sam2.modeling.sam.transformer as transformer

    if getattr(transformer.apply_rotary_enc, '_fused', False):
        return
    previous = transformer.apply_rotary_enc
    tables: dict = {}

    def fusable(x, freqs_cis):
        return (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-2] > 0
                and x.shape[-2] % freqs_cis.shape[0] == 0 and x.shape[-1] == 2 * freqs_cis.shape[1])

    def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
        if not (freqs_cis.is_cuda and freqs_cis.ndim == 2 and fusable(xq, freqs_cis)
                and (xk.shape[-2] == 0 or fusable(xk, freqs_cis))
                and (repeat_freqs_k or xk.shape[-2] in (0, xq.shape[-2]))
                and xq.shape[-2] == freqs_cis.shape[0]):
            return previous(xq, xk, freqs_cis, repeat_freqs_k)
        key = (id(freqs_cis), freqs_cis.data_ptr())
        if key not in tables:
            if len(tables) >= 16:
                tables.clear()
            tables[key] = (freqs_cis, freqs_cis.real.contiguous(), freqs_cis.imag.contiguous())
        _, cos, sin = tables[key]
        xq_out = _rotate(xq, cos, sin)
        return xq_out, (xk if xk.shape[-2] == 0 else _rotate(xk, cos, sin))

    apply_rotary_enc._fused = True
    transformer.apply_rotary_enc = apply_rotary_enc


def install(predictor=None) -> None:
    """ab.py patch entry point."""
    install_fused_rope()
