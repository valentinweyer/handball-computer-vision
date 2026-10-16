"""Rejected experimental LayerNorm2d fusion; not an exact-output winner.

The corrected autocast-aware kernel still differed on 58,190 / 262,144 values
for shape [1, 4, 256, 256], scale 1, with maximum absolute error
2.384185791015625e-07. Exactness failed; no tolerances were relaxed. The
remaining difference may come from CUDA sqrt/divide or affine operation
semantics. Retained only to reproduce this negative result; do not promote.


Under CUDA autocast SAM2's BF16 LayerNorm2d computes BF16 mean/difference,
then pow(2) autocast-promotes to FP32. Variance, sqrt, divide and affine stay
FP32. A normal fused LayerNorm changes that computation. This patch keeps
both original torch.mean operations and the BF16 subtraction round boundary,
combining only the surrounding elementwise operations into two Triton kernels.
It does not retain or reuse any activation buffers across invocations.

GPU identity/timing microtest (serialize with all other GPU measurements):
    python -m experiments.sam2_profile.patches.layernorm2d_fused --microtest
"""
from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _square_difference(x_ptr, mean_ptr, squared_ptr, n, C: tl.constexpr,
                       HW: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    mean_offs = (offs // (C * HW)) * HW + offs % HW
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + mean_offs, mask=mask, other=0).to(tl.float32)
    difference = (x - mean).to(tl.bfloat16).to(tl.float32)
    square = difference * difference
    tl.store(squared_ptr + offs, square, mask=mask)


@triton.jit
def _normalize_affine(x_ptr, mean_ptr, variance_ptr, weight_ptr, bias_ptr,
                      out_ptr, n, C: tl.constexpr, HW: tl.constexpr,
                      EPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    channel = (offs // HW) % C
    mean_offs = (offs // (C * HW)) * HW + offs % HW
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + mean_offs, mask=mask, other=0).to(tl.float32)
    variance = tl.load(variance_ptr + mean_offs, mask=mask, other=0).to(tl.float32)
    difference = (x - mean).to(tl.bfloat16).to(tl.float32)
    adjusted_variance = variance + EPS
    denominator = tl.sqrt_rn(adjusted_variance)
    normalized = tl.div_rn(difference, denominator)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0)
    # Disable FP fusion at launch: eager performs separate FP32 mul and add.
    result = weight * normalized + bias
    tl.store(out_ptr + offs, result, mask=mask)


def _make_forward(previous):
    def forward(self, x):
        if (torch.is_grad_enabled() or not torch.is_autocast_enabled('cuda')
                or not x.is_cuda or x.dtype != torch.bfloat16
                or x.ndim != 4 or not x.is_contiguous() or x.numel() == 0
                or self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32
                or self.weight.device != x.device or self.bias.device != x.device
                or self.weight.ndim != 1 or self.bias.ndim != 1
                or self.weight.numel() != x.shape[1] or self.bias.numel() != x.shape[1]
                or not self.weight.is_contiguous() or not self.bias.is_contiguous()):
            return previous(x)
        n, c, hw = x.numel(), x.shape[1], x.shape[2] * x.shape[3]
        mean = x.mean(1, keepdim=True)
        squared = torch.empty_like(x, dtype=torch.float32)
        _square_difference[(triton.cdiv(n, 1024),)](
            x, mean, squared, n, C=c, HW=hw, BLOCK=1024, enable_fp_fusion=False)
        variance = squared.mean(1, keepdim=True)
        out = torch.empty_like(x, dtype=torch.float32)
        _normalize_affine[(triton.cdiv(n, 1024),)](
            x, mean, variance, self.weight, self.bias, out, n, C=c, HW=hw,
            EPS=self.eps, BLOCK=1024, enable_fp_fusion=False)
        return out
    return forward


def install(predictor) -> None:
    """Refuse promotion of the measured inexact experiment."""
    raise RuntimeError(
        "Rejected LayerNorm2d experiment: torch.equal failed for 58,190/262,144 "
        "values (max_abs=2.384185791015625e-07). Use --microtest only to "
        "reproduce the negative result; do not stack this patch.")


def _install_candidate(predictor) -> None:
    from sam2.modeling.sam2_utils import LayerNorm2d
    for module in predictor.modules():
        if isinstance(module, LayerNorm2d) and not getattr(module, '_layernorm2d_fused_patch', False):
            module.forward = MethodType(_make_forward(module.forward), module)
            module._layernorm2d_fused_patch = True


def _microtest():
    import json
    from sam2.modeling.sam2_utils import LayerNorm2d

    def bench(fn, iterations=30):
        for _ in range(5):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations

    torch.manual_seed(13)
    cases = [(1, 4, 256, 256), (1, 16, 256, 256), (1, 64, 128, 128),
             (1, 64, 256, 256), (1, 256, 64, 64), (2, 256, 16, 32)]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for shape in cases:
            module = LayerNorm2d(shape[1]).eval().cuda()
            module.weight.uniform_(-2, 2)
            module.bias.uniform_(-2, 2)
            original = module.forward
            _install_candidate(module)
            for scale in (1.0, 0.001, 100.0):
                x = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale
                expected = original(x)
                actual = module(x)
                row = dict(shape=shape, scale=scale, equal=torch.equal(expected, actual),
                           changed=int((expected != actual).sum()),
                           max_abs=float((expected - actual).abs().max()),
                           original_dtype=str(expected.dtype), dtype=str(actual.dtype))
                print(json.dumps(row), flush=True)
                if not row['equal'] or expected.dtype != actual.dtype:
                    raise SystemExit('LayerNorm2d equality failed')
            base_ms = bench(lambda: original(x))
            candidate_ms = bench(lambda: module(x))
            print(json.dumps(dict(shape=shape, base_ms=base_ms, candidate_ms=candidate_ms,
                                  speedup=base_ms / candidate_ms)), flush=True)
            # Non-contiguous and FP32 activations retain the original implementation.
            for fallback_x in (x.transpose(2, 3), x.float()):
                assert torch.equal(original(fallback_x), module(fallback_x)), 'fallback differs'
    print(json.dumps({'all_equal': True, 'shapes': len(cases)}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--microtest', action='store_true', required=True)
    parser.parse_args()
    _microtest()
