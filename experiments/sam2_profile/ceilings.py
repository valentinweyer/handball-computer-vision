"""Measure achievable BF16 tensor throughput and DRAM bandwidth on the local GPU.

These are the two roofs the per-op roofline in roofline.py is judged against.
"""
import json
import sys

import torch
import torch.nn.functional as F


def bench(fn, iters=20, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best  # ms


def main():
    dev, bf = 'cuda', torch.bfloat16
    out = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'gemm': [], 'bandwidth': [], 'sdpa': []}
    for m, n, k in [(8192, 8192, 8192), (4096, 4096, 4096), (16384, 4096, 1024),
                    (4096, 256, 256), (4096, 2048, 256), (28736, 256, 64), (65536, 1152, 288)]:
        a = torch.randn(m, k, device=dev, dtype=bf)
        b = torch.randn(k, n, device=dev, dtype=bf)
        ms = bench(lambda: a @ b)
        flops = 2 * m * n * k
        byts = (m * k + k * n + m * n) * 2
        out['gemm'].append({'mnk': [m, n, k], 'ms': ms, 'tflops': flops / ms / 1e9, 'gbs': byts / ms / 1e6})
    for mb in [64, 256, 1024]:
        n = mb * 2**20 // 4
        x = torch.empty(n, device=dev, dtype=torch.float32).uniform_()
        y = torch.empty_like(x)
        ms = bench(lambda: y.copy_(x))
        out['bandwidth'].append({'op': 'copy', 'MB': mb, 'gbs': 2 * n * 4 / ms / 1e6})
        ms = bench(lambda: x.sum())
        out['bandwidth'].append({'op': 'sum', 'MB': mb, 'gbs': n * 4 / ms / 1e6})
        ms = bench(lambda: torch.add(x, 1.0, out=y))
        out['bandwidth'].append({'op': 'add_scalar', 'MB': mb, 'gbs': 2 * n * 4 / ms / 1e6})
    with torch.inference_mode():
        for m, n, d, h in [(4096, 28736, 256, 1), (4096, 4096, 256, 1), (8192, 8192, 64, 16)]:
            q = torch.randn(1, h, m, d, device=dev, dtype=bf)
            k = torch.randn(1, h, n, d, device=dev, dtype=bf)
            v = torch.randn(1, h, n, d, device=dev, dtype=bf)
            ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            out['sdpa'].append({'m_n_d_h': [m, n, d, h], 'ms': ms, 'tflops': 4 * h * m * n * d / ms / 1e9})
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
