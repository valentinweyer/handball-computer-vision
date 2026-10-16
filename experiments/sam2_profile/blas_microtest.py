"""Compare installed cuBLAS choices on measured SAM2 linear shapes, one GPU job."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shapes = defaultdict(lambda: {'ms': 0, 'count': 0})
    profile = Path('runs/sam2_gpu_bottleneck/roofline/ops.jsonl')
    for line in profile.open():
        row = json.loads(line)
        if row['op'] not in ('aten.addmm.default', 'aten.mm.default'):
            continue
        ins = row['shapes'][1:3] if row['op'] == 'aten.addmm.default' else row['shapes'][:2]
        if len(ins) != 2 or any(len(x) != 2 for x in ins):
            continue
        (m, k), (k2, n) = ins
        if k != k2:
            continue
        key = (m, n, k, row['op'] == 'aten.addmm.default')
        shapes[key]['ms'] += row['ms']
        shapes[key]['count'] += 1
    selected = sorted(shapes, key=lambda k: shapes[k]['ms'], reverse=True)[:16]
    torch.manual_seed(0)
    results = []
    original = torch.backends.cuda.preferred_blas_library()
    try:
        with torch.inference_mode():
            for m, n, k, bias in selected:
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                weight = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
                b = torch.randn(n, device='cuda', dtype=torch.bfloat16) if bias else None
                outputs = {}
                timings = {}
                for backend in ('cublas', 'cublaslt'):
                    torch.backends.cuda.preferred_blas_library(backend)
                    for _ in range(5):
                        y = F.linear(x, weight, b)
                    torch.cuda.synchronize()
                    outputs[backend] = y.clone()
                    samples = []
                    for _ in range(3):
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(20):
                            y = F.linear(x, weight, b)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / 20)
                    timings[backend] = sorted(samples)[1]
                a, b = outputs.values()
                row = {'mnk': [m, n, k], 'bias': bias, 'profile': shapes[(m,n,k,bias)],
                       'ms': timings, 'identical': torch.equal(a, b),
                       'different': int((a != b).sum().item()),
                       'max_abs': float((a.float()-b.float()).abs().max().item())}
                results.append(row)
                print(json.dumps(row), flush=True)
    finally:
        torch.backends.cuda.preferred_blas_library(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'torch':torch.__version__, 'rows':results}, indent=2)+'\n')

if __name__ == '__main__':
    main()
