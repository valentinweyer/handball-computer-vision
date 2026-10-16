"""Classify the ops in roofline.py's ops.jsonl against measured GB10 roofs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

# Measured by ceilings.py on this stack (runs/sam2_gpu_bottleneck/ceilings.json).
GEMM_TFLOPS, ATTN_TFLOPS, DRAM_GBS = 64.0, 95.0, 235.0


def group(module):
    parts = module.split('.')
    if parts[0] == 'memory_attention' and len(parts) >= 3 and parts[1] == 'layers':
        sub = parts[3] if len(parts) > 3 else '(layer)'
        return 'memory_attention.' + {'cross_attn_image': 'cross_attn', 'self_attn': 'self_attn'}.get(sub, 'ffn/norm/other')
    return parts[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ops', type=Path)
    args = p.parse_args()
    ops = [json.loads(l) for l in args.ops.open()]
    frames = sorted({o['frame'] for o in ops})
    n = len(frames)
    by_group = defaultdict(lambda: defaultdict(float))
    by_class = defaultdict(float)
    by_op = defaultdict(lambda: [0.0, 0, 0.0, 0.0])
    total = 0.0
    for o in ops:
        ms = o['ms'] / n
        total += ms
        peak = ATTN_TFLOPS if 'attention' in o['op'] else GEMM_TFLOPS
        t_c = o['flops'] / (peak * 1e9) / n
        t_m = o['bytes'] / (DRAM_GBS * 1e6) / n
        if o['flops'] == 0 and o['bytes'] == 0:
            cls = 'no-data (views/sync)'
        else:
            side = 'compute' if t_c >= t_m else 'memory'
            eff = max(t_c, t_m) / ms if ms > 0 else 1.0
            cls = f"{side}-side, {'>=50%' if eff >= .5 else '20-50%' if eff >= .2 else '<20%'} of roof"
        by_class[cls] += ms
        g = by_group[group(o['module'])]
        g['ms'] += ms; g['flops'] += o['flops'] / n; g['bytes'] += o['bytes'] / n; g['t_c'] += t_c; g['t_m'] += t_m
        g['ops'] += 1 / n
        b = by_op[o['op']]
        b[0] += ms; b[1] += 1; b[2] += o['flops'] / n; b[3] += o['bytes'] / n
    print(f'frames {frames}; summed per-op GPU time {total:.1f} ms/frame; ops/frame {len(ops)/n:.0f}\n')
    print(f"{'module group':32s} {'ms':>7s} {'%':>5s} {'ops':>6s} {'GFLOP':>7s} {'MB':>7s} {'TF/s':>6s} {'GB/s':>6s} {'FLOP/B':>7s} {'roof ms':>8s} {'eff':>5s}")
    for k, g in sorted(by_group.items(), key=lambda x: -x[1]['ms']):
        roof = max(g['t_c'], g['t_m'])  # group-level: assumes perfect fusion within the group
        print(f"{k:32s} {g['ms']:7.1f} {100*g['ms']/total:5.1f} {g['ops']:6.0f} {g['flops']/1e9:7.1f} {g['bytes']/1e6:7.0f} "
              f"{g['flops']/g['ms']/1e9:6.1f} {g['bytes']/g['ms']/1e6:6.0f} {g['flops']/max(g['bytes'],1):7.0f} "
              f"{g['t_c']+g['t_m']:8.1f} {(g['t_c']+g['t_m'])/g['ms']:5.2f}")
    print('\nper-op classification (time share):')
    for k, v in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f'  {k:34s} {v:7.1f} ms  {100*v/total:5.1f}%')
    print('\ntop ops by time:')
    print(f"{'op':52s} {'ms':>7s} {'calls/fr':>8s} {'TF/s':>6s} {'GB/s':>6s} {'FLOP/B':>7s}")
    for k, (ms, c, f, b) in sorted(by_op.items(), key=lambda x: -x[1][0])[:22]:
        print(f"{k[:52]:52s} {ms:7.1f} {c/n:8.0f} {f/ms/1e9:6.1f} {b/ms/1e6:6.0f} {f/max(b,1):7.0f}")
    small = [o for o in ops if o['ms'] < 0.02 and not o['view']]
    print(f"\nnon-view ops < 20 us: {len(small)/n:.0f}/frame, {sum(o['ms'] for o in small)/n:.1f} ms/frame")


if __name__ == '__main__':
    main()
