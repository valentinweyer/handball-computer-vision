"""Attribute nsys kernel time to SAM2 modules and frames, and join with roofline FLOPs/bytes.

Inputs: the sqlite export of `nsys profile -t cuda,nvtx` over `roofline.py --mode plain`
and the ops.jsonl from `roofline.py --mode roofline`. Kernels are attributed to the
innermost NVTX module range open on the launching thread at launch time.
"""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from experiments.sam2_profile.analyze_roofline import ATTN_TFLOPS, DRAM_GBS, GEMM_TFLOPS, group


def kernel_class(name):
    n = name.lower()
    if 'flash' in n or 'fmha' in n:
        return 'attention'
    if 'gemm' in n or 'gemv' in n or 'splitkreduce' in n:
        return 'gemm' if 'magma_sgemm' not in n else 'gemm_fp32'
    if 'conv' in n or 'dgrad' in n or 'fprop' in n:
        return 'conv'
    return 'memory'


def op_class(op):
    if 'attention' in op:
        return 'attention'
    if any(k in op for k in ('addmm', '.mm.', 'bmm', 'aten.mm')):
        return 'gemm'
    if 'convolution' in op:
        return 'conv'
    return 'memory'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sqlite', type=Path)
    p.add_argument('ops', type=Path)
    p.add_argument('--steady-min-frame', type=int, default=8, help='memory bank is full from here')
    args = p.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute('select id, value from StringIds'))
    nvtx = db.execute('select start, end, text, globalTid from NVTX_EVENTS where eventType=59 and end is not null').fetchall()
    frames = sorted((s, e) for s, e, t, _ in nvtx if t == 'frame')
    ranges = defaultdict(list)  # tid -> [(start, end, name)]
    for s, e, t, tid in nvtx:
        if t and t.startswith('m:'):
            ranges[tid].append((s, e, t[2:]))
    for v in ranges.values():
        v.sort()
    launches = dict(((c, (s, tid)) for c, s, tid in db.execute(
        'select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')))
    kernels = db.execute('select start, end, demangledName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()

    def module_at(tid, t):
        rs = ranges.get(tid, [])
        i = bisect.bisect_right(rs, (t, float('inf'), ''))
        best = None
        for j in range(i - 1, max(i - 400, -1), -1):  # innermost = latest start that still contains t
            s, e, name = rs[j]
            if s <= t <= e:
                best = name
                break
        return best or '(predictor)'

    frame_starts = [s for s, _ in frames]
    per_frame = defaultdict(lambda: {'kernel_ns': 0, 'busy': []})
    group_ns = defaultdict(lambda: defaultdict(int))
    class_ns = defaultdict(lambda: defaultdict(int))
    cross_ns = defaultdict(lambda: defaultdict(int))
    for ks, ke, name_id, corr in kernels:
        launch = launches.get(corr)
        if launch is None:
            continue
        ls, tid = launch
        fi = bisect.bisect_right(frame_starts, ls) - 1
        if fi < 0 or ls > frames[fi][1]:
            continue
        name = strings[name_id]
        pf = per_frame[fi]
        pf['kernel_ns'] += ke - ks
        pf['busy'].append((ks, ke))
        g = group(module_at(tid, ls))
        group_ns[fi][g] += ke - ks
        class_ns[fi][kernel_class(name)] += ke - ks
        cross_ns[fi][(g, kernel_class(name))] += ke - ks
    rows = []
    for fi, (s, e) in enumerate(frames):
        busy = sorted(per_frame[fi]['busy'])
        union, cur_s, cur_e = 0, None, None
        for a, b in busy:  # union of kernel intervals, clipped to the frame window
            a, b = max(a, s), min(b, e)
            if b <= a:
                continue
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    union += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None:
            union += cur_e - cur_s
        rows.append({'frame': fi + 1, 'wall_ms': (e - s) / 1e6, 'busy_ms': union / 1e6,
                     'kernel_ms': per_frame[fi]['kernel_ns'] / 1e6})
    steady = [r for r in rows if r['frame'] >= args.steady_min_frame and r['wall_ms'] < 1.5 * sorted(x['wall_ms'] for x in rows)[len(rows) // 2]]
    sidx = [r['frame'] - 1 for r in steady]
    n = len(steady)
    wall = sum(r['wall_ms'] for r in steady) / n
    busy = sum(r['busy_ms'] for r in steady) / n
    print(f"steady frames {[r['frame'] for r in steady]}")
    print(f"wall {wall:.1f} ms/frame; GPU busy {busy:.1f} ms ({100*busy/wall:.1f}%); idle {wall-busy:.1f} ms\n")

    ops = [json.loads(l) for l in args.ops.open()]
    nops = len({o['frame'] for o in ops})
    gf = defaultdict(lambda: [0.0, 0.0])
    cf = defaultdict(lambda: [0.0, 0.0])
    for o in ops:
        gf[group(o['module'])][0] += o['flops'] / nops
        gf[group(o['module'])][1] += o['bytes'] / nops
        cf[op_class(o['op'])][0] += o['flops'] / nops
        cf[op_class(o['op'])][1] += o['bytes'] / nops

    def table(title, ns_by_frame, fb, peak_of):
        keys = {k for fi in sidx for k in ns_by_frame[fi]}
        print(f"{title:32s} {'GPU ms':>7s} {'%busy':>6s} {'GFLOP':>7s} {'MB':>7s} {'TF/s':>6s} {'GB/s':>6s} {'FLOP/B':>7s} {'%TFpk':>6s} {'%BWpk':>6s}")
        for k in sorted(keys, key=lambda k: -sum(ns_by_frame[fi][k] for fi in sidx)):
            ms = sum(ns_by_frame[fi][k] for fi in sidx) / n / 1e6
            f, b = fb.get(k, (0.0, 0.0))
            tf, gbs = f / ms / 1e9, b / ms / 1e6
            print(f"{k:32s} {ms:7.1f} {100*ms/busy:6.1f} {f/1e9:7.1f} {b/1e6:7.0f} {tf:6.1f} {gbs:6.0f} {f/max(b,1):7.0f} "
                  f"{100*tf/peak_of(k):6.0f} {100*gbs/DRAM_GBS:6.0f}")
        print()

    table('module group', group_ns, gf, lambda k: GEMM_TFLOPS)
    table('kernel class', class_ns, cf, lambda k: ATTN_TFLOPS if k == 'attention' else GEMM_TFLOPS)
    classes = ['attention', 'gemm', 'conv', 'memory']
    print(f"{'GPU ms by module x kernel class':32s} " + ' '.join(f'{c:>9s}' for c in classes))
    for g in sorted({g for fi in sidx for g, _ in cross_ns[fi]}, key=lambda g: -sum(group_ns[fi][g] for fi in sidx)):
        cells = [sum(cross_ns[fi][(g, c)] for fi in sidx) / n / 1e6 for c in classes]
        print(f'{g:32s} ' + ' '.join(f'{v:9.1f}' for v in cells))


if __name__ == '__main__':
    main()
