"""Per-op roofline of SAM2 propagation, without hardware counters.

Hardware counters (ncu) and CUPTI activity tracing inside PyTorch are blocked
for non-admin users on this machine (RmProfilingAdminOnly=1). This measures
what can be measured without them:

  --mode roofline  For each aten op on the sampled frames: FLOPs (PyTorch's
                   flop-counter formulas), compulsory bytes (unique input and
                   output memory spans), GPU time (CUDA events around the op)
                   and the owning module. Writes ops.jsonl.
  --mode plain     No instrumentation except one NVTX range per frame, for
                   running under `nsys profile -t cuda,nvtx` to get GPU busy
                   fraction and the kernel mix.

Compulsory bytes are a lower bound on DRAM traffic (cache re-reads are not
counted, L2 hits are counted as DRAM). Event timing in roofline mode adds CPU
overhead between ops, so it measures per-op GPU time, not the frame's wall time.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--assets-root', type=Path, default=Path('/home/valentinweyer/projects/handball-computer-vision'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['roofline', 'plain', 'rope-ab'], required=True)
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--sample-frames', type=int, nargs='+', default=[25, 26],
                   help='frames to instrument in roofline mode (avoid checkpoint frames, multiples of 10)')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.assets_root.resolve()
    for path in [base / 'sam2-upstream', ROOT / 'src', ROOT]:
        sys.path.insert(0, str(path))
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2, install_rope_repeat_cache
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    torch.manual_seed(0)
    checkpoint = base / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=None, device='cpu', vos_optimized=False,
        hydra_overrides_extra=['++model.compile_image_encoder=false'])
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    predictor.load_state_dict(weights, strict=True)
    del weights
    predictor.eval().cuda()
    predictor.fill_hole_area = 0
    install_rope_repeat_cache()
    team = TeamModel.load(base / 'outputs/team_correction_mcbyte/.Han-Ber4_team.pkl', device='cuda')
    cache = load_detection_cache(base / 'outputs/team_dataset/.Han-Ber4_detections_v1.npz')
    files = sorted((base / 'data/cache/frames/Han-Ber4_cached').glob('*.jpg'), key=lambda x: int(x.stem))
    video = base / 'data/raw/Han-Ber4_cached.mp4'

    @contextmanager
    def bounded_cache(count):
        with tempfile.TemporaryDirectory(prefix='sam2-roofline-') as tmp:
            for f in files[:count]:
                (Path(tmp) / f.name).symlink_to(f.resolve())
            yield Path(tmp)

    def drive(directory):
        return drive_sam2(video, lambda fid: person_detections(cache, fid), team,
            str(checkpoint), frame_cache_dir=directory, max_frames=args.frames,
            predictor_factory=lambda _: predictor, desc='SAM2 roofline')

    # Warm cuDNN/cuBLAS heuristics and the allocator before anything is measured.
    with bounded_cache(12) as directory:
        _, _, frames = drive(directory)
        for _ in frames:
            pass
    torch.cuda.synchronize()

    meta = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'mode': args.mode,
            'frames': args.frames}
    if args.mode == 'plain':
        # One NVTX range per module (no syncs), so nsys can project kernel time
        # onto the module tree the roofline run attributes FLOPs and bytes to.
        for name, module in predictor.named_modules():
            if name:
                module.register_forward_pre_hook(lambda m, a, name=name: (torch.cuda.nvtx.range_push('m:' + name), None)[1])
                module.register_forward_hook(lambda m, a, o: (torch.cuda.nvtx.range_pop(), None)[1])
        frame_ms = []
        with bounded_cache(args.frames) as directory:
            _, _, frames = drive(directory)
            torch.cuda.cudart().cudaProfilerStart()
            while True:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                torch.cuda.nvtx.range_push('frame')
                try:
                    result = next(frames)
                except StopIteration:
                    torch.cuda.nvtx.range_pop()
                    break
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                frame_ms.append({'frame': result.frame_idx, 'ms': (time.perf_counter() - t0) * 1e3,
                                 'objects': len(result.player_ids)})
            torch.cuda.cudart().cudaProfilerStop()
        meta['frame_ms'] = frame_ms
    elif args.mode == 'rope-ab':
        meta['rope_ab'] = run_rope_ab(torch, bounded_cache, drive, args)
    else:
        meta['ops'] = run_roofline(torch, predictor, bounded_cache, drive, args)
    (args.output / 'summary.json').write_text(json.dumps(meta, indent=1) + '\n')


def run_rope_ab(torch, bounded_cache, drive, args):
    """Baseline, fused RoPE, baseline again: per-frame time and exact output equality."""
    import hashlib
    import numpy as np
    import sam2.modeling.sam.transformer as transformer
    from experiments.sam2_profile.fused_rope import install_fused_rope

    def run():
        rows = []
        with bounded_cache(args.frames) as directory:
            _, _, frames = drive(directory)
            while True:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = next(frames, None)
                torch.cuda.synchronize()
                if result is None:
                    break
                rows.append({'frame': result.frame_idx, 'ms': (time.perf_counter() - t0) * 1e3,
                             'ids': result.player_ids.tolist(), 'boxes': result.boxes.tolist(),
                             'masks': hashlib.sha256(np.packbits(result.masks).tobytes()).hexdigest()})
        return rows

    baseline_fn = transformer.apply_rotary_enc
    runs = {'baseline': run()}
    install_fused_rope()
    runs['fused'] = run()
    transformer.apply_rotary_enc = baseline_fn
    runs['baseline_again'] = run()

    def compare(a, b):
        keys = ('frame', 'ids', 'boxes', 'masks')
        return {k: sum(x[k] != y[k] for x, y in zip(a, b)) for k in keys} | {'frames': (len(a), len(b))}

    def timing(rows):
        ms = np.array([r['ms'] for r in rows])
        return {'mean_ms': float(ms.mean()), 'median_ms': float(np.median(ms)), 'fps': float(1e3 / ms.mean())}

    out = {name: timing(rows) for name, rows in runs.items()}
    out['diff_fused_vs_baseline'] = compare(runs['baseline'], runs['fused'])
    out['diff_baseline_repeat'] = compare(runs['baseline'], runs['baseline_again'])
    print('ROPE_AB', json.dumps(out, indent=1), flush=True)
    return out


def run_roofline(torch, predictor, bounded_cache, drive, args):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from torch.utils.flop_counter import flop_registry

    stack = ['(predictor)']
    for name, module in predictor.named_modules():
        if not name:
            continue
        module.register_forward_pre_hook(lambda m, a, name=name: stack.append(name))
        module.register_forward_hook(lambda m, a, o: (stack.pop(), None)[1])

    def span_bytes(t):
        if t.numel() == 0:
            return 0
        extent = sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride())) + 1
        return extent * t.element_size()

    aten = torch.ops.aten
    DECOMPOSE = {aten.linear, aten.scaled_dot_product_attention, aten.matmul, aten.einsum, aten.to,
                 aten.type_as, aten.reshape, aten.flatten, aten.contiguous, aten.conv2d,
                 aten.conv_transpose2d, aten.layer_norm}
    records = []

    class OpRecorder(TorchDispatchMode):
        frame = None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            # Composite ops (linear, sdpa, to, reshape) reach the mode undecomposed;
            # decompose them so FLOPs and copies are attributed to the real kernels.
            # Allowlist only: others (upsample_*.vec) have Python decompositions
            # that run different, slower kernels than production does.
            if func._overloadpacket in DECOMPOSE:
                with self:
                    out = func.decompose(*args, **kwargs)
                if out is not NotImplemented:
                    return out
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = func(*args, **kwargs)
            end.record()
            is_view = any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)
            seen, nbytes, shapes = set(), 0, []
            for t in tree_flatten((args, kwargs, out))[0]:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
                    if key not in seen:
                        seen.add(key)
                        nbytes += span_bytes(t)
                        shapes.append(list(t.shape))
            packet = func._overloadpacket
            flops = int(flop_registry[packet](*args, **kwargs, out_val=out)) if packet in flop_registry else 0
            records.append({'frame': self.frame, 'module': stack[-1], 'op': str(func), 'view': is_view,
                            'flops': flops, 'bytes': 0 if is_view else nbytes, 'shapes': shapes[:4],
                            'events': (start, end)})
            return out

    sample = set(args.sample_frames)
    recorder = OpRecorder()
    with bounded_cache(args.frames) as directory:
        _, _, frames = drive(directory)
        fid = 1
        while True:
            if fid in sample:
                torch.cuda.synchronize()
                recorder.frame = fid
                with recorder:
                    result = next(frames, None)
                torch.cuda.synchronize()
            else:
                result = next(frames, None)
            if result is None:
                break
            fid += 1
    for r in records:
        start, end = r.pop('events')
        r['ms'] = start.elapsed_time(end)
    with open(args.output / 'ops.jsonl', 'w') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')
    return len(records)


if __name__ == '__main__':
    main()
