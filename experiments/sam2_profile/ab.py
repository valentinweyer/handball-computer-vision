"""A/B harness for SAM2 speed patches: one configuration per process.

Run a configuration (baseline = no --patch) on a reference clip:

    python -m experiments.sam2_profile.ab run --case han --output runs/sam2_gpu_bottleneck/ab/base_1.json
    python -m experiments.sam2_profile.ab run --case han \
        --patch experiments.sam2_profile.fused_rope:install \
        --output runs/sam2_gpu_bottleneck/ab/rope_1.json

Compare two runs (exact identity of ids, boxes, mask hashes; timing):

    python -m experiments.sam2_profile.ab compare base_1.json rope_1.json

A patch is `module:function`; the function is called as fn(predictor) after
the production predictor is built and production's own RoPE patch is
installed, and before `drive_sam2` is looked up, so a patch may replace
`handball_cv.tracking.sam2_driver.drive_sam2` or anything else it reaches.
Patches are applied in the order given.

Timing: the loop is not synchronised per frame (so patches that overlap CPU
and GPU work are credited); `wall_s` is the loop's wall time with one
synchronize at the end. GPU timings are only meaningful if nothing else is
using the GPU; run configurations one at a time and alternate them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path('/home/valentinweyer/projects/handball-computer-vision')
CASES = {  # stem, detections, team model (relative to outputs/); as experiments/cutie/cases.py
    'han': ('Han-Ber4_cached', 'team_dataset/.Han-Ber4_detections_v1.npz', 'team_correction_mcbyte/.Han-Ber4_team.pkl'),
    'felix': ('FelixClaar', 'team_comparison/.FelixClaar_detections_v1.npz', 'team_comparison/.FelixClaar_team.pkl'),
    'bhc': ('BHC-FAG_window_cached', 'team_dataset/.BHC-FAG_window_detections_v1.npz', 'team_confidence_v2/.BHC-FAG_team.pkl'),
}


def run(args):
    os.environ.setdefault('TRITON_PTXAS_PATH', str(ASSETS / '.venv/lib/python3.11/site-packages/torch/bin/ptxas'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    for path in [ASSETS / 'sam2-upstream', ROOT / 'src', ROOT]:
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from handball_cv.teams.model import TeamModel
    import handball_cv.tracking.sam2_driver as sam2_driver
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    stem, detections, team_path = CASES[args.case]
    torch.manual_seed(0)
    checkpoint = ASSETS / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=None, device='cpu', vos_optimized=False,
        hydra_overrides_extra=['++model.compile_image_encoder=false'])
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    predictor.load_state_dict(weights, strict=True)
    del weights
    predictor.eval().cuda()
    predictor.fill_hole_area = 0
    sam2_driver.install_rope_repeat_cache()
    for spec in args.patch:
        module, fn = spec.split(':')
        getattr(importlib.import_module(module), fn)(predictor)
    team = TeamModel.load(ASSETS / 'outputs' / team_path, device='cuda')
    cache = load_detection_cache(ASSETS / 'outputs' / detections)
    frame_dir = ASSETS / 'data/cache/frames' / stem
    video = ASSETS / 'data/raw' / f'{stem}.mp4'

    def drive(max_frames):
        return sam2_driver.drive_sam2(video, lambda fid: person_detections(cache, fid), team,
            str(checkpoint), frame_cache_dir=frame_dir, max_frames=max_frames,
            predictor_factory=lambda _: predictor, desc=f'ab {args.case}')

    _, _, frames = drive(args.warmup)  # warm kernels, autotuning, graphs, allocator
    for _ in frames:
        pass
    torch.cuda.synchronize()

    rows = []
    _, _, frames = drive(args.frames)
    torch.cuda.synchronize()
    start = last = time.perf_counter()
    for result in frames:
        now = time.perf_counter()
        rows.append({'frame': int(result.frame_idx), 'ms': (now - last) * 1e3,
                     'ids': [int(i) for i in result.player_ids], 'boxes': np.asarray(result.boxes).tolist(),
                     'masks': hashlib.sha256(np.packbits(result.masks).tobytes()).hexdigest()})
        last = time.perf_counter()  # exclude this bookkeeping from the next frame
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    out = {'case': args.case, 'patches': args.patch, 'frames': len(rows), 'wall_s': wall,
           'fps': len(rows) / wall, 'device': torch.cuda.get_device_name(),
           'revision': subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'gpu_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                           capture_output=True, text=True).stdout,
           'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'gpu_processes')}))


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    diff = {k: 0 for k in ('ids', 'boxes', 'masks')}
    first = None
    for x, y in zip(a['rows'], b['rows']):
        for k in diff:
            if x[k] != y[k]:
                diff[k] += 1
                first = first if first is not None else x['frame']
    ms = lambda r: sorted(x['ms'] for x in r['rows'])
    out = {'identical': len(a['rows']) == len(b['rows']) and not any(diff.values()),
           'frames': (len(a['rows']), len(b['rows'])), 'frames_differing': diff, 'first_differing_frame': first,
           'wall_s': (a['wall_s'], b['wall_s']), 'fps': (a['fps'], b['fps']), 'speedup': a['wall_s'] / b['wall_s'],
           'median_ms': (ms(a)[len(ms(a)) // 2], ms(b)[len(ms(b)) // 2])}
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--case', choices=sorted(CASES), default='han')
    r.add_argument('--patch', action='append', default=[], help='module:function, called with the predictor')
    r.add_argument('--frames', type=int, default=None, help='max frames (default: whole clip)')
    r.add_argument('--warmup', type=int, default=12)
    r.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    args = p.parse_args()
    run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    main()
