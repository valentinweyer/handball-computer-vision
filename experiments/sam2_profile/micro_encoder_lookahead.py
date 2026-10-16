"""Exact trained-encoder side-stream microtest; root runs GPU serially.

  python -m experiments.sam2_profile.micro_encoder_lookahead

Builds and loads checkpoint on CPU before moving the eager model to CUDA.
No production files or input caches are modified.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ASSETS = Path('/home/valentinweyer/projects/handball-computer-vision')
ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
for path in (ASSETS / 'sam2-upstream', ROOT / 'src'):
    sys.path.insert(0, str(path))

import torch
from sam2.build_sam import build_sam2_video_predictor
from handball_cv.tracking.sam2_driver import LazyFrames
from experiments.sam2_profile.patches import encoder_lookahead


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), (a.shape, a.dtype, b.dtype)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def state(paths):
    result = {key: {} for key in ('obj_id_to_idx', 'obj_idx_to_id', 'point_inputs_per_obj',
                                  'mask_inputs_per_obj', 'output_dict_per_obj',
                                  'temp_output_dict_per_obj', 'frames_tracked_per_obj')}
    result.update(images=LazyFrames(paths, 1024, torch.device('cuda')), num_frames=len(paths),
                  device=torch.device('cuda'), cached_features={}, obj_ids=[])
    return result


def main():
    torch.manual_seed(0)
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=None, device='cpu', vos_optimized=False,
        hydra_overrides_extra=['++model.compile_image_encoder=false'])
    checkpoint = ASSETS / 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt'
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    predictor.load_state_dict(weights, strict=True)
    del weights
    predictor.eval().cuda()
    original = predictor._get_image_feature
    paths = sorted((ASSETS / 'data/cache/frames/Han-Ber4_cached').glob('*.jpg'), key=lambda p: int(p.stem))[:4]
    baseline, candidate = state(paths), state(paths)
    expected = []
    with torch.inference_mode():
        expected.append(original(baseline, 0, 1))  # init_state computes FP32 frame 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for fid in (0, 1, 1, 2, 3):
                expected.append(original(baseline, fid, 1))
    torch.cuda.synchronize()
    encoder_lookahead.install(predictor)
    encoder_lookahead.install(predictor)
    with torch.inference_mode():
        same(expected[0], predictor._get_image_feature(candidate, 0, 1))
        assert predictor._encoder_lookahead_counters['scheduled'] == 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for expected_frame, fid in zip(expected[1:], (0, 1, 1, 2, 3)):
                same(expected_frame, predictor._get_image_feature(candidate, fid, 1))
        # A fresh session must never consume another session's cached tensor.
        fresh = state(paths)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            same(original(baseline, 1, 1), predictor._get_image_feature(fresh, 1, 1))
        # Switch to FP32 while a BF16 future result is queued: original fallback.
        same(original(baseline, 2, 1), predictor._get_image_feature(fresh, 2, 1))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            same(original(baseline, 1, 1), predictor._get_image_feature(fresh, 1, 1))
            predictor.reset_state(fresh)
            assert encoder_lookahead._TOKEN not in fresh
            same(original(baseline, 2, 1), predictor._get_image_feature(fresh, 2, 1))
            token = encoder_lookahead._FORWARD_ALLOWED.set(False)
            try:
                same(original(baseline, 1, 1), predictor._get_image_feature(fresh, 1, 1))
            finally:
                encoder_lookahead._FORWARD_ALLOWED.reset(token)
    torch.cuda.synchronize()
    print('Trained encoder: torch.equal for FP32 init, BF16 sequential/repeat, fresh state, context switch, reset, reverse fallback')
    print(predictor._encoder_lookahead_counters)
    assert predictor._encoder_lookahead_counters['consumed'] >= 3


if __name__ == '__main__':
    main()
