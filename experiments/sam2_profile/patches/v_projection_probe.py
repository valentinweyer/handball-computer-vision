"""Diagnostic only: can BF16 SAM2 value projections be cached per memory frame?

This does not install a predictor patch. It compares the current concatenated
linear operation with separate frame projections for all four trained memory
attention layers. Run serially on the GPU:
    python -m experiments.sam2_profile.patches.v_projection_probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CHECKPOINT = Path('/home/valentinweyer/projects/handball-computer-vision/'
                  'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    torch.manual_seed(23)
    state = None
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)['model']
    print(json.dumps({'checkpoint': str(args.checkpoint) if state is not None else None,
                      'trained_weights': state is not None}), flush=True)
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for layer in range(4):
            key = f'memory_attention.layers.{layer}.cross_attn_image.v_proj'
            if state is None:
                weight = torch.randn(256, 64, device='cuda') / 8
                bias = torch.randn(256, device='cuda') / 8
            else:
                weight = state[key + '.weight'].cuda()
                bias = state[key + '.bias'].cuda()
            for memories in (1, 3, 7):
                for pointers in (0, 4, 16, 64):
                    chunks = [torch.randn(1, 4096, 64, device='cuda', dtype=torch.bfloat16)
                              for _ in range(memories)]
                    if pointers:
                        # The pointer path can be FP32; autocast must match the
                        # cast of the promoted concatenation, not assume BF16.
                        chunks.append(torch.randn(1, pointers, 64, device='cuda'))
                    combined = torch.cat(chunks, dim=1)
                    expected = F.linear(combined, weight, bias)
                    actual = torch.cat([F.linear(chunk, weight, bias) for chunk in chunks], dim=1)
                    different = expected != actual
                    changed = int(different.sum())
                    first = different.nonzero()[0].tolist() if changed else None
                    row = dict(layer=layer, memories=memories, pointers=pointers,
                               input_dtype=str(combined.dtype), equal=torch.equal(expected, actual),
                               changed=changed, values=expected.numel(),
                               first_differing_index=first,
                               max_abs=float((expected - actual).abs().max()))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del chunks, combined, expected, actual, different
    print(json.dumps({'all_equal': all(row['equal'] for row in rows),
                      'cases': len(rows), 'cases_differing': sum(not row['equal'] for row in rows)}), flush=True)


if __name__ == '__main__':
    main()
