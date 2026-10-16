"""Synthetic GPU lifecycle proof for the standalone Cutie adapter."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from experiments.cutie.adapter import build_models, ROOT
    import cv2
    import numpy as np
    import torch
    predictor, loading = build_models(args.device)
    with tempfile.TemporaryDirectory(prefix='cutie-smoke-') as tmp:
        image = np.zeros((192, 320, 3), np.uint8)
        masks = []
        for x, color in [(20, (60, 160, 220)), (120, (160, 220, 60)), (220, (220, 60, 160))]:
            image[20:170, x:x+55] = color
            m = np.zeros(image.shape[:2], bool)
            m[20:170, x:x+55] = True
            masks.append(m)
        for fid in range(5):
            cv2.imwrite(str(Path(tmp)/f'{fid:05}.jpg'), image)
        state = predictor.init_state(tmp)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
        with torch.inference_mode(), precision:
            for oid, mask in zip([11, 22, 33], masks):
                predictor.add_new_mask(state, 0, oid, mask)
            list(predictor.propagate_in_video(state, 0, 1))
            predictor.add_new_mask(state, 1, 11, masks[0])
            corrected = list(predictor.propagate_in_video(state, 1, 0))[-1]
            assert corrected[1] == [11, 22, 33]
            assert state['core'].last_mask.shape[1] == 3
            core = state['core']
            previous = core.last_mask.clone()
            survivor_summary = core.memory.obj_v[33].clone()
            predictor.remove_object(state, 22)
            assert core.object_manager.all_obj_ids == [11, 33]
            assert torch.equal(core.last_mask, previous[:, [0, 2]])
            assert torch.equal(core.memory.obj_v[33], survivor_summary)
            assert 22 not in core.memory.obj_v and 22 not in core.memory.sensory
            predictor.add_new_mask(state, 1, 44, masks[1])
            result = list(predictor.propagate_in_video(state, 1, 1))[-1]
            assert result[1] == [11, 33, 44]
            predictor.remove_object(state, 11)
            assert 11 not in core.memory.obj_v
            predictor.add_new_mask(state, 2, 11, masks[0])
            result = list(predictor.propagate_in_video(state, 2, 1))[-1]
            assert result[1] == [33, 44, 11]
            for oid in result[1].copy():
                predictor.remove_object(state, oid)
            result = list(predictor.propagate_in_video(state, 3, 1))[-1]
            assert result[1] == []
            predictor.reset_state(state)
            predictor.add_new_points_or_box(state, 0, 91, [20, 20, 75, 170])
            predictor.add_new_points_or_box(state, 0, 92, [120, 20, 175, 170])
            result = list(predictor.propagate_in_video(state, 0, 1))[-1]
            assert result[1] == [91, 92]
            assert torch.isfinite(result[2]).all()
    if args.device == 'cpu':
        assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'device': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu',
              'strict_loading': loading, 'tests': ['middle-ID deletion preserves survivor channels and memory',
              'partial correction of an existing player preserves all channels',
              'late addition', 'remove/re-add same ID starts fresh object summary',
              'remove all objects', 'reset', 'batched SAM2 image-head box seeds',
              'propagation after every operation'], 'input': 'synthetic images only'}
    (ROOT/f'.local/cutie/setup_{args.device}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
