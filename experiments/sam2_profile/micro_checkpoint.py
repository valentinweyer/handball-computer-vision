"""CPU-only correctness microtests; --cuda additionally times dense PE on GPU.

Run with the production virtualenv from the profiling worktree root:
  python -m experiments.sam2_profile.micro_checkpoint
  python -m experiments.sam2_profile.micro_checkpoint --cuda
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch

ASSETS = Path('/home/valentinweyer/projects/handball-computer-vision')
sys.path.insert(0, str(ASSETS / 'sam2-upstream'))
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.sam2_video_predictor import SAM2VideoPredictor
from experiments.sam2_profile.patches import checkpoint_outputs, dense_pe_cache


@contextmanager
def discarded():
    token = checkpoint_outputs._DRIVER_ACTIVE.set(True)
    try:
        yield
    finally:
        checkpoint_outputs._DRIVER_ACTIVE.reset(token)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def predictor():
    obj = SAM2VideoPredictor.__new__(SAM2VideoPredictor)
    torch.nn.Module.__init__(obj)
    obj.image_size = 8
    obj.add_all_frames_to_correct_as_cond = False
    obj.non_overlap_masks = False
    obj.clear_non_cond_mem_around_input = False

    def inference(self, inference_state, frame_idx, point_inputs, mask_inputs, **kwargs):
        logits = torch.arange(16).reshape(1, 1, 4, 4).float() + frame_idx
        if point_inputs is not None:
            logits = logits + point_inputs['point_coords'].sum()
        if mask_inputs is not None:
            logits = logits + mask_inputs.sum()
        out = {'pred_masks': logits, 'maskmem_features': None,
               'maskmem_pos_enc': None, 'obj_ptr': torch.ones(1, 4),
               'object_score_logits': torch.ones(1, 1)}
        return out, logits

    def memory(self, batch_size, **kwargs):
        return torch.ones(batch_size, 2, 2, 2), [torch.zeros(batch_size, 2, 2, 2)]

    obj._run_single_frame_inference = MethodType(inference, obj)
    obj._run_memory_encoder = MethodType(memory, obj)
    state = {key: {} for key in ('obj_id_to_idx', 'obj_idx_to_id', 'point_inputs_per_obj',
                                'mask_inputs_per_obj', 'output_dict_per_obj',
                                'temp_output_dict_per_obj', 'frames_tracked_per_obj')}
    state.update(obj_ids=[], device=torch.device('cpu'), storage_device=torch.device('cpu'),
                 video_height=12, video_width=20, constants={})
    return obj, state


def check_driver_scope():
    flags = []

    def original():
        flags.append(checkpoint_outputs._DRIVER_ACTIVE.get())

        def frames():
            flags.append(checkpoint_outputs._DRIVER_ACTIVE.get())
            yield 1
            flags.append(checkpoint_outputs._DRIVER_ACTIVE.get())

        return None, None, frames()

    driver = SimpleNamespace(drive_sam2=original)
    checkpoint_outputs._wrap_driver(driver)
    checkpoint_outputs._wrap_driver(driver)
    _, _, frames = driver.drive_sam2()
    assert not checkpoint_outputs._DRIVER_ACTIVE.get()
    assert next(frames) == 1
    assert not checkpoint_outputs._DRIVER_ACTIVE.get()
    assert list(frames) == []
    assert not checkpoint_outputs._DRIVER_ACTIVE.get()
    assert flags == [True, True, True]


def check_outputs():
    baseline, state_a = predictor()
    patched, state_b = predictor()
    checkpoint_outputs._patch_predictor(patched)
    checkpoint_outputs._patch_predictor(patched)  # idempotence
    box = torch.tensor([1., 2., 10., 11.])

    def call(name, *args, **kwargs):
        with discarded():
            out_a = getattr(baseline, name)(state_a, *args, **kwargs)
            out_b = getattr(patched, name)(state_b, *args, **kwargs)
        same(state_a, state_b)
        if name.startswith('add_new'):
            assert out_a[2].shape[-2:] == (12, 20)
            assert out_b[2] is None
        return out_a, out_b

    call('add_new_points_or_box', frame_idx=0, obj_id=1, box=box)
    call('add_new_points_or_box', frame_idx=0, obj_id=2, box=box)
    baseline.propagate_in_video_preflight(state_a)
    patched.propagate_in_video_preflight(state_b)
    same(state_a, state_b)
    for state in (state_a, state_b):
        for idx in state['frames_tracked_per_obj']:
            state['frames_tracked_per_obj'][idx][10] = {'reverse': False}
    call('add_new_points_or_box', frame_idx=10, obj_id=1, box=box)
    call('add_new_mask', frame_idx=10, obj_id=2, mask=torch.eye(12, 20, dtype=torch.bool))
    call('remove_object', obj_id=1)
    call('add_new_points_or_box', frame_idx=10, obj_id=1, box=box)  # reset/re-add
    call('add_new_points_or_box', frame_idx=20, obj_id=3, box=box)  # entrant
    # Direct predictor API still returns full-resolution editing previews.
    same(baseline.add_new_points_or_box(state_a, frame_idx=21, obj_id=3, box=box),
         patched.add_new_points_or_box(state_b, frame_idx=21, obj_id=3, box=box))
    same(state_a, state_b)
    with discarded():
        try:
            patched.add_new_points_or_box(state_b, frame_idx=22, obj_id=3)
        except ValueError:
            pass
        else:
            raise AssertionError('expected validation error')
    assert not checkpoint_outputs._PROMPT_PREVIEW_UNUSED.get()
    baseline.reset_state(state_a)
    patched.reset_state(state_b)
    same(state_a, state_b)
    call('add_new_points_or_box', frame_idx=30, obj_id=5, box=box)
    call('remove_object', obj_id=5)  # final object resets the session
    print('checkpoint preview: exact state for seed, correction, mask, remove/re-add, entrant, reset; API intact')


def check_dense(device):
    encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16).eval().to(device)
    previous = encoder.get_dense_pe
    dense_pe_cache._install_encoder(encoder)
    dense_pe_cache._install_encoder(encoder)
    with torch.inference_mode():
        for enabled in (False, True):
            with torch.autocast(device, enabled=enabled, dtype=torch.bfloat16):
                expected = previous()
                actual = encoder.get_dense_pe()
                assert torch.equal(expected, actual)
                assert actual is not encoder.get_dense_pe()
                actual.zero_()
                assert torch.equal(expected, encoder.get_dense_pe())
        encoder.pe_layer.positional_encoding_gaussian_matrix.add_(0.01)
        assert torch.equal(previous(), encoder.get_dense_pe())
        encoder.image_embedding_size = (32, 64)
        assert torch.equal(previous(), encoder.get_dense_pe())
        encoder.image_embedding_size = (64, 64)
        encoder.train()
        assert torch.equal(previous(), encoder.get_dense_pe())
        encoder.eval()
    with torch.no_grad():
        actual = encoder.get_dense_pe()
        assert torch.equal(previous(), actual) and not torch.is_inference(actual)
    with torch.enable_grad():
        assert torch.equal(previous(), encoder.get_dense_pe())
    print(f'dense PE ({device}): exact in FP32/BF16, bounded reuse, mutation/shape/train/gradient fallback')
    if device == 'cuda':
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoder.get_dense_pe()
            other_stream = torch.cuda.Stream()
            with torch.cuda.stream(other_stream):
                assert torch.equal(previous(), encoder.get_dense_pe())
            other_stream.synchronize()
            for label, fn in (('original', previous), ('cached', encoder.get_dense_pe)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    fn()
                end.record()
                end.synchronize()
                print(f'{label} dense PE: {start.elapsed_time(end)/200:.4f} ms/call')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    check_driver_scope()
    check_outputs()
    check_dense('cuda' if args.cuda else 'cpu')
