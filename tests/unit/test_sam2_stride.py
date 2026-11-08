"""Reduced-frame-rate SAM2 helpers: which frames run, and the skipped frames' geometry."""
import numpy as np
import pytest

from handball_cv.tracking.sam2_driver import (
    interpolate_skipped,
    shift_mask,
    track_frame_schedule,
)


def test_stride_one_is_the_unchanged_path():
    assert track_frame_schedule(100, 1, 10, 5) is None


def test_schedule_keeps_stride_checkpoints_reads_and_last_frame():
    frames = track_frame_schedule(23, 2, 10, 5)
    assert frames == [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 22]
    assert track_frame_schedule(24, 2, 10)[-1] == 23   # last frame always runs


def test_schedule_rejects_nonpositive_stride():
    with pytest.raises(ValueError):
        track_frame_schedule(10, 0)


def test_shift_mask_moves_and_crops():
    mask = np.zeros((5, 6), bool)
    mask[1:3, 1:3] = True
    moved = shift_mask(mask, 2, 1)
    assert moved.sum() == 4 and moved[2:4, 3:5].all()
    assert shift_mask(mask, -2, 0).sum() == 2          # half leaves the frame
    assert not shift_mask(mask, 10, 0).any()


def test_interpolate_skipped_carries_only_players_on_both_sides():
    h, w = 20, 40
    masks_a = np.zeros((2, h, w), bool)
    masks_a[0, 5:10, 2:6] = True     # player 7
    masks_a[1, 5:10, 30:34] = True   # player 9, gone by frame 4
    boxes_a = np.array([[2, 5, 5, 9], [30, 5, 33, 9]], float)
    boxes_b = np.array([[10, 5, 13, 9], [0, 0, 5, 5]], float)   # player 7, new player 3
    out = list(interpolate_skipped(0, [7, 9], boxes_a, masks_a, 4, [7, 3], boxes_b))
    assert [f for f, *_ in out] == [1, 2, 3]
    frame, ids, boxes, masks = out[1]
    assert ids.tolist() == [7]
    assert np.allclose(boxes[0], [6, 5, 9, 9])          # halfway
    assert masks.shape == (1, h, w) and masks[0, 5:10, 6:10].all() and masks[0].sum() == 20


def test_interpolate_skipped_adjacent_frames_yield_nothing():
    masks = np.zeros((1, 4, 4), bool)
    masks[0, 1, 1] = True
    box = np.array([[1, 1, 1, 1]], float)
    assert list(interpolate_skipped(5, [1], box, masks, 6, [1], box)) == []


def test_interpolate_skipped_keeps_a_lost_player_listed_but_empty():
    masks_a = np.zeros((1, 6, 6), bool)          # player 4 currently has no mask
    boxes_a = np.zeros((1, 4))
    boxes_b = np.array([[1, 1, 3, 3]], float)
    (frame, ids, boxes, masks), = interpolate_skipped(10, [4], boxes_a, masks_a, 12, [4], boxes_b)
    assert frame == 11 and ids.tolist() == [4]
    assert not boxes.any() and not masks.any()
