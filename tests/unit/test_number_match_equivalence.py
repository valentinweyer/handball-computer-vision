"""match_numbers_to_players computes IoS on box-local slices; it must equal the
dense supervision computation it replaced, value for value and in order."""
import numpy as np
import supervision as sv

from handball_cv.jersey.identity import MATCH_IOS_MIN, match_numbers_to_players


def dense_reference(player_masks, number_xyxy, frame_shape):
    h, w = frame_shape[:2]
    number_masks = sv.xyxy_to_mask(boxes=np.asarray(number_xyxy).reshape(-1, 4), resolution_wh=(w, h))
    iou = sv.mask_iou_batch(
        masks_true=player_masks, masks_detection=number_masks, overlap_metric=sv.OverlapMetric.IOS
    )
    rows, cols = np.where(iou > MATCH_IOS_MIN)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    pairs.sort(key=lambda rc: iou[rc[0], rc[1]], reverse=True)
    return pairs


def random_case(rng, h=90, w=160, players=6, numbers=5):
    masks = np.zeros((players, h, w), dtype=bool)
    for m in masks:
        y0, x0 = rng.integers(0, h - 10), rng.integers(0, w - 10)
        m[y0:y0 + rng.integers(5, 40), x0:x0 + rng.integers(5, 30)] = True
        m &= rng.random((h, w)) < 0.97          # ragged, like a real mask
    boxes = []
    for _ in range(numbers):
        anchor = masks[rng.integers(players)]
        ys, xs = np.nonzero(anchor) if anchor.any() else ([0], [0])
        cx, cy = rng.choice(xs), rng.choice(ys)
        bw, bh = rng.uniform(0.5, 12, size=2)
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    boxes = np.array(boxes) + rng.normal(0, 1.5, size=(numbers, 4))
    return masks, boxes


def test_matches_dense_supervision_on_random_scenes():
    rng = np.random.default_rng(0)
    for _ in range(300):
        masks, boxes = random_case(rng)
        assert match_numbers_to_players(masks, boxes, masks.shape[1:]) == \
            dense_reference(masks, boxes, masks.shape[1:])


def test_edge_boxes_and_empty_masks():
    masks = np.zeros((3, 20, 30), dtype=bool)
    masks[0, 2:8, 3:9] = True
    masks[2, 15:20, 25:30] = True                 # touches the bottom-right corner
    boxes = np.array([
        [3.7, 2.2, 6.9, 5.99],                    # fractional: int() truncation
        [-4.0, -3.0, 4.0, 3.0],                   # partly outside, negative
        [24.0, 14.0, 45.0, 40.0],                 # clipped by the frame
        [10.0, 10.0, 9.0, 12.0],                  # inverted: empty box
        [40.0, 30.0, 50.0, 40.0],                 # wholly outside
    ])
    assert match_numbers_to_players(masks, boxes, (20, 30)) == \
        dense_reference(masks, boxes, (20, 30))


def test_ties_keep_dense_order():
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[:, 2:8, 2:8] = True                     # identical players
    boxes = np.array([[3, 3, 5, 5], [4, 4, 6, 6]], dtype=float)
    assert match_numbers_to_players(masks, boxes, (10, 10)) == \
        dense_reference(masks, boxes, (10, 10))
