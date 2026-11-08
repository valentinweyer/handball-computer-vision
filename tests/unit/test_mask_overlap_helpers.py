"""The checkpoint's cached mask stats must equal the full-frame scans they replaced."""
import numpy as np
import supervision as sv

from handball_cv.tracking.sam2_manager import mask_area_and_bbox, mask_intersection


def random_masks(rng, count=8, h=60, w=80):
    masks = np.zeros((count, h, w), dtype=bool)
    for m in masks[:-1]:                      # the last one stays empty
        y0, x0 = rng.integers(0, h - 5), rng.integers(0, w - 5)
        m[y0:y0 + rng.integers(1, 30), x0:x0 + rng.integers(1, 30)] = True
        m &= rng.random((h, w)) < 0.9
    return masks


def test_area_and_box_match_full_scans():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for m in random_masks(rng):
            area, box = mask_area_and_bbox(m)
            assert area == int(m.sum())
            if not m.any():
                assert box is None
                continue
            y0, y1, x0, x1 = box
            assert list(sv.mask_to_xyxy(m[None])[0]) == [x0, y0, x1 - 1, y1 - 1]


def test_intersection_matches_full_frame_and():
    rng = np.random.default_rng(2)
    for _ in range(50):
        masks = random_masks(rng)
        stats = [mask_area_and_bbox(m) for m in masks]
        for i in range(len(masks)):
            for j in range(len(masks)):
                if stats[i][1] is None or stats[j][1] is None:
                    continue
                assert mask_intersection(masks[i], stats[i][1], masks[j], stats[j][1]) \
                    == int((masks[i] & masks[j]).sum())
