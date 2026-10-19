"""The crop painter must draw exactly what `sv.MaskAnnotator` drew.

It replaced the annotator for speed alone, so any pixel difference -- most
likely where two players overlap -- would be a regression in every render.
"""
import numpy as np
import supervision as sv

from handball_cv.tracking.geometry_cache import crop_mask
from handball_cv.visualization.mask_overlay import tint_masks
from handball_cv.visualization.palette import PALETTE_HEX


def blob(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_matches_mask_annotator_where_players_overlap():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    masks = np.stack([
        blob(frame.shape[:2], 10, 60, 10, 50),   # large, drawn first
        blob(frame.shape[:2], 30, 80, 40, 70),   # overlaps the first
        blob(frame.shape[:2], 35, 45, 45, 55),   # small, inside the second
        blob(frame.shape[:2], 5, 15, 100, 110),  # equal area to the next one
        blob(frame.shape[:2], 70, 80, 100, 110),
    ])
    masks[1] &= rng.random(frame.shape[:2]) > 0.2  # ragged, not a box
    colors = np.array([0, 1, 2, 3, 4])
    palette = sv.ColorPalette.from_hex(PALETTE_HEX)
    boxes = np.array([[x.min(), y.min(), x.max(), y.max()]
                      for y, x in (np.nonzero(m) for m in masks)], dtype=float)

    expected = sv.MaskAnnotator(color=palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX).annotate(
        frame.copy(), sv.Detections(xyxy=boxes, class_id=colors, mask=masks),
        custom_color_lookup=colors)
    got = frame.copy()
    tint_masks(got, [crop_mask(m) for m in masks],
               [palette.by_idx(int(c)).as_bgr() for c in colors], opacity=0.5)

    np.testing.assert_array_equal(got, expected)


def test_missing_masks_draw_nothing():
    frame = np.full((20, 20, 3), 77, dtype=np.uint8)
    tint_masks(frame, [None, crop_mask(np.zeros((20, 20), bool))], [(0, 0, 255)] * 2, opacity=0.5)
    assert (frame == 77).all()
