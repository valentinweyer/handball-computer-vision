"""Tint player masks onto a frame, touching only the pixels the masks cover.

Same picture as `sv.MaskAnnotator` -- each mask filled with its colour, larger
masks first so a smaller player in front stays visible, the colour layer blended
over the frame at `opacity` -- but painted from crops. Handed full-frame masks,
`MaskAnnotator` boolean-indexes every one of them over the box spanning all the
players, which at 1080p with ~13 players was 35 of the render's 52 ms a frame;
painting each mask into its own crop is ~2 ms.
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from handball_cv.tracking.geometry_cache import MaskCrop

__all__ = ["tint_masks"]


def tint_masks(frame_bgr: np.ndarray, crops: Sequence[MaskCrop | None],
               colors_bgr: Sequence[tuple], opacity: float) -> None:
    """Blend each crop's colour into `frame_bgr` in place; None entries draw nothing."""
    if len(crops) != len(colors_bgr):
        raise ValueError(f"{len(crops)} masks but {len(colors_bgr)} colours")
    present = [crop for crop in crops if crop is not None]
    if not present:
        return
    x0 = min(crop.x for crop in present)
    y0 = min(crop.y for crop in present)
    x1 = max(crop.x + crop.mask.shape[1] for crop in present)
    y1 = max(crop.y + crop.mask.shape[0] for crop in present)
    roi = frame_bgr[y0:y1, x0:x1]
    painted = roi.copy()
    areas = np.array([0 if crop is None else np.count_nonzero(crop.mask) for crop in crops])
    # MaskAnnotator's own order, ties included, so overlaps resolve as before.
    for index in np.flip(np.argsort(areas)):
        crop = crops[index]
        if crop is None:
            continue
        h, w = crop.mask.shape
        painted[crop.y - y0:crop.y - y0 + h, crop.x - x0:crop.x - x0 + w][crop.mask] = colors_bgr[index]
    # Unpainted pixels blend with themselves and come back unchanged.
    roi[:] = cv2.addWeighted(painted, opacity, roi, 1 - opacity, 0)
