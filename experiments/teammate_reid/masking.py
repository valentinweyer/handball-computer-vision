"""Hide jersey digits so an appearance model cannot learn to read them.

Two modes, both applied identically to gallery, query and training crops:

- ``digits``: every number-detector box inside the player box is covered by a
  square patch of standard size (a fixed fraction of the crop width, grown only
  if the box is larger), filled with a neutral constant. A crop with no number
  box gets one decoy patch at a deterministic torso position, so neither the
  patch's presence nor its width (one digit vs two) is an identity cue.
  Limitation: digits the detector missed (small front numbers, shorts and
  sleeve numbers, partial views) stay visible.
- ``strict``: ``digits`` plus a fixed torso-and-shorts band, independent of any
  detection. This removes nearly all printed numbers at the cost of most of the
  shirt; if a result survives ``strict`` it is not coming from digits.

No body keypoints are used; regions are fixed fractions of the player box.
"""
from __future__ import annotations

import zlib

import numpy as np

FILL = np.array((124, 116, 104), dtype=np.uint8)   # ImageNet mean: ~zero after normalisation
PATCH_FRAC = 0.45            # patch side as a share of the crop width
PATCH_GROW = 1.25            # a box larger than the patch is covered with this margin
DECOY_X = (0.38, 0.62)       # decoy patch centre range, as shares of crop width / height
DECOY_Y = (0.25, 0.42)
STRICT_BAND = (0.12, 0.12, 0.88, 0.62)   # x1, y1, x2, y2 as shares of the crop
MODES = ("raw", "digits", "strict")


def _square(cx: float, cy: float, side: float, w: int, h: int) -> tuple[int, int, int, int]:
    half = side / 2
    return (max(0, int(np.floor(cx - half))), max(0, int(np.floor(cy - half))),
            min(w, int(np.ceil(cx + half))), min(h, int(np.ceil(cy + half))))


def digit_patches(crop_w: int, crop_h: int, digit_boxes: list, seed_key: str) -> tuple[list, dict]:
    """Patch rectangles in crop pixels, and whether any patch had to grow."""
    side0 = PATCH_FRAC * crop_w
    patches, grown = [], 0
    for x1, y1, x2, y2 in digit_boxes:
        need = PATCH_GROW * max(x2 - x1, y2 - y1)
        side = max(side0, need)
        grown += side > side0
        patches.append(_square((x1 + x2) / 2, (y1 + y2) / 2, side, crop_w, crop_h))
    if not patches:
        rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
        cx = rng.uniform(*DECOY_X) * crop_w
        cy = rng.uniform(*DECOY_Y) * crop_h
        patches.append(_square(cx, cy, side0, crop_w, crop_h))
    return patches, {"digit_boxes": len(digit_boxes), "grown": int(grown),
                     "decoy": not digit_boxes}


def apply_mask(crop_rgb: np.ndarray, digit_boxes: list, mode: str, seed_key: str) -> tuple[np.ndarray, dict]:
    """`crop_rgb` with digits hidden; `digit_boxes` are in crop pixel coordinates."""
    if mode not in MODES:
        raise ValueError(mode)
    out = crop_rgb.copy()
    if mode == "raw":
        return out, {"digit_boxes": len(digit_boxes), "grown": 0, "decoy": False}
    h, w = out.shape[:2]
    patches, info = digit_patches(w, h, digit_boxes, seed_key)
    for x1, y1, x2, y2 in patches:
        out[y1:y2, x1:x2] = FILL
    if mode == "strict":
        bx1, by1, bx2, by2 = STRICT_BAND
        out[int(by1 * h):int(np.ceil(by2 * h)), int(bx1 * w):int(np.ceil(bx2 * w))] = FILL
    return out, info


def crop_with_digits(frame_rgb: np.ndarray, box: list, digit_boxes: list) -> tuple[np.ndarray, list]:
    """The runtime crop (raw player box, as `sam2_manager._appearance` cuts it)
    and the digit boxes moved into its coordinates, clipped to it."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=float).round().astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = frame_rgb[y1:y2, x1:x2]
    local = []
    for dx1, dy1, dx2, dy2 in digit_boxes:
        lx1, ly1 = max(0.0, dx1 - x1), max(0.0, dy1 - y1)
        lx2, ly2 = min(float(x2 - x1), dx2 - x1), min(float(y2 - y1), dy2 - y1)
        if lx2 > lx1 and ly2 > ly1:
            local.append((lx1, ly1, lx2, ly2))
    return crop, local
