"""Court mapping over a whole clip: keypoints, camera motion, then the tracked fit.

The three steps `CourtTracker` needs, each over a frame sequence, so a caller
can feed frames from wherever it has them -- a JPEG folder, or the video
decoded directly -- and cache any step's output separately. Keypoints and
motion are the expensive half (a model pass, and optical flow); the fit is
cheap and deterministic, so it is what gets re-run when the tracker changes.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .homography import CourtFit, CourtTracker
from .keypoints import court_vertices

__all__ = ["infer_keypoints", "measure_motion", "fit_court"]


def infer_keypoints(model, frames_rgb: Iterable[np.ndarray]) -> list:
    """All 37 slots per frame, unfiltered: `[[(slot, x, y, confidence), ...], ...]`.

    Unfiltered so the confidence threshold stays the fit's choice, not the cache's.
    """
    return [[list(point) for point in model.infer(frame)[0]] for frame in frames_rgb]


def measure_motion(frames_rgb: Iterable[np.ndarray]) -> list:
    """Per frame, the 3x3 image homography from the previous frame, or None.

    The first frame has no predecessor; a frame the estimator cannot relate to
    the one before (a cut, no texture) is None too, which the tracker treats as
    "fall back to the landmarks".
    """
    from .camera_motion import estimate_camera_motion

    motion, previous = [], None
    for frame in frames_rgb:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        matrix = None if previous is None else estimate_camera_motion(previous, gray)
        motion.append(None if matrix is None else np.asarray(matrix).tolist())
        previous = gray
    return motion


def fit_court(keypoints: list, motion: list, image_size: tuple[int, int],
              **tracker_options) -> list[CourtFit]:
    """One `CourtFit` per frame, carried through the clip in order."""
    tracker = CourtTracker(court_vertices(), image_size, **tracker_options)
    fits = []
    for index, points in enumerate(keypoints):
        step = motion[index] if index < len(motion) else None
        fits.append(tracker.update(
            [tuple(point) for point in points],
            motion=None if step is None else np.asarray(step, dtype=np.float64),
        ))
    return fits
