"""Inter-frame camera motion, measured from the pictures rather than the court.

:class:`~handball_cv.court.homography.CourtTracker` carries its estimate forward
through the camera's motion, and how that motion is measured decides whether the
carried estimate helps or drags. Reading it off the court landmarks is the
obvious route and the weakest one, because it fails hardest in exactly the views
that depend on the prior: a camera holding the centre circle offers a handful of
collinear landmarks, enough to approximate a similarity and not enough to be
right about a panning perspective camera. That approximation error accumulates
over a long stretch with no goal in sight.

Measuring it from the frames themselves has none of that dependence -- the floor
advertising, the boards and the stands are all texture, and there is no shortage
of it anywhere in the picture. Measured on the Melsungen clip, swapping the
landmark-derived similarity for this cut frames that reject more than half their
players from 17 to 6 and let a stronger prior improve every axis at once instead
of trading jitter against lag.

Players move independently of the camera and would bias the fit, so the
homography is estimated with RANSAC: the background is the overwhelming majority
of tracked points, so the dominant motion is the camera's and the players fall
out as outliers. No player masks are needed.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["estimate_camera_motion"]

_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def estimate_camera_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    max_corners: int = 1200,
    quality: float = 0.01,
    min_distance: int = 12,
    ransac_px: float = 2.0,
    min_inliers: int = 8,
) -> np.ndarray | None:
    """Homography carrying `previous_gray`'s pixels onto `current_gray`.

    Pass the result straight to ``CourtTracker.update(..., motion=...)``.

    Returns None when the frames cannot be related -- too little texture, a shot
    cut, or too few points surviving RANSAC. The tracker treats that as "no
    measured motion" and falls back to its own weaker estimate, so a None is
    degraded behaviour rather than a failure.

    Args:
        previous_gray: single-channel previous frame.
        current_gray: single-channel current frame.
        ransac_px: inlier threshold, in pixels. Tight on purpose -- the
            background moves rigidly, and anything looser starts admitting
            players.
        min_inliers: reject the estimate below this many agreeing points.
    """
    if previous_gray is None or current_gray is None:
        return None
    if previous_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("estimate_camera_motion expects single-channel frames")

    corners = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance, blockSize=7,
    )
    if corners is None or len(corners) < min_inliers:
        return None

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, corners, None,
        winSize=(21, 21), maxLevel=3, criteria=_LK_CRITERIA,
    )
    if tracked is None or status is None:
        return None

    kept = status.ravel() == 1
    before = corners[kept].reshape(-1, 2)
    after = tracked[kept].reshape(-1, 2)
    if len(before) < min_inliers:
        return None

    matrix, mask = cv2.findHomography(before, after, cv2.RANSAC, ransac_px)
    if matrix is None or mask is None or int(mask.sum()) < min_inliers:
        return None
    return matrix.astype(np.float64)
