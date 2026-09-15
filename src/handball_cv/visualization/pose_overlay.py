"""Draw body keypoints and their limbs onto a frame."""
from __future__ import annotations

import cv2
import numpy as np

from handball_cv.pose.vitpose import SKELETON

# Left limbs warm, right limbs cool, centre/head neutral. Colouring by side is
# what makes a skeleton readable at handball scale: players are ~60x140 px in
# 1080p, so which arm is which is otherwise guesswork.
LEFT_BGR = (80, 180, 255)
RIGHT_BGR = (255, 180, 80)
CENTRE_BGR = (230, 230, 230)
LEFT_INDICES = frozenset((1, 3, 5, 7, 9, 11, 13, 15))
RIGHT_INDICES = frozenset((2, 4, 6, 8, 10, 12, 14, 16))


def _side_colour(index: int) -> tuple[int, int, int]:
    if index in LEFT_INDICES:
        return LEFT_BGR
    if index in RIGHT_INDICES:
        return RIGHT_BGR
    return CENTRE_BGR


def draw_pose(
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    min_score: float = 0.3,
    radius: int = 2,
    thickness: int = 2,
) -> None:
    """Draw one person's ``(17, 3)`` keypoints in place.

    Points below ``min_score`` are dropped, and a limb is drawn only when both
    of its ends survived: half a limb drawn to a guessed joint reads as a real
    measurement, which is exactly the impression a demo should not give.
    """
    visible = keypoints[:, 2] >= min_score
    for start, end in SKELETON:
        if not (visible[start] and visible[end]):
            continue
        cv2.line(
            frame_bgr,
            tuple(np.round(keypoints[start, :2]).astype(int)),
            tuple(np.round(keypoints[end, :2]).astype(int)),
            _side_colour(end),
            thickness,
            cv2.LINE_AA,
        )
    for index in np.flatnonzero(visible):
        cv2.circle(
            frame_bgr,
            tuple(np.round(keypoints[index, :2]).astype(int)),
            radius,
            _side_colour(int(index)),
            -1,
            cv2.LINE_AA,
        )
