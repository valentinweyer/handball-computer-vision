"""Ball candidates from motion, with nothing learned and nothing labelled.

The ball is the one object on court that moves independently of both the camera
and the players, so it is separable by motion alone -- which matters because the
detector has no ball class and the labels to train one do not exist (277 of 1033
images in the sibling repo's export carry a ball box, so the rest would teach
"no ball" on exactly the frames where it is hard to see).

Three frames, not two
---------------------
Differencing `t` against `t-1` alone puts residual in two places: where the ball
was, and where it is. Which of the pair is the ball at `t` is not recoverable
from that difference. Taking the per-pixel *minimum* of the differences against
`t-1` and `t+1` keeps only what differs from both neighbours, which is where the
ball is now -- the ghost at the old position matches `t-1` and drops out.

A volume, not a surface
-----------------------
The obvious spatial prior is the court, and on this footage it is a good one:
the residual is overwhelmingly crowd and bench, and the playing surface is
nearly clean. But a handball in flight leaves the court region in the image
entirely -- unlike football, where the ball stays near the ground. So the prior
has to be the volume play happens in: the court, extended *upward* to cover a
thrown ball, and not extended downward where the bench sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

__all__ = [
    "BallCandidate",
    "CandidateFrame",
    "ball_candidates",
    "believable_motion",
    "overlay_mask",
    "play_volume_mask",
]

#: Court colour on this broadcast footage, sampled from Melsungen frame 710:
#: hue ~109, saturation ~120, value ~228. The advertising boards share the hue
#: but sit at value <90, and the crowd is darker still, so brightness is what
#: actually separates the playing surface from everything behind it.
_COURT_LOW = (98, 60, 150)
_COURT_HIGH = (122, 200, 255)


@dataclass(frozen=True)
class BallCandidate:
    """One accepted motion blob, in the coordinates of its own frame."""

    x: float
    y: float
    radius: float
    area: float
    strength: float
    """Mean residual over the blob -- how strongly it differs from both neighbours."""


@dataclass(frozen=True)
class CandidateFrame:
    """Everything one frame contributed, including what it threw away."""

    candidates: tuple[BallCandidate, ...]
    rejected: Mapping[str, int]
    """Blob counts by rejection reason. ``outside_volume`` is the play-volume
    blind spot: it is counted rather than silently masked so the cost of the
    spatial prior stays visible."""


def play_volume_mask(
    frame_bgr: np.ndarray,
    *,
    rise_px: int = 320,
    spread_px: int = 24,
    fill_px: int = 91,
) -> np.ndarray:
    """The region a ball can occupy: the court, extended upward.

    ``rise_px`` is deliberately generous. A ball thrown three or four metres
    above the floor projects well above the court region, and the whole point of
    this mask is to admit that while still excluding the bench below the court
    and the deep stands above it. Tightening it would recreate the blind spot it
    exists to avoid; the cost of leaving it loose is crowd, which the trajectory
    stage is there to reject.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    court = cv2.inRange(hsv, _COURT_LOW, _COURT_HIGH)

    # Players stand on the court and are not court-coloured, so the raw mask is
    # full of person-shaped holes. Close them before taking the largest region,
    # or the "court" becomes whichever fragment happens to be biggest.
    closing = np.ones((fill_px, fill_px), np.uint8)
    court = cv2.morphologyEx(court, cv2.MORPH_CLOSE, closing)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(court, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        court = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Anchoring a vertical kernel at its top makes dilation reach upward only:
    # each mask pixel spreads to the rows above it, never below.
    if rise_px > 0:
        court = cv2.dilate(court, np.ones((rise_px, 1), np.uint8), anchor=(0, 0))
    if spread_px > 0:
        court = cv2.dilate(court, np.ones((3, spread_px), np.uint8))
    return court


def overlay_mask(
    sample_gray: Sequence[np.ndarray],
    *,
    max_variation: float = 6.0,
    dilate_px: int = 11,
    max_coverage: float = 0.25,
) -> np.ndarray | None:
    """Burned-in broadcast graphics: the pixels that never change.

    A score bug is painted at fixed coordinates, so compensating the camera
    turns it into a flawless projectile -- it crosses the compensated frame at
    exactly the pan speed, dead straight, for as long as the pan lasts. Track
    geometry cannot refuse it, and requiring a trajectory to actually travel in
    the picture only catches the long ones: during a whip pan the residual
    around a fixed glyph smears far enough to clear that bar over five frames.

    What no overlay can do is change. Sample frames from across the clip and the
    arena is different in every one, because the camera moved; the score bug is
    identical in all of them. So the overlay is simply the low-variance pixels.

    Returns None when the mask would cover more than ``max_coverage`` of the
    frame, which means the sample held too little camera motion to tell an
    overlay from a wall -- better to exclude nothing than to blank the court.
    """
    if len(sample_gray) < 3:
        return None
    variation = np.stack([f.astype(np.float32) for f in sample_gray]).std(axis=0)
    mask = (variation < max_variation).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if float(mask.mean()) / 255.0 > max_coverage:
        return None
    return cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))


def believable_motion(
    matrix: np.ndarray | None,
    shape: tuple[int, int],
    *,
    max_corner_shift: float = 250.0,
    max_area_ratio: float = 2.0,
) -> bool:
    """Is this camera-motion estimate one a real camera could have produced?

    ``estimate_camera_motion`` fits with RANSAC and returns whatever agrees with
    eight points. In a frame with little rigid texture those eight can agree on
    something wild, and the result is not a near-identity pan but a transform
    that hurls the picture across the plane. Nothing downstream notices: warping
    by it produces residual everywhere, and -- worse -- it makes two blobs 698
    pixels apart in the image look like a 15 px/frame trajectory, which is
    exactly how a 37-frame "ball" was assembled out of the scoreboard and a
    patch of court on the first full-clip run.

    Between adjacent frames the camera can pan hard but cannot teleport, so a
    believable estimate moves the image corners by a bounded amount and roughly
    preserves area.
    """
    if matrix is None:
        return False
    height, width = shape
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    moved = project_points_2d(corners, matrix)
    if not np.all(np.isfinite(moved)):
        return False
    if float(np.hypot(*(moved - np.array(corners, float)).T).max()) > max_corner_shift:
        return False
    diagonal_a, diagonal_b = moved[2] - moved[0], moved[3] - moved[1]
    area = abs(float(diagonal_a[0] * diagonal_b[1] - diagonal_a[1] * diagonal_b[0])) / 2.0
    original = float(width * height)
    return 1.0 / max_area_ratio <= area / original <= max_area_ratio


def project_points_2d(points, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.c_[array, np.ones(len(array))] @ np.asarray(matrix, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3]


def _is_misregistered(
    current: np.ndarray,
    warped_previous: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    shift: int,
    match: float,
) -> bool:
    """Is this blob new content, or old content the warp failed to line up?

    Camera compensation is never exact, and a few pixels of error along a
    painted floor marking produces residual that is the right size and the right
    shape to be a ball. The difference is not how it looks but whether it was
    already there: search the warped previous frame around the blob, and if the
    same patch turns up a few pixels away, nothing new happened here. Where the
    ball actually is, the previous frame holds bare floor -- the ball has moved
    -- so there is nothing to match.
    """
    left, top, width, height = box
    rows, columns = current.shape
    x1, y1 = max(0, left - 2), max(0, top - 2)
    x2, y2 = min(columns, left + width + 2), min(rows, top + height + 2)
    patch = current[y1:y2, x1:x2]
    if patch.size == 0 or float(patch.std()) < 4.0:
        return False        # featureless: nothing to match either way

    sx1, sy1 = max(0, x1 - shift), max(0, y1 - shift)
    sx2, sy2 = min(columns, x2 + shift), min(rows, y2 + shift)
    search = warped_previous[sy1:sy2, sx1:sx2]
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return False
    scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    return float(scores.max()) >= match


def _warp_onto(source: np.ndarray, motion: np.ndarray, shape: tuple[int, int]):
    """Warp `source` by `motion`, and report which pixels it actually covered."""
    height, width = shape
    warped = cv2.warpPerspective(source, motion, (width, height))
    support = cv2.warpPerspective(
        np.full(source.shape[:2], 255, np.uint8), motion, (width, height)
    )
    return warped, support


def ball_candidates(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    motion_previous: np.ndarray,
    motion_next: np.ndarray,
    *,
    play_volume: np.ndarray | None = None,
    overlay: np.ndarray | None = None,
    person_boxes: Sequence[Sequence[float]] = (),
    min_radius: float = 4.0,
    max_radius: float = 26.0,
    threshold: int = 18,
    person_pad: float = 6.0,
    max_aspect: float = 4.0,
    registration_shift: int = 6,
    registration_match: float = 0.8,
) -> CandidateFrame:
    """Motion blobs in `current_gray` that could be the ball.

    Args:
        motion_previous: homography carrying `previous_gray` onto the current
            frame, as returned by
            :func:`handball_cv.court.camera_motion.estimate_camera_motion`.
        motion_next: homography carrying `next_gray` onto the current frame.
            The caller usually already holds the forward motion from the current
            frame to the next and can invert it, so a sequence costs one
            estimate per frame rather than two.
        person_boxes: ``xyxy`` person detections. A ball held or touched is not
            separable from the player holding it, and limbs produce ball-sized
            residuals -- a running player's feet were the single largest source
            of false tracks before this filter existed.
        registration_match: reject a blob when the warped previous frame
            already contains it within ``registration_shift`` pixels, which
            means the residual is camera-compensation error along a static
            marking rather than anything that moved. Set to 1.0 or above to
            disable. This costs recall only where the ball passes directly over
            a marking that resembles it.
    """
    if current_gray.ndim != 2 or previous_gray.ndim != 2 or next_gray.ndim != 2:
        raise ValueError("ball_candidates expects single-channel frames")

    shape = current_gray.shape[:2]
    warped_previous, support_previous = _warp_onto(previous_gray, motion_previous, shape)
    warped_next, support_next = _warp_onto(next_gray, motion_next, shape)

    # The minimum is what makes this a three-frame difference rather than two:
    # the ghost at the ball's old position agrees with t-1 and is suppressed.
    residual = np.minimum(
        cv2.absdiff(warped_previous, current_gray),
        cv2.absdiff(warped_next, current_gray),
    )
    residual[(support_previous < 255) | (support_next < 255)] = 0

    _, binary = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_area = np.pi * min_radius ** 2 * 0.5
    max_area = np.pi * max_radius ** 2 * 2.0
    boxes = [tuple(float(v) for v in box[:4]) for box in person_boxes]

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    accepted: list[BallCandidate] = []
    rejected = {
        "area": 0, "shape": 0, "outside_volume": 0, "overlay": 0, "person": 0,
        "misregistered": 0,
    }
    for index in range(1, count):
        left, top, width, height, area = (int(v) for v in stats[index])
        if not min_area <= area <= max_area:
            rejected["area"] += 1
            continue
        if max(width, height) > max_radius * 4 or (
            max(width, height) / max(1, min(width, height)) > max_aspect
        ):
            rejected["shape"] += 1
            continue

        x, y = (float(v) for v in centroids[index])
        row, column = int(round(y)), int(round(x))
        if play_volume is not None and not play_volume[row, column]:
            rejected["outside_volume"] += 1
            continue
        if overlay is not None and overlay[row, column]:
            rejected["overlay"] += 1
            continue
        if any(
            x1 - person_pad <= x <= x2 + person_pad
            and y1 - person_pad <= y <= y2 + person_pad
            for x1, y1, x2, y2 in boxes
        ):
            rejected["person"] += 1
            continue

        if _is_misregistered(
            current_gray, warped_previous, (left, top, width, height),
            shift=registration_shift, match=registration_match,
        ):
            rejected["misregistered"] += 1
            continue

        blob = labels[top:top + height, left:left + width] == index
        accepted.append(
            BallCandidate(
                x=x,
                y=y,
                radius=float(np.sqrt(area / np.pi)),
                area=float(area),
                strength=float(
                    np.mean(residual[top:top + height, left:left + width][blob])
                ),
            )
        )
    return CandidateFrame(candidates=tuple(accepted), rejected=rejected)
