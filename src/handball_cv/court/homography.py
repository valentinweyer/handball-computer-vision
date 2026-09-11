"""Court homography estimation with temporal continuity.

Solving each frame independently does not work on broadcast handball, and the
reason is a shortage of evidence rather than noise. Measured over the 60-second
Melsungen clip, no frame's RANSAC inliers span more than 55% of the court and
91% span under a quarter of it, so every homography is extrapolated roughly
fourfold beyond the landmarks it was fitted to. Two symptoms follow:

- **Jitter.** Re-solving eight degrees of freedom from scratch every frame moves
  the projected court by a median 32 px between adjacent frames, 361 px at p90.
  A broadcast camera does not move like that.
- **Collapse.** When the camera holds the centre circle with neither goal in
  view, the visible landmarks are the centre line -- collinear by construction --
  plus the two circle extremes 180 cm off it. There is no rank left to fix the
  remaining directions, the fit goes degenerate, and the court projects to a
  line. Frame 1310 of that clip is the case, and its reprojection error is
  1.0 px: five near-collinear points are trivial to fit perfectly, so residual
  is not merely a weak signal here but an actively misleading one.

Neither is fixable by a better gate. The only gate that caught every bad frame
kept 3% of the data, because almost no frame is well-conditioned on its own.
What is missing is constraint, and the free source of it is time.

So this module regularises each frame's fit toward a prediction carried from the
previous frame. Because the court is planar and the camera pans, tilts and zooms
about a roughly fixed centre, consecutive frames are themselves related by a
homography, and carrying an estimate forward is principled rather than a
smoothing hack. The prediction enters as extra correspondences at a low weight,
which is Tikhonov regularisation toward it and is why a single uniform weight is
enough: where the real landmarks constrain a direction their contribution to the
normal equations dwarfs the prior, and where they constrain nothing the prior is
all that is there. A centre-only frame keeps the scale and orientation it
inherited from frames that could see a goal.

`cv2.findHomography` takes no weights, so `fit_homography` below is a weighted
DLT. RANSAC still runs first, on the real landmarks alone, to throw out bad
detections before the prior is allowed to speak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .keypoints import court_points

__all__ = ["CourtFit", "CourtTracker", "fit_homography", "robust_fit"]


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to the origin, mean radius to sqrt(2).

    Court coordinates are in centimetres and image coordinates in pixels, so the
    raw DLT matrix mixes terms spanning several orders of magnitude and its SVD
    loses precision. Conditioning both sides first is what makes the direct
    solution usable.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_distance = float(np.linalg.norm(centred, axis=1).mean())
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]]
    )
    return centred * scale, transform


def fit_homography(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray | None:
    """Weighted direct linear transform mapping `source` onto `target`.

    Weights scale each correspondence's rows in the DLT matrix, so a
    correspondence at weight w contributes as though seen w times (the rows
    carry sqrt(w), since the solution minimises a sum of squares). This is the
    piece `cv2.findHomography` does not provide and the whole reason the prior
    can be blended in at a controlled strength.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 4 or len(source) != len(target):
        return None

    src_n, src_t = _normalize(source)
    dst_n, dst_t = _normalize(target)

    rows = np.zeros((2 * len(source), 9), dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src_n, dst_n)):
        rows[2 * i] = (-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u)
        rows[2 * i + 1] = (0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v)
    if weights is not None:
        scale = np.repeat(np.sqrt(np.asarray(weights, dtype=np.float64)), 2)
        rows = rows * scale[:, None]

    try:
        _, _, vt = np.linalg.svd(rows)
    except np.linalg.LinAlgError:
        return None
    target_inverse = _safe_inverse(dst_t)
    if target_inverse is None:
        return None
    matrix = target_inverse @ vt[-1].reshape(3, 3) @ src_t
    if abs(matrix[2, 2]) < 1e-12 or not np.all(np.isfinite(matrix)):
        return None
    return matrix / matrix[2, 2]


def _safe_inverse(matrix: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def robust_fit(
    source: np.ndarray, target: np.ndarray, threshold_px: float
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC fit, returning the homography and a boolean inlier mask."""
    if len(source) < 4:
        return None, np.zeros(len(source), dtype=bool)
    matrix, mask = cv2.findHomography(
        np.asarray(source, np.float32), np.asarray(target, np.float32),
        cv2.RANSAC, threshold_px,
    )
    if matrix is None or mask is None:
        return None, np.zeros(len(source), dtype=bool)
    return matrix.astype(np.float64), mask.ravel().astype(bool)


@dataclass
class CourtFit:
    """One frame's court mapping, and enough context to judge it.

    `source` says where the estimate came from: "keypoints" when this frame's
    own landmarks carried it, "propagated" when they could not and the carried
    prediction stood in, "none" when the tracker abstains.
    """

    homography: np.ndarray | None = None      # court -> image
    inverse: np.ndarray | None = None         # image -> court
    inliers: int = 0
    reprojection_px: float = float("nan")
    coverage_cm: tuple[float, float] = (0.0, 0.0)
    source: str = "none"
    age: int = 0                              # frames since real landmarks last carried it

    @property
    def usable(self) -> bool:
        return self.homography is not None

    def to_court(self, image_points: np.ndarray) -> np.ndarray:
        """Map image points to court coordinates. Empty in, empty out."""
        points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        if self.inverse is None or points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(points, self.inverse.astype(np.float32)).reshape(-1, 2)


class CourtTracker:
    """Per-frame court homography, regularised toward the previous frame.

    Feed it one frame's keypoints at a time, in order. Each call returns a
    :class:`CourtFit` for that frame.

    Args:
        vertices: ``CourtConfiguration.vertices``, in the unit court coordinates
            should come back in.
        image_size: ``(width, height)`` of the frames. Carried constraints are
            kept only where the prediction puts them near the picture -- see
            `_carried_points`.
        min_confidence: keypoints below this are ignored.
        ransac_px: RANSAC reprojection threshold for the real-landmark stage.
        prior_weight: weight of each carried correspondence against 1.0 for a
            real landmark. Small on purpose -- it should fill the directions the
            landmarks leave free without dragging on the ones they fix.
        prior_grid: how many court points to carry, as (along length, across
            width). They only need to span the court, not be dense.
        identity_gate_px: reject a landmark sitting further than this from where
            the carried estimate places it, before it can vote. Loose on
            purpose, and None disables it -- see `_agreeing_slots`.
        max_age: after this many consecutive frames with no usable landmarks,
            stop propagating and abstain rather than keep projecting a stale
            camera.
    """

    def __init__(
        self,
        vertices: Sequence[Sequence[float]],
        image_size: tuple[int, int],
        *,
        min_confidence: float = 0.5,
        ransac_px: float = 20.0,
        prior_weight: float = 0.08,
        prior_grid: tuple[int, int] = (7, 4),
        prior_margin: float = 1.0,
        identity_gate_px: float | None = 400.0,
        max_age: int = 50,
    ) -> None:
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._image_size = (float(image_size[0]), float(image_size[1]))
        self._prior_margin = prior_margin
        self._identity_gate_px = identity_gate_px
        self._min_confidence = min_confidence
        self._ransac_px = ransac_px
        self._prior_weight = prior_weight
        self._max_age = max_age
        self._homography: np.ndarray | None = None
        self._previous_points: dict[int, tuple[float, float]] = {}
        self._age = 0

        width = self._vertices[:, 0].max() - self._vertices[:, 0].min()
        height = self._vertices[:, 1].max() - self._vertices[:, 1].min()
        xs = np.linspace(self._vertices[:, 0].min(), self._vertices[:, 0].min() + width, prior_grid[0])
        ys = np.linspace(self._vertices[:, 1].min(), self._vertices[:, 1].min() + height, prior_grid[1])
        self._grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)

    def _predict(
        self, current: dict[int, tuple[float, float]], motion: np.ndarray | None
    ) -> np.ndarray | None:
        """Carry the previous homography forward through this frame's camera motion.

        A caller that can measure the inter-frame motion from the pictures
        themselves should pass it as `motion`; that is strictly better, because
        the fallback below reads the motion off the court landmarks, and those
        are exactly what a centre-only view does not have enough of. The
        fallback estimates a similarity rather than a homography -- four degrees
        of freedom need only two points, so it at least survives a collinear
        view -- but a similarity is only an approximation to what a panning
        perspective camera does, and the error accumulates over a long stretch
        with no goal in sight.
        """
        if self._homography is None:
            return None
        if motion is not None:
            return np.asarray(motion, dtype=np.float64) @ self._homography
        shared = sorted(set(current) & set(self._previous_points))
        if len(shared) < 2:
            return self._homography
        before = np.array([self._previous_points[s] for s in shared], dtype=np.float32)
        after = np.array([current[s] for s in shared], dtype=np.float32)
        motion, _ = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=self._ransac_px
        )
        if motion is None:
            return self._homography
        full = np.vstack([motion, [0.0, 0.0, 1.0]])
        return full @ self._homography

    def _carried_points(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid points the prediction places near the picture, and where it puts them.

        Most of the court is off-screen on a broadcast frame, and a grid corner
        the prediction throws tens of thousands of pixels away is not evidence
        about this frame -- it is the previous frame's extrapolation, carrying a
        squared error large enough to dominate the fit however low its weight.
        Keeping only what lands within `prior_margin` frame-widths of the
        picture is what confines the prior to saying something observable.
        """
        projected = cv2.perspectiveTransform(
            self._grid.reshape(-1, 1, 2).astype(np.float32), prediction.astype(np.float32)
        ).reshape(-1, 2)
        width, height = self._image_size
        keep = (
            np.all(np.isfinite(projected), axis=1)
            & (np.abs(projected[:, 0] - width / 2) <= width * (0.5 + self._prior_margin))
            & (np.abs(projected[:, 1] - height / 2) <= height * (0.5 + self._prior_margin))
        )
        return self._grid[keep], projected[keep]

    def _agreeing_slots(
        self,
        slots: list[int],
        observed: dict[int, tuple[float, float]],
        prediction: np.ndarray,
    ) -> list[int]:
        """Drop landmarks whose claimed identity the carried estimate contradicts.

        A frame with no goal in view cannot say which end of the court it is
        looking at -- the court is symmetric, so one goal area's arc is the
        other's. The detector answers per frame and so has to guess, and on the
        measured clip half its detections in such a view contradict the other
        half. RANSAC alone cannot settle it, because a consistent majority of
        wrong identities outvotes a correct minority.

        The previous frame does know which end it was looking at. So a landmark
        claiming a position far from where the carried estimate puts it is
        rejected before it can vote. The gate is deliberately loose: it is there
        to catch a landmark on the wrong half of a 40 m court, not to second-
        guess localisation.

        Falls back to the ungated set when it would leave too little to fit, so
        a stale or wrong prediction cannot starve the solve entirely.
        """
        placed = cv2.perspectiveTransform(
            court_points(slots, self._vertices).reshape(-1, 1, 2).astype(np.float32),
            prediction.astype(np.float32),
        ).reshape(-1, 2)
        seen = np.array([observed[s] for s in slots], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            distance = np.linalg.norm(placed - seen, axis=1)
        keep = np.isfinite(distance) & (distance <= self._identity_gate_px)
        return [s for s, k in zip(slots, keep) if k] if keep.sum() >= 4 else slots

    def update(
        self,
        keypoints: Iterable[tuple[int, float, float, float]],
        motion: np.ndarray | None = None,
    ) -> CourtFit:
        """Advance the tracker by one frame.

        Args:
            keypoints: ``(slot, x, y, confidence)`` for this frame, where `slot`
                is the model's keypoint index -- a prediction's ``class_id``,
                not its ``class_name``.
            motion: optional 3x3 image-to-image homography carrying the previous
                frame's pixels onto this one. Supplying one measured from the
                frames themselves is the accurate path; without it the tracker
                falls back to reading the motion off the court landmarks, which
                is weakest in precisely the views that need the prior most.
        """
        observed = {
            int(slot): (float(x), float(y))
            for slot, x, y, confidence in keypoints
            if confidence >= self._min_confidence
        }
        prediction = self._predict(observed, motion)

        slots = sorted(observed)
        if prediction is not None and self._identity_gate_px is not None and slots:
            slots = self._agreeing_slots(slots, observed, prediction)
        image_points = np.array([observed[s] for s in slots], dtype=np.float64) if slots else np.zeros((0, 2))
        court = court_points(slots, self._vertices) if slots else np.zeros((0, 2))

        matrix, mask = robust_fit(court, image_points, self._ransac_px)
        if matrix is None:
            return self._propagate(prediction, observed)

        court_in, image_in = court[mask], image_points[mask]
        source_pts, target_pts = court_in, image_in
        weights = np.ones(len(court_in), dtype=np.float64)
        if prediction is not None:
            grid, carried = self._carried_points(prediction)
            if len(grid):
                source_pts = np.vstack([court_in, grid])
                target_pts = np.vstack([image_in, carried])
                weights = np.concatenate([weights, np.full(len(grid), self._prior_weight)])

        blended = fit_homography(source_pts, target_pts, weights)
        if blended is None:
            blended = matrix

        self._homography = blended
        self._previous_points = observed
        self._age = 0
        return self._describe(blended, court_in, image_in, "keypoints")

    def _propagate(
        self, prediction: np.ndarray | None, observed: dict[int, tuple[float, float]]
    ) -> CourtFit:
        """No usable landmarks this frame: coast on the prediction, or abstain."""
        self._previous_points = observed
        if prediction is None or self._age >= self._max_age:
            self._homography = None if prediction is None else self._homography
            self._age += 1
            return CourtFit(age=self._age)
        self._homography = prediction
        self._age += 1
        return self._describe(prediction, np.zeros((0, 2)), np.zeros((0, 2)), "propagated")

    def _describe(
        self, matrix: np.ndarray, court_in: np.ndarray, image_in: np.ndarray, source: str
    ) -> CourtFit:
        inverse = _safe_inverse(matrix)
        reprojection = float("nan")
        coverage = (0.0, 0.0)
        if len(court_in):
            projected = cv2.perspectiveTransform(
                court_in.reshape(-1, 1, 2).astype(np.float32), matrix.astype(np.float32)
            ).reshape(-1, 2)
            reprojection = float(np.median(np.linalg.norm(projected - image_in, axis=1)))
            coverage = (
                float(court_in[:, 0].max() - court_in[:, 0].min()),
                float(court_in[:, 1].max() - court_in[:, 1].min()),
            )
        return CourtFit(
            homography=matrix,
            inverse=inverse,
            inliers=len(court_in),
            reprojection_px=reprojection,
            coverage_cm=coverage,
            source=source,
            age=self._age,
        )
