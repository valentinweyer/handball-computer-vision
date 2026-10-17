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

Time can also hold a wrong answer in place, which the first full broadcast
(10 minutes of Kiel-Lemgo) showed in two ways:

- **Lock-in.** Once the carried estimate is wrong, the identity gate rejects
  the correct landmarks for disagreeing with it -- 6 of 14 per frame for 9 s
  there, while 11 of 14 agreed with each other on a court metres away. So a
  frame whose own landmarks outvote the gated set, without flipping the
  court's orientation, now wins and drops the prior for that frame.
- **No court at all.** In a lights-off timeout the keypoint model finds court
  points on silhouettes, and the old tracker fitted 95% of those frames. Now
  a frame whose keypoints mostly disagree is not fitted, and a run of such
  frames means no court is in view: the tracker abstains and forgets its fit.

Refitted from the stored keypoints, player feet more than 5 m outside the
court went from 4.7% to 1.8% of rows in play on that broadcast, and frame-
to-frame jitter at p90 from 26 to 11 px; the three short clips stayed level
or improved (BHC-FAG 1.7 -> 1.4%, Melsungen 2.0 -> 2.0%, FelixClaar 5.0 ->
4.8%). Some of what remains is people who really are off the court.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .keypoints import court_points

__all__ = ["CourtFit", "CourtTracker", "fit_homography", "robust_fit"]

# Frames that must agree on the court's orientation before it is enforced.
ORIENTATION_EVIDENCE = 3


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


def _orientation(matrix: np.ndarray, image_point: tuple[float, float]) -> tuple[int, int]:
    """Which way round the court sits in the picture: (handedness, depth direction).

    Both read off the image-to-court map at `image_point` -- the picture's
    centre, which a broadcast camera keeps on the floor -- because a court
    point far outside the picture can sit behind the camera, where the signs
    mean nothing. Handedness is the sign of that map's Jacobian; depth
    direction is which way court y runs as one moves up the picture. A camera
    move changes neither. The two ways the court's symmetry can fool the
    keypoint labels each flip one: mirroring the ends flips handedness, a half
    turn reverses the depth direction.
    """
    inverse = _safe_inverse(matrix)
    if inverse is None:
        return 0, 0
    x, y = image_point
    probe = np.array([[[x, y]], [[x + 1.0, y]], [[x, y - 1.0]]], dtype=np.float64)
    court = cv2.perspectiveTransform(probe, inverse).reshape(-1, 2)
    right, up = court[1] - court[0], court[2] - court[0]
    return int(np.sign(right[0] * up[1] - right[1] * up[0])), int(np.sign(up[1]))


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
    prediction stood in, "none" when the tracker abstains. `court_in_view` is
    False when the recent frames show no court at all -- a timeout huddle with
    the lights off, a close-up -- which is a different reason to abstain from a
    court the tracker merely lost.
    """

    homography: np.ndarray | None = None      # court -> image
    inverse: np.ndarray | None = None         # image -> court
    inliers: int = 0
    reprojection_px: float = float("nan")
    coverage_cm: tuple[float, float] = (0.0, 0.0)
    source: str = "none"
    age: int = 0                              # frames since real landmarks last carried it
    court_in_view: bool = True

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
        min_agreement: the share of a frame's confident keypoints that must
            agree on one court for the frame to be fitted at all. Below it the
            frame is treated as having no usable landmarks. On the Kiel
            broadcast, under half agreed in 85% of the frames of a lights-off
            timeout (the model finding court points on silhouettes) against
            1.8% of frames of play.
        view_window: frames over which "is there a court in view" is decided.
            One frame cannot settle it -- inside that timeout 15% of frames
            still agree by chance -- so the court counts as in view while at
            least half of the last `view_window` frames agreed. Out of view,
            the tracker abstains and forgets its fit rather than carry it
            through a stretch it cannot see.
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
        min_agreement: float = 0.5,
        view_window: int = 25,
    ) -> None:
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._image_size = (float(image_size[0]), float(image_size[1]))
        self._prior_margin = prior_margin
        self._identity_gate_px = identity_gate_px
        self._min_confidence = min_confidence
        self._ransac_px = ransac_px
        self._prior_weight = prior_weight
        self._max_age = max_age
        self._min_agreement = min_agreement
        self._recent_agreement: deque[bool] = deque(maxlen=view_window)
        self._homography: np.ndarray | None = None
        # Which way round the court sits in the picture (see `_orientation`) is a
        # property of the video -- the camera's side of the court and the keypoint
        # model's labelling -- not of one fit. So it is a running vote over every
        # frame whose keypoints mostly agree, never a single fit's answer: on two
        # broadcasts 97% of such frames vote the same way, and a stray first
        # frame cannot decide what every later one is held to.
        self._orientation_votes: Counter = Counter()
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

        May leave too little to fit; `update` then treats the prediction as
        contradicted, so a stale or wrong one cannot starve the solve.
        """
        placed = cv2.perspectiveTransform(
            court_points(slots, self._vertices).reshape(-1, 1, 2).astype(np.float32),
            prediction.astype(np.float32),
        ).reshape(-1, 2)
        seen = np.array([observed[s] for s in slots], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            distance = np.linalg.norm(placed - seen, axis=1)
        keep = np.isfinite(distance) & (distance <= self._identity_gate_px)
        return [s for s, k in zip(slots, keep) if k]

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
        consensus = self._consensus(slots, observed)
        # Whether this frame's keypoints agree on any court at all, before the
        # prediction has a say: a mirrored labelling agrees with itself, points
        # found on a dark huddle do not.
        agrees = consensus is not None and len(consensus[1]) >= self._min_agreement * len(observed)
        self._recent_agreement.append(agrees)
        if 2 * sum(self._recent_agreement) < len(self._recent_agreement):
            self._homography = None
            self._previous_points = observed
            self._age += 1
            return CourtFit(age=self._age, court_in_view=False)
        if not agrees:
            return self._propagate(prediction, observed)
        self._orientation_votes[self._orientation_of(consensus[0])] += 1

        use_prior = prediction is not None
        if prediction is not None and self._identity_gate_px is not None:
            gated = self._agreeing_slots(slots, observed, prediction)
            held = self._consensus(gated, observed) if gated != slots else consensus
            if self._prediction_contradicted(consensus, held):
                use_prior = False
            else:
                consensus = held

        if consensus is None or not self._keeps_orientation(consensus[0]):
            return self._propagate(prediction, observed)

        matrix, court_in, image_in = consensus
        source_pts, target_pts = court_in, image_in
        weights = np.ones(len(court_in), dtype=np.float64)
        if use_prior:
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

    def _consensus(
        self, slots: list[int], observed: dict[int, tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """RANSAC over `slots`: the homography and its inliers, court and image side."""
        if len(slots) < 4:
            return None
        court = court_points(slots, self._vertices)
        image = np.array([observed[s] for s in slots], dtype=np.float64)
        matrix, mask = robust_fit(court, image, self._ransac_px)
        if matrix is None:
            return None
        return matrix, court[mask], image[mask]

    def _prediction_contradicted(self, ungated, gated) -> bool:
        """The frame's own landmarks agree on a court the carried estimate rules out.

        The identity gate exists for a coherent mislabelling: a mirrored story
        that outvotes the true one. A mirror shows up as a flip of the court's
        orientation in the picture, which no camera move produces, so an
        ungated consensus that outvotes the gated one *and* keeps the
        orientation is not that case -- it is the carried estimate that is
        wrong, and holding to it would reject exactly the landmarks that could
        correct it, frame after frame.
        """
        if ungated is None or not self._keeps_orientation(ungated[0]):
            return False
        return gated is None or len(ungated[1]) > len(gated[1])

    def _orientation_of(self, matrix: np.ndarray) -> tuple[int, int]:
        return _orientation(matrix, (self._image_size[0] / 2, self._image_size[1] / 2))

    def _keeps_orientation(self, matrix: np.ndarray) -> bool:
        """True unless the video has settled on an orientation and `matrix` has another."""
        if not self._orientation_votes:
            return True
        (leader, votes), = self._orientation_votes.most_common(1)
        settled = votes >= ORIENTATION_EVIDENCE and votes > sum(self._orientation_votes.values()) / 2
        return not settled or self._orientation_of(matrix) == leader

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
