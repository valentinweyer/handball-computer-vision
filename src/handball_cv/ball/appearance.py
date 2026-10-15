"""What the ball looks like in *this* video, learned from its own flight frames.

A held ball moves with its holder, so it produces no residual and motion cannot
see it. Appearance can -- but only if we know what the ball looks like, and
nothing in this project does: there is no ball class, and the 277 ball boxes in
the sibling repo's export are too few and too biased toward easy frames to train
one.

The way out is that the clip has already shown us the ball by the time anyone is
holding it. Trajectories found by motion are high-precision -- they are accepted
on physics, not appearance -- so they are a supply of free, unlabelled examples
of this ball, under this lighting, from this camera. Calibrating on them is the
same anonymous per-video pattern `teams/model.py` uses for kit colour, and for
the same reason: nothing global has to be labelled, and nothing transfers a
mistake between videos.

Scale comes from the players
----------------------------
A ball's size in pixels depends on depth and zoom, neither of which is known
without a homography -- and court geometry on this footage is only trustworthy
when a goal is in frame. But a person is a known size standing at the same
depth, and the detector already found every person. So the size prior is the
ratio of ball radius to person-box height, measured from the flight detections
rather than assumed, and it stays correct through a zoom because both terms move
together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

__all__ = ["BallAppearance", "BallObservation", "calibrate"]

_HUE_BINS, _SATURATION_BINS = 30, 32
_RANGES = [0, 180, 0, 256]

#: Ball radius as a fraction of person-box height, measured over the 389 ball
#: annotations in the sibling repo's export that share an image with a person
#: box (p50 0.0822, p10-p90 0.058-0.113). Comparing heights keeps the figure
#: usable despite that export stretching 1920x1080 into 640x640, which squashes
#: x and leaves y alone. It runs above the naive 19cm-ball-over-180cm-player
#: figure because a detection box is not a standing player: it tightens around a
#: crouched or running one, and the ball's own box absorbs its motion blur.
#: Only a fallback -- a clip with enough flight sightings measures its own.
_MEASURED_PLAYER_RATIO = 0.0822


@dataclass(frozen=True)
class BallObservation:
    """One confident sighting, used only to calibrate."""

    patch: np.ndarray
    """BGR crop centred on the ball."""

    radius: float
    person_height: float | None = None
    """Height of the nearest person box, when there is one within reach."""


@dataclass(frozen=True)
class BallAppearance:
    """A per-video ball model. Never constructed without enough evidence."""

    histogram: np.ndarray
    player_ratio: float
    samples: int

    def radius_for(self, person_box: Sequence[float]) -> float:
        """Expected ball radius for a ball held by this person."""
        x1, y1, x2, y2 = (float(v) for v in person_box[:4])
        return max(2.0, self.player_ratio * (y2 - y1))

    def resemblance(
        self,
        frame_bgr: np.ndarray,
        x: float,
        y: float,
        radius: float,
    ) -> float:
        """How ball-coloured the disc at ``(x, y)`` is, on the 0-255 scale.

        `locate` searches a person box for the ball; this asks about one place
        someone else already picked. A tracker propagating a mask forward has a
        position and needs to know whether it is still on the ball -- on the
        BHC-FAG test clip SAM2 drifts onto players' heads and a goalkeeper's
        jersey, and dark hair against a pale ball is exactly the difference a
        hue/saturation model built from this clip's own sightings can see.

        Averaged over the disc rather than read at the centre pixel, for the
        same reason `locate` blurs: the ball is a concentration of ball-coloured
        pixels the size of a ball, and a single matching pixel is noise.
        """
        height, width = frame_bgr.shape[:2]
        span = max(2, int(round(radius)))
        left, top = int(max(0, x - span)), int(max(0, y - span))
        right, bottom = int(min(width, x + span + 1)), int(min(height, y + span + 1))
        if right - left < 2 or bottom - top < 2:
            return 0.0

        region = frame_bgr[top:bottom, left:right]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        likelihood = cv2.calcBackProject([hsv], [0, 1], self.histogram, _RANGES, 1)

        mask = np.zeros(likelihood.shape, np.uint8)
        cv2.circle(mask, (int(x) - left, int(y) - top), span, 255, -1)
        if not mask.any():
            return 0.0
        return float(cv2.mean(likelihood, mask=mask)[0])

    def locate(
        self,
        frame_bgr: np.ndarray,
        person_box: Sequence[float],
        *,
        min_score: float = 96.0,
        upper_fraction: float = 0.62,
    ) -> tuple[float, float, float] | None:
        """Best ball-coloured spot in the upper part of a person box.

        Returns ``(x, y, score)`` in image coordinates, or None when nothing in
        the box looks enough like the ball -- which is the common case and a
        correct answer. A ball tucked against a body or hidden by an arm is not
        visible, and reporting a position anyway would put a confident wrong
        number where a gap belongs.

        Only the upper part of the box is searched: a carried ball is at hand
        and chest height, and the legs below are where most of a jersey-coloured
        false match would otherwise come from.
        """
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in person_box[:4])
        left, top = int(max(0, x1)), int(max(0, y1))
        right = int(min(width, x2))
        bottom = int(min(height, y1 + (y2 - y1) * upper_fraction))
        if right - left < 4 or bottom - top < 4:
            return None

        region = frame_bgr[top:bottom, left:right]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        likelihood = cv2.calcBackProject([hsv], [0, 1], self.histogram, _RANGES, 1)

        # A ball is a *concentration* of ball-coloured pixels the size of a ball,
        # not a single matching pixel -- averaging over that disc is what makes
        # a scattering of similar pixels in a jersey score low.
        span = max(3, int(round(self.radius_for(person_box) * 2)) | 1)
        smoothed = cv2.blur(likelihood, (span, span))
        _, peak, _, location = cv2.minMaxLoc(smoothed)
        if peak < min_score:
            return None
        return float(left + location[0]), float(top + location[1]), float(peak)


def calibrate(
    observations: Sequence[BallObservation],
    *,
    min_samples: int = 12,
    default_ratio: float = _MEASURED_PLAYER_RATIO,
) -> BallAppearance | None:
    """Build the ball model from confident flight sightings, or abstain.

    Returns None below ``min_samples``. A model fitted on three glimpses would
    be worse than no model, because it would still be consulted and would still
    be believed -- so a clip where motion never found the ball reliably gets no
    held-ball search at all.
    """
    usable = [o for o in observations if o.patch is not None and o.patch.size]
    if len(usable) < min_samples:
        return None

    histogram = np.zeros((_HUE_BINS, _SATURATION_BINS), np.float32)
    for observation in usable:
        patch = observation.patch
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Mask to the ball itself; a square patch is mostly court around it, and
        # calibrating on that would build a model of the floor.
        mask = np.zeros(patch.shape[:2], np.uint8)
        centre = (patch.shape[1] // 2, patch.shape[0] // 2)
        cv2.circle(mask, centre, max(2, int(observation.radius)), 255, -1)
        histogram += cv2.calcHist(
            [hsv], [0, 1], mask, [_HUE_BINS, _SATURATION_BINS], _RANGES
        )
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)

    ratios = [
        o.radius / o.person_height
        for o in usable
        if o.person_height and o.person_height > 0
    ]
    return BallAppearance(
        histogram=histogram,
        player_ratio=float(np.median(ratios)) if ratios else default_ratio,
        samples=len(usable),
    )
