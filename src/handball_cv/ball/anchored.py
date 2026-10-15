"""Detector-anchored ball tracking: RF-DETR finds it, SAM2 keeps it, a gate drops it.

The motion pipeline in `candidates.py` finds only balls that move independently
of the players, which is the flight regime and nothing else. The sibling repo's
`ball-v1` checkpoint can see a ball that is being carried -- badly (35.4% recall
on held balls against 58.9% in flight, measured there) but not never. SAM2
propagation covers what neither reaches: once anchored on a ball, it follows it
through a catch and a carry, because propagation is not detection and does not
have to re-find anything.

Measured on the BHC-FAG test clip (500 frames, a match in neither the ball
checkpoint's train nor its valid split), anchoring at 0.25 and propagating with
no gate covers 480/500 frames -- and most of that is wrong. SAM2 drifts onto
players' heads and, in one 46-frame run, onto the goalkeeper's orange jersey,
where the mask grew to 1310 px^2. A ball at that zoom is ~415 px^2. Read on a
contact sheet, roughly 40% of sampled propagated frames were not the ball.

Nothing about the propagated mask alone reveals this. What does is that the
track stops agreeing with anything else:

* **Area.** A mask far outside a plausible ball size is not a ball, whatever it
  is doing. This is a hard per-frame stop rather than a strike, because there is
  no ball reading at 1310 px^2 that later recovers -- the mask has already
  swallowed a torso.
* **Confirmation.** The detector fires near a real ball far more often than near
  a head, and the per-video appearance model separates a white ball from dark
  hair by colour. Either counts as the track earning itself for another frame;
  after `max_unconfirmed` consecutive frames with neither, it lapses.

The lapse is the same shape as `resolve_possession`'s: possession asserted by
association has to keep being confirmed, or it falls back to absent. The reason
is the same too. A tracker that never gives up does not report the ball for
longer, it reports a wrong position for longer, and the events this feeds are
defined by where possession changed.

`resolve_possession` itself is deliberately *not* reused here. It exists to fill
frames motion cannot see by attributing the ball to whoever the trajectory ended
beside -- an inference SAM2 does not need, because it observed those frames
directly. Held frames here are held because the ball was seen inside a person
box, not because a previous track ended nearby.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from handball_cv.ball.tracking import BallFrame, BallState, BallTrack, TrackPoint

__all__ = [
    "AnchorConfig",
    "Propagated",
    "accepted_prefix",
    "anchor_candidates",
    "build_tracks",
    "classify_frames",
    "fill_path_gaps",
]


@dataclass(frozen=True)
class AnchorConfig:
    """Thresholds for anchoring, and for how a track keeps earning itself.

    The defaults are the BHC-FAG measurements, not guesses. `anchor_threshold`
    is 0.25 because SAM2 needs only one seed per track: on the Melsungen clip
    all 18 human-verified ball tracks had at least one detection at 0.25, while
    only 15 did at 0.5 -- three whole trajectories were unreachable at the
    higher bar. The area band brackets a ~23 px ball (~415 px^2) with enough
    room for motion blur without admitting the 860-1310 px^2 masks that a drift
    onto a torso produces.
    """

    anchor_threshold: float = 0.25
    support_threshold: float = 0.25
    support_radius: float = 35.0
    """How near a detection must be to count as confirming the propagated mask."""

    max_unconfirmed: int = 5
    """Consecutive frames with neither detector nor appearance support before a
    track lapses. A pass crosses the court in well under this; a drift onto a
    head survives indefinitely without it.

    5 rather than a more generous figure because coverage bought past it is
    bought by guessing. Swept on the BHC-FAG test clip, with the square-crop
    confirmation pass on, and scoring what share of reported frames any
    detection actually agrees with:

        max_unconfirmed   coverage   confirmed share
                      3      65.6%             90.2%
                      5      66.2%             89.1%
                     10      69.8%             83.4%
                     20      75.0%             69.3%

    The extra 8.8 points of coverage between 5 and 20 cost 20 points of
    agreement, which is the drift in `docs/ball-tracking.md` being reported as
    ball positions. 3 and 5 are nearly the same trade; 5 is the looser of the
    two that still holds ~90%."""

    min_area: float = 150.0
    max_area: float = 600.0
    min_resemblance: float = 96.0
    """Back-projection score for the appearance model to count as confirming,
    on the same 0-255 scale `BallAppearance.locate` thresholds."""


@dataclass(frozen=True)
class Propagated:
    """One frame of SAM2 output, reduced to what the gate needs."""

    frame: int
    x: float
    y: float
    area: float


def anchor_candidates(
    detections: Mapping[int, Sequence[tuple[Sequence[float], float]]],
    *,
    threshold: float,
) -> list[tuple[int, tuple[float, ...], float]]:
    """Every frame that could seed a track, best box first within each frame.

    Returned in frame order as `(frame, box, score)`. Handball has one ball, so
    only the highest-scoring box on a frame is ever a seed: a second box on the
    same frame is a false positive or the same ball twice, and seeding both
    would start two tracks that then compete to explain one object.
    """
    out: list[tuple[int, tuple[float, ...], float]] = []
    for frame in sorted(detections):
        best_box, best_score = None, threshold
        for box, score in detections[frame]:
            if score >= best_score:
                best_box, best_score = box, score
        if best_box is not None:
            out.append((int(frame), tuple(float(v) for v in best_box), float(best_score)))
    return out


def _supported(
    frame: int,
    x: float,
    y: float,
    sources: Sequence[Mapping[int, Sequence[tuple[Sequence[float], float]]]],
    config: AnchorConfig,
) -> bool:
    """Is there a detection near this propagated position, in any source?"""
    for detections in sources:
        for box, score in detections.get(frame, ()):
            if score < config.support_threshold:
                continue
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            if float(np.hypot(cx - x, cy - y)) <= config.support_radius:
                return True
    return False


def accepted_prefix(
    observations: Sequence[Propagated],
    confirmed: Sequence[bool],
    config: AnchorConfig,
) -> int:
    """How many leading propagated frames are still the ball.

    Pure, and the whole of the gate's logic: a propagation is truncated, never
    reordered or bridged, so the answer is a prefix length.

    Both an implausible area and a run past `max_unconfirmed` end the track, and
    both trim back to the last frame something independent agreed with -- so a
    track is kept exactly as far as it was vouched for. Unconfirmed frames in
    the *middle* survive, because a later confirmation vouches for them
    retrospectively; an unconfirmed tail does not, because nothing ever did.
    That asymmetry is the point: the frames a track spends drifting away are
    precisely the ones with no confirmation, and keeping them because the lapse
    is only noticed twenty frames later is how a wrong position gets published.
    """
    if len(observations) != len(confirmed):
        raise ValueError("observations and confirmed must be the same length")

    last_confirmed = 0
    stale = 0
    for index, observation in enumerate(observations):
        if not (config.min_area <= observation.area <= config.max_area):
            break
        if confirmed[index]:
            stale = 0
            last_confirmed = index + 1
            continue
        stale += 1
        if stale > config.max_unconfirmed:
            break
    return last_confirmed


def build_tracks(
    detections: Mapping[int, Sequence[tuple[Sequence[float], float]]],
    propagate: Callable[[int, Sequence[float]], Iterable[Propagated]],
    *,
    config: AnchorConfig | None = None,
    resembles: Callable[[int, float, float], bool] | None = None,
    support_detections: Mapping[int, Sequence[tuple[Sequence[float], float]]] | None = None,
    n_frames: int | None = None,
) -> list[BallTrack]:
    """Anchor, propagate, gate, repeat.

    `propagate(frame, box)` yields `Propagated` from `frame` forward; it is
    injected so the gate is testable without SAM2 or a GPU, the same seam
    `drive_sam2` uses for its predictor. `resembles(frame, x, y)` reports
    whether the per-video appearance model recognises the ball there, and is
    optional: a clip with too few confident sightings to calibrate on gets no
    appearance gate rather than a model fitted on three glimpses.

    `support_detections` may confirm a propagated frame but never seed a track.
    It exists for boxes found by re-running the detector on a crop around a
    *predicted* position: at 1920x1080 a 23 px ball reaches a 704 square input
    as 8.4 px, and on a 352 px crop it reaches it as 46 px, so the same
    checkpoint sees an object it is otherwise shown below its own patch size.
    That is the structural fix `compare_split_detectors.py` measured for jersey
    numbers (hosted numbers 27 -> 38, false unhosted 10 -> 0, same speed).

    Kept out of anchoring because those boxes are conditioned on where a
    previous propagation went. The verdict is still independent -- the detector
    either sees a ball in the crop or it does not, exactly as a jersey number is
    detected inside a person crop -- but allowing them to *seed* would let a
    drifted track nominate its own starting point, and no new track should begin
    only where an old one already was.

    Anchors already covered by an accepted track are skipped, so one ball
    produces one track across the frames it was held for.
    """
    config = config or AnchorConfig()
    sources = [detections] if support_detections is None else [detections, support_detections]
    tracks: list[BallTrack] = []
    covered_to = -1

    for frame, box, _score in anchor_candidates(detections, threshold=config.anchor_threshold):
        if frame <= covered_to:
            continue
        if n_frames is not None and frame >= n_frames:
            break

        observations = [
            observation for observation in propagate(frame, box)
            if n_frames is None or observation.frame < n_frames
        ]
        if not observations:
            continue

        confirmed = [
            _supported(o.frame, o.x, o.y, sources, config)
            or bool(resembles and resembles(o.frame, o.x, o.y))
            for o in observations
        ]
        keep = accepted_prefix(observations, confirmed, config)
        if keep <= 0:
            continue

        points = tuple(
            TrackPoint(frame=o.frame, x=o.x, y=o.y, radius=float(np.sqrt(o.area / np.pi)))
            for o in observations[:keep]
        )
        tracks.append(BallTrack(points=points, mean_speed=_mean_speed(points)))
        covered_to = points[-1].frame

    return tracks


def fill_path_gaps(
    path: Sequence,
    propagate: Callable[[int, Sequence[float]], Iterable[Propagated]],
    *,
    tolerance: float = 45.0,
    box_half: float = 11.5,
) -> list[BallTrack]:
    """Join consecutive one-ball path points by propagating between them.

    `path` is `path.select_path`'s output: positions already known to lie on a
    single reachable trajectory. Propagating from one to the next is therefore a
    question with a checkable answer -- did the mask arrive where the next
    sighting says the ball was? -- rather than the open-ended "is it still the
    ball?" that `build_tracks` has to keep re-asking of a track with no known
    destination.

    That end check is the whole point. A propagation that drifts onto a head
    stays with that head, so it arrives somewhere the next detection is not, and
    the entire segment is discarded rather than trimmed back to its last
    confirmation. Anchored at both ends, a filled frame is supported by evidence
    on both sides of it, which no forward-only gate can offer.

    Segments that fail the check are dropped, leaving the gap absent. Endpoints
    are always kept: they are detections, and their support does not depend on
    whether the span next to them could be bridged.
    """
    tracks: list[BallTrack] = []
    pending: list[TrackPoint] = []

    for current, following in zip(path, list(path[1:]) + [None]):
        if not pending or pending[-1].frame != current.frame:
            pending.append(TrackPoint(frame=current.frame, x=current.x, y=current.y,
                                      radius=box_half))
        if following is None or following.frame == current.frame + 1:
            continue

        box = (current.x - box_half, current.y - box_half,
               current.x + box_half, current.y + box_half)
        bridge = [
            observation for observation in propagate(current.frame, box)
            if current.frame < observation.frame <= following.frame
        ]
        landed = bridge and bridge[-1].frame == following.frame and float(
            np.hypot(bridge[-1].x - following.x, bridge[-1].y - following.y)
        ) <= tolerance

        if landed:
            pending.extend(
                TrackPoint(frame=o.frame, x=o.x, y=o.y,
                           radius=float(np.sqrt(o.area / np.pi)))
                for o in bridge[:-1]
            )
        else:
            tracks.append(BallTrack(points=tuple(pending),
                                    mean_speed=_mean_speed(pending)))
            pending = []

    if pending:
        tracks.append(BallTrack(points=tuple(pending), mean_speed=_mean_speed(pending)))
    return tracks


def _mean_speed(points: Sequence[TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    steps = [
        float(np.hypot(b.x - a.x, b.y - a.y)) / max(1, b.frame - a.frame)
        for a, b in zip(points, points[1:])
    ]
    return float(np.mean(steps))


def classify_frames(
    tracks: Sequence[BallTrack],
    person_boxes: Sequence[Sequence[Sequence[float]]],
) -> tuple[BallFrame, ...]:
    """Turn accepted tracks into per-frame states, abstaining everywhere else.

    A tracked position inside a person box is `HELD` and names that person;
    outside every box it is `FLIGHT`. This is a reading of where the ball was
    seen, not an inference about who must have it -- which is why a frame with
    no track is `ABSENT` and stays that way. Roughly a third of frames in a
    handball broadcast genuinely have no visible ball, and filling them in is
    how a confident wrong position gets manufactured.
    """
    frames = [
        BallFrame(frame=index, state=BallState.ABSENT)
        for index in range(len(person_boxes))
    ]
    for track in tracks:
        for point in track.points:
            if not 0 <= point.frame < len(frames):
                continue
            holder = _containing_box(point.x, point.y, person_boxes[point.frame])
            frames[point.frame] = BallFrame(
                frame=point.frame,
                state=BallState.FLIGHT if holder is None else BallState.HELD,
                x=point.x,
                y=point.y,
                holder=holder,
            )
    return tuple(frames)


def _containing_box(x: float, y: float, boxes: Sequence[Sequence[float]]) -> int | None:
    """Index of the smallest person box containing the point, if any.

    Smallest, because players overlap constantly and the tighter box is the
    nearer player -- the one actually holding it.
    """
    best_index, best_area = None, float("inf")
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best_index, best_area = index, area
    return best_index
