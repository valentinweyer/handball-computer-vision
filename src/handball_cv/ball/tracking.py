"""Chaining ball candidates into trajectories, and deciding who has the ball.

Per-frame motion blobs are far too noisy to threshold into a ball: on this
footage the crowd alone contributes tens of ball-sized residuals per frame. What
separates the ball from all of them is not how any single blob looks but how it
*moves* -- the crowd jiggles in place while the ball translates smoothly at
10-30 px/frame. So a blob is accepted because of where its neighbours are, never
on its own merits, which is what makes this track-before-detect rather than
detect-then-track.

Compensation is pairwise, not accumulated
-----------------------------------------
Motion has to be measured with the camera taken out of it, and the tempting way
to do that is to map every frame into one shared reference frame by composing
the inter-frame homographies. Over a few dozen frames that works. Over 1498 it
fails quietly and badly: the composed transform drifts, distances stop meaning
anything, and points hundreds of pixels apart in the picture land next to each
other in the reference frame. Measured on the first full-clip run, chains were
being accepted whose *image* steps reached 698 px while their compensated steps
passed a 70 px gate.

So each step is compensated against its own neighbour instead. Positions and
velocities are carried forward one frame at a time through that frame's own
homography, and nothing is ever composed.

Burned-in graphics
------------------
Compensating the camera turns every *static overlay* into a perfect projectile.
The scoreboard is painted at fixed pixels, so while the arena slides underneath
it, the score bug travels across the compensated frame at exactly the pan speed,
dead straight, for as long as the pan lasts. On the first full-clip run the four
longest "ball" trajectories were all glyphs in the score bug, one of them 37
frames. Nothing about the trajectory gives it away -- it is more ballistic than
the ball.

What gives it away is that it never goes anywhere. The residual around a fixed
glyph reshapes as the pan speed changes, so its centroid does wobble by a few
pixels a frame and a per-step threshold cannot separate it from a slow ball.
Over a whole trajectory the difference is stark: measured on frames 500-800,
the score-bug chains walked 12-78 px of path and finished 6-11 px from where
they started, while the ball covered 423 px of path and ended 408 px away. A
ball goes somewhere; an overlay only fidgets.

Abstention is a state
---------------------
A ball that is occluded, out of frame, or never acquired is reported as absent,
not interpolated through and not quietly attributed to the nearest player.
Interpolating is how a confident wrong position gets manufactured, and a wrong
position is worse here than no position: the events this feeds -- passes,
throws -- are defined by *where* possession changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "BallFrame",
    "BallState",
    "BallTrack",
    "TrackPoint",
    "chain_tracks",
    "project_points",
    "resolve_possession",
]


@dataclass(frozen=True)
class TrackPoint:
    """One candidate, in the image coordinates of its own frame.

    ``frame`` indexes the window being tracked, not the clip: it pairs
    positionally with the per-frame sequences passed alongside, and the caller
    adds the clip offset back when it needs to open a file.
    """

    frame: int
    x: float
    y: float
    radius: float = 0.0
    """Measured blob radius, carried so the appearance model can calibrate scale
    on what the ball actually was rather than on an assumed size."""


@dataclass(frozen=True)
class BallTrack:
    """A run of frames over which the ball moved independently of the players."""

    points: tuple[TrackPoint, ...]
    mean_speed: float = 0.0
    """Mean camera-compensated step, in pixels per frame."""

    @property
    def first_frame(self) -> int:
        return self.points[0].frame

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame


class BallState(str, Enum):
    FLIGHT = "flight"
    """Moving independently -- a pass, throw, shot, or loose ball."""

    HELD = "held"
    """Attributed to a person box. No independent motion to measure."""

    ABSENT = "absent"
    """Not known. Occluded, out of frame, or never acquired."""


@dataclass(frozen=True)
class BallFrame:
    """Where the ball is on one frame, or an explicit statement that we do not know."""

    frame: int
    state: BallState
    x: float | None = None
    y: float | None = None
    holder: int | None = None
    """Index into that frame's person boxes, when the ball is held."""


def project_points(points: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(array):
        return array
    homogeneous = np.c_[array, np.ones(len(array))] @ np.asarray(matrix, np.float64).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _carry(point, matrix: np.ndarray) -> np.ndarray:
    """Where a point in frame `t` lands in frame `t+1` if only the camera moved."""
    return project_points([point], matrix)[0]


def chain_tracks(
    per_frame: Sequence[Sequence[TrackPoint]],
    motions: Sequence[np.ndarray | None],
    *,
    min_speed: float = 9.0,
    max_speed: float = 70.0,
    tolerance: float = 9.0,
    min_length: int = 5,
    min_image_travel: float = 30.0,
    max_image_speed: float = 120.0,
) -> tuple[BallTrack, ...]:
    """Link candidates into constant-velocity trajectories.

    Args:
        motions: ``motions[i]`` is the homography carrying frame ``i``'s pixels
            onto frame ``i + 1``. A None entry means those two frames could not
            be related, and no trajectory crosses it.

    The speed gate applies to *every* step, not just the first. Gating only the
    opening pair lets a chain start at ball-like speed and then decay into a
    cluster of crowd blobs that happen to sit within tolerance of each other,
    which is the dominant false-positive mode when it is missing.

    A frame that offers no candidate within ``tolerance`` of the predicted
    position ends the track. Nothing is bridged: a gap means the evidence
    stopped, and a trajectory drawn through it would be invention.

    ``max_image_speed`` states the other half of the same fact: whatever the
    compensation says, a ball does not cross 700 px of picture between two
    frames. It is the backstop for a camera-motion estimate that was wrong in a
    way the compensated speed alone cannot reveal.

    ``min_image_travel`` rejects trajectories that only exist once the camera is
    compensated away, by requiring the track to get that far from where it began
    *in the picture* at some point. A burned-in graphic never does. Excursion is
    used rather than total path length so that a chain fidgeting in place cannot
    accumulate its way past the threshold.
    """
    used: list[set[int]] = [set() for _ in per_frame]
    tracks: list[BallTrack] = []

    def motion(index: int):
        return motions[index] if 0 <= index < len(motions) else None

    for start in range(len(per_frame) - min_length + 1):
        opening = motion(start)
        if opening is None:
            continue
        for first in range(len(per_frame[start])):
            if first in used[start]:
                continue
            origin = per_frame[start][first]
            carried = _carry((origin.x, origin.y), opening)
            best: list[tuple[int, TrackPoint]] | None = None
            best_speeds: list[float] = []

            for second, nxt in enumerate(per_frame[start + 1]):
                if second in used[start + 1]:
                    continue
                step = np.subtract((nxt.x, nxt.y), carried)
                opening_speed = float(np.hypot(*step))
                if not min_speed <= opening_speed <= max_speed:
                    continue
                if float(np.hypot(nxt.x - origin.x, nxt.y - origin.y)) > max_image_speed:
                    continue

                chain = [(first, origin), (second, nxt)]
                speeds = [opening_speed]
                velocity = step
                cursor = start + 2
                while cursor < len(per_frame) and per_frame[cursor]:
                    onward = motion(cursor - 1)
                    if onward is None:
                        break
                    last = chain[-1][1]
                    here = _carry((last.x, last.y), onward)
                    predicted = _carry(
                        (last.x + velocity[0], last.y + velocity[1]), onward
                    )
                    options = np.array([[p.x, p.y] for p in per_frame[cursor]])
                    distance = np.hypot(*(options - predicted).T)
                    pick = int(np.argmin(distance))
                    travelled = float(np.hypot(*(options[pick] - here)))
                    if (
                        distance[pick] > tolerance
                        or pick in used[cursor]
                        or not min_speed <= travelled <= max_speed
                        or float(np.hypot(*(options[pick] - [last.x, last.y])))
                        > max_image_speed
                    ):
                        break
                    chain.append((pick, per_frame[cursor][pick]))
                    speeds.append(travelled)
                    # Ease the carried velocity toward the latest step so a
                    # curving ballistic path is followed without chasing one
                    # noisy frame. Both terms are in this frame's coordinates.
                    velocity = 0.5 * (predicted - here) + 0.5 * (options[pick] - here)
                    cursor += 1

                if len(chain) < min_length:
                    continue
                seen = np.array([[p.x, p.y] for _, p in chain])
                excursion = float(np.hypot(*(seen - seen[0]).T).max())
                if excursion < min_image_travel:
                    continue                    # fidgets but never travels: an overlay
                if best is None or len(chain) > len(best):
                    best, best_speeds = chain, speeds

            if best is not None:
                for offset, (index, _) in enumerate(best):
                    used[start + offset].add(index)
                tracks.append(BallTrack(
                    points=tuple(point for _, point in best),
                    mean_speed=float(np.mean(best_speeds)),
                ))
                break
    return tuple(tracks)


def _nearest_box(x: float, y: float, boxes: Sequence[Sequence[float]], reach: float):
    """Index of the box containing or closest to a point, within `reach`."""
    best_index, best_distance = None, reach
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        gap = float(np.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)))
        if gap <= best_distance:
            best_index, best_distance = index, gap
    return best_index


def _carry_position(box: Sequence[float]) -> tuple[float, float]:
    """Where to put a held ball inside a person box, absent anything better.

    Upper-middle: a carried ball sits around chest and hand height, and this is
    a stated fallback rather than a measurement -- the appearance search
    supplies a real position when it can, and this is what stands in when it
    abstains.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (x1 + x2) / 2.0, y1 + (y2 - y1) * 0.35


def resolve_possession(
    tracks: Sequence[BallTrack],
    person_boxes: Sequence[Sequence[Sequence[float]]],
    *,
    reach: float = 60.0,
    max_unconfirmed: int = 50,
    confirmations: Mapping[int, tuple[float, float]] | None = None,
) -> tuple[BallFrame, ...]:
    """Fill the frames between trajectories with possession, or with absence.

    A trajectory ending beside a person is the ball arriving at that person, and
    from there it is carried rather than thrown -- invisible to motion, because
    it no longer moves independently of its holder. So possession is asserted by
    association and then has to keep earning itself: `confirmations` carries the
    frames where an appearance search actually found the ball, and after
    `max_unconfirmed` frames with none the state falls back to absent rather
    than claiming a holder nobody has verified.
    """
    confirmed = confirmations or {}
    frames = [BallFrame(frame=index, state=BallState.ABSENT)
              for index in range(len(person_boxes))]

    for track in tracks:
        for point in track.points:
            if 0 <= point.frame < len(frames):
                frames[point.frame] = BallFrame(
                    frame=point.frame, state=BallState.FLIGHT, x=point.x, y=point.y,
                )

    for track in sorted(tracks, key=lambda t: t.last_frame):
        end = track.points[-1]
        cursor = end.frame + 1
        position = (end.x, end.y)
        stale = 0
        while cursor < len(frames) and frames[cursor].state is BallState.ABSENT:
            holder = _nearest_box(position[0], position[1], person_boxes[cursor], reach)
            if holder is None:
                break
            seen = confirmed.get(cursor)
            stale = 0 if seen is not None else stale + 1
            if stale > max_unconfirmed:
                break
            position = seen if seen is not None else _carry_position(
                person_boxes[cursor][holder]
            )
            frames[cursor] = BallFrame(
                frame=cursor, state=BallState.HELD,
                x=position[0], y=position[1], holder=holder,
            )
            cursor += 1
    return tuple(frames)
