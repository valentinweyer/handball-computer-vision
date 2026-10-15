"""One ball, one trajectory: choosing it globally instead of one anchor at a time.

`anchored.py` decides each track on its own merits -- anchor on a detection,
propagate, keep it while something confirms it. That produces a correct-ish set
of segments and a badly wrong picture of the ball, because nothing ever asks
whether two segments can both be true. Measured on the BHC-FAG test clip, the
gaps between consecutive accepted segments are short (median 2 frames, max 13)
while the jumps across them are not (median 255 px, p90 690 px). At 25 fps that
is 60 px/frame where a handball flies at 10-30, so **30 of 50 consecutive
segment pairs cannot both be the ball**.

Camera motion is not the explanation. Estimated pairwise over the same clip the
pan is 0.8 px/frame (p90 1.8), and compensating the jumps changes the median
from 255 px to 258 px. The segments really are describing different objects.

What greedy anchoring lacks is the one constraint this sport hands us for free:
there is exactly one ball, and between two frames it goes somewhere reachable.
That turns "is this detection good enough?" -- a question about one box -- into
"which sequence of detections is the ball?", a question with a single global
answer. This module computes that answer as a shortest path.

The graph
---------
Each candidate position is a node. An edge joins a candidate at frame `t` to one
at frame `t + k` (k up to `max_gap`) when the implied speed is reachable, and
costs more the less the evidence supports it and the further the ball had to
travel. The best path from before the clip to after it is the trajectory, and
frames it does not visit are absent -- not interpolated, because a frame the
optimum skipped is a frame where the ball was not seen, which is a state and not
a hole.

This is the same argument `tracking.py` makes for motion blobs -- "a blob is
accepted because of where its neighbours are, never on its own merits" -- moved
up a level, from blobs to whole detections, and solved exactly rather than
greedily. Being a DAG (edges only ever go forward in time) the optimum is one
pass of dynamic programming, with no risk of the local minima a greedy scan
falls into.

Skipping is priced, not free
----------------------------
A path could otherwise explain the whole clip by visiting three easy detections
and declaring the other 497 frames absent. `absent_cost` per skipped frame is
what stops that: skipping has to be cheaper than the bad alternatives but not
cheaper than a decent detection, so the optimum reaches for the ball wherever
the evidence is reasonable and lets go where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["Candidate", "PathConfig", "select_path", "candidates_from_detections"]


@dataclass(frozen=True)
class Candidate:
    """One possible ball position on one frame."""

    frame: int
    x: float
    y: float
    score: float
    source: str = "detection"


@dataclass(frozen=True)
class PathConfig:
    """Costs for the one-ball path.

    `max_speed` is 45 px/frame rather than a handball's 10-30: the ball is
    fastest on a shot, the frame rate is not guaranteed, and the cost term
    already discourages fast steps without forbidding them. The bar exists to
    rule out the 255 px hops between unrelated objects, not to enforce physics.
    """

    max_speed: float = 45.0
    """Largest reachable displacement per frame, in pixels."""

    max_gap: int = 15
    """Longest run of unseen frames the path may step over in one edge."""

    absent_cost: float = 1.1
    """Charged per skipped frame. Swept on the BHC-FAG test clip, measuring
    coverage, how many segments it breaks into, and the implied speed needed to
    get from one segment to the next -- the last being whether the segments can
    all be the same ball:

        absent_cost   coverage   segments   inter-segment px/frame p50/p90
                0.7      52.4%         27                          17 / 42
                0.9      64.8%         27                          22 / 42
                1.1      70.4%         31                          27 / 44

    1.1 because coherence is not what is being traded here. `max_speed` bounds
    the inter-segment figure by construction, so it stays under 45 at every
    setting, and coverage is simply better at the top of the range.

    At 1.1 a skipped frame costs more than any admitted detection, so score
    stops being the discriminator and reachability does all the work. That is
    deliberate: on this clip the detector's confidence is poorly calibrated
    (median 0.43 against 0.76 on a valid-split match), while geometry is not."""

    speed_cost: float = 0.012
    """Per px/frame. Ranks two otherwise equal candidates by which one the ball
    could have reached more easily."""


def candidates_from_detections(
    *sources: Mapping[int, Sequence[tuple[Sequence[float], float]]],
    threshold: float = 0.2,
    merge_radius: float = 20.0,
) -> dict[int, list[Candidate]]:
    """Per-frame candidates from one or more detection caches.

    Boxes from different passes that describe the same object are merged, best
    score winning: the full-frame and square-crop passes agree on 144 frames of
    BHC-FAG, and leaving both in would let the path score one sighting twice.
    """
    per_frame: dict[int, list[Candidate]] = {}
    for source in sources:
        for frame, detections in source.items():
            bucket = per_frame.setdefault(int(frame), [])
            for box, score in detections:
                if score < threshold:
                    continue
                x = (float(box[0]) + float(box[2])) / 2.0
                y = (float(box[1]) + float(box[3])) / 2.0
                for index, existing in enumerate(bucket):
                    if float(np.hypot(existing.x - x, existing.y - y)) <= merge_radius:
                        if score > existing.score:
                            bucket[index] = Candidate(int(frame), x, y, float(score))
                        break
                else:
                    bucket.append(Candidate(int(frame), x, y, float(score)))
    return {frame: bucket for frame, bucket in sorted(per_frame.items()) if bucket}


def select_path(
    candidates: Mapping[int, Sequence[Candidate]],
    n_frames: int,
    config: PathConfig | None = None,
) -> list[Candidate]:
    """The single best ball trajectory through `candidates`, in frame order.

    Returns the visited candidates only. Frames between two visited ones were
    skipped by the optimum and are absent; the caller decides whether anything
    may fill them, and `anchored.classify_frames` deliberately does not.
    """
    config = config or PathConfig()
    flat: list[Candidate] = [c for frame in sorted(candidates) for c in candidates[frame]]
    if not flat:
        return []

    # Sorted by frame so every edge points forward and one left-to-right sweep
    # settles each node's best predecessor.
    flat.sort(key=lambda c: c.frame)
    by_frame: dict[int, list[int]] = {}
    for index, candidate in enumerate(flat):
        by_frame.setdefault(candidate.frame, []).append(index)

    # Entering the path costs the frames skipped before the first sighting, so a
    # late start is not free -- otherwise every path would begin at its easiest
    # detection.
    best = [config.absent_cost * c.frame + _emission(c) for c in flat]
    previous: list[int | None] = [None] * len(flat)

    frames = sorted(by_frame)
    for frame in frames:
        for later in frames:
            step = later - frame
            if step < 1:
                continue
            if step > config.max_gap + 1:
                break
            for i in by_frame[frame]:
                for j in by_frame[later]:
                    distance = float(np.hypot(flat[j].x - flat[i].x, flat[j].y - flat[i].y))
                    speed = distance / step
                    if speed > config.max_speed:
                        continue
                    cost = (best[i]
                            + config.absent_cost * (step - 1)
                            + config.speed_cost * speed
                            + _emission(flat[j]))
                    if cost < best[j]:
                        best[j] = cost
                        previous[j] = i

    # Leaving the path costs the frames after the last sighting, symmetrically.
    end = min(range(len(flat)),
              key=lambda i: best[i] + config.absent_cost * (n_frames - 1 - flat[i].frame))

    path: list[Candidate] = []
    cursor: int | None = end
    while cursor is not None:
        path.append(flat[cursor])
        cursor = previous[cursor]
    return list(reversed(path))


def _emission(candidate: Candidate) -> float:
    """What it costs to claim this candidate is the ball.

    `1 - score`, so a 0.9 detection costs 0.1 and a 0.25 one costs 0.75 -- more
    than `absent_cost`, which is what makes a weak lone detection lose to a gap
    unless its neighbours support it.
    """
    return 1.0 - float(candidate.score)
