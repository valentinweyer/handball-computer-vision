"""Offline tracklet refinement: split mixed identities, connect fragments.

An implementation of the two operations from GTA (Global Tracklet Association,
ACCV 2024 -- https://arxiv.org/abs/2411.08216), which the SoccerTrack 2025
winner used as a post-pass over an online tracker:

  * **split**   -- a tracklet that slid from one player onto another is cut
                   into per-identity pieces.
  * **connect** -- tracklets that are the same player, separated by a gap, are
                   merged back together.

This is deliberately *not* a tracker. It runs after MCByte on a finished clip,
which is the right shape for this project: the pipeline is offline (cached
detections, rendered output), so a global view of the whole clip is available
and is strictly more informative than online association.

Measured motivation on FelixClaar: 7 of 8 identity failures were McByte keeping
one tracker_id alive while it slid onto a different player -- exactly what
split addresses -- and disabling Cutie masks reduced those switches but cost
fragmentation, which is exactly what connect repays.

The central risk is over-splitting. A player's appearance embedding also moves
with pose, motion blur and occlusion, so "the embeddings form two clusters" is
by itself weak evidence of an identity change. The guard used here is
**temporal segregation**: a genuine identity switch puts one identity before a
changepoint and the other after it, while pose variation interleaves. Clusters
that interleave in time are treated as one identity and not split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Cosine distance between L2-normalised embeddings, in [0, 2].
SPLIT_DISTANCE_THRESHOLD = 0.45
# A split must leave both sides with enough samples to be believable.
SPLIT_MIN_SAMPLES = 4
# Fraction of each cluster that must fall on its own side of the changepoint
# for the two clusters to count as temporally segregated rather than interleaved.
SPLIT_MIN_SEGREGATION = 0.85
# Representative embeddings closer than this may describe the same player.
CONNECT_DISTANCE_THRESHOLD = 0.30
# Tracklets overlapping by more than this many frames cannot be one player.
CONNECT_MAX_OVERLAP_FRAMES = 0


def normalise(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot products are cosine similarities."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        embeddings = embeddings.reshape(len(embeddings), -1)
    if not len(embeddings):
        # `reshape(0, -1)` cannot infer a width from zero elements.
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


@dataclass
class Tracklet:
    """One contiguous run of detections the tracker considered a single object."""

    tracklet_id: int
    source_tracker_id: int
    frames: np.ndarray            # (N,) int, ascending
    boxes: np.ndarray             # (N, 4) xyxy
    embeddings: np.ndarray        # (N, D)
    parts: list = field(default_factory=list)  # provenance after a split

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def __len__(self) -> int:
        return len(self.frames)

    def representative(self) -> np.ndarray:
        """Median of the normalised embeddings -- robust to a few bad crops."""
        unit = normalise(self.embeddings)
        if not len(unit):
            return np.zeros(self.embeddings.shape[1:], dtype=np.float32)
        median = np.median(unit, axis=0)
        return median / max(float(np.linalg.norm(median)), 1e-8)

    def slice(self, mask: np.ndarray, tracklet_id: int) -> "Tracklet":
        return Tracklet(
            tracklet_id=tracklet_id,
            source_tracker_id=self.source_tracker_id,
            frames=self.frames[mask],
            boxes=self.boxes[mask],
            embeddings=self.embeddings[mask],
            parts=list(self.parts) + [self.tracklet_id],
        )


def _two_cluster_labels(unit: np.ndarray) -> np.ndarray:
    """Split embeddings into two groups by their dominant axis of variation.

    A full agglomerative clustering is unnecessary: we only ever ask whether a
    tracklet holds *two* identities, and the leading principal direction of a
    two-identity tracklet separates them. Sign of the projection is the label.
    """
    centred = unit - unit.mean(axis=0, keepdims=True)
    # Leading right singular vector == first principal direction.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ vt[0]
    return (projection > 0).astype(int)


def _cluster_distance(unit: np.ndarray, labels: np.ndarray) -> float:
    """Cosine distance between the two cluster centroids."""
    a, b = unit[labels == 0], unit[labels == 1]
    if not len(a) or not len(b):
        return 0.0
    ca = a.mean(axis=0); ca /= max(float(np.linalg.norm(ca)), 1e-8)
    cb = b.mean(axis=0); cb /= max(float(np.linalg.norm(cb)), 1e-8)
    return float(1.0 - np.dot(ca, cb))


def _segregation(labels: np.ndarray) -> tuple[float, int]:
    """How cleanly the two labels separate in time, and where they divide.

    Returns (score, changepoint_index). Score 1.0 means every sample of one
    cluster precedes every sample of the other -- an identity switch. Around
    0.5 means the clusters interleave -- pose variation within one identity.
    """
    best_score, best_index = 0.0, 0
    total = len(labels)
    for index in range(1, total):
        left, right = labels[:index], labels[index:]
        # Fraction correctly placed if the split point is here, either polarity.
        agree = (left == 0).sum() + (right == 1).sum()
        score = max(agree, total - agree) / total
        if score > best_score:
            best_score, best_index = score, index
    return best_score, best_index


def split_tracklet(
    tracklet: Tracklet,
    next_id: int,
    distance_threshold: float = SPLIT_DISTANCE_THRESHOLD,
    min_samples: int = SPLIT_MIN_SAMPLES,
    min_segregation: float = SPLIT_MIN_SEGREGATION,
) -> list[Tracklet]:
    """Cut a tracklet at an identity changepoint, or return it unchanged.

    Returns one tracklet when no split is warranted, two when it is.
    """
    if len(tracklet) < 2 * min_samples:
        return [tracklet]
    unit = normalise(tracklet.embeddings)
    labels = _two_cluster_labels(unit)
    if labels.sum() < min_samples or (len(labels) - labels.sum()) < min_samples:
        return [tracklet]
    if _cluster_distance(unit, labels) < distance_threshold:
        return [tracklet]
    score, changepoint = _segregation(labels)
    # Interleaved clusters are pose variation, not a new person.
    if score < min_segregation:
        return [tracklet]
    if changepoint < min_samples or len(tracklet) - changepoint < min_samples:
        return [tracklet]
    before = np.zeros(len(tracklet), dtype=bool)
    before[:changepoint] = True
    return [
        tracklet.slice(before, tracklet.tracklet_id),
        tracklet.slice(~before, next_id),
    ]


def split_all(tracklets: list, **kwargs) -> list:
    """Apply `split_tracklet` across a set, allocating ids for new pieces."""
    next_id = max((t.tracklet_id for t in tracklets), default=0) + 1
    out = []
    for tracklet in tracklets:
        pieces = split_tracklet(tracklet, next_id, **kwargs)
        if len(pieces) > 1:
            next_id += 1
        out.extend(pieces)
    return out


def _overlap(a: Tracklet, b: Tracklet) -> int:
    """Number of frames the two tracklets both occupy."""
    return len(np.intersect1d(a.frames, b.frames))


def connect_tracklets(
    tracklets: list,
    distance_threshold: float = CONNECT_DISTANCE_THRESHOLD,
    max_overlap_frames: int = CONNECT_MAX_OVERLAP_FRAMES,
) -> dict:
    """Group tracklets that describe one player. -> {tracklet_id: group_id}.

    Two tracklets that are visible at the same time are necessarily different
    people, no matter how similar they look -- that constraint does most of the
    work in a same-uniform sport, where appearance alone is weak.
    """
    order = sorted(tracklets, key=lambda t: (t.start, t.tracklet_id))
    group_of = {t.tracklet_id: t.tracklet_id for t in order}
    members = {t.tracklet_id: [t] for t in order}
    reps = {t.tracklet_id: t.representative() for t in order}

    for index, tracklet in enumerate(order):
        for other in order[:index]:
            root = group_of[other.tracklet_id]
            if root == group_of[tracklet.tracklet_id]:
                continue
            if any(_overlap(tracklet, m) > max_overlap_frames for m in members[root]):
                continue
            distance = float(1.0 - np.dot(reps[tracklet.tracklet_id], reps[root]))
            if distance > distance_threshold:
                continue
            # Merge this tracklet's group into `root`.
            moving = group_of[tracklet.tracklet_id]
            for member in members[moving]:
                group_of[member.tracklet_id] = root
            members[root].extend(members[moving])
            stacked = np.stack([reps[m.tracklet_id] for m in members[root]])
            merged = stacked.mean(axis=0)
            reps[root] = merged / max(float(np.linalg.norm(merged)), 1e-8)
            members[moving] = []
            break
    return group_of
