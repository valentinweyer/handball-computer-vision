"""SORT with an appearance term in association -- an experiment, not a default.

Plain `trackers.SORTTracker` with one change: before the Hungarian step, each
track/detection IoU is combined with the cosine distance between the track's
appearance (an EMA of its matched detections' features) and the detection's.

  * `gate`: IoU kept only where the appearance distance is within `gate`;
    otherwise the pair cannot match (DeepSORT-style veto).
  * `fuse`: BoT-SORT's rule -- where IoU >= the SORT threshold and the
    distance is within `gate`, the score is max(IoU, 1 - distance).

The appearance cue is generic (kit-colour histogram or PRTReID), computed
inside the tracker experiment; it never reads the team model, so it does not
couple team classification into tracking (CLAUDE.md hard constraints).
Features arrive precomputed, aligned to the detection order of each frame.
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import SORTTracker


class AppearanceSORT(SORTTracker):
    search_space = {}   # not tunable through the trackers registry

    def __init__(self, features: list[np.ndarray], mode: str, gate: float,
                 ema: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"gate", "fuse"}:
            raise ValueError(mode)
        self._features, self.mode, self.gate, self.ema = features, mode, gate, ema
        self._cursor = 0
        self._frame_feats = None

    def update(self, detections: sv.Detections, frame=None, timestamp=None):
        self._frame_feats = self._features[self._cursor]
        self._cursor += 1
        if len(self._frame_feats) != len(detections):
            raise RuntimeError("feature cache does not match this frame's detections")
        return super().update(detections, None, timestamp)

    def _get_associated_indices(self, iou_matrix, detection_boxes):
        feats = self._frame_feats
        score = iou_matrix.copy()
        if len(self.tracks) and len(detection_boxes):
            tracks = np.stack([
                t.appearance if getattr(t, "appearance", None) is not None
                else np.zeros(feats.shape[1], np.float32) for t in self.tracks
            ])
            known = np.linalg.norm(tracks, axis=1) > 0
            distance = 1.0 - tracks @ feats.T
            allowed = (distance <= self.gate) | ~known[:, None]
            if self.mode == "gate":
                score = np.where(allowed, score, 0.0)
            else:
                fused = np.where(known[:, None], np.maximum(score, 1.0 - distance), score)
                score = np.where(allowed & (score >= self.minimum_iou_threshold), fused, 0.0)
        matched, unmatched_tracks, unmatched_detections = super()._get_associated_indices(
            score, detection_boxes)
        for row, col in matched:
            self._absorb(self.tracks[row], feats[col])
        return matched, unmatched_tracks, unmatched_detections

    def _spawn_new_tracklets(self, confidences, detection_boxes, unmatched_detections):
        before = len(self.tracks)
        super()._spawn_new_tracklets(confidences, detection_boxes, unmatched_detections)
        spawned = [d for d in unmatched_detections
                   if confidences[d] >= self.track_activation_threshold]
        for track, col in zip(self.tracks[before:], spawned):
            self._absorb(track, self._frame_feats[col])

    def _absorb(self, track, feature: np.ndarray) -> None:
        current = getattr(track, "appearance", None)
        mixed = feature if current is None else self.ema * current + (1 - self.ema) * feature
        track.appearance = mixed / max(np.linalg.norm(mixed), 1e-9)
