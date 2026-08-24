"""Soft team gates for ByteTrack and McByte association.

Detections carry cheap, tracker-independent evidence in ``Detections.data``:

``team_probability``
    Probability of semantic team B (team A is ``1 - p``).
``team_quality``
    Crop/overlap evidence weight in ``[0, 1]``.

Only a stable track and a high-confidence, high-quality contradictory detection
are forbidden from matching. Uncertain detections remain fully available to
motion/IoU association, avoiding the brittle behavior of two hard-split
trackers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, McByteTracker
from trackers.utils.iou import BaseIoU, IoU

TEAM_PROBABILITY_KEY = "team_probability"
TEAM_QUALITY_KEY = "team_quality"
DEFAULT_STABLE_CONFIDENCE = 0.80
DEFAULT_MIN_QUALITY = 0.40
EVIDENCE_PRIOR = 0.5


@dataclass
class AssociationTeamEvidence:
    scores: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )

    def update(self, probability_b: float, quality: float) -> None:
        if not np.isfinite(probability_b) or not np.isfinite(quality):
            return
        probability_b = float(np.clip(probability_b, 0.0, 1.0))
        quality = float(np.clip(quality, 0.0, 1.0))
        certainty = abs(2.0 * probability_b - 1.0)
        weight = quality * certainty
        self.scores += weight * np.array(
            [1.0 - probability_b, probability_b], dtype=float
        )

    @property
    def team_id(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def confidence(self) -> float:
        posterior = self.scores + EVIDENCE_PRIOR
        return float(posterior.max() / posterior.sum())


class TeamGatedIoU(BaseIoU):
    """Standard IoU with owner-provided contradiction gates."""

    def __init__(self, owner, base_iou: BaseIoU | None = None) -> None:
        self.owner = owner
        self.base_iou = base_iou if base_iou is not None else IoU()

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        similarity = self.base_iou.compute(boxes_1, boxes_2)
        return self.owner._apply_team_gate(similarity, boxes_1, boxes_2)

    def normalize_for_fusion(self, similarity_matrix: np.ndarray) -> np.ndarray:
        return self.base_iou.normalize_for_fusion(similarity_matrix)


class _TeamGateMixin:
    def _init_team_gate(
        self,
        base_iou: BaseIoU | None,
        stable_confidence: float,
        min_quality: float,
    ) -> None:
        self.team_stable_confidence = float(stable_confidence)
        self.team_min_quality = float(min_quality)
        self._association_team_evidence: dict[int, AssociationTeamEvidence] = {}
        self._team_detection_boxes = np.empty((0, 4), dtype=float)
        self._team_detection_probability = np.empty(0, dtype=float)
        self._team_detection_quality = np.empty(0, dtype=float)
        self.team_gate_rejections = 0
        self.iou = TeamGatedIoU(self, base_iou)

    @staticmethod
    def _data_vector(
        detections: sv.Detections, key: str, default: float,
    ) -> np.ndarray:
        value = detections.data.get(key)
        if value is None:
            return np.full(len(detections), default, dtype=float)
        value = np.asarray(value, dtype=float).reshape(-1)
        if len(value) != len(detections):
            raise ValueError(
                f"detections.data[{key!r}] has {len(value)} rows for "
                f"{len(detections)} detections"
            )
        return value

    def _prepare_team_detections(self, detections: sv.Detections) -> None:
        self._team_detection_boxes = np.asarray(
            detections.xyxy, dtype=float
        ).reshape(-1, 4)
        self._team_detection_probability = self._data_vector(
            detections, TEAM_PROBABILITY_KEY, np.nan
        )
        self._team_detection_quality = self._data_vector(
            detections, TEAM_QUALITY_KEY, 0.0
        )

    @staticmethod
    def _nearest_box_indices(
        query: np.ndarray, reference: np.ndarray,
    ) -> np.ndarray:
        if len(query) == 0 or len(reference) == 0:
            return np.full(len(query), -1, dtype=int)
        scale = np.maximum(
            reference[:, 2:] - reference[:, :2], 1.0
        ).mean(axis=1)
        difference = np.abs(
            query[:, None, :] - reference[None, :, :]
        ).mean(axis=2) / scale[None, :]
        indices = difference.argmin(axis=1)
        best = difference[np.arange(len(query)), indices]
        return np.where(best <= 1e-3, indices, -1)

    def _track_objects_for_boxes(self, boxes: np.ndarray) -> list[object | None]:
        tracks = list(self.tracks)
        if not tracks:
            return [None] * len(boxes)
        states = np.asarray([track.get_state_bbox() for track in tracks])
        indices = self._nearest_box_indices(boxes, states)
        return [tracks[index] if index >= 0 else None for index in indices]

    def _apply_team_gate(
        self,
        similarity: np.ndarray,
        track_boxes: np.ndarray,
        detection_boxes: np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(similarity, dtype=float).copy()
        if not result.size:
            return result
        track_objects = self._track_objects_for_boxes(track_boxes)
        detection_indices = self._nearest_box_indices(
            detection_boxes, self._team_detection_boxes
        )
        for row, track in enumerate(track_objects):
            if track is None:
                continue
            evidence = self._association_team_evidence.get(id(track))
            if (
                evidence is None
                or evidence.confidence < self.team_stable_confidence
            ):
                continue
            for column, detection_index in enumerate(detection_indices):
                if detection_index < 0:
                    continue
                probability_b = self._team_detection_probability[detection_index]
                quality = self._team_detection_quality[detection_index]
                if not np.isfinite(probability_b) or quality < self.team_min_quality:
                    continue
                detection_confidence = max(probability_b, 1.0 - probability_b)
                detection_team = int(probability_b >= 0.5)
                if (
                    detection_confidence >= self.team_stable_confidence
                    and detection_team != evidence.team_id
                    and result[row, column] > 0
                ):
                    result[row, column] = 0.0
                    self.team_gate_rejections += 1
        return result

    def _update_team_evidence(self, tracked: sv.Detections) -> None:
        if tracked.tracker_id is None or not len(tracked):
            return
        probability = self._data_vector(
            tracked, TEAM_PROBABILITY_KEY, np.nan
        )
        quality = self._data_vector(tracked, TEAM_QUALITY_KEY, 0.0)
        tracks_by_id = {
            int(track.tracker_id): track
            for track in self.tracks
            if int(track.tracker_id) >= 0
        }
        for tracker_id, probability_b, crop_quality in zip(
            tracked.tracker_id, probability, quality
        ):
            track = tracks_by_id.get(int(tracker_id))
            if track is None:
                continue
            evidence = self._association_team_evidence.setdefault(
                id(track), AssociationTeamEvidence()
            )
            evidence.update(probability_b, crop_quality)
        alive = {id(track) for track in self.tracks}
        self._association_team_evidence = {
            key: value for key, value in self._association_team_evidence.items()
            if key in alive
        }

    def association_team(self, tracker_id: int) -> tuple[int | None, float]:
        for track in self.tracks:
            if int(track.tracker_id) == int(tracker_id):
                evidence = self._association_team_evidence.get(id(track))
                if evidence is not None:
                    return evidence.team_id, evidence.confidence
        return None, 0.5


class TeamGatedByteTrackTracker(_TeamGateMixin, ByteTrackTracker):
    search_space = {}

    def __init__(
        self, *args,
        team_stable_confidence: float = DEFAULT_STABLE_CONFIDENCE,
        team_min_quality: float = DEFAULT_MIN_QUALITY,
        **kwargs,
    ) -> None:
        base_iou = kwargs.pop("iou", None)
        super().__init__(*args, iou=base_iou, **kwargs)
        self._init_team_gate(
            base_iou, team_stable_confidence, team_min_quality
        )

    def update(
        self, detections: sv.Detections, frame=None, timestamp=None,
    ) -> sv.Detections:
        self._prepare_team_detections(detections)
        tracked = super().update(
            detections, frame=frame, timestamp=timestamp
        )
        self._update_team_evidence(tracked)
        return tracked


class TeamGatedMcByteTracker(_TeamGateMixin, McByteTracker):
    search_space = {}

    def __init__(
        self, *args,
        team_stable_confidence: float = DEFAULT_STABLE_CONFIDENCE,
        team_min_quality: float = DEFAULT_MIN_QUALITY,
        **kwargs,
    ) -> None:
        base_iou = kwargs.pop("iou", None)
        super().__init__(*args, iou=base_iou, **kwargs)
        self._init_team_gate(
            base_iou, team_stable_confidence, team_min_quality
        )

    def update(
        self, detections: sv.Detections, frame=None, timestamp=None,
    ) -> sv.Detections:
        self._prepare_team_detections(detections)
        tracked = super().update(
            detections, frame=frame, timestamp=timestamp
        )
        self._update_team_evidence(tracked)
        return tracked
