"""Per-video detection caches: every class the detector produced, per frame.

A cache is an npz of all boxes concatenated, with `offsets[i]:offsets[i + 1]`
the rows of frame `i`. It keeps every class so that a later consumer never
forces a re-detection; each consumer filters for what it wants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv

__all__ = [
    "GOALKEEPER_CLASS_ID", "NUMBER_CLASS_ID", "PERSON_CLASS_IDS",
    "frame_detections", "load_detection_cache", "number_detections", "person_detections",
]

GOALKEEPER_CLASS_ID = 1


def load_detection_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "offsets": data["offsets"],
            "boxes": data["boxes"],
            "confidence": data["confidence"],
            "class_id": data["class_id"],
        }


def frame_detections(cache: dict[str, np.ndarray], frame_index: int) -> sv.Detections:
    start, end = cache["offsets"][frame_index:frame_index + 2]
    start, end = int(start), int(end)
    return sv.Detections(
        xyxy=cache["boxes"][start:end].copy(),
        confidence=cache["confidence"][start:end].copy(),
        class_id=cache["class_id"][start:end].astype(int, copy=True),
    )


# Project-wide class ids in the detection caches: 1 goalkeeper, 2 player,
# 3 referee, 4 jersey number.
PERSON_CLASS_IDS = (1, 2)
NUMBER_CLASS_ID = 4


def person_detections(cache: dict[str, np.ndarray], frame_index: int) -> sv.Detections:
    """Only the people a tracker should follow: goalkeepers and field players.

    The caches deliberately store every class the detector produced so a later
    question never forces a re-detection, which makes filtering the consumer's
    job. Feeding a tracker the unfiltered cache makes it follow referees and --
    since the caches gained a jersey-number class -- the number boxes too.
    """
    detections = frame_detections(cache, frame_index)
    if detections.class_id is None:
        return detections
    return detections[np.isin(detections.class_id, PERSON_CLASS_IDS)]


def number_detections(cache: dict[str, np.ndarray], frame_index: int) -> np.ndarray:
    """Class-4 jersey-number boxes for this frame, as (N, 4) xyxy."""
    detections = frame_detections(cache, frame_index)
    if detections.class_id is None:
        return detections.xyxy
    return detections.xyxy[detections.class_id == NUMBER_CLASS_ID]
