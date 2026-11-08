"""One pass over the video for every per-frame model: detector, court keypoints, motion.

Decoding a 1080p frame costs about as much as running the court model on it,
so the three per-frame consumers share one sequential decode instead of each
reading the video. The detector runs only on every `detect_stride`-th frame:
tracking reads detections at its checkpoints and number reads at its OCR
cadence and nowhere else, so frames between them would be detected for
nothing (`detection_stride_guard` checks the cadences agree). Camera motion
is CPU work and runs on a worker thread while the GPU models see the frame.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from handball_cv.court.sequence import MotionMeter

__all__ = ["detection_stride_guard", "scan_video"]

CACHE_SCHEMA = 1


def detection_stride_guard(stride: int, *cadences: int) -> None:
    """Fail unless every cadence lands only on frames the detector scored.

    A frame the detector skipped and a frame with nothing on it look identical
    in the cache, so a cadence off the stride would silently read empty frames.
    """
    for cadence in cadences:
        if cadence % stride:
            raise ValueError(f"cadence {cadence} is not a multiple of the detection stride {stride}: "
                             f"it would read frames the detector never scored")


def scan_video(video, *, detector=None, court_model=None, motion: bool = True,
               detect_stride: int = 5, threshold: float = 0.3, detector_id: str = "",
               desc: str = "scan") -> dict:
    """Run whichever per-frame stages are given over `video` (a `VideoFrames`).

    Returns `{"detections": cache dict | None, "keypoints": list | None,
    "motion": list | None}`. The detection cache has the schema every
    consumer here already reads (`offsets` spanning every frame, `boxes`,
    `confidence`, `class_id` in project ids) plus `stride`, so a skipped frame
    is distinguishable from an empty one.
    """
    count = len(video)
    offsets, boxes, scores, classes = [0], [], [], []
    keypoints = [] if court_model is not None else None
    meter = MotionMeter() if motion else None
    steps = [] if motion else None
    pool = ThreadPoolExecutor(max_workers=1) if motion else None
    pending = None
    try:
        for index in tqdm(range(count), desc=desc):
            frame = video.rgb(index)
            if meter is not None:
                if pending is not None:
                    steps.append(pending.result())
                pending = pool.submit(meter.step, frame)
            if detector is not None:
                if index % detect_stride == 0:
                    xyxy, confidence, class_id = detector.infer(frame, threshold)
                    boxes.append(np.asarray(xyxy, dtype=np.float64).reshape(-1, 4))
                    scores.append(np.asarray(confidence, dtype=np.float64))
                    classes.append(np.asarray(class_id, dtype=np.int64))
                    offsets.append(offsets[-1] + len(xyxy))
                else:
                    offsets.append(offsets[-1])
            if keypoints is not None:
                keypoints.append([list(point) for point in court_model.infer(frame)[0]])
        if pending is not None:
            steps.append(pending.result())
    finally:
        if pool is not None:
            pool.shutdown()

    detections = None
    if detector is not None:
        detections = {
            "schema": np.asarray(CACHE_SCHEMA),
            "detector_id": np.asarray(detector_id),
            "threshold": np.asarray(threshold),
            "stride": np.asarray(detect_stride),
            "scored_frames": np.asarray(len(range(0, count, detect_stride))),
            "total_frames": np.asarray(count),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "boxes": np.concatenate(boxes) if boxes else np.zeros((0, 4)),
            "confidence": np.concatenate(scores) if scores else np.zeros(0),
            "class_id": np.concatenate(classes) if classes else np.zeros(0, dtype=np.int64),
        }
    return {"detections": detections, "keypoints": keypoints, "motion": steps}
