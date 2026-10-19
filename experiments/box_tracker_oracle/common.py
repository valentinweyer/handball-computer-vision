"""Clips, detection sources and per-frame appearance features for the
box-tracker oracle study (`docs/tracking-evaluation.md` §8.13).

Two detection arms per clip: `detector` (the cached RF-DETR run every frame)
and `oracle` (the reference's own boxes, every identity). Appearance features
are computed once per (clip, arm, feature) on CPU and cached, aligned to the
exact detection order the tracker receives, so every association variant
reuses them instead of re-embedding.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from scripts.evaluate_tracker_identity import (
    MATCH_IOU,
    iou_matrix,
    load_reference,
    oracle_frame_detections,
    parse_labels,
)
from scripts.render_raw_team_classification import frame_detections, load_detection_cache

MAIN = Path("/home/valentinweyer/projects/handball-computer-vision")
OUT = MAIN / "runs/tracker_eval/oracle_2026-10-20"

# clip -> (video stem, detection cache, reference dir, labels dir, non-player ids)
CLIPS = {
    "felix": ("FelixClaar", "outputs/team_comparison/.FelixClaar_detections_v1.npz",
              ".FelixClaar_ref_masks", "FelixClaar_reference", ()),
    "han": ("Han-Ber4_cached", "outputs/team_dataset/.Han-Ber4_detections_v1.npz",
            ".Han-Ber4_sam2_masks", "HanBer4_sam2_reference", (12, 14)),
    "bhc": ("BHC-FAG_window_cached", "outputs/team_dataset/.BHC-FAG_window_detections_v1.npz",
            ".BHC-FAG_window_ref_masks", "BHC-FAG_window_reference", ()),
}


class Clip:
    def __init__(self, name: str):
        stem, detections, reference, labels, exclude = CLIPS[name]
        self.name = name
        self.video = MAIN / "data/raw" / f"{stem}.mp4"
        self.detection_cache = MAIN / detections
        self.reference_dir = MAIN / "source" / reference
        self.valid = parse_labels(MAIN / "runs/tracklet_labels" / labels / "answers.txt")
        self.keep = set(self.valid) - set(exclude)
        self.reference = load_reference(self.reference_dir, 0, self.valid)
        self._oracle = None
        self._cache = None
        self.fps = sv.VideoInfo.from_video_path(str(self.video)).fps

    @property
    def oracle_boxes(self) -> dict:
        if self._oracle is None:
            self._oracle = load_reference(self.reference_dir, 0, None)
        return self._oracle

    def detections_for(self, arm: str):
        """Per-frame detections for one arm.

        `filled` is the detector's own output plus the reference box of every
        player it missed (no detection at IoU >= 0.5): perfect recall with the
        detector's geometry, separating "missed players" from "imprecise boxes".
        """
        if arm == "oracle":
            boxes = self.oracle_boxes
            return lambda f: oracle_frame_detections(boxes, f)
        if self._cache is None:
            self._cache = load_detection_cache(self.detection_cache)
        cache = self._cache
        if arm == "detector":
            return lambda f: frame_detections(cache, f)
        if arm != "filled":
            raise ValueError(arm)
        boxes = self.oracle_boxes

        def filled(f):
            detected = frame_detections(cache, f)
            truth = boxes.get(f, {})
            if not truth:
                return detected
            reference = np.stack([truth[i] for i in sorted(truth)])
            people = detected.xyxy[np.isin(detected.class_id, (1, 2))]
            missed = (np.ones(len(reference), bool) if not len(people)
                      else iou_matrix(reference, people).max(axis=1) < MATCH_IOU)
            if not missed.any():
                return detected
            extra = sv.Detections(xyxy=reference[missed], confidence=np.ones(missed.sum()),
                                  class_id=np.full(missed.sum(), 2))
            return sv.Detections.merge([detected, extra])
        return filled

    def frames_rgb(self):
        for frame_bgr in sv.get_video_frames_generator(str(self.video)):
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(max(x2, x1 + 2), w), min(max(y2, y1 + 2), h)
    return frame[y1:y2, x1:x2]


def colour_feature(frame_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Hellinger-normalised HSV histogram of the box's torso region.

    Torso = central 50% of the width, 15-55% of the height: mostly shirt,
    little floor or neighbour. A kit-colour cue, deliberately cheap.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso = np.array([x1 + 0.25 * w, y1 + 0.15 * h, x2 - 0.25 * w, y1 + 0.55 * h])
    patch = crop(frame_rgb, torso)
    hsv = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = np.sqrt(hist / max(hist.sum(), 1.0))
    return (hist / max(np.linalg.norm(hist), 1e-9)).astype(np.float32)


def prtreid_encoder(threads: int = 6):
    import torch

    from handball_cv.embeddings.prtreid import PRTReIDBackend

    torch.set_num_threads(threads)
    backend = PRTReIDBackend(
        source_root=MAIN / "prtreid-upstream",
        checkpoint=MAIN / "models/prtreid/prtreid-soccernet-baseline.pth.tar",
        device="cpu", feature_kind="global",
    )

    def encode(frame_rgb, boxes):
        crops = [np.ascontiguousarray(crop(frame_rgb, b)) for b in boxes]
        feats = backend.encode_images(crops, batch_size=32)
        return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)

    return encode


def features(clip: Clip, arm: str, kind: str) -> list[np.ndarray]:
    """-> per-frame (N_f, D) unit features aligned to that frame's detections."""
    path = OUT / "features" / f"{clip.name}_{arm}_{kind}.npz"
    if path.exists():
        with np.load(path) as data:
            feats, offsets = data["feats"], data["offsets"]
        return [feats[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    if kind == "colour":
        encode = lambda frame, boxes: (
            np.stack([colour_feature(frame, b) for b in boxes]) if len(boxes)
            else np.zeros((0, 128), np.float32))
    elif kind == "prtreid":
        encode = prtreid_encoder()
    else:
        raise ValueError(kind)
    detections_for = clip.detections_for(arm)
    per_frame = []
    for index, frame in enumerate(clip.frames_rgb()):
        boxes = detections_for(index).xyxy
        per_frame.append(encode(frame, boxes) if len(boxes) else None)
    dim = next(f.shape[1] for f in per_frame if f is not None)
    per_frame = [f if f is not None else np.zeros((0, dim), np.float32) for f in per_frame]
    offsets = np.cumsum([0] + [len(f) for f in per_frame])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feats=np.concatenate(per_frame), offsets=offsets)
    return per_frame
