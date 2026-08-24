"""Tracker-independent data contract for handball team classification.

The unit of annotation is an RF-DETR detection in one source frame.  Track
IDs are deliberately absent: they may be attached later as noisy temporal
evidence, but can never define the ground-truth team label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from team_model import crop_quality, torso_boxes, torso_contamination


DATASET_SCHEMA_VERSION = 1
ANNOTATION_CODES = frozenset({"A", "B", "G", "R", "O", "M", "X"})
TEAM_LABELS = frozenset({"A", "B"})
ROLE_LABELS = frozenset({"field", "goalkeeper", "referee", "other", "unknown"})
QUALITY_LABELS = frozenset({"clean", "occluded", "mixed", "unusable"})

CODE_TO_ANNOTATION = {
    "A": {"team": "A", "role": "field", "quality": "clean"},
    "B": {"team": "B", "role": "field", "quality": "clean"},
    "G": {"team": None, "role": "goalkeeper", "quality": "clean"},
    "R": {"team": None, "role": "referee", "quality": "clean"},
    "O": {"team": None, "role": "other", "quality": "clean"},
    "M": {"team": None, "role": "unknown", "quality": "mixed"},
    "X": {"team": None, "role": "unknown", "quality": "unusable"},
}


def empty_annotation() -> dict:
    return {"code": None, "team": None, "role": None, "quality": None}


def annotation_from_code(code: str, quality: str | None = None) -> dict:
    """Expand a single-key label into explicit team/role/quality fields."""
    normalized = code.upper()
    if normalized not in ANNOTATION_CODES:
        raise ValueError(
            f"unknown annotation code {code!r}; expected one of "
            f"{sorted(ANNOTATION_CODES)}"
        )
    annotation = {"code": normalized, **CODE_TO_ANNOTATION[normalized]}
    if quality is not None:
        if quality not in QUALITY_LABELS:
            raise ValueError(f"unknown quality label: {quality!r}")
        annotation["quality"] = quality
    return annotation


def validate_annotation(annotation: dict) -> None:
    code = annotation.get("code")
    if code is None:
        if any(annotation.get(key) is not None for key in ("team", "role", "quality")):
            raise ValueError("an unlabeled annotation cannot contain derived labels")
        return
    if code not in ANNOTATION_CODES:
        raise ValueError(f"unknown annotation code: {code!r}")
    if annotation.get("team") not in TEAM_LABELS | {None}:
        raise ValueError(f"unknown team label: {annotation.get('team')!r}")
    if annotation.get("role") not in ROLE_LABELS:
        raise ValueError(f"unknown role label: {annotation.get('role')!r}")
    if annotation.get("quality") not in QUALITY_LABELS:
        raise ValueError(f"unknown quality label: {annotation.get('quality')!r}")
    if annotation["team"] is not None and annotation["role"] != "field":
        raise ValueError("only field-player annotations may carry team A/B")


def load_detection_cache(path: Path | str) -> dict[str, np.ndarray | str | int]:
    """Load and strictly validate the flattened RF-DETR detection cache."""
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as raw:
        required = {"offsets", "boxes", "confidence", "class_id"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"{cache_path} is missing cache arrays: {sorted(missing)}")
        cache = {key: raw[key].copy() for key in required}
        cache["schema"] = int(raw["schema"]) if "schema" in raw.files else 0
        cache["detector_id"] = (
            str(raw["detector_id"]) if "detector_id" in raw.files else "unknown"
        )
        cache["total_frames"] = (
            int(raw["total_frames"])
            if "total_frames" in raw.files
            else len(cache["offsets"]) - 1
        )

    offsets = np.asarray(cache["offsets"])
    boxes = np.asarray(cache["boxes"])
    confidence = np.asarray(cache["confidence"])
    class_id = np.asarray(cache["class_id"])
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("cache offsets must be a one-dimensional frame index")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("cache offsets must start at zero and be monotonic")
    if int(offsets[-1]) != len(boxes):
        raise ValueError("cache offsets do not cover the box array")
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"expected boxes with shape (N, 4), got {boxes.shape}")
    if len(confidence) != len(boxes) or len(class_id) != len(boxes):
        raise ValueError("box, confidence, and class arrays have different lengths")
    if int(cache["total_frames"]) + 1 != len(offsets):
        raise ValueError("cache total_frames disagrees with offsets")
    return cache


def frame_arrays(
    cache: dict[str, np.ndarray | str | int], frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes, confidence, classes and stable flat indices for a frame."""
    total_frames = int(cache["total_frames"])
    if frame_index < 0 or frame_index >= total_frames:
        raise IndexError(f"frame {frame_index} outside [0, {total_frames})")
    offsets = np.asarray(cache["offsets"])
    start, end = (int(value) for value in offsets[frame_index:frame_index + 2])
    return (
        np.asarray(cache["boxes"])[start:end].copy(),
        np.asarray(cache["confidence"])[start:end].copy(),
        np.asarray(cache["class_id"])[start:end].astype(int, copy=True),
        np.arange(start, end, dtype=np.int64),
    )


def evenly_spaced_detection_frames(
    cache: dict[str, np.ndarray | str | int], count: int,
) -> list[int]:
    """Select deterministic, evenly distributed frames containing detections."""
    if count <= 0:
        raise ValueError("frame count must be positive")
    offsets = np.asarray(cache["offsets"])
    available = np.flatnonzero(np.diff(offsets) > 0)
    if not len(available):
        return []
    take = min(count, len(available))
    # Floor makes the tie behavior explicit (NumPy's rint uses banker's
    # rounding, which otherwise shifts a midpoint to a later frame).
    positions = np.floor(np.linspace(0, len(available) - 1, take)).astype(int)
    return available[np.unique(positions)].astype(int).tolist()


def sample_id(video_id: str, frame_index: int, detection_index: int) -> str:
    safe_video_id = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in video_id
    ).strip("-")
    return f"{safe_video_id}-f{frame_index:06d}-d{detection_index:03d}"


def build_sample_records(
    video_path: Path | str,
    cache: dict[str, np.ndarray | str | int],
    frame_indices: Iterable[int],
    class_ids: set[int] | None = None,
    max_per_frame: int | None = None,
) -> list[dict]:
    """Build detector-only sample metadata without decoding the video."""
    video = Path(video_path).resolve()
    records: list[dict] = []
    for frame_index in sorted(set(int(value) for value in frame_indices)):
        boxes, confidence, detected_classes, flat_indices = frame_arrays(
            cache, frame_index
        )
        if not len(boxes):
            continue
        contamination = torso_contamination(boxes)
        eligible = np.arange(len(boxes))
        if class_ids is not None:
            eligible = eligible[np.isin(detected_classes[eligible], list(class_ids))]
        # If a crowded frame must be truncated, retain large/confident crops and
        # a few contaminated cases rather than relying on detector ordering.
        if max_per_frame is not None and len(eligible) > max_per_frame:
            widths = boxes[eligible, 2] - boxes[eligible, 0]
            heights = boxes[eligible, 3] - boxes[eligible, 1]
            importance = (
                np.log1p(np.maximum(widths * heights, 0.0))
                + confidence[eligible]
                + np.minimum(contamination[eligible], 1.0)
            )
            eligible = eligible[np.argsort(importance)[-max_per_frame:]]
        for detection_index in sorted(eligible.astype(int).tolist()):
            box = boxes[detection_index].astype(float)
            width, height = box[2] - box[0], box[3] - box[1]
            records.append({
                "sample_id": sample_id(video.stem, frame_index, detection_index),
                "video_id": video.stem,
                "video_path": str(video),
                "frame_index": frame_index,
                "detection_index": detection_index,
                "flat_detection_index": int(flat_indices[detection_index]),
                "bbox_xyxy": box.tolist(),
                "bbox_width": float(width),
                "bbox_height": float(height),
                "detector_confidence": float(confidence[detection_index]),
                "detector_class_id": int(detected_classes[detection_index]),
                "torso_contamination": float(contamination[detection_index]),
                "crop_path": None,
                "torso_path": None,
                "preview_path": None,
                "crop_quality": None,
                "annotation": empty_annotation(),
            })
    return records


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(box, dtype=float).copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height)
    return clipped


def _expand_box(
    box: np.ndarray, scale_x: float, scale_y: float, width: int, height: int,
) -> np.ndarray:
    center = (box[:2] + box[2:]) / 2.0
    size = (box[2:] - box[:2]) * np.array([scale_x, scale_y])
    return _clip_box(np.r_[center - size / 2.0, center + size / 2.0], width, height)


def _crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.rint(box).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image.size == 0:
        return canvas
    source_h, source_w = image.shape[:2]
    scale = min(width / max(source_w, 1), height / max(source_h, 1))
    target_w = max(1, round(source_w * scale))
    target_h = max(1, round(source_h * scale))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    x0, y0 = (width - target_w) // 2, (height - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = resized
    return canvas


def _preview(
    frame_bgr: np.ndarray, box: np.ndarray, person: np.ndarray, torso: np.ndarray,
    record: dict,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    context_box = _expand_box(box, 3.0, 2.0, width, height)
    context = _crop(frame_bgr, context_box).copy()
    if context.size:
        local = box.copy()
        local[[0, 2]] -= context_box[0]
        local[[1, 3]] -= context_box[1]
        x1, y1, x2, y2 = np.rint(local).astype(int)
        cv2.rectangle(context, (x1, y1), (x2, y2), (0, 220, 255), 3)
    panel_h = 360
    composite = np.hstack([
        _fit_panel(context, 560, panel_h),
        _fit_panel(person, 210, panel_h),
        _fit_panel(torso, 210, panel_h),
    ])
    cv2.rectangle(composite, (0, 0), (composite.shape[1], 43), (12, 12, 12), -1)
    caption = (
        f"{record['video_id']}  frame {record['frame_index']}  "
        f"det {record['detection_index']}  conf {record['detector_confidence']:.2f}  "
        f"overlap {record['torso_contamination']:.2f}"
    )
    cv2.putText(
        composite, caption, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (245, 245, 245), 1, cv2.LINE_AA,
    )
    return composite


def render_sample_assets(
    video_path: Path | str, records: list[dict], output_dir: Path | str,
) -> None:
    """Decode selected frames once and write full-person crops plus previews."""
    if not records:
        return
    video = Path(video_path).resolve()
    output = Path(output_dir)
    crop_dir = output / "crops"
    torso_dir = output / "torsos"
    preview_dir = output / "previews"
    crop_dir.mkdir(parents=True, exist_ok=True)
    torso_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict]] = {}
    for record in records:
        by_frame.setdefault(int(record["frame_index"]), []).append(record)
    final_frame = max(by_frame)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    try:
        frame_index = 0
        while frame_index <= final_frame:
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"video ended before requested frame {final_frame}: {video}"
                )
            for record in by_frame.get(frame_index, []):
                frame_h, frame_w = frame_bgr.shape[:2]
                box = _clip_box(np.asarray(record["bbox_xyxy"]), frame_w, frame_h)
                person_box = _expand_box(box, 1.10, 1.06, frame_w, frame_h)
                person = _crop(frame_bgr, person_box)
                torso_box = _clip_box(torso_boxes(box[None])[0], frame_w, frame_h)
                torso = _crop(frame_bgr, torso_box)
                crop_path = crop_dir / f"{record['sample_id']}.jpg"
                torso_path = torso_dir / f"{record['sample_id']}.jpg"
                preview_path = preview_dir / f"{record['sample_id']}.jpg"
                if person.size == 0 or torso.size == 0:
                    raise RuntimeError(f"empty crop for {record['sample_id']}")
                if not cv2.imwrite(str(crop_path), person):
                    raise RuntimeError(f"could not write {crop_path}")
                if not cv2.imwrite(str(torso_path), torso):
                    raise RuntimeError(f"could not write {torso_path}")
                preview = _preview(frame_bgr, box, person, torso, record)
                if not cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError(f"could not write {preview_path}")
                torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
                quality = crop_quality(torso_rgb)
                record["crop_path"] = crop_path.relative_to(output).as_posix()
                record["torso_path"] = torso_path.relative_to(output).as_posix()
                record["preview_path"] = preview_path.relative_to(output).as_posix()
                record["crop_quality"] = {
                    "accepted": bool(quality.accepted),
                    "score": float(quality.score),
                    "width": int(quality.width),
                    "height": int(quality.height),
                    "contrast": float(quality.contrast),
                    "sharpness": float(quality.sharpness),
                }
            frame_index += 1
    finally:
        capture.release()


def new_manifest(
    video_path: Path | str,
    cache_path: Path | str,
    cache: dict[str, np.ndarray | str | int],
    samples: list[dict],
) -> dict:
    video = Path(video_path).resolve()
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "task": "tracker_independent_team_classification",
        "video_id": video.stem,
        "video_path": str(video),
        "video_size": video.stat().st_size,
        "video_mtime_ns": video.stat().st_mtime_ns,
        "detection_cache": str(Path(cache_path).resolve()),
        "detector_id": str(cache["detector_id"]),
        "annotation_guide": {
            "A": "field player, anonymous team A within this video",
            "B": "field player, anonymous team B within this video",
            "G": "goalkeeper",
            "R": "referee or official",
            "O": "other person",
            "M": "mixed crop; target jersey is contaminated by another person",
            "X": "unusable or target absent",
        },
        "samples": samples,
    }


def validate_manifest(manifest: dict, require_complete: bool = False) -> None:
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported dataset schema: {manifest.get('schema_version')!r}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    identifiers = set()
    for sample in samples:
        identifier = sample.get("sample_id")
        if not identifier or identifier in identifiers:
            raise ValueError(f"missing or duplicate sample id: {identifier!r}")
        identifiers.add(identifier)
        annotation = sample.get("annotation", empty_annotation())
        validate_annotation(annotation)
        if require_complete and annotation.get("code") is None:
            raise ValueError(f"sample {identifier} is not labeled")


def read_manifest(path: Path | str, require_complete: bool = False) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest, require_complete=require_complete)
    return manifest


def write_manifest(path: Path | str, manifest: dict) -> None:
    validate_manifest(manifest)
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")


def annotation_counts(manifest: dict) -> dict[str, int]:
    counts = {code: 0 for code in sorted(ANNOTATION_CODES)}
    counts["unlabeled"] = 0
    for sample in manifest["samples"]:
        code = sample.get("annotation", {}).get("code")
        counts[code if code is not None else "unlabeled"] += 1
    return counts
