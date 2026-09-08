"""Render raw per-detection team predictions with no tracker or temporal state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from handball_cv.teams.model import (
    MAX_TORSO_CONTAMINATION,
    MIN_TEAM_VOTE_CONFIDENCE,
    TeamModel,
    torso_boxes,
    torso_contamination,
)

FIELD_PLAYER_CLASS_ID = 2
GOALKEEPER_CLASS_ID = 1
TEAM_BGR = {0: (255, 190, 0), 1: (0, 120, 255)}
UNCERTAIN_BGR = (0, 220, 255)
REJECT_BGR = (40, 40, 255)
GOALKEEPER_BGR = (180, 180, 180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    since the caches gained a jersey-number class -- the number boxes too: on the
    60s Melsungen window, 16689 of 36686 cached detections are numbers, so 45% of
    what the tracker was asked to follow were not people at all.
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


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    font_scale = 0.50 * scale
    thickness = max(1, round(1.5 * scale))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - height - 7),
        (x + width + 8, y + baseline + 3),
        (16, 16, 16),
        -1,
    )
    cv2.rectangle(
        frame,
        (x, y - height - 7),
        (x + width + 8, y + baseline + 3),
        color,
        max(1, round(scale)),
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_header(
    frame: np.ndarray,
    source: Path,
    frame_index: int,
    fps: float,
    counts: dict[str, int],
    agreement: float,
) -> None:
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    header_height = min(height, round(86 * scale))
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, header_height), (8, 12, 18), -1)
    cv2.addWeighted(shade, 0.84, frame, 0.16, 0, frame)
    thickness = max(1, round(1.5 * scale))
    cv2.putText(
        frame,
        f"{source.name}   {frame_index / fps:05.2f}s",
        (round(18 * scale), round(31 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72 * scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"RAW PER-DETECTION TEAM CLASSIFICATION | NO TRACKER | "
        f"NO TEMPORAL VOTING | fit agreement {agreement:.2f}",
        (round(18 * scale), round(64 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47 * scale,
        (190, 205, 215),
        max(1, round(scale)),
        cv2.LINE_AA,
    )
    summary = (
        f"A {counts['a']}   B {counts['b']}   "
        f"uncertain {counts['uncertain']}   rejected {counts['rejected']}   "
        f"GK {counts['gk']}"
    )
    cv2.putText(
        frame,
        summary,
        (max(round(18 * scale), width - round(660 * scale)), round(31 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47 * scale,
        (235, 235, 235),
        max(1, round(scale)),
        cv2.LINE_AA,
    )


def render(
    source: Path,
    cache_path: Path,
    model_path: Path,
    output_path: Path,
    device: str,
) -> dict:
    source = source.resolve()
    cache = load_detection_cache(cache_path)
    model = TeamModel.load(model_path, device=device)
    info = sv.VideoInfo.from_video_path(str(source))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_name(f"{output_path.stem}_preview.jpg")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")

    totals = {"a": 0, "b": 0, "uncertain": 0, "rejected": 0, "gk": 0}
    frames_written = 0
    preview = None
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=info.total_frames,
        desc=f"raw team {source.stem}",
    )):
        detections = frame_detections(cache, frame_index)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated = frame_bgr.copy()
        counts = {key: 0 for key in totals}
        scale = max(info.width / 1920.0, 1.0)
        line = max(2, round(3 * scale))

        field_indices = np.flatnonzero(
            detections.class_id == FIELD_PLAYER_CLASS_ID
        )
        teams = np.empty(0, dtype=int)
        confidence = np.empty(0, dtype=float)
        quality = np.empty(0, dtype=float)
        overlap = np.empty(0, dtype=float)
        if len(field_indices):
            field_boxes = detections.xyxy[field_indices]
            _embeddings, teams, confidence, quality = model.observe(
                frame_rgb, field_boxes, detections.xyxy
            )
            overlap = torso_contamination(field_boxes, detections.xyxy)

        field_slot = {int(index): slot for slot, index in enumerate(field_indices)}
        all_torsos = torso_boxes(detections.xyxy)
        for index, (box, torso, class_id) in enumerate(zip(
            detections.xyxy, all_torsos, detections.class_id
        )):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            tx1, ty1, tx2, ty2 = np.rint(torso).astype(int)
            if class_id == GOALKEEPER_CLASS_ID:
                color = GOALKEEPER_BGR
                label = "DETECTOR: GOALKEEPER"
                counts["gk"] += 1
            else:
                slot = field_slot[index]
                team = int(teams[slot])
                raw_confidence = float(confidence[slot])
                crop_quality = float(quality[slot])
                contamination = float(overlap[slot])
                rejected = crop_quality <= 0.0
                uncertain = raw_confidence < MIN_TEAM_VOTE_CONFIDENCE
                if rejected:
                    color = REJECT_BGR
                    label = (
                        f"RAW {'A' if team == 0 else 'B'} | REJECTED | "
                        f"c={raw_confidence:.2f} q={crop_quality:.2f} "
                        f"ov={contamination:.0%}"
                    )
                    counts["rejected"] += 1
                elif uncertain:
                    color = UNCERTAIN_BGR
                    label = (
                        f"RAW {'A' if team == 0 else 'B'} | UNCERTAIN | "
                        f"c={raw_confidence:.2f} q={crop_quality:.2f}"
                    )
                    counts["uncertain"] += 1
                else:
                    color = TEAM_BGR[team]
                    label = (
                        f"RAW {'A' if team == 0 else 'B'} | "
                        f"c={raw_confidence:.2f} q={crop_quality:.2f}"
                    )
                    counts["a" if team == 0 else "b"] += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line)
            cv2.rectangle(
                annotated,
                (tx1, ty1),
                (tx2, ty2),
                color,
                max(1, round(1.5 * scale)),
            )
            draw_text(
                annotated,
                label,
                (max(0, x1), max(round(105 * scale), y1 - round(5 * scale))),
                color,
                scale,
            )

        draw_header(
            annotated,
            source,
            frame_index,
            info.fps,
            counts,
            model.visual_color_agreement,
        )
        writer.write(annotated)
        frames_written += 1
        for key in totals:
            totals[key] += counts[key]
        if frame_index == info.total_frames // 2:
            preview = annotated.copy()

    writer.release()
    if preview is not None:
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(
                preview,
                (1600, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)

    result = {
        "source": str(source),
        "output": str(output_path),
        "preview": str(preview_path),
        "frames": frames_written,
        "visual_color_fit_agreement": model.visual_color_agreement,
        "detections": totals,
        "tracker": None,
        "temporal_voting": False,
    }
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    result = render(
        args.video,
        args.detections,
        args.team_model,
        args.output,
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
