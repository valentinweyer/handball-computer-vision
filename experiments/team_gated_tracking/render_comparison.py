"""Render tracker-independent, per-video team-classification diagnostics.

The detector is run once and cached. Each video gets its own anonymous two-team
fit from clean field-player torso crops. ByteTrack is used only to aggregate
confidence across a short tracklet and make the overlay readable; the team
features themselves do not depend on the tracker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model
from tqdm import tqdm

from handball_cv.tracking.identity import IdentityManager
from handball_cv.detection.rfdetr_onnx import OfflineRFDETR
from experiments.team_gated_tracking.tracker import (
    TEAM_PROBABILITY_KEY,
    TEAM_QUALITY_KEY,
    TeamGatedByteTrackTracker,
)
from handball_cv.teams.calibration import (
    ROLE_REFEREE,
    CalibratedTeamObserver,
    PrototypeTeamCalibrator,
)
from handball_cv.teams.model import (
    MAX_TORSO_CONTAMINATION,
    MIN_STABLE_TEAM_CONFIDENCE,
    TeamModel,
    TeamModelCacheStale,
    crop_quality,
    torso_boxes,
    torso_contamination,
)

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_ID = "player-and-handball-detection-3z9xf/3"
DETECTION_CACHE_SCHEMA = 1
GOALKEEPER_CLASS_ID = 1
FIELD_PLAYER_CLASS_ID = 2
TEAM_BGR = {0: (255, 190, 0), 1: (0, 120, 255)}
TEAM_NAMES = {0: "TEAM A", 1: "TEAM B"}
GK_BGR = (180, 180, 180)
REJECT_BGR = (40, 40, 255)
UNCERTAIN_BGR = (0, 220, 255)
EXCLUDED_BGR = (220, 80, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "team_comparison",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.90)
    parser.add_argument("--max-fit-crops", type=int, default=1200)
    parser.add_argument(
        "--offline-weights", type=Path,
        help="Cached RF-DETR ONNX weights; avoids all remote model resolution.",
    )
    parser.add_argument(
        "--calibration-dir", type=Path,
        default=ROOT / "outputs" / "team_comparison" / "calibration",
    )
    return parser.parse_args()


def detect_players(model, frame_rgb: np.ndarray, confidence: float, iou: float):
    if isinstance(model, OfflineRFDETR):
        boxes, scores, class_ids = model.infer_detections(
            frame_rgb, confidence=confidence
        )
        detections = sv.Detections(
            xyxy=boxes, confidence=scores, class_id=class_ids
        )
        return detections[np.isin(
            detections.class_id, [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]
        )]
    result = model.infer(
        frame_rgb, confidence=confidence, iou_threshold=iou
    )[0]
    detections = sv.Detections.from_inference(result)
    if detections.class_id is None:
        return sv.Detections.empty()
    return detections[np.isin(
        detections.class_id, [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]
    )]


def cache_is_current(path: Path, source: Path, total_frames: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["schema"]) == DETECTION_CACHE_SCHEMA
                and int(data["source_size"]) == source.stat().st_size
                and int(data["source_mtime_ns"]) == source.stat().st_mtime_ns
                and int(data["total_frames"]) == total_frames
                and str(data["detector_id"]) == DETECTOR_ID
            )
    except Exception:
        return False


def build_detection_cache(
    source: Path, cache_path: Path, model, confidence: float, iou: float,
):
    info = sv.VideoInfo.from_video_path(str(source))
    if cache_is_current(cache_path, source, info.total_frames):
        print(f"detections: reuse {cache_path}")
        return

    all_boxes, all_confidence, all_classes = [], [], []
    offsets = [0]
    frames = sv.get_video_frames_generator(str(source))
    for frame_bgr in tqdm(frames, total=info.total_frames, desc=f"detect {source.stem}"):
        detections = detect_players(
            model, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), confidence, iou
        )
        all_boxes.append(np.asarray(detections.xyxy, dtype=np.float32))
        det_confidence = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        all_confidence.append(np.asarray(det_confidence, dtype=np.float32))
        all_classes.append(np.asarray(detections.class_id, dtype=np.int16))
        offsets.append(offsets[-1] + len(detections))

    boxes = np.concatenate(all_boxes) if offsets[-1] else np.empty((0, 4), np.float32)
    scores = np.concatenate(all_confidence) if offsets[-1] else np.empty(0, np.float32)
    classes = np.concatenate(all_classes) if offsets[-1] else np.empty(0, np.int16)
    np.savez_compressed(
        cache_path,
        schema=np.array(DETECTION_CACHE_SCHEMA),
        detector_id=np.array(DETECTOR_ID),
        source_size=np.array(source.stat().st_size),
        source_mtime_ns=np.array(source.stat().st_mtime_ns),
        total_frames=np.array(info.total_frames),
        offsets=np.asarray(offsets, dtype=np.int64),
        boxes=boxes,
        confidence=scores,
        class_id=classes,
    )
    print(f"detections: wrote {cache_path} ({len(boxes)} player boxes)")


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


def collect_fit_crops(
    source: Path, cache: dict[str, np.ndarray], stride: int, max_crops: int,
) -> list[np.ndarray]:
    crops = []
    frames = sv.get_video_frames_generator(str(source))
    for frame_index, frame_bgr in enumerate(frames):
        if frame_index % stride:
            continue
        detections = frame_detections(cache, frame_index)
        if len(detections) == 0:
            continue
        contamination = torso_contamination(detections.xyxy)
        field_indices = np.flatnonzero(detections.class_id == FIELD_PLAYER_CLASS_ID)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for index in field_indices:
            if contamination[index] > MAX_TORSO_CONTAMINATION:
                continue
            crop = sv.crop_image(
                frame_rgb, torso_boxes(detections.xyxy[index:index + 1])[0]
            )
            if crop_quality(crop).accepted:
                crops.append(crop)
            if len(crops) >= max_crops:
                return crops
    return crops


def load_or_fit_team_model(
    source: Path, cache: dict[str, np.ndarray], model_path: Path,
    device: str, max_crops: int,
) -> tuple[TeamModel, int]:
    if model_path.exists():
        try:
            return TeamModel.load(model_path, device=device), -1
        except TeamModelCacheStale as error:
            print(f"team cache stale: {error}")
    info = sv.VideoInfo.from_video_path(str(source))
    stride = max(1, round(info.fps / 5.0))  # sample at about 5 Hz
    crops = collect_fit_crops(source, cache, stride, max_crops)
    print(f"team fit {source.stem}: {len(crops)} clean field-player crops")
    team_model = TeamModel.fit_from_crops(crops, device=device)
    team_model.save(model_path)
    return team_model, len(crops)


def text_box(
    image: np.ndarray, text: str, origin: tuple[int, int],
    color: tuple[int, int, int], scale: float,
) -> None:
    font_scale = 0.52 * scale
    thickness = max(1, round(1.5 * scale))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        image, (x, y - height - 7), (x + width + 8, y + baseline + 3),
        (20, 20, 20), -1,
    )
    cv2.rectangle(
        image, (x, y - height - 7), (x + width + 8, y + baseline + 3),
        color, max(1, round(scale)),
    )
    cv2.putText(
        image, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA,
    )


def draw_header(
    image: np.ndarray, source: Path, frame_index: int, fps: float,
    counts: dict[str, int], agreement: float, calibrated: bool = False,
) -> None:
    height, width = image.shape[:2]
    scale = max(width / 1920.0, 1.0)
    header_h = min(height, round(86 * scale))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_h), (8, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(1.6 * scale))
    cv2.putText(
        image, f"{source.name}   {frame_index / fps:05.2f}s",
        (round(18 * scale), round(31 * scale)), font, 0.72 * scale,
        (245, 245, 245), thick, cv2.LINE_AA,
    )
    mode = (
        "clicked prototypes + team-gated association"
        if calibrated else "unsupervised jersey color"
    )
    details = (
        f"Per-video {mode} | short-tracklet evidence | "
        f"appearance agreement {agreement:.2f}"
    )
    cv2.putText(
        image, details, (round(18 * scale), round(64 * scale)), font,
        0.47 * scale, (190, 200, 210), max(1, round(scale)), cv2.LINE_AA,
    )
    legend = [
        (TEAM_BGR[0], f"A {counts['a']}"),
        (TEAM_BGR[1], f"B {counts['b']}"),
        (UNCERTAIN_BGR, f"uncertain {counts['uncertain']}"),
        (REJECT_BGR, f"rejected/overlap {counts['rejected']}"),
        (EXCLUDED_BGR, f"excluded {counts['excluded']}"),
        (GK_BGR, f"GK {counts['gk']}"),
    ]
    x = width - round(805 * scale)
    y = round(31 * scale)
    for color, label in legend:
        cv2.rectangle(
            image, (x, y - round(13 * scale)),
            (x + round(18 * scale), y + round(3 * scale)), color, -1,
        )
        cv2.putText(
            image, label, (x + round(25 * scale), y), font,
            0.43 * scale, (235, 235, 235), max(1, round(scale)), cv2.LINE_AA,
        )
        x += round((65 + len(label) * 7) * scale)


def render_video(
    source: Path, cache: dict[str, np.ndarray], team_model: TeamModel,
    output_path: Path, preview_path: Path,
    calibrator: PrototypeTeamCalibrator | None = None,
) -> dict:
    info = sv.VideoInfo.from_video_path(str(source))
    tracker = (
        TeamGatedByteTrackTracker(frame_rate=info.fps)
        if calibrator is not None
        else sv.ByteTrack(frame_rate=info.fps)
    )
    observer = (
        CalibratedTeamObserver(team_model, calibrator)
        if calibrator is not None else team_model
    )
    identity = IdentityManager(
        observer, goalkeeper_class_id=GOALKEEPER_CLASS_ID
    )
    preview_index = info.total_frames // 2
    preview = None
    excluded_total = 0

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")

    frames = sv.get_video_frames_generator(str(source))
    for frame_index, frame_bgr in enumerate(tqdm(
        frames, total=info.total_frames, desc=f"render {source.stem}"
    )):
        all_detections = frame_detections(cache, frame_index)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        excluded = sv.Detections.empty()
        if calibrator is not None and len(all_detections):
            (
                probability_b, _certainty, role, role_confidence,
                cheap_quality,
            ) = calibrator.predict_frame(
                frame_rgb, all_detections.xyxy, all_detections.xyxy
            )
            is_goalkeeper = (
                all_detections.class_id == GOALKEEPER_CLASS_ID
            )
            probability_b = probability_b.astype(float)
            cheap_quality = cheap_quality.astype(float)
            probability_b[is_goalkeeper] = np.nan
            cheap_quality[is_goalkeeper] = 0.0
            all_detections.data[TEAM_PROBABILITY_KEY] = probability_b
            all_detections.data[TEAM_QUALITY_KEY] = cheap_quality
            exclude_mask = (
                (all_detections.class_id == FIELD_PLAYER_CLASS_ID)
                & (role == ROLE_REFEREE)
                & (role_confidence >= 0.15)
                & (cheap_quality > 0)
            )
            excluded = all_detections[exclude_mask]
            detections = all_detections[~exclude_mask]
            tracked = tracker.update(detections)
        else:
            detections = all_detections
            tracked = (
                tracker.update(detections)
                if calibrator is not None
                else tracker.update_with_detections(detections)
            )
        tracked = tracked[tracked.tracker_id >= 0]
        excluded_total += len(excluded)
        logical_frame = int(round(frame_index * 25.0 / info.fps))
        player_ids = identity.update(logical_frame, frame_rgb, tracked)

        if len(tracked):
            torso = torso_boxes(tracked.xyxy)
            contamination = torso_contamination(
                tracked.xyxy, all_detections.xyxy
            )
            qualities = []
            for box, overlap in zip(torso, contamination):
                crop = sv.crop_image(frame_rgb, box)
                quality = crop_quality(crop).score
                quality *= float(np.clip(
                    1.0 - overlap / MAX_TORSO_CONTAMINATION, 0.0, 1.0
                ))
                qualities.append(quality)
            qualities = np.asarray(qualities)
        else:
            torso = np.empty((0, 4))
            contamination = np.empty(0)
            qualities = np.empty(0)

        counts = {
            "a": 0, "b": 0, "uncertain": 0, "rejected": 0,
            "excluded": len(excluded), "gk": 0,
        }
        annotated = frame_bgr.copy()
        scale = max(info.width / 1920.0, 1.0)
        line = max(2, round(3 * scale))
        for box in excluded.xyxy:
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), EXCLUDED_BGR, line
            )
            text_box(
                annotated, "EXCLUDED ROLE",
                (max(0, x1), max(round(105 * scale), y1 - 5)),
                EXCLUDED_BGR, scale,
            )
        for i, (box, torso_box, player_id) in enumerate(
            zip(tracked.xyxy, torso, player_ids)
        ):
            player = identity.players[int(player_id)]
            is_gk = player.is_goalkeeper
            team = player.voted_team_id
            stable = player.team_confidence >= MIN_STABLE_TEAM_CONFIDENCE
            rejected = qualities[i] <= 0.0
            if is_gk:
                color = GK_BGR
                label = f"P{player_id} | GK"
                counts["gk"] += 1
            else:
                color = TEAM_BGR[team] if stable else UNCERTAIN_BGR
                label = (
                    f"P{player_id} | {TEAM_NAMES[team]} | "
                    f"{player.team_confidence:.0%} | n={player.team_observations}"
                )
                counts["a" if team == 0 else "b"] += 1
                if not stable:
                    counts["uncertain"] += 1
            if rejected:
                counts["rejected"] += 1

            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line)
            tx1, ty1, tx2, ty2 = np.rint(torso_box).astype(int)
            torso_color = REJECT_BGR if rejected else color
            cv2.rectangle(
                annotated, (tx1, ty1), (tx2, ty2), torso_color,
                max(1, round(1.5 * scale)),
            )
            if rejected:
                label += f" | REJECT q={qualities[i]:.2f} ov={contamination[i]:.0%}"
            else:
                label += f" | q={qualities[i]:.2f}"
            text_box(
                annotated, label,
                (max(0, x1), max(round(105 * scale), y1 - round(5 * scale))),
                torso_color, scale,
            )

        draw_header(
            annotated, source, frame_index, info.fps, counts,
            team_model.visual_color_agreement,
            calibrated=calibrator is not None,
        )
        writer.write(annotated)
        if frame_index == preview_index:
            preview = annotated.copy()

    writer.release()
    if preview is not None:
        max_width = 1600
        if preview.shape[1] > max_width:
            ratio = max_width / preview.shape[1]
            preview = cv2.resize(
                preview, (max_width, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)

    players = list(identity.players.values()) + identity.retired
    return {
        "source": str(source),
        "output": str(output_path),
        "preview": str(preview_path),
        "tracks": len(players),
        "stable_field_tracks": sum(
            (not player.is_goalkeeper)
            and player.team_confidence >= MIN_STABLE_TEAM_CONFIDENCE
            for player in players
        ),
        "uncertain_field_tracks": sum(
            (not player.is_goalkeeper)
            and player.team_confidence < MIN_STABLE_TEAM_CONFIDENCE
            for player in players
        ),
        "goalkeeper_tracks": sum(player.is_goalkeeper for player in players),
        "visual_color_fit_agreement": team_model.visual_color_agreement,
        "excluded_role_detections": excluded_total,
        "team_gate_rejections": getattr(
            tracker, "team_gate_rejections", 0
        ),
    }


def make_comparison_preview(results: list[dict], target: Path) -> None:
    images = [cv2.imread(result["preview"]) for result in results]
    images = [image for image in images if image is not None]
    if not images:
        return
    target_height = 540
    resized = [
        cv2.resize(
            image,
            (round(image.shape[1] * target_height / image.shape[0]), target_height),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    cv2.imwrite(str(target), cv2.hconcat(resized))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "notebooks" / ".env")
    os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")
    if args.offline_weights is not None:
        detector = OfflineRFDETR(args.offline_weights, device=args.device)
        print(f"detector: offline ONNX {args.offline_weights}")
    else:
        roboflow_key = os.getenv("ROBOFLOW_API_KEY")
        detector = get_model(model_id=DETECTOR_ID, api_key=roboflow_key)

    results = []
    for source_arg in args.videos:
        source = source_arg.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        stem = source.stem
        detection_path = args.output_dir / f".{stem}_detections_v1.npz"
        team_path = args.output_dir / f".{stem}_team.pkl"
        calibration_path = (
            args.calibration_dir / f".{stem}_prototype_team.pkl"
        )
        calibrator = (
            PrototypeTeamCalibrator.load(calibration_path)
            if calibration_path.exists() else None
        )
        suffix = "_calibrated" if calibrator is not None else ""
        output_path = (
            args.output_dir / f"{stem}_team_overlay{suffix}.mp4"
        )
        preview_path = args.output_dir / f"{stem}_preview{suffix}.jpg"
        build_detection_cache(
            source, detection_path, detector, args.confidence, args.iou
        )
        detection_cache = load_detection_cache(detection_path)
        team_model, fit_crops = load_or_fit_team_model(
            source, detection_cache, team_path, args.device, args.max_fit_crops
        )
        result = render_video(
            source, detection_cache, team_model, output_path, preview_path,
            calibrator=calibrator,
        )
        result["fit_crops"] = fit_crops
        results.append(result)
        print(json.dumps(result, indent=2))

    comparison_path = (
        args.output_dir / "comparison_preview_calibrated.jpg"
    )
    make_comparison_preview(results, comparison_path)
    summary_path = args.output_dir / "summary_calibrated.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"comparison preview: {comparison_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
