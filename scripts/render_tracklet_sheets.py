"""Render chronological ByteTrack tracklet crop sheets from cached detections."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from handball_cv.teams.model import (
    MAX_TORSO_CONTAMINATION,
    crop_quality,
    torso_boxes,
    torso_contamination,
)

ROOT = Path(__file__).resolve().parents[1]
CELL_W = 180
CELL_H = 205
LABEL_W = 310
HEADER_H = 76
ROW_H = 230
SAMPLES_PER_TRACK = 8
FIELD_BGR = (80, 220, 80)
GK_BGR = (190, 190, 190)
REJECT_BGR = (40, 40, 240)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "outputs" / "team_comparison",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "team_comparison" / "tracklets",
    )
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_TRACK)
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "offsets": data["offsets"],
            "boxes": data["boxes"],
            "confidence": data["confidence"],
            "class_id": data["class_id"],
        }


def detections_at(cache: dict[str, np.ndarray], frame_index: int) -> sv.Detections:
    start, end = cache["offsets"][frame_index:frame_index + 2]
    start, end = int(start), int(end)
    return sv.Detections(
        xyxy=cache["boxes"][start:end].copy(),
        confidence=cache["confidence"][start:end].copy(),
        class_id=cache["class_id"][start:end].astype(int, copy=True),
    )


def collect_tracklets(
    cache: dict[str, np.ndarray], fps: float, frame_count: int,
) -> dict[int, list[dict]]:
    tracker = sv.ByteTrack(frame_rate=fps)
    tracklets: dict[int, list[dict]] = defaultdict(list)
    for frame_index in tqdm(range(frame_count), desc="associate tracklets"):
        detections = detections_at(cache, frame_index)
        tracked = tracker.update_with_detections(detections)
        if not len(tracked):
            continue
        overlap = torso_contamination(tracked.xyxy, detections.xyxy)
        for box, class_id, track_id, contamination in zip(
            tracked.xyxy, tracked.class_id, tracked.tracker_id, overlap
        ):
            width, height = box[2] - box[0], box[3] - box[1]
            area = max(float(width * height), 0.0)
            score = np.log1p(area) * max(0.05, 1.0 - float(contamination))
            tracklets[int(track_id)].append({
                "frame": frame_index,
                "box": np.asarray(box, dtype=float),
                "class_id": int(class_id),
                "overlap": float(contamination),
                "selection_score": float(score),
            })
    return dict(tracklets)


def choose_samples(observations: list[dict], count: int) -> list[dict]:
    """Choose the cleanest item from each chronological bin."""
    selected = []
    for chunk in np.array_split(np.arange(len(observations)), count):
        if not len(chunk):
            continue
        best_index = max(
            (int(index) for index in chunk),
            key=lambda index: observations[index]["selection_score"],
        )
        selected.append(dict(observations[best_index]))
    return selected


def crop_with_torso(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = 0.05 * (x2 - x1)
    pad_y = 0.03 * (y2 - y1)
    display = np.array([
        max(0, x1 - pad_x), max(0, y1 - pad_y),
        min(width, x2 + pad_x), min(height, y2 + pad_y),
    ])
    dx1, dy1, dx2, dy2 = np.rint(display).astype(int)
    crop = frame[dy1:dy2, dx1:dx2].copy()
    if crop.size == 0:
        return crop
    torso = torso_boxes(box[None])[0]
    tx1, ty1, tx2, ty2 = np.rint(torso).astype(int)
    cv2.rectangle(
        crop, (tx1 - dx1, ty1 - dy1), (tx2 - dx1, ty2 - dy1),
        (255, 255, 255), 2,
    )
    return crop


def fit_cell(image: np.ndarray) -> np.ndarray:
    cell = np.full((CELL_H, CELL_W, 3), 24, dtype=np.uint8)
    if image.size == 0:
        return cell
    available_h = CELL_H - 24
    height, width = image.shape[:2]
    scale = min((CELL_W - 10) / width, (available_h - 8) / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (CELL_W - new_w) // 2
    y0 = (available_h - new_h) // 2
    cell[y0:y0 + new_h, x0:x0 + new_w] = resized
    return cell


def extract_selected_crops(
    video: Path, selected_by_track: dict[int, list[dict]],
) -> None:
    requests: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for track_id, samples in selected_by_track.items():
        for sample_index, sample in enumerate(samples):
            requests[sample["frame"]].append((track_id, sample_index))

    frames = sv.get_video_frames_generator(str(video))
    for frame_index, frame in enumerate(frames):
        for track_id, sample_index in requests.get(frame_index, []):
            sample = selected_by_track[track_id][sample_index]
            sample["crop"] = crop_with_torso(frame, sample["box"])
            torso = torso_boxes(sample["box"][None])[0]
            torso_rgb = sv.crop_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), torso
            )
            quality = crop_quality(torso_rgb)
            overlap_weight = float(np.clip(
                1.0 - sample["overlap"] / MAX_TORSO_CONTAMINATION,
                0.0, 1.0,
            ))
            sample["quality"] = quality.score * overlap_weight
            sample["accepted"] = bool(quality.accepted and overlap_weight > 0)


def majority_goalkeeper(observations: list[dict]) -> bool:
    classes = np.array([item["class_id"] for item in observations])
    return bool(np.mean(classes == 1) >= 0.5)


def render_sheet(
    video: Path, tracklets: dict[int, list[dict]], samples_per_track: int,
    output_path: Path, manifest_path: Path,
) -> None:
    info = sv.VideoInfo.from_video_path(str(video))
    ordered = sorted(
        tracklets.items(), key=lambda item: (item[1][0]["frame"], item[0])
    )
    selected = {
        track_id: choose_samples(observations, samples_per_track)
        for track_id, observations in ordered
    }
    extract_selected_crops(video, selected)

    width = LABEL_W + samples_per_track * CELL_W
    height = HEADER_H + len(ordered) * ROW_H
    sheet = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(
        sheet, f"{video.name}: ByteTrack tracklets ({len(ordered)})",
        (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
        (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "One row = one tracklet; crops run left to right. White box = jersey crop. Red = rejected/overlap.",
        (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (180, 195, 205), 1, cv2.LINE_AA,
    )

    manifest = []
    for row_index, (track_id, observations) in enumerate(ordered):
        y0 = HEADER_H + row_index * ROW_H
        cv2.line(sheet, (0, y0), (width, y0), (65, 65, 65), 1)
        is_gk = majority_goalkeeper(observations)
        base_color = GK_BGR if is_gk else FIELD_BGR
        start, end = observations[0]["frame"], observations[-1]["frame"]
        cv2.putText(
            sheet, f"row {row_index + 1:02d}   track {track_id}",
            (10, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
            base_color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, "GOALKEEPER" if is_gk else "FIELD PLAYER",
            (10, y0 + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            base_color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, f"f{start}-{end}  {start / info.fps:.2f}-{end / info.fps:.2f}s",
            (10, y0 + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (195, 195, 195), 1, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, f"detections: {len(observations)}",
            (10, y0 + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (195, 195, 195), 1, cv2.LINE_AA,
        )

        selected_samples = selected[track_id]
        serialized_samples = []
        for sample_index, sample in enumerate(selected_samples):
            x0 = LABEL_W + sample_index * CELL_W
            cell = fit_cell(sample.get("crop", np.empty((0, 0, 3), np.uint8)))
            sheet[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
            accepted = sample.get("accepted", False)
            border = base_color if accepted else REJECT_BGR
            cv2.rectangle(
                sheet, (x0 + 2, y0 + 2),
                (x0 + CELL_W - 3, y0 + CELL_H - 3), border, 2,
            )
            footer = (
                f"f{sample['frame']} q{sample.get('quality', 0):.2f} "
                f"ov{sample['overlap']:.0%}"
            )
            cv2.putText(
                sheet, footer, (x0 + 6, y0 + CELL_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, border, 1, cv2.LINE_AA,
            )
            serialized_samples.append({
                "frame": sample["frame"],
                "quality": sample.get("quality", 0.0),
                "overlap": sample["overlap"],
                "accepted": accepted,
            })
        manifest.append({
            "row": row_index + 1,
            "track_id": track_id,
            "start_frame": start,
            "end_frame": end,
            "detection_count": len(observations),
            "goalkeeper": is_gk,
            "samples": serialized_samples,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"failed to write {output_path}")
    manifest_path.write_text(json.dumps({
        "video": str(video),
        "fps": info.fps,
        "tracklet_count": len(ordered),
        "tracklets": manifest,
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_arg in args.videos:
        video = video_arg.resolve()
        info = sv.VideoInfo.from_video_path(str(video))
        cache_path = args.cache_dir / f".{video.stem}_detections_v1.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cache = load_cache(cache_path)
        tracklets = collect_tracklets(cache, info.fps, info.total_frames)
        output_path = args.output_dir / f"{video.stem}_tracklets.jpg"
        manifest_path = args.output_dir / f"{video.stem}_tracklets.json"
        render_sheet(
            video, tracklets, args.samples, output_path, manifest_path
        )
        print(f"wrote {output_path} ({len(tracklets)} tracklets)")
        print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
