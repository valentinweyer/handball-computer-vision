"""Generate easy per-video prototype sheets and save clicked calibrations.

Examples:
    python calibrate_teams.py candidates data/raw/FelixClaar.mp4 data/raw/Hannover.mp4
    python calibrate_teams.py fit --video data/raw/Hannover.mp4 --team-a 3 --team-b 8 --referee 1 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from team_calibration import PrototypeTeamCalibrator, spatial_jersey_features
from team_model import (
    MAX_TORSO_CONTAMINATION,
    crop_quality,
    torso_boxes,
    torso_contamination,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "outputs" / "team_comparison"
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "calibration"
FIELD_PLAYER_CLASS_ID = 2
SHEET_COLS = 6
CELL_W = 290
CELL_H = 310
HEADER_H = 90


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "offsets": data["offsets"],
            "boxes": data["boxes"],
            "class_id": data["class_id"],
        }


def frame_arrays(cache: dict[str, np.ndarray], frame_index: int):
    start, end = cache["offsets"][frame_index:frame_index + 2]
    start, end = int(start), int(end)
    return (
        cache["boxes"][start:end],
        cache["class_id"][start:end].astype(int),
    )


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image.size == 0:
        return canvas
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def collect_pool(
    video: Path, cache: dict[str, np.ndarray], max_crops: int = 1200,
):
    info = sv.VideoInfo.from_video_path(str(video))
    stride = max(1, round(info.fps / 5.0))
    records, crops = [], []
    frames = sv.get_video_frames_generator(str(video))
    for frame_index, frame_bgr in enumerate(tqdm(
        frames, total=info.total_frames, desc=f"pool {video.stem}"
    )):
        if frame_index % stride:
            continue
        boxes, class_ids = frame_arrays(cache, frame_index)
        if not len(boxes):
            continue
        overlap = torso_contamination(boxes)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for detection_index in np.flatnonzero(class_ids == FIELD_PLAYER_CLASS_ID):
            if overlap[detection_index] > MAX_TORSO_CONTAMINATION:
                continue
            torso = torso_boxes(boxes[detection_index:detection_index + 1])[0]
            crop_rgb = sv.crop_image(frame_rgb, torso)
            quality = crop_quality(crop_rgb)
            if not quality.accepted:
                continue
            records.append({
                "frame": frame_index,
                "detection_index": int(detection_index),
                "box": boxes[detection_index].astype(float).tolist(),
                "quality": quality.score,
                "overlap": float(overlap[detection_index]),
            })
            crops.append(crop_rgb)
            if len(crops) >= max_crops:
                return records, crops
    return records, crops


def diverse_candidates(records, crops, count: int):
    if len(crops) < count:
        count = len(crops)
    features = spatial_jersey_features(crops)
    scaled = StandardScaler().fit_transform(features)
    components = min(10, len(crops) - 1, scaled.shape[1])
    projection = PCA(n_components=components, random_state=0).fit_transform(scaled)
    model = KMeans(n_clusters=count, random_state=0, n_init=20).fit(projection)
    chosen = []
    for cluster_id, center in enumerate(model.cluster_centers_):
        indices = np.flatnonzero(model.labels_ == cluster_id)
        distances = np.linalg.norm(projection[indices] - center, axis=1)
        # Prefer a representative medoid, then use quality as a gentle tie-break.
        objective = distances / np.maximum(
            np.array([records[index]["quality"] for index in indices]), 0.1
        )
        chosen_index = int(indices[objective.argmin()])
        chosen.append((len(indices), chosen_index))
    chosen.sort(reverse=True)
    return [(records[index], crops[index], size) for size, index in chosen]


def read_frame(video: Path, frame_index: int) -> np.ndarray:
    # Random seeking is unreliable for some long-GOP H.264 inputs. Candidate
    # generation is infrequent and the clips are short, so deterministic
    # sequential decode is the safer behavior.
    for index, frame in enumerate(
        sv.get_video_frames_generator(str(video))
    ):
        if index == frame_index:
            return frame
    raise RuntimeError(f"could not read frame {frame_index} from {video}")


def full_crop(frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = 0.05 * (x2 - x1), 0.03 * (y2 - y1)
    expanded = np.array([
        max(0, x1 - pad_x), max(0, y1 - pad_y),
        min(width, x2 + pad_x), min(height, y2 + pad_y),
    ])
    return sv.crop_image(frame_bgr, expanded)


def write_candidate_sheet(video: Path, candidates, path: Path, json_path: Path):
    rows = (len(candidates) + SHEET_COLS - 1) // SHEET_COLS
    sheet = np.full(
        (HEADER_H + rows * CELL_H, SHEET_COLS * CELL_W, 3),
        20, dtype=np.uint8,
    )
    cv2.putText(
        sheet, f"{video.name}: choose one clean crop for each team",
        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(
        sheet, "Reply A=<id> B=<id>; optionally R=<id,id> for referees/officials",
        (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (80, 220, 80), 1, cv2.LINE_AA,
    )
    serialized = []
    frame_cache = {}
    for candidate_id, (record, torso_rgb, cluster_size) in enumerate(candidates, 1):
        row, column = divmod(candidate_id - 1, SHEET_COLS)
        x0, y0 = column * CELL_W, HEADER_H + row * CELL_H
        frame_index = record["frame"]
        if frame_index not in frame_cache:
            frame_cache[frame_index] = read_frame(video, frame_index)
        frame_bgr = frame_cache[frame_index]
        person = full_crop(frame_bgr, np.asarray(record["box"]))
        torso_bgr = cv2.cvtColor(torso_rgb, cv2.COLOR_RGB2BGR)
        sheet[y0 + 42:y0 + 264, x0 + 5:x0 + 115] = fit_image(
            person, 110, 222
        )
        sheet[y0 + 42:y0 + 264, x0 + 120:x0 + 285] = fit_image(
            torso_bgr, 165, 222
        )
        cv2.rectangle(
            sheet, (x0 + 2, y0 + 2), (x0 + CELL_W - 3, y0 + CELL_H - 3),
            (70, 70, 70), 1,
        )
        cv2.putText(
            sheet, f"ID {candidate_id:02d}", (x0 + 10, y0 + 29),
            cv2.FONT_HERSHEY_SIMPLEX, 0.67, (80, 220, 80), 2, cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"f{frame_index}  q={record['quality']:.2f}  support={cluster_size}",
            (x0 + 9, y0 + 290), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            (195, 195, 195), 1, cv2.LINE_AA,
        )
        serialized.append({
            "id": candidate_id,
            **record,
            "cluster_support": cluster_size,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write {path}")
    json_path.write_text(json.dumps({
        "video": str(video.resolve()),
        "source_size": video.stat().st_size,
        "source_mtime_ns": video.stat().st_mtime_ns,
        "candidates": serialized,
    }, indent=2) + "\n")


def seed_crop(video: Path, candidate: dict) -> np.ndarray:
    frame_bgr = read_frame(video, int(candidate["frame"]))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    torso = torso_boxes(np.asarray(candidate["box"])[None])[0]
    return sv.crop_image(frame_rgb, torso)


def command_candidates(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_arg in args.videos:
        video = video_arg.resolve()
        cache_path = args.cache_dir / f".{video.stem}_detections_v1.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        records, crops = collect_pool(video, load_cache(cache_path))
        candidates = diverse_candidates(records, crops, args.count)
        sheet = args.output_dir / f"{video.stem}_candidates.jpg"
        metadata = args.output_dir / f"{video.stem}_candidates.json"
        write_candidate_sheet(video, candidates, sheet, metadata)
        print(f"wrote {sheet}")
        print(f"wrote {metadata}")


def command_fit(args) -> None:
    video = args.video.resolve()
    candidate_path = args.output_dir / f"{video.stem}_candidates.json"
    data = json.loads(candidate_path.read_text())
    candidates = {item["id"]: item for item in data["candidates"]}
    requested = [args.team_a, args.team_b, *args.referee]
    missing = [candidate_id for candidate_id in requested if candidate_id not in candidates]
    if missing:
        raise ValueError(f"unknown candidate ids: {missing}")
    cache_path = args.cache_dir / f".{video.stem}_detections_v1.npz"
    _records, pool_crops = collect_pool(video, load_cache(cache_path))
    seed_crops = {
        0: [seed_crop(video, candidates[args.team_a])],
        1: [seed_crop(video, candidates[args.team_b])],
    }
    if args.referee:
        seed_crops[2] = [
            seed_crop(video, candidates[candidate_id])
            for candidate_id in args.referee
        ]
    selection = {
        "video": str(video),
        "team_a": args.team_a,
        "team_b": args.team_b,
        "referee": args.referee,
    }
    calibrator = PrototypeTeamCalibrator.fit(
        pool_crops, seed_crops, metadata=selection
    )
    model_path = args.output_dir / f".{video.stem}_prototype_team.pkl"
    selection_path = args.output_dir / f"{video.stem}_selection.json"
    calibrator.save(model_path)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"wrote {model_path}")
    print(f"wrote {selection_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    candidate_parser = commands.add_parser("candidates")
    candidate_parser.add_argument("videos", nargs="+", type=Path)
    candidate_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    candidate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    candidate_parser.add_argument("--count", type=int, default=18)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--video", type=Path, required=True)
    fit_parser.add_argument("--team-a", type=int, required=True)
    fit_parser.add_argument("--team-b", type=int, required=True)
    fit_parser.add_argument("--referee", type=int, nargs="*", default=[])
    fit_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    fit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "candidates":
        command_candidates(args)
    else:
        command_fit(args)


if __name__ == "__main__":
    main()
