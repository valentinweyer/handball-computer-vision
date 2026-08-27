"""Can split detect the mixed tracklets a human labelled, using masked crops?

Two numbers per tracklet:

  * **oracle**       -- centroid distance between the span before the human's
                        changepoint and the span after it, with a window around
                        the changepoint excluded. During body contact the box
                        holds both players, so those crops are blends and drag
                        the two spans together; excluding them measures what is
                        actually there to find.
  * **unsupervised** -- what `split_tracklet` computes on its own (leading
                        principal direction, then temporal segregation). This is
                        what the algorithm can use without labels.

A threshold is only viable if the unsupervised statistic on labelled-mixed
tracklets sits clear of the same statistic on labelled-clean ones.

Usage:
    python -m scripts.evaluate_tracklet_split data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --team-model outputs/team_comparison/.FelixClaar_team.pkl \\
        --mixed 0:103 3:86 4:75 6:82 7:82 9:164 10:140 \\
        --clean 1 2 5 8 11 12 13 14 15 16
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.embeddings.prtreid import PRTReIDBackend
from handball_cv.tracking.tracklets import (
    _segregation,
    _two_cluster_labels,
    normalise,
)
from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRTREID_ROOT = ROOT / "prtreid-upstream"
DEFAULT_PRTREID_CKPT = ROOT / "models/prtreid/prtreid-soccernet-baseline.pth.tar"
MIN_MASK_PIXELS = 60
MIN_BOX_W, MIN_BOX_H = 12, 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument(
        "--mixed", nargs="+", default=(),
        help="labelled mixed tracklets as tracker_id:changepoint_frame",
    )
    parser.add_argument("--clean", type=int, nargs="+", default=())
    parser.add_argument(
        "--exclude", type=int, default=15,
        help="frames either side of the changepoint to drop as blended",
    )
    parser.add_argument("--masked", choices=("on", "off"), default="on")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prtreid-root", type=Path, default=DEFAULT_PRTREID_ROOT)
    parser.add_argument("--prtreid-checkpoint", type=Path, default=DEFAULT_PRTREID_CKPT)
    return parser.parse_args()


def collect(args, wanted: set) -> tuple[dict, dict]:
    """-> ({tracker_id: [crop, ...]}, {tracker_id: [frame, ...]})."""
    cache = load_detection_cache(args.detections)
    info = sv.VideoInfo.from_video_path(str(args.video))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps, lost_track_buffer=30, track_activation_threshold=0.7,
        enable_mask_manager=True, mask_config=McByteMaskConfig(device=args.device),
        minimum_mask_average_confidence=0.6, minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    crops, frames = defaultdict(list), defaultdict(list)
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(args.video)),
        total=info.total_frames, desc="collect",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(frame_detections(cache, frame_index), frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        mask_output = getattr(tracker, "_last_mask_output", None)
        masks = (
            np.asarray(mask_output.masks, dtype=bool)
            if mask_output is not None and mask_output.masks is not None
            and len(mask_output.masks) else None
        )
        height, width = frame_rgb.shape[:2]
        for box, tracker_id in zip(tracked.xyxy, tracked.tracker_id):
            tracker_id = int(tracker_id)
            if tracker_id not in wanted:
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                continue
            crop = frame_rgb[y1:y2, x1:x2]
            if args.masked == "on":
                if masks is None:
                    continue
                row = mask_output.tracklet_mask_dict.get(tracker_id)
                if row is None or not (0 <= int(row) < len(masks)):
                    continue
                mask = masks[int(row)][y1:y2, x1:x2]
                if mask.sum() < MIN_MASK_PIXELS:
                    continue
                crop = crop.copy()
                crop[~mask] = 0
            crops[tracker_id].append(crop)
            frames[tracker_id].append(frame_index)
    return crops, frames


def centroid(unit: np.ndarray) -> np.ndarray:
    c = unit.mean(axis=0)
    return c / max(float(np.linalg.norm(c)), 1e-8)


def main() -> None:
    args = parse_args()
    mixed = {}
    for item in args.mixed:
        tracker_id, changepoint = item.split(":")
        mixed[int(tracker_id)] = int(changepoint)
    clean = set(args.clean)
    wanted = set(mixed) | clean

    crops, frames = collect(args, wanted)
    backend = PRTReIDBackend(
        source_root=args.prtreid_root, checkpoint=args.prtreid_checkpoint,
        device=args.device, feature_kind="global",
    )

    rows = []
    for tracker_id in sorted(wanted):
        if len(crops.get(tracker_id, [])) < 16:
            continue
        unit = normalise(backend.encode_images(crops[tracker_id]))
        frame_array = np.asarray(frames[tracker_id])
        labels = _two_cluster_labels(unit)
        segregation, changepoint_index = _segregation(labels)
        a, b = unit[labels == 0], unit[labels == 1]
        unsupervised = (
            float(1.0 - np.dot(centroid(a), centroid(b)))
            if len(a) >= 2 and len(b) >= 2 else float("nan")
        )
        oracle = float("nan")
        if tracker_id in mixed:
            cut = mixed[tracker_id]
            before = unit[frame_array <= cut - args.exclude]
            after = unit[frame_array >= cut + args.exclude]
            if len(before) >= 4 and len(after) >= 4:
                oracle = float(1.0 - np.dot(centroid(before), centroid(after)))
        rows.append({
            "tracker_id": tracker_id,
            "label": "MIXED" if tracker_id in mixed else "clean",
            "n": len(unit),
            "oracle": oracle,
            "unsupervised": unsupervised,
            "segregation": segregation,
            "detected_frame": int(frame_array[min(changepoint_index, len(frame_array) - 1)]),
            "true_frame": mixed.get(tracker_id),
        })

    print(f"\nmasked={args.masked}  exclusion=+/-{args.exclude} frames\n")
    print(f'{"tid":>4} {"label":>6} {"n":>4} {"oracle":>8} {"unsup":>8} {"seg":>6} {"det@":>6} {"true@":>6}')
    for row in rows:
        print(
            f'{row["tracker_id"]:>4} {row["label"]:>6} {row["n"]:>4} '
            f'{row["oracle"]:>8.4f} {row["unsupervised"]:>8.4f} '
            f'{row["segregation"]:>6.3f} {row["detected_frame"]:>6} '
            f'{"-" if row["true_frame"] is None else row["true_frame"]:>6}'
        )

    mixed_stat = [r["unsupervised"] for r in rows if r["label"] == "MIXED"]
    clean_stat = [r["unsupervised"] for r in rows if r["label"] == "clean"]
    oracles = [r["oracle"] for r in rows if r["label"] == "MIXED" and r["oracle"] == r["oracle"]]
    if oracles:
        print(f'\noracle (mixed, labelled changepoint): min={min(oracles):.4f} '
              f'median={np.median(oracles):.4f} max={max(oracles):.4f}')
    if mixed_stat and clean_stat:
        print(f'unsupervised MIXED: min={min(mixed_stat):.4f} median={np.median(mixed_stat):.4f}')
        print(f'unsupervised clean: min={min(clean_stat):.4f} median={np.median(clean_stat):.4f} max={max(clean_stat):.4f}')
        separable = min(mixed_stat) > max(clean_stat)
        print(
            f'\nthreshold separates mixed from clean: {"YES" if separable else "NO"}'
            + (f'  (any value in {max(clean_stat):.4f}..{min(mixed_stat):.4f})' if separable else
               f'  (overlap: worst mixed {min(mixed_stat):.4f} <= best clean {max(clean_stat):.4f})')
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
