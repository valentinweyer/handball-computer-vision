"""Render one sheet per tracklet so a human can mark it clean or mixed.

Every tracking conclusion in this project currently rests on
`suspected_id_switch`, which is a heuristic: it fires when a *settled* team
label flips. It cannot see a swap between two players on the same team, and it
is suppressed by fragmentation -- a track that breaks instead of sliding starts
a fresh provisional identity and never registers a "switch" at all. So the
counter can fall while identity gets no better, which makes masks-on vs
masks-off comparisons built on it untrustworthy.

This produces the ground truth that settles it. For each tracker_id it renders
crops sampled evenly across the tracklet's whole lifetime, in temporal order,
each labelled with its frame number. A tracklet that changes hands is obvious
by eye -- that is how `tracker_id 3` on FelixClaar was confirmed to hold a
white-shirted #13 until ~frame 90 and a navy player afterwards.

Answer format, one line per tracklet, in `answers.txt`:

    <tracker_id> C              # clean: one player throughout
    <tracker_id> M <frame>      # mixed: changes hands at about <frame>
    <tracker_id> X              # unclear / too occluded to judge

Usage:
    python -m scripts.label_tracklet_identity data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --masks on --out runs/tracklet_labels/FelixClaar_masks_on
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

from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

TILE_HEIGHT = 280
LABEL_STRIP = 26
COLUMNS = 6
MIN_CROP_SIDE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--masks", choices=("on", "off"), default="on",
        help="Cutie mask conditioning. Label BOTH settings to compare them "
             "honestly -- that comparison is the point of this script.",
    )
    parser.add_argument(
        "--samples", type=int, default=12,
        help="crops per tracklet, spread evenly over its lifetime",
    )
    parser.add_argument(
        "--min-length", type=int, default=8,
        help="skip tracklets shorter than this many frames",
    )
    return parser.parse_args()


def collect(video: Path, cache_path: Path, device: str, enable_masks: bool) -> dict:
    """-> {tracker_id: [(frame_index, box), ...]} in frame order."""
    cache = load_detection_cache(cache_path)
    info = sv.VideoInfo.from_video_path(str(video))
    if enable_masks and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        enable_mask_manager=enable_masks,
        mask_config=McByteMaskConfig(device=device) if enable_masks else None,
        minimum_mask_average_confidence=0.6,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    per_track = defaultdict(list)
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(video)),
        total=info.total_frames, desc=f"track {video.stem}",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(frame_detections(cache, frame_index), frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        for box, tracker_id in zip(tracked.xyxy, tracked.tracker_id):
            per_track[int(tracker_id)].append((frame_index, box.copy()))
    return per_track


def evenly_spaced(items: list, count: int) -> list:
    """`count` items spread across `items`, always including first and last."""
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in dict.fromkeys(indices.tolist())]


def crop_tile(frame_bgr: np.ndarray, box: np.ndarray, pad: int = 6) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    scale = TILE_HEIGHT / crop.shape[0]
    return cv2.resize(
        crop, (max(1, int(crop.shape[1] * scale)), TILE_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def build_sheet(tiles: list, labels: list, title: str) -> np.ndarray:
    columns = min(COLUMNS, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    cell_width = max(t.shape[1] for t in tiles) + 8
    cell_height = TILE_HEIGHT + LABEL_STRIP
    sheet = np.full(
        (rows * cell_height + 34, columns * cell_width, 3), 28, dtype=np.uint8
    )
    cv2.putText(
        sheet, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row, column = divmod(index, columns)
        top = 34 + row * cell_height
        left = column * cell_width
        sheet[top:top + TILE_HEIGHT, left:left + tile.shape[1]] = tile
        cv2.putText(
            sheet, label, (left + 4, top + TILE_HEIGHT + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA,
        )
    return sheet


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per_track = collect(
        args.video, args.detections, args.device, args.masks == "on"
    )
    kept = {
        tracker_id: entries for tracker_id, entries in per_track.items()
        if len(entries) >= args.min_length
    }
    wanted = {
        tracker_id: evenly_spaced(entries, args.samples)
        for tracker_id, entries in kept.items()
    }
    # One pass over the video collecting only the frames some sheet needs.
    needed = defaultdict(list)
    for tracker_id, entries in wanted.items():
        for frame_index, box in entries:
            needed[frame_index].append((tracker_id, box))

    tiles = defaultdict(list)
    labels = defaultdict(list)
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(args.video)),
        total=sv.VideoInfo.from_video_path(str(args.video)).total_frames,
        desc="crops",
    )):
        for tracker_id, box in needed.get(frame_index, ()):
            tile = crop_tile(frame_bgr, box)
            if tile is None:
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            tiles[tracker_id].append(tile)
            labels[tracker_id].append(f"f{frame_index}  {x2 - x1}x{y2 - y1}")

    manifest = {}
    for tracker_id in sorted(tiles):
        entries = kept[tracker_id]
        span = f"{entries[0][0]}-{entries[-1][0]}"
        title = (
            f"tracker_id {tracker_id}   frames {span}   "
            f"n={len(entries)}   masks={args.masks}"
        )
        sheet = build_sheet(tiles[tracker_id], labels[tracker_id], title)
        path = args.out / f"tracklet_{tracker_id:03d}.jpg"
        cv2.imwrite(str(path), sheet)
        manifest[tracker_id] = {
            "sheet": path.name, "start": entries[0][0],
            "end": entries[-1][0], "detections": len(entries),
        }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    template = args.out / "answers.txt"
    if not template.exists():
        lines = [
            "# One line per tracklet. C = clean (one player), "
            "M <frame> = mixed (changes hands near <frame>), X = unclear.",
            f"# video={args.video}  masks={args.masks}",
        ]
        lines += [
            f"{tracker_id} " for tracker_id in sorted(manifest)
        ]
        template.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {len(manifest)} tracklet sheets to {args.out}")
    print(f"skipped {len(per_track) - len(kept)} tracklets shorter than {args.min_length} frames")
    print(f"fill in {template}")


if __name__ == "__main__":
    main()
