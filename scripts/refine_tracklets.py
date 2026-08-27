"""Run MCByte, then refine its tracklets offline with GTA-style split/connect.

Phase 1 harness: builds tracklets from a cached-detection clip, embeds every
detection with the SoccerNet PRTReID model, applies split and connect, and
reports what changed against the identity failures we already know about.

Reports, per clip:
  * how many tracklets were split (a split means "this tracker_id held more
    than one player"), with the frame each cut landed on
  * how many tracklets were connected back into one player
  * whether the known-bad cases were caught

Usage:
    python -m scripts.refine_tracklets data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --expect-split 3
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
    CONNECT_DISTANCE_THRESHOLD,
    SPLIT_DISTANCE_THRESHOLD,
    Tracklet,
    connect_tracklets,
    split_all,
)
from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRTREID_ROOT = Path("/tmp/handball-prtreid")
DEFAULT_PRTREID_CKPT = ROOT / "models/prtreid/prtreid-soccernet-baseline.pth.tar"
MIN_CROP_SIDE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--masks", choices=("on", "off"), default="off",
        help="Cutie mask conditioning. Measured to cause identity switches on "
             "FelixClaar (6 -> 2 with masks off), so off is the default here.",
    )
    parser.add_argument("--split-distance", type=float, default=SPLIT_DISTANCE_THRESHOLD)
    parser.add_argument("--connect-distance", type=float, default=CONNECT_DISTANCE_THRESHOLD)
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--no-connect", action="store_true")
    parser.add_argument(
        "--expect-split", type=int, nargs="*", default=(),
        help="tracker_ids known to hold more than one player; reported as a check",
    )
    parser.add_argument("--prtreid-root", type=Path, default=DEFAULT_PRTREID_ROOT)
    parser.add_argument("--prtreid-checkpoint", type=Path, default=DEFAULT_PRTREID_CKPT)
    return parser.parse_args()


def build_tracklets(
    video: Path, cache_path: Path, backend, device: str, enable_masks: bool,
) -> list:
    """Track the clip and collect one embedded Tracklet per tracker_id."""
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

    frames_by_track = defaultdict(list)
    boxes_by_track = defaultdict(list)
    crops_by_track = defaultdict(list)
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(video)),
        total=info.total_frames, desc=f"track {video.stem}",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(frame_detections(cache, frame_index), frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        height, width = frame_rgb.shape[:2]
        for box, tracker_id in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
                continue
            frames_by_track[int(tracker_id)].append(frame_index)
            boxes_by_track[int(tracker_id)].append([x1, y1, x2, y2])
            crops_by_track[int(tracker_id)].append(frame_rgb[y1:y2, x1:x2])

    tracklets = []
    for tracker_id in sorted(frames_by_track):
        crops = crops_by_track[tracker_id]
        embeddings = backend.encode_images(crops)
        tracklets.append(Tracklet(
            tracklet_id=tracker_id,
            source_tracker_id=tracker_id,
            frames=np.asarray(frames_by_track[tracker_id], dtype=int),
            boxes=np.asarray(boxes_by_track[tracker_id], dtype=float),
            embeddings=embeddings,
        ))
    return tracklets


def main() -> None:
    args = parse_args()
    backend = PRTReIDBackend(
        source_root=args.prtreid_root,
        checkpoint=args.prtreid_checkpoint,
        device=args.device,
        feature_kind="global",
    )
    tracklets = build_tracklets(
        args.video, args.detections, backend, args.device, args.masks == "on"
    )
    before = {t.tracklet_id: (t.start, t.end, len(t)) for t in tracklets}

    refined = tracklets
    if not args.no_split:
        refined = split_all(refined, distance_threshold=args.split_distance)
    splits = defaultdict(list)
    for tracklet in refined:
        if tracklet.parts:
            splits[tracklet.parts[-1]].append(tracklet)

    groups = {}
    if not args.no_connect:
        groups = connect_tracklets(refined, distance_threshold=args.connect_distance)

    merged = defaultdict(list)
    for tracklet_id, group_id in groups.items():
        merged[group_id].append(tracklet_id)
    connected = {g: ids for g, ids in merged.items() if len(ids) > 1}

    print(f"\ntracklets in : {len(tracklets)}")
    print(f"tracklets out: {len(refined)}")
    print(f"players after connect: {len(set(groups.values())) if groups else len(refined)}")

    print(f"\nSPLIT ({len(splits)} tracker_ids cut):")
    for source, pieces in sorted(splits.items()):
        start, end, count = before[source]
        spans = ", ".join(f"{p.start}-{p.end}" for p in sorted(pieces, key=lambda p: p.start))
        print(f"  tracker_id {source} (frames {start}-{end}, n={count}) -> {spans}")
    if not splits:
        print("  none")

    print(f"\nCONNECT ({len(connected)} groups with >1 tracklet):")
    for group_id, ids in sorted(connected.items()):
        print(f"  group {group_id}: tracklets {sorted(ids)}")
    if not connected:
        print("  none")

    if args.expect_split:
        print("\nCHECK against known mixed-identity tracker_ids:")
        for tracker_id in args.expect_split:
            caught = tracker_id in splits
            print(f"  tracker_id {tracker_id}: {'SPLIT (caught)' if caught else 'NOT split (missed)'}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "video": str(args.video),
            "masks": args.masks,
            "tracklets_in": len(tracklets),
            "tracklets_out": len(refined),
            "players_after_connect": len(set(groups.values())) if groups else len(refined),
            "split": {str(k): [[p.start, p.end] for p in v] for k, v in splits.items()},
            "connected": {str(k): sorted(v) for k, v in connected.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
