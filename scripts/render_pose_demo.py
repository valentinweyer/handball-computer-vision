"""Render body-keypoint skeletons over a clip, on top of cached detections.

A demo of one stage only: what body pose looks like on this footage. It reuses
the project's cached RF-DETR person boxes as the sole source of who is on the
frame and asks ViTPose to describe each of those boxes, so nothing here can
invent a player the detector did not find.

Deliberately stateless -- no tracker, no identity, no team. Pose is attached to
a box and nothing else, which keeps the demo honest about what it measures and
keeps it off every invariant the tracking and team code owns.

Usage:
    python -m scripts.render_pose_demo data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --output runs/pose_demo/FelixClaar_pose.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import cv2
from tqdm import tqdm

from handball_cv.pose.vitpose import ViTPoseEstimator
from handball_cv.visualization.pose_overlay import draw_pose
from scripts.render_raw_team_classification import (
    load_detection_cache,
    person_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default="models/vitpose-base-simple")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="drop keypoints the model is not sure about")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many frames")
    return parser.parse_args()


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    args = parse_args()
    cache = load_detection_cache(args.detections)
    estimator = ViTPoseEstimator(args.checkpoint, args.device)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cached_frames = len(cache["offsets"]) - 1
    total = min(cached_frames, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.limit is not None:
        total = min(total, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    drawn = 0
    try:
        for index in tqdm(range(total), desc="pose"):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            boxes = person_detections(cache, index).xyxy
            keypoints = estimator.estimate(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), boxes
            )
            for person in keypoints:
                draw_pose(frame_bgr, person, min_score=args.min_score)
            drawn += len(keypoints)
            writer.write(frame_bgr)
    finally:
        capture.release()
        writer.release()

    # mp4v is not portable enough to open everywhere; every other render in
    # this repo ships h264, so match it.
    h264 = args.output.with_stem(args.output.stem + "_h264")
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(args.output),
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(h264)],
        check=True,
    )
    print(f"{drawn} skeletons over {total} frames -> {h264}")


if __name__ == "__main__":
    main()
