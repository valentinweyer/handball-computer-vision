"""Find and follow the ball on a clip, and show the working.

The detector has no ball class, and the labels to add one do not exist -- so
this finds the ball the way a person following a replay does, from motion and
continuity rather than from appearance in any single frame. Nothing here is
trained and nothing is labelled.

Three stages, each supplying what the next one cannot get for itself:

  1. camera-compensated three-frame differencing proposes candidates,
  2. constant-velocity chaining accepts the ones that lie on a plausible
     trajectory -- the crowd contributes tens of ball-sized blobs per frame and
     none of them travel, so this is what does the real rejecting,
  3. the accepted trajectories are used to calibrate what this ball looks like
     in this video, which is then the only way to place it while a player is
     carrying it, since a held ball has no motion of its own to find.

Reads the frame cache rather than the mp4: it is what the tracking path already
uses, and re-decoding would risk differencing frames the rest of the project
never sees.

Usage:
    python -m scripts.render_ball_demo --clip Melsungen_window \\
        --detections outputs/team_dataset/.Melsungen_window_detections_v1.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

from handball_cv.ball.appearance import BallObservation, calibrate
from handball_cv.ball.candidates import (
    ball_candidates, believable_motion, overlay_mask, play_volume_mask,
)
from handball_cv.ball.tracking import (
    BallState, TrackPoint, chain_tracks, resolve_possession,
)
from handball_cv.court.camera_motion import estimate_camera_motion
from handball_cv.teams.dataset import frame_arrays, load_detection_cache

FRAME_CACHE = Path("data/cache/frames")

# Goalkeeper, player AND referee. render_raw_team_classification's
# PERSON_CLASS_IDS excludes referees because a tracker should not follow them;
# the question here is the opposite one -- whose limbs throw off ball-sized
# residuals -- and a referee's do. Class 4 is jersey numbers, which are not
# bodies and would mask out part of every torso.
BODY_CLASS_IDS = (1, 2, 3)

FLIGHT_COLOUR = (0, 0, 255)
HELD_COLOUR = (0, 210, 255)
CANDIDATE_COLOUR = (140, 140, 140)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ball_demo"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--min-speed", type=float, default=9.0)
    parser.add_argument("--max-speed", type=float, default=70.0)
    parser.add_argument("--tolerance", type=float, default=9.0)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--min-image-travel", type=float, default=30.0,
                        help="how far a trajectory must get from its start in the "
                             "picture; a burned-in scoreboard never gets anywhere")
    parser.add_argument("--rise-px", type=int, default=320,
                        help="how far above the court a thrown ball is still believed")
    parser.add_argument("--ignore-person-boxes", action="store_true",
                        help="propose candidates on players too, to measure what the "
                             "person detector is actually worth to ball tracking")
    parser.add_argument("--no-volume", action="store_true",
                        help="drop the play-volume prior, to measure what it costs")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def body_boxes(cache, index: int) -> np.ndarray:
    boxes, _, class_ids, _ = frame_arrays(cache, index)
    return boxes[np.isin(class_ids, BODY_CLASS_IDS)]


def broadcast_overlay(files, first: int, count: int, samples: int = 40):
    """Sample across the window so the camera has demonstrably moved between them."""
    step = max(1, count // samples)
    frames = [cv2.imread(str(files[index]), cv2.IMREAD_GRAYSCALE)
              for index in range(first, first + count, step)]
    return overlay_mask([f for f in frames if f is not None])


def collect_candidates(files, cache, args):
    """Stream the clip once, proposing candidates and tracking the camera.

    One homography per frame, not two: the motion from `t` to `t+1` is measured
    on this step and inverted to serve as `t+1`'s backward motion on the next.
    """
    first, last = args.start, args.start + args.count
    per_frame: list[list[TrackPoint]] = []
    boxes_by_frame: list[np.ndarray] = []
    motions: list[np.ndarray | None] = []
    rejected = {}
    breaks = 0

    overlay = broadcast_overlay(files, first, args.count)
    previous = cv2.imread(str(files[first - 1]), cv2.IMREAD_GRAYSCALE)
    current_bgr = cv2.imread(str(files[first]))
    current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    backward = estimate_camera_motion(previous, current)
    if not believable_motion(backward, current.shape[:2]):
        backward = None

    for index in tqdm(range(first, last), desc="candidates"):
        following_bgr = cv2.imread(str(files[index + 1]))
        following = cv2.cvtColor(following_bgr, cv2.COLOR_BGR2GRAY)
        forward = estimate_camera_motion(current, following)
        if not believable_motion(forward, current.shape[:2]):
            forward = None
        boxes = body_boxes(cache, index)
        boxes_by_frame.append(boxes)

        motions.append(forward)
        if backward is None or forward is None:
            # No measured motion means these frames cannot be related at all, so
            # there is nothing honest to say about this one. An empty list also
            # terminates any chain running through it, which is the point.
            per_frame.append([])
            breaks += 1
        else:
            volume = None if args.no_volume else play_volume_mask(
                current_bgr, rise_px=args.rise_px
            )
            frame = ball_candidates(
                current, previous, following, backward, np.linalg.inv(forward),
                play_volume=volume, overlay=overlay,
                person_boxes=() if args.ignore_person_boxes else boxes,
            )
            for reason, count in frame.rejected.items():
                rejected[reason] = rejected.get(reason, 0) + count
            per_frame.append([
                TrackPoint(frame=index - first, x=c.x, y=c.y, radius=c.radius)
                for c in frame.candidates
            ])

        previous, current, current_bgr = current, following, following_bgr
        backward = forward

    return per_frame, boxes_by_frame, motions, rejected, breaks


def calibrate_from_tracks(files, tracks, boxes_by_frame, first, *, half=22):
    """Learn this ball's colour and size from the trajectories already accepted."""
    observations = []
    for track in tracks:
        for point in track.points:
            frame = cv2.imread(str(files[first + point.frame]))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            x, y = int(round(point.x)), int(round(point.y))
            patch = frame[max(0, y - half):min(height, y + half),
                          max(0, x - half):min(width, x + half)]
            if patch.size == 0:
                continue
            boxes = boxes_by_frame[point.frame]
            heights = [float(b[3] - b[1]) for b in boxes
                       if abs((b[0] + b[2]) / 2 - x) < 300]
            observations.append(BallObservation(
                patch=patch, radius=point.radius,
                person_height=float(np.median(heights)) if heights else None,
            ))
    return calibrate(observations), len(observations)


def contact_sheet(files, points, path: Path, first: int, *,
                  half=150, columns=6, tile=260):
    """Crops around each point, tiled. A 27px ball is not checkable at 1080p."""
    tiles = []
    for point in points:
        frame = cv2.imread(str(files[first + point.frame]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        left = int(max(0, min(width - 2 * half, point.x - half)))
        top = int(max(0, min(height - 2 * half, point.y - half)))
        crop = frame[top:top + 2 * half, left:left + 2 * half].copy()
        cv2.circle(crop, (int(point.x - left), int(point.y - top)), 26, FLIGHT_COLOUR, 2)
        cv2.putText(crop, str(first + point.frame), (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(crop, (tile, tile)))
    if not tiles:
        return
    blank = np.zeros((tile, tile, 3), np.uint8)
    rows = [np.hstack(tiles[i:i + columns]
                      + [blank] * (columns - len(tiles[i:i + columns])))
            for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows))


def render(files, states, per_frame, boxes_by_frame, first, output: Path, fps: float):
    sample = cv2.imread(str(files[first]))
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for offset, state in enumerate(tqdm(states, desc="render")):
            frame = cv2.imread(str(files[first + offset]))
            if frame is None:
                break
            for box in boxes_by_frame[offset]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 150, 0), 1)
            for point in per_frame[offset]:
                cv2.circle(frame, (int(point.x), int(point.y)), 10,
                           CANDIDATE_COLOUR, 1)
            if state.state is BallState.FLIGHT:
                cv2.circle(frame, (int(state.x), int(state.y)), 24, FLIGHT_COLOUR, 3)
            elif state.state is BallState.HELD:
                cv2.circle(frame, (int(state.x), int(state.y)), 20, HELD_COLOUR, 3)
            cv2.putText(frame, f"f{first + offset}  ball: {state.state.value}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        FLIGHT_COLOUR if state.state is BallState.FLIGHT
                        else HELD_COLOUR, 3)
            writer.write(frame)
    finally:
        writer.release()

    portable = output.with_stem(output.stem + "_h264")
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(output),
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(portable)],
        check=True,
    )
    return portable


def main() -> None:
    args = parse_args()
    files = sorted((FRAME_CACHE / args.clip).glob("*.jpg"))
    if not files:
        raise SystemExit(f"no cached frames for {args.clip}")
    cache = load_detection_cache(args.detections)
    available = min(len(files) - 1, int(cache["total_frames"]))
    args.start = max(1, args.start)
    args.count = min(args.count or available, available - args.start)

    per_frame, boxes_by_frame, motions, rejected, breaks = collect_candidates(
        files, cache, args
    )
    tracks = chain_tracks(
        per_frame, motions, min_speed=args.min_speed, max_speed=args.max_speed,
        tolerance=args.tolerance, min_length=args.min_length,
        min_image_travel=args.min_image_travel,
    )

    model, sightings = calibrate_from_tracks(files, tracks, boxes_by_frame, args.start)

    # A first pass without appearance says who would be holding the ball; the
    # second lets the appearance model either confirm that or let it lapse.
    provisional = resolve_possession(tracks, boxes_by_frame)
    confirmations = {}
    if model is not None:
        for state in tqdm(provisional, desc="held-ball search"):
            if state.state is not BallState.HELD or state.holder is None:
                continue
            frame = cv2.imread(str(files[args.start + state.frame]))
            if frame is None:
                continue
            found = model.locate(frame, boxes_by_frame[state.frame][state.holder])
            if found is not None:
                confirmations[state.frame] = (found[0], found[1])
    states = resolve_possession(tracks, boxes_by_frame, confirmations=confirmations)

    out = args.output_dir
    sheets = out / f"{args.clip}_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    # Sheets are numbered by rank, so a run that finds fewer tracks than the last
    # one would otherwise leave the tail of the old run lying beside the new.
    for stale in sheets.glob("track_*.jpg"):
        stale.unlink()
    for number, track in enumerate(sorted(tracks, key=lambda t: -len(t.points))):
        contact_sheet(files, track.points,
                      sheets / f"track_{number:03d}.jpg", args.start)

    counts = {state.value: sum(1 for s in states if s.state is state)
              for state in BallState}
    summary = {
        "clip": args.clip,
        "frames": args.count,
        "first_frame": args.start,
        "candidates": sum(len(p) for p in per_frame),
        "rejected": rejected,
        "camera_motion_breaks": breaks,
        "appearance_model": None if model is None else {
            "samples": model.samples,
            "player_ratio": round(model.player_ratio, 4),
        },
        "appearance_sightings_offered": sightings,
        "held_confirmations": len(confirmations),
        "states": counts,
        "tracks": [
            {"first": args.start + t.first_frame, "last": args.start + t.last_frame,
             "length": len(t.points), "mean_speed": round(t.mean_speed, 1)}
            for t in sorted(tracks, key=lambda t: -len(t.points))
        ],
    }
    (out / f"{args.clip}_summary.json").write_text(json.dumps(summary, indent=2))
    # Per-track points, kept separate: states.json flattens overlapping tracks
    # into one position per frame, so measuring a track's own step lengths there
    # silently measures the gap between two different tracks instead.
    (out / f"{args.clip}_tracks.json").write_text(json.dumps([
        [{"frame": args.start + p.frame, "x": p.x, "y": p.y, "radius": p.radius}
         for p in t.points]
        for t in sorted(tracks, key=lambda t: -len(t.points))
    ]))
    (out / f"{args.clip}_states.json").write_text(json.dumps([
        {"frame": args.start + s.frame, "state": s.state.value,
         "x": s.x, "y": s.y, "holder": s.holder}
        for s in states
    ]))

    if not args.no_video:
        video = render(files, states, per_frame, boxes_by_frame, args.start,
                       out / f"{args.clip}_ball.mp4", args.fps)
        print(f"-> {video}")
    print(json.dumps({k: summary[k] for k in
                      ("candidates", "rejected", "states", "appearance_model",
                       "held_confirmations", "camera_motion_breaks")}, indent=2))
    print(f"{len(tracks)} tracks; sheets in {sheets}")


if __name__ == "__main__":
    main()
