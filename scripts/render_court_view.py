"""Render a clip with the court model's keypoints, the fitted court, and a top-down view.

Left: the frame with the court template warped onto it through the tracked
homography (``CourtTracker``, regularised by measured camera motion) and the
model's confident keypoints numbered by template KP. Right: the court from
above with every cached person detection's feet mapped through the same fit.

    python -m scripts.render_court_view --clip Melsungen_window \\
        --motion runs/court_mapping/Melsungen_motion.json \\
        --detections outputs/team_dataset/.Melsungen_window_detections_v1.npz

Keypoints are cached unfiltered (all 37 slots with confidence) in the same
format as ``runs/court_mapping/Melsungen_keypoints.json``, so a redraw needs no
inference, and passing that older file as ``--keypoints`` renders the YOLO
version 3 through the identical pipeline for comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sports import MeasurementUnit
from sports.handball import CourtConfiguration, League, draw_court
from tqdm import tqdm

from handball_cv.court.camera_motion import estimate_camera_motion
from handball_cv.court.homography import CourtTracker
from handball_cv.court.keypoints import KEYPOINT_TO_VERTEX
from handball_cv.tracking.sam2_driver import ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_ROOT = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
DEFAULT_WEIGHTS = FINETUNE_ROOT / "runs/court-keypoints-roboflow-v4/weights.onnx"
MODEL_ID = "keypointv333-uwois-xprdi/4"

FRAME_W, FRAME_H = 1280, 720      # left panel
SIDE_W = 640                      # right panel
TEMPLATE_SCALE, TEMPLATE_PAD = 0.15, 20   # top-down view, px per cm
LINE_SCALE, LINE_PAD = 0.5, 0             # template warped onto the frame
CONFIDENCE = 0.5

# Detection cache classes (outputs/*_detections_v1.npz): 1 goalkeeper,
# 2 player, 3 referee, 4 jersey number.
PEOPLE = {1: ("goalkeeper", (0, 200, 255)), 2: ("player", (255, 255, 255)), 3: ("referee", (60, 60, 60))}
LINE_COLOUR = (255, 220, 0)
KEYPOINT_COLOUR = (255, 0, 255)


def court_config() -> CourtConfiguration:
    return CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)


def to_template(config: CourtConfiguration, scale: float, pad: float) -> np.ndarray:
    """Court cm -> ``draw_court`` pixels (y flipped, as ``draw_court`` draws)."""
    return np.array([[scale, 0.0, pad], [0.0, -scale, config.court_width * scale + pad], [0.0, 0.0, 1.0]])


def load_or_infer(args, frames: list[Path]) -> list[list[tuple[int, float, float, float]]]:
    if args.keypoints.exists():
        cached = json.loads(args.keypoints.read_text())
        print(f"keypoints from {args.keypoints} ({cached.get('model')})")
        return cached["keypoints"]
    from handball_cv.court.rfdetr_keypoints import OfflineCourtKeypoints
    model = OfflineCourtKeypoints(args.weights)
    keypoints = []
    for path in tqdm(frames, desc="court keypoints"):
        points, _ = model.infer(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        keypoints.append([list(p) for p in points])
    args.keypoints.parent.mkdir(parents=True, exist_ok=True)
    args.keypoints.write_text(json.dumps({"model": MODEL_ID, "conf": CONFIDENCE, "clip": args.clip,
                                          "keypoints": keypoints}))
    return keypoints


def load_or_measure_motion(args, frames: list[Path]) -> list:
    if args.motion and args.motion.exists():
        return json.loads(args.motion.read_text())
    motion, previous = [None], None
    for i, path in enumerate(tqdm(frames, desc="camera motion")):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        if i:
            matrix = estimate_camera_motion(previous, gray)
            motion.append(None if matrix is None else matrix.tolist())
        previous = gray
    if args.motion:
        args.motion.write_text(json.dumps(motion))
    return motion


def feet_by_frame(path: Path | None, count: int) -> list[list[tuple[float, float, int]]]:
    if path is None:
        return [[] for _ in range(count)]
    data = np.load(path, allow_pickle=True)
    offsets, boxes, classes = data["offsets"], data["boxes"], data["class_id"]
    out = []
    for t in range(count):
        rows = range(int(offsets[t]), int(offsets[t + 1]))
        out.append([((boxes[r, 0] + boxes[r, 2]) / 2, boxes[r, 3], int(classes[r]))
                    for r in rows if int(classes[r]) in PEOPLE])
    return out


def render(args) -> None:
    frames = sorted(args.frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames_dir}")
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    config = court_config()
    vertices = np.asarray(config.vertices, dtype=np.float64)

    keypoints = load_or_infer(args, frames)
    motion = load_or_measure_motion(args, frames)
    feet = feet_by_frame(args.detections, len(frames))

    # A line-only template, warped onto each frame through the fit.
    line_layer = draw_court(config, scale=LINE_SCALE, padding=LINE_PAD, line_thickness=3,
                            background_color=sv.Color.BLACK)
    line_mask = (cv2.cvtColor(line_layer, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    template_from_court = to_template(config, LINE_SCALE, LINE_PAD)
    court_from_template = np.linalg.inv(template_from_court)

    side_court = draw_court(config, scale=TEMPLATE_SCALE, padding=TEMPLATE_PAD, line_thickness=2)
    side_from_court = to_template(config, TEMPLATE_SCALE, TEMPLATE_PAD)
    side_court = cv2.resize(side_court, (SIDE_W, round(side_court.shape[0] * SIDE_W / side_court.shape[1])))
    side_scale = SIDE_W / (config.court_length * TEMPLATE_SCALE + 2 * TEMPLATE_PAD)
    side_from_court = np.diag([side_scale, side_scale, 1.0]) @ side_from_court

    tracker = CourtTracker(vertices, (width, height))
    raw = args.output.with_suffix(".raw.mp4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (FRAME_W + SIDE_W, FRAME_H))
    sources = {"keypoints": 0, "propagated": 0, "none": 0}
    font = cv2.FONT_HERSHEY_SIMPLEX
    sx, sy = FRAME_W / width, FRAME_H / height

    for t, path in enumerate(tqdm(frames, desc="render")):
        frame = cv2.imread(str(path))
        m = motion[t] if t < len(motion) else None
        fit = tracker.update([tuple(k) for k in keypoints[t]], motion=None if m is None else np.array(m))
        sources[fit.source] += 1

        if fit.usable:
            warp = fit.homography @ court_from_template
            mask = cv2.warpPerspective(line_mask, warp, (width, height), flags=cv2.INTER_NEAREST)
            frame[mask > 0] = (0.35 * frame[mask > 0] + 0.65 * np.array(LINE_COLOUR)).astype(np.uint8)
        for slot, x, y, conf in keypoints[t]:
            if conf >= CONFIDENCE:
                p = (int(round(x)), int(round(y)))
                cv2.drawMarker(frame, p, KEYPOINT_COLOUR, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
                cv2.putText(frame, f"{KEYPOINT_TO_VERTEX[int(slot)] + 1:02d}", (p[0] + 9, p[1] - 9),
                            font, 0.8, KEYPOINT_COLOUR, 2, cv2.LINE_AA)
        left = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

        side = np.full((FRAME_H, SIDE_W, 3), 30, np.uint8)
        court = side_court.copy()
        if fit.usable:
            # The camera's view: the frame border mapped onto the court.
            border = np.concatenate([
                np.stack([np.linspace(0, width, 40), np.full(40, height)], 1),
                np.stack([np.full(40, width), np.linspace(height, 0, 40)], 1),
                np.stack([np.linspace(width, 0, 40), np.zeros(40)], 1),
                np.stack([np.zeros(40), np.linspace(0, height, 40)], 1)])
            on_court = fit.to_court(border)
            sane = np.all(np.abs(on_court - (2000, 1000)) < 6000, axis=1)
            if sane.sum() > 3:
                poly = cv2.perspectiveTransform(on_court[sane].reshape(-1, 1, 2).astype(np.float32),
                                                side_from_court.astype(np.float32)).astype(np.int32)
                cv2.polylines(court, [poly], True, LINE_COLOUR, 1, cv2.LINE_AA)
            if feet[t]:
                pts = fit.to_court(np.array([(x, y) for x, y, _ in feet[t]]))
                px = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), side_from_court.astype(np.float32)).reshape(-1, 2)
                for (x, y), (_, _, cls) in zip(px, feet[t]):
                    if 0 <= x < court.shape[1] and 0 <= y < court.shape[0]:
                        cv2.circle(court, (int(x), int(y)), 6, PEOPLE[cls][1], -1, cv2.LINE_AA)
                        cv2.circle(court, (int(x), int(y)), 6, (0, 0, 0), 1, cv2.LINE_AA)
        side[20:20 + court.shape[0]] = court

        y = court.shape[0] + 60
        lines = [
            (f"frame {t}", (235, 235, 235)),
            (f"court fit: {fit.source}" + (f", {fit.inliers} inliers" if fit.source == "keypoints" else
                                            f", carried {fit.age} frames" if fit.source == "propagated" else ""),
             (235, 235, 235)),
            (f"model: {args.label}", (235, 235, 235)),
            ("", None),
            ("left: court lines through the fit", LINE_COLOUR),
            (f"left: model keypoints (conf >= {CONFIDENCE}), number = KP", KEYPOINT_COLOUR),
            ("right: feet of cached detections:", (235, 235, 235)),
        ] + [(f"   {name}", colour) for name, colour in PEOPLE.values()]
        for text, colour in lines:
            if text:
                cv2.putText(side, text, (16, y), font, 0.6, colour, 1 if text.startswith("   ") else 2, cv2.LINE_AA)
            y += 30
        writer.write(np.hstack([left, side]))
    writer.release()

    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(args.output)], check=True)
    raw.unlink(missing_ok=True)
    print(f"wrote {args.output}\ncourt fit per frame: {sources}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True, help="name under data/cache/frames/")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--keypoints", type=Path, default=None,
                        help="keypoint cache; read if present, else written after inference")
    parser.add_argument("--motion", type=Path, default=None,
                        help="camera-motion cache; read if present, else measured (and written if given)")
    parser.add_argument("--detections", type=Path, default=None, help="person detection cache (.npz)")
    parser.add_argument("--label", default="v4 RF-DETR keypoints")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.frames_dir = args.frames_dir or ROOT / "data/cache/frames" / args.clip
    args.keypoints = args.keypoints or ROOT / "runs/court_view" / f"{args.clip}_v4_keypoints.json"
    args.output = args.output or ROOT / "runs/court_view" / f"{args.clip}_v4_court_view_h264.mp4"
    render(args)


if __name__ == "__main__":
    main()
