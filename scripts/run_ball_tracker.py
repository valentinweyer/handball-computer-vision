#!/usr/bin/env python
"""Track the ball by anchoring the RF-DETR ball checkpoint and propagating SAM2.

    SAM2_UPSTREAM_DIR=sam2-upstream python -m scripts.run_ball_tracker \\
        --frames data/cache/frames/BHC-FAG_window_cached \\
        --ball-detections outputs/ball_cache/.BHC-FAG_window_cached_ball_v1.npz \\
        --person-detections outputs/team_dataset/.BHC-FAG_window_detections_v1.npz \\
        --output runs/ball_anchored/BHC-FAG_window_cached.json

Ball boxes come from `scripts/cache_ball_detections.py`, which has to run under
the sibling repo's interpreter; everything here runs in this project's venv.

`--compare` additionally scores the two arms this tracker has to beat to be
worth having -- the detector's own boxes, and the same propagation with the gate
switched off -- on the same frames, so the gate's value is measured rather than
assumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# SAM2 has no importable package name until its checkout is on sys.path, and
# `handball_cv.ball.anchored` stays import-clean of it -- but the propagator
# built here needs it, so the guard runs in the entry point, as it does in
# scripts/run_sam2_reprompt_tracker.py.
SAM2_UPSTREAM_DIR = Path(os.getenv("SAM2_UPSTREAM_DIR", PROJECT_ROOT / "sam2-upstream"))
if not SAM2_UPSTREAM_DIR.is_dir():
    raise SystemExit(
        "Needs an external facebookresearch/sam2 checkout. Set SAM2_UPSTREAM_DIR "
        f"or place one at {PROJECT_ROOT / 'sam2-upstream'}."
    )
sys.path.insert(0, str(SAM2_UPSTREAM_DIR))

import cv2  # noqa: E402
import torch  # noqa: E402

from handball_cv.ball.anchored import (  # noqa: E402
    AnchorConfig, Propagated, anchor_candidates, build_tracks, classify_frames,
    fill_path_gaps,
)
from handball_cv.ball.path import (  # noqa: E402
    PathConfig, candidates_from_detections, select_path,
)
from handball_cv.ball.appearance import BallObservation, calibrate  # noqa: E402
from handball_cv.ball.tracking import BallState, BallTrack, TrackPoint  # noqa: E402

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_DEFAULT = str(
    PROJECT_ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
)


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {"offsets": data["offsets"], "boxes": data["boxes"],
                "confidence": data["confidence"], "class_id": data["class_id"]}


def per_frame(cache: dict, n_frames: int) -> dict[int, list[tuple[tuple[float, ...], float]]]:
    """`{frame: [(box, score), ...]}` from a detection cache."""
    out: dict[int, list[tuple[tuple[float, ...], float]]] = {}
    for frame in range(n_frames):
        start, end = int(cache["offsets"][frame]), int(cache["offsets"][frame + 1])
        out[frame] = [
            (tuple(float(v) for v in cache["boxes"][i]), float(cache["confidence"][i]))
            for i in range(start, end)
        ]
    return out


def person_boxes_per_frame(cache: dict, n_frames: int) -> list[list[tuple[float, ...]]]:
    return [[box for box, _score in frame] for frame in
            (per_frame(cache, n_frames)[i] for i in range(n_frames))]


def build_appearance(frame_paths, ball_dets, threshold: float):
    """Calibrate the per-video ball model on this clip's confident sightings.

    The motion pipeline calibrates on its own accepted trajectories because it
    has nothing else. Here there is something better: the detector is high
    precision (0.82 on its own test split), so its confident boxes are ball
    pixels under this clip's lighting and camera -- the same anonymous
    per-video pattern, from a cleaner source. Below `min_samples` `calibrate`
    abstains and the appearance gate is simply absent.
    """
    observations = []
    for frame, detections in ball_dets.items():
        for box, score in detections:
            if score < threshold:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            radius = max(2.0, (box[2] - box[0]) / 2.0)
            image = cv2.imread(str(frame_paths[frame]))
            if image is None:
                continue
            span = int(round(radius * 2))
            x0, y0 = int(max(0, cx - span)), int(max(0, cy - span))
            patch = image[y0:int(cy + span) + 1, x0:int(cx + span) + 1]
            if patch.size:
                observations.append(BallObservation(patch=patch, radius=radius))
    return calibrate(observations)


def make_propagator(frames_dir: Path, checkpoint: str, min_area: float, max_area: float):
    """A `propagate(frame, box)` closure over one SAM2 session.

    The session is built once and `reset_state` re-used per anchor: SAM2's state
    holds the whole clip's image features, and rebuilding it per track would
    re-encode every frame. Propagation is cut off as soon as the mask leaves the
    plausible area band -- the gate would truncate there anyway, and running on
    costs GPU time on a mask already known to be wrong.
    """
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(SAM2_CONFIG, checkpoint)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))

    def propagate(frame: int, box):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=1, box=np.asarray(box, dtype=np.float32)
            )
            for fid, _ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=frame):
                mask = (logits[0] > 0).cpu().numpy().squeeze()
                area = int(mask.sum())
                if not (min_area <= area <= max_area):
                    return
                ys, xs = np.nonzero(mask)
                yield Propagated(frame=int(fid), x=float(xs.mean()),
                                 y=float(ys.mean()), area=float(area))

    return propagate


FLIGHT_COLOUR = (0, 255, 0)
HELD_COLOUR = (0, 200, 255)


def ungated_tracks(ball_dets, propagate, config, n_frames: int) -> list[BallTrack]:
    """The baseline the gate has to beat: propagate and keep everything.

    Deliberately not `build_tracks` with a huge `max_unconfirmed` -- the prefix
    rule also trims unconfirmed tails, and that trimming is itself part of the
    gate, so routing this through it would compare the gate against a slightly
    weaker gate rather than against no gate. The only stop here is the mask
    leaving the plausible area band, which `propagate` applies for cost reasons
    before the gate ever sees it.
    """
    tracks: list[BallTrack] = []
    cursor = -1
    for frame, box, _score in anchor_candidates(
            ball_dets, threshold=config.anchor_threshold):
        if frame <= cursor:
            continue
        points = tuple(
            TrackPoint(frame=o.frame, x=o.x, y=o.y,
                       radius=float(np.sqrt(o.area / np.pi)))
            for o in propagate(frame, box) if o.frame < n_frames
        )
        if not points:
            continue
        tracks.append(BallTrack(points=points))
        cursor = points[-1].frame
    return tracks


def render(frame_paths, frames, ball_dets, output: Path, fps: float,
           anchor_threshold: float) -> None:
    """Draw what the tracker believes, next to what the detector alone saw.

    Both are drawn because the question this clip has to answer is whether
    propagation is adding anything: a green/amber ring with no magenta box near
    it is a frame the detector missed and SAM2 covered -- or a frame SAM2 is
    drifting on, which is exactly what needs eyes.
    """
    from handball_cv.tracking.sam2_driver import ffmpeg_exe
    import subprocess

    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    raw = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for index, path in enumerate(frame_paths):
        image = cv2.imread(str(path))
        for box, score in ball_dets.get(index, ()):
            if score < anchor_threshold:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(image, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (255, 0, 255), 2)
            cv2.putText(image, f"det {score:.2f}", (x1 - 8, y2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        state = frames[index]
        if state.state is not BallState.ABSENT:
            colour = FLIGHT_COLOUR if state.state is BallState.FLIGHT else HELD_COLOUR
            centre = (int(round(state.x)), int(round(state.y)))
            cv2.circle(image, centre, 26, colour, 2)
            cv2.putText(image, state.state.value, (centre[0] - 24, centre[1] - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        else:
            cv2.putText(image, "absent", (16, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (60, 60, 255), 2)
        cv2.putText(image, f"frame {index}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(image)
    writer.release()

    # mp4v does not play in most browsers; the repo's other renders ship h264.
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)], check=True)
    raw.unlink(missing_ok=True)
    print(f"wrote {output}")


def coverage(frames) -> dict:
    states = [f.state for f in frames]
    return {
        "frames": len(states),
        "with_position": sum(1 for s in states if s is not BallState.ABSENT),
        "flight": sum(1 for s in states if s is BallState.FLIGHT),
        "held": sum(1 for s in states if s is BallState.HELD),
        "absent": sum(1 for s in states if s is BallState.ABSENT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--ball-detections", required=True, type=Path)
    parser.add_argument("--person-detections", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument("--anchor-threshold", type=float, default=0.25)
    parser.add_argument("--max-unconfirmed", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=150.0)
    parser.add_argument("--max-area", type=float, default=600.0)
    parser.add_argument("--calibration-threshold", type=float, default=0.5,
                        help="detector score a box must reach to calibrate appearance")
    parser.add_argument("--appearance", action="store_true",
                        help="use the per-video colour model as a second confirmation "
                             "source. OFF by default: measured on BHC-FAG it passes 43%% "
                             "of true balls and 29%% of random court positions, because "
                             "one team wears white and so does the ball. Useful only on "
                             "a clip where the ball's colour is actually distinctive.")
    parser.add_argument("--compare", action="store_true",
                        help="also score the detector alone and the ungated propagation")
    parser.add_argument("--render", type=Path,
                        help="write an annotated h264 mp4 to this path")
    parser.add_argument("--render-ungated", type=Path,
                        help="also render the ungated baseline, for comparison by eye")
    parser.add_argument("--dump-proposals", type=Path,
                        help="write the ungated propagation's positions as {frame: [x, y]}, "
                             "the input to cache_ball_detections.py --proposals")
    parser.add_argument("--crop-detections", type=Path,
                        help="a cache from cache_ball_detections.py --proposals; "
                             "confirms propagated frames but never seeds a track")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--continuous", action="store_true",
                        help="select ONE ball trajectory globally, then bridge its gaps "
                             "by propagating between consecutive sightings, instead of "
                             "accepting each anchor's track on its own merits")
    parser.add_argument("--path-threshold", type=float, default=0.2,
                        help="lowest detection score admitted as a path candidate")
    parser.add_argument("--absent-cost", type=float, default=1.1,
                        help="cost per skipped frame; higher reaches for weaker sightings")
    args = parser.parse_args()

    frame_paths = sorted(args.frames.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frame_paths:
        raise SystemExit(f"no frames in {args.frames}")
    n_frames = len(frame_paths)

    ball_dets = per_frame(load_cache(args.ball_detections), n_frames)
    if args.person_detections:
        persons = person_boxes_per_frame(load_cache(args.person_detections), n_frames)
    else:
        persons = [[] for _ in range(n_frames)]

    config = AnchorConfig(
        anchor_threshold=args.anchor_threshold,
        max_unconfirmed=args.max_unconfirmed,
        min_area=args.min_area,
        max_area=args.max_area,
    )

    appearance = None
    if args.appearance:
        appearance = build_appearance(frame_paths, ball_dets, args.calibration_threshold)
        print(f"appearance model: "
              f"{'calibrated on %d sightings' % appearance.samples if appearance else 'abstained (too few sightings)'}")

    resembles = None
    if appearance is not None:
        cache: dict[int, np.ndarray] = {}

        def resembles(frame: int, x: float, y: float) -> bool:
            image = cache.get(frame)
            if image is None:
                image = cv2.imread(str(frame_paths[frame]))
                cache.clear()
                cache[frame] = image
            radius = float(np.sqrt(config.max_area / np.pi))
            return appearance.resemblance(image, x, y, radius) >= config.min_resemblance

    propagate = make_propagator(args.frames, args.checkpoint,
                                args.min_area, args.max_area)

    if args.dump_proposals:
        # The ungated propagation is a poor tracker and a good proposal set: it
        # covers nearly every frame, and crop detection only needs somewhere to
        # look, not a position it can be held to.
        baseline = ungated_tracks(ball_dets, propagate, config, n_frames)
        proposals = {
            str(point.frame): [round(point.x, 1), round(point.y, 1)]
            for track in baseline for point in track.points
        }
        args.dump_proposals.parent.mkdir(parents=True, exist_ok=True)
        args.dump_proposals.write_text(json.dumps(proposals))
        print(f"wrote {args.dump_proposals}: {len(proposals)} proposals "
              f"over {n_frames} frames")
        return

    crop_dets = None
    if args.crop_detections:
        crop_dets = per_frame(load_cache(args.crop_detections), n_frames)
        found = sum(1 for f in crop_dets.values() if f)
        print(f"crop detections: {found}/{n_frames} frames carry a box")

    if args.continuous:
        sources = [ball_dets] + ([crop_dets] if crop_dets else [])
        candidates = candidates_from_detections(*sources, threshold=args.path_threshold)
        path = select_path(candidates, n_frames,
                           PathConfig(absent_cost=args.absent_cost))
        print(f"one-ball path: {len(path)} sightings on a single reachable trajectory "
              f"out of {sum(len(v) for v in candidates.values())} candidates")
        tracks = fill_path_gaps(path, propagate)
    else:
        tracks = build_tracks(ball_dets, propagate, config=config,
                              resembles=resembles, support_detections=crop_dets,
                              n_frames=n_frames)
    frames = classify_frames(tracks, persons)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "frames_dir": str(args.frames),
        "ball_detections": str(args.ball_detections),
        "config": {
            "anchor_threshold": config.anchor_threshold,
            "max_unconfirmed": config.max_unconfirmed,
            "min_area": config.min_area, "max_area": config.max_area,
            "appearance": bool(appearance),
        },
        "tracks": [
            {"start": t.first_frame, "end": t.last_frame,
             "len": len(t.points), "mean_speed": round(t.mean_speed, 2)}
            for t in tracks
        ],
        "coverage": coverage(frames),
        "positions": {
            str(f.frame): {"state": f.state.value, "x": f.x, "y": f.y, "holder": f.holder}
            for f in frames if f.state is not BallState.ABSENT
        },
    }, indent=1))

    if args.render:
        args.render.parent.mkdir(parents=True, exist_ok=True)
        render(frame_paths, frames, ball_dets, args.render, args.fps,
               config.anchor_threshold)

    if args.render_ungated:
        args.render_ungated.parent.mkdir(parents=True, exist_ok=True)
        baseline = ungated_tracks(ball_dets, propagate, config, n_frames)
        render(frame_paths, classify_frames(baseline, persons), ball_dets,
               args.render_ungated, args.fps, config.anchor_threshold)
        kept = sum(len(t.points) for t in baseline)
        print(f"ungated: {len(baseline)} tracks, {kept}/{n_frames} = {kept/n_frames:.1%}")

    cov = coverage(frames)
    print(f"gated:  {len(tracks)} tracks, "
          f"{cov['with_position']}/{n_frames} = {cov['with_position']/n_frames:.1%} "
          f"(flight {cov['flight']}, held {cov['held']})")
    print(f"wrote {args.output}")

    if args.compare:
        detector_only = sum(
            1 for f in range(n_frames)
            if any(s >= config.anchor_threshold for _b, s in ball_dets[f])
        )
        ungated_frames = sum(
            len(t.points) for t in ungated_tracks(ball_dets, propagate, config, n_frames)
        )
        print("\narm                       frames with a position")
        print(f"  detector alone @{config.anchor_threshold:<5}   "
              f"{detector_only}/{n_frames} = {detector_only/n_frames:.1%}")
        print(f"  propagation, no gate      "
              f"{ungated_frames}/{n_frames} = {ungated_frames/n_frames:.1%}")
        print(f"  propagation, gated        "
              f"{cov['with_position']}/{n_frames} = {cov['with_position']/n_frames:.1%}")


if __name__ == "__main__":
    main()
