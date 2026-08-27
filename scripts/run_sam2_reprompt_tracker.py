"""Drive the real SAM2 video predictor + TrackManager reprompting policy end
to end on a cached-detection clip, dumping per-frame (frame_index,
tracker_id, box) triples for scripts.evaluate_tracker_identity to replay.

Why this script exists: docs/tracking-evaluation.md measured two SAM2-adjacent
configurations -- MCByte's box association merely nudged by SAM/Cutie masks
(worst of seven trackers), and naive seed-once mask propagation with no
re-detection (checked only by a drift-consistency proxy, not the project's
rigorous per-frame scorer). Neither is "SAM2 as primary tracker" in the sense
the mask-memory-does-the-association design actually means. The configuration
closest to that -- and to src/handball_cv/tracking/sam2_manager.TrackManager's
own purpose -- adds periodic detector-checkpoint reprompting to fix the
lifecycle gaps (new players, long-horizon drift) naive propagation has. That
configuration has never been run through evaluate_tracker_identity.py. This
script produces the dump that lets it be.

Uses the SAME cached detections every other tracker in evaluate_tracker_identity
consumes -- for both frame-0 seeding and the periodic checkpoint reprompt
source -- so SAM2 is not being compared against a stronger (live) detection
input than the box trackers get.

No homography/court filtering here (unlike experiments/sam2_baseline/run_pipeline.py,
which this is adapted from): court_test_fn always accepts, matching the box
trackers, which also add every unmatched detection as a new track without a
court-membership check.

Usage:
    SAM2_UPSTREAM_DIR=sam2-upstream python -m scripts.run_sam2_reprompt_tracker \\
        data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --team-model outputs/team_comparison/.FelixClaar_team.pkl \\
        --output runs/sam2_reprompt/FelixClaar.npz
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM2_UPSTREAM_DIR = Path(os.getenv("SAM2_UPSTREAM_DIR", PROJECT_ROOT / "sam2-upstream"))
if not SAM2_UPSTREAM_DIR.is_dir():
    raise FileNotFoundError(
        "Needs an external facebookresearch/sam2 checkout. Set SAM2_UPSTREAM_DIR "
        "to its path before running this module."
    )
# Must run before any other import -- see the identical guard in
# scripts/evaluate_tracker_identity.py and experiments/sam2_baseline/run_pipeline.py.
sys.path.insert(0, str(SAM2_UPSTREAM_DIR))

import shutil

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from handball_cv.teams.model import TeamModel
from handball_cv.tracking.sam2_manager import TrackManager
from sam2.build_sam import build_sam2_video_predictor


def _ffmpeg_exe() -> str:
    """Bare `ffmpeg` if on PATH, else the imageio-ffmpeg bundled static binary.

    This host has no system ffmpeg and no passwordless sudo to install one;
    imageio-ffmpeg ships a working static binary as a plain pip package.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

GOALKEEPER_CLASS_ID = 1

SAM2_CHECKPOINT_DEFAULT = str(
    PROJECT_ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECK_EVERY_DEFAULT = 10  # matches run_pipeline.py's detector-checkpoint cadence


def load_detection_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "offsets": data["offsets"],
            "boxes": data["boxes"],
            "confidence": data["confidence"],
            "class_id": data["class_id"],
        }


def frame_detections(cache: dict, frame_index: int) -> sv.Detections:
    start, end = cache["offsets"][frame_index:frame_index + 2]
    start, end = int(start), int(end)
    return sv.Detections(
        xyxy=cache["boxes"][start:end].copy(),
        confidence=cache["confidence"][start:end].copy(),
        class_id=cache["class_id"][start:end].astype(int, copy=True),
    )


def masks_from_logits(mask_logits: torch.Tensor) -> np.ndarray:
    """(N, 1, H, W) logits -> (N, H, W) bool, edge-fragment filtered."""
    masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
    return np.array([
        sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
        for m in masks
    ])


def _json_default(value):
    """Event dicts may carry numpy scalars (obj_id/frame from array iteration)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path,
                         help="existing TeamModel.save() cache; not refit here")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY_DEFAULT)
    parser.add_argument("--frame-cache-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="truncate to the first N frames, for a quick smoke test")
    args = parser.parse_args()

    cache = load_detection_cache(args.detections)

    frame_cache_dir = args.frame_cache_dir or (
        PROJECT_ROOT / "data/cache/frames" / args.video.stem
    )
    if not frame_cache_dir.exists() or not any(frame_cache_dir.glob("*.jpg")):
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(args.video),
             "-q:v", "2", "-start_number", "0", str(frame_cache_dir / "%05d.jpg")],
            check=True,
        )
    frame_files = sorted(frame_cache_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    num_frames = len(frame_files)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    def read_frame(idx: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(frame_files[idx])), cv2.COLOR_BGR2RGB)

    team_model = TeamModel.load(args.team_model, device=args.device)
    predictor = build_sam2_video_predictor(SAM2_CONFIG, args.checkpoint)

    frame0 = read_frame(0)
    det0 = frame_detections(cache, 0)
    if len(det0) == 0:
        raise RuntimeError("no cached detections on frame 0")

    track_manager = TrackManager(team_model, court_test_fn=lambda box: True)
    is_gk0 = det0.class_id == GOALKEEPER_CLASS_ID
    obj_ids0 = track_manager.seed(0, det0.xyxy, frame0, is_gk0)

    state = predictor.init_state(video_path=str(frame_cache_dir))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for oid, xyxy in zip(obj_ids0, det0.xyxy):
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=oid, box=np.asarray(xyxy, dtype=np.float32)
            )

    per_frame_boxes: dict[int, dict[int, np.ndarray]] = {
        0: dict(zip(obj_ids0, det0.xyxy))
    }

    next_new_fid = 1
    last_masks_by_id: dict[int, np.ndarray] = {}
    last_frame = frame0

    print(f"Tracking {num_frames} frames with SAM2 + periodic reprompting "
          f"(checkpoint every {args.check_every} frames)...")
    pbar = tqdm(total=num_frames - 1, desc="SAM2 reprompt")
    t = 0
    while t < num_frames - 1:
        chunk_len = min(args.check_every, num_frames - 1 - t)
        chunk_end = t

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for fid, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=t, max_frame_num_to_track=chunk_len):
                if fid < next_new_fid or fid >= num_frames:
                    continue  # repeated boundary frame, or past our truncation

                masks = masks_from_logits(mask_logits)
                track_manager.update_from_propagation(fid, obj_ids, masks)

                boxes_this = sv.mask_to_xyxy(masks=masks)
                per_frame_boxes[fid] = dict(zip(obj_ids, boxes_this))

                chunk_end = fid
                if fid == t + chunk_len:
                    last_masks_by_id = dict(zip(obj_ids, masks))
                    last_frame = read_frame(fid)
                pbar.update(1)

        next_new_fid = chunk_end + 1
        t = chunk_end
        if t >= num_frames - 1:
            break

        # ── detector checkpoint: decide add / remove / reprompt ──
        live_obj_ids = list(last_masks_by_id.keys())
        live_masks = (
            np.array([last_masks_by_id[oid] for oid in live_obj_ids])
            if live_obj_ids else np.empty((0, 0, 0), dtype=bool)
        )
        det = frame_detections(cache, chunk_end)
        det_is_goalkeeper = det.class_id == GOALKEEPER_CLASS_ID

        actions = track_manager.checkpoint(
            chunk_end, last_frame, live_obj_ids, live_masks, det.xyxy, det_is_goalkeeper
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for action in actions:
                if action["type"] == "remove":
                    predictor.remove_object(state, obj_id=action["obj_id"])
                elif action["type"] == "reprompt":
                    predictor.add_new_points_or_box(
                        state, frame_idx=chunk_end, obj_id=action["obj_id"],
                        box=np.asarray(action["box"], dtype=np.float32),
                        clear_old_points=True,
                    )
                elif action["type"] == "add":
                    predictor.add_new_points_or_box(
                        state, frame_idx=chunk_end, obj_id=action["obj_id"],
                        box=np.asarray(action["box"], dtype=np.float32),
                    )
                elif action["type"] == "reset":
                    # Body-swap, not drift: memory is contaminated with the
                    # wrong player's appearance, so reprompting in place would
                    # seed the "correction" from that wrong mask. Tear the
                    # object down and re-add it fresh under the same obj_id.
                    if len(state["obj_id_to_idx"]) > 1:
                        predictor.remove_object(state, obj_id=action["obj_id"])
                        predictor.add_new_points_or_box(
                            state, frame_idx=chunk_end, obj_id=action["obj_id"],
                            box=np.asarray(action["box"], dtype=np.float32),
                        )
                    else:
                        # Removing the only live object resets the whole
                        # session (see SAM2VideoPredictor.remove_object) --
                        # fall back to an in-place reprompt instead.
                        predictor.add_new_points_or_box(
                            state, frame_idx=chunk_end, obj_id=action["obj_id"],
                            box=np.asarray(action["box"], dtype=np.float32),
                            clear_old_points=True,
                        )
    pbar.close()

    print(f"track lifecycle events: {len(track_manager.events)}")
    for e in track_manager.events:
        print(f"  frame {e['frame']:>4}  {e['type']:<16} obj_id={e['obj_id']}")

    events_path = args.output.with_name(args.output.stem + "_events.json")
    events_path.write_text(json.dumps(track_manager.events, indent=2, default=_json_default))
    print("events dumped ->", events_path)

    # ── flatten to (frame_index, tracker_id, box) rows ───────────────────────
    rows_frame, rows_id, rows_box = [], [], []
    for fid in range(num_frames):
        for oid, box in per_frame_boxes.get(fid, {}).items():
            rows_frame.append(fid)
            rows_id.append(int(oid))
            rows_box.append(box)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_index=np.array(rows_frame, dtype=int),
        tracker_id=np.array(rows_id, dtype=int),
        boxes=np.array(rows_box, dtype=float).reshape(-1, 4),
        source=str(args.video),
    )
    print("dumped ->", args.output)


if __name__ == "__main__":
    main()
