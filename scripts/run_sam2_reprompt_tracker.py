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

The actual predictor/TrackManager loop lives in
`handball_cv.tracking.sam2_driver.drive_sam2`, shared with
`scripts.evaluate_number_pipeline --tracker sam2` -- both need the identical
propagation + periodic reprompting sequence, and this script only adds the
tracking-only concern of flattening it to a (frame_index, tracker_id, box) dump.

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

import numpy as np
import supervision as sv

from handball_cv.teams.model import TeamModel
from handball_cv.tracking.sam2_driver import CHECK_EVERY_DEFAULT, drive_sam2

GOALKEEPER_CLASS_ID = 1

SAM2_CHECKPOINT_DEFAULT = str(
    PROJECT_ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
)


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
    parser.add_argument("--checkpoint-policy", default="reprompt",
                         choices=["reprompt", "reset_reseed"],
                         help="how checkpoint decisions reach the predictor; "
                              "reset_reseed discards SAM2's memory bank each time, "
                              "reproducing the pattern EdgeTAM would force. See "
                              "handball_cv.tracking.sam2_driver.drive_sam2.")
    args = parser.parse_args()

    cache = load_detection_cache(args.detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    frame_cache_dir = args.frame_cache_dir or (
        PROJECT_ROOT / "data/cache/frames" / args.video.stem
    )

    print(f"Tracking with SAM2 + periodic reprompting "
          f"(checkpoint every {args.check_every} frames)...")
    track_manager, seed_boxes, frames = drive_sam2(
        args.video, lambda idx: frame_detections(cache, idx), team_model,
        checkpoint=args.checkpoint, check_every=args.check_every,
        frame_cache_dir=frame_cache_dir, max_frames=args.max_frames,
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        checkpoint_policy=args.checkpoint_policy,
        desc=f"SAM2 {args.checkpoint_policy}",
    )

    per_frame_boxes: dict[int, dict[int, np.ndarray]] = {0: seed_boxes}
    for result in frames:
        per_frame_boxes[result.frame_idx] = dict(zip(result.player_ids, result.boxes))

    print(f"track lifecycle events: {len(track_manager.events)}")
    for e in track_manager.events:
        print(f"  frame {e['frame']:>4}  {e['type']:<16} obj_id={e['obj_id']}")

    events_path = args.output.with_name(args.output.stem + "_events.json")
    events_path.write_text(json.dumps(track_manager.events, indent=2, default=_json_default))
    print("events dumped ->", events_path)

    # ── flatten to (frame_index, tracker_id, box) rows ───────────────────────
    num_frames = max(per_frame_boxes) + 1 if per_frame_boxes else 0
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
