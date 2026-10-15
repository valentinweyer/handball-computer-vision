#!/usr/bin/env python
"""Cache local RF-DETR ball boxes for a clip, one npz, same schema as the other
detection caches this project uses.

Separate pass for the same reason `cache_number_detections.py` is one: `rfdetr`
is installed only in the sibling fine-tuning project's venv, not this one, so
ball detection cannot share a process with tracking (SAM2 and the rest live
here). Run it with the SIBLING repo's interpreter:

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" \\
        scripts/cache_ball_detections.py \\
        --frames data/cache/frames/BHC-FAG_window_cached \\
        --output outputs/ball_cache/.BHC-FAG_window_cached_ball_v1.npz

Reads the JPEG frame cache rather than the mp4, so the boxes line up frame for
frame with what the SAM2 path propagates over -- a detection pass that decoded
the video instead could land a frame off and would be silently wrong.

Every box is stored down to `--threshold` (default 0.20), well below any
threshold the tracker uses. Choosing an operating point is a read-time decision
and this pass is the expensive one: re-running it because the cache was written
at 0.5 and the answer turned out to be 0.25 costs a full GPU pass for nothing.

Crop mode
---------
With `--proposals`, the same checkpoint runs on a square crop around each
proposed position instead of on the whole frame:

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" \\
        scripts/cache_ball_detections.py \\
        --frames data/cache/frames/BHC-FAG_window_cached \\
        --proposals runs/ball_anchored/BHC-FAG_proposals.json \\
        --output outputs/ball_cache/.BHC-FAG_window_cached_ball_crop_v1.npz

This is the resolution fix, not a different model. RF-DETR Large resizes its
input to a 704 square, so from 1920x1080 a 23 px ball arrives 8.4 px wide --
narrower than the backbone's 16 px patch. On a 352 px crop the same ball
arrives 46 px wide. `compare_split_detectors.py` measured this structural
change for jersey numbers (hosted 27 -> 38, false unhosted 10 -> 0, same
speed); the ball has the same problem and more headroom.

The crop is square so the 704 resize does not distort it, and emitted boxes are
translated back to full-frame coordinates, so the npz is interchangeable with a
full-frame one and nothing downstream needs to know which pass produced it.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Separate checkout, so the default is "next to this one"; RFDETR_FINETUNE_DIR
# overrides it (see .env.example).
FINETUNE_ROOT = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
DEFAULT_CHECKPOINT = FINETUNE_ROOT / "runs/ball-v1-s42/checkpoint_best_total.pth"

# `ball-v1` is a single-class checkpoint: Ball is index 0 there. We write this
# project's own id so consumers never need to know which model produced the
# cache, matching what cache_number_detections.py does for jersey numbers.
BALL_CLASS_ID = 5


def _crop(image, centre, side: int):
    """Square crop of `side` px around `centre`, clamped inside the frame.

    Clamped rather than padded, so the crop is always real pixels at real
    scale: a padded edge crop would put a black border through the resize and
    change the apparent size of a ball near the touchline.
    """
    height, width = image.shape[:2]
    side = min(side, height, width)
    half = side // 2
    x0 = int(min(max(0, round(centre[0]) - half), width - side))
    y0 = int(min(max(0, round(centre[1]) - half), height - side))
    return image[y0:y0 + side, x0:x0 + side], (x0, y0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True, type=Path,
                        help="directory of numbered JPEGs (the SAM2 frame cache)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--proposals", type=Path,
                        help="JSON {frame: [x, y]} of positions to look around; "
                             "switches this pass from full-frame to crop mode")
    parser.add_argument("--crop", type=int, default=352,
                        help="side of the square crop in crop mode (default 352: "
                             "a 23px ball reaches the 704 input at 46px)")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    files = sorted(args.frames.glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(pretrain_weights=str(args.checkpoint))

    proposals: dict[int, tuple[float, float]] = {}
    if args.proposals:
        import json
        proposals = {
            int(frame): (float(xy[0]), float(xy[1]))
            for frame, xy in json.loads(args.proposals.read_text()).items()
        }
        print(f"crop mode: {len(proposals)} proposals, {args.crop}px square crops")

    boxes: list[list[float]] = []
    scores: list[float] = []
    offsets = [0]
    for index, path in enumerate(files):
        bgr = None
        origin = (0, 0)
        if args.proposals:
            # No proposal means nowhere to look. A full-frame pass on that frame
            # would be a different measurement silently mixed into this cache.
            if index in proposals:
                bgr = cv2.imread(str(path))
                if bgr is not None:
                    bgr, origin = _crop(bgr, proposals[index], args.crop)
        else:
            bgr = cv2.imread(str(path))

        if bgr is None:
            offsets.append(len(boxes))
            continue
        result = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                               threshold=args.threshold)
        for box, score in zip(result.xyxy, result.confidence):
            boxes.append([float(box[0]) + origin[0], float(box[1]) + origin[1],
                          float(box[2]) + origin[0], float(box[3]) + origin[1]])
            scores.append(float(score))
        offsets.append(len(boxes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Same key names and shapes as the player-detection caches, so
    # `run_sam2_reprompt_tracker.load_detection_cache` reads this unchanged.
    np.savez(
        args.output,
        schema=np.int64(1),
        detector_id=str(args.checkpoint.parent.name),
        total_frames=np.int64(len(files)),
        offsets=np.asarray(offsets, dtype=np.int64),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.full(len(boxes), BALL_CLASS_ID, dtype=np.int16),
        threshold=np.float32(args.threshold),
    )
    hit = sum(1 for i in range(len(files)) if offsets[i + 1] > offsets[i])
    print(f"wrote {args.output}: {len(boxes)} boxes over {len(files)} frames "
          f"({hit} frames with a detection at >={args.threshold})")


if __name__ == "__main__":
    main()
