"""Cache local RF-DETR jersey-number boxes for a video, one npz, same schema as the
player-detection caches this project already uses.

This exists as a separate pass because `rfdetr` is installed only in the sibling
fine-tuning project's venv, not this one, so number detection cannot run in the same
process as tracking (McByte/SAM/Cutie/EasyOCR all live here). Caching also makes the
detector pass reusable and keeps the measurement loop fast and deterministic.

Run with the SIBLING repo's interpreter, which has rfdetr:

    /home/valentinweyer/projects/rfdetr-handball-finetune/.venv/bin/python3 \
        scripts/cache_number_detections.py data/raw/FelixClaar.mp4 \
        --output outputs/number_cache/.FelixClaar_numbers_v1.npz

Full 90-minute broadcasts are 270k frames, which is far more than a sampled
evaluation set needs, so `--stride N` scores every Nth frame and grabs the rest
without decoding. The stride is stored in the npz because a skipped frame and a
genuinely empty frame look identical in `offsets`.

The emitted npz is readable by scripts.render_raw_team_classification.load_detection_cache,
so downstream code treats it exactly like any other cached detection set. Every stored
box is a jersey number; class_id is written as the *project-wide* NUMBER_CLASS_ID rather
than the checkpoint's own index, so consumers don't need to know which model produced it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/home/valentinweyer/projects/rfdetr-handball-finetune/runs/"
    "large-handball-v4-w-numbers-no-ball/checkpoint_best_ema.pth"
)
# Index inside the fine-tuned checkpoint (exports/classes.json in the sibling repo),
# NOT the hosted Roboflow model's class numbering.
CHECKPOINT_JERSEY_CLASS_ID = 3
# What we write into the cache: the id the rest of this project means by "number box".
NUMBER_CLASS_ID = 4
INFERENCE_SHAPE = (704, 704)  # the checkpoint's training resolution


def detect_all_frames(
    video: Path,
    checkpoint: Path,
    threshold: float,
    shape: tuple[int, int],
    stride: int = 1,
) -> dict:
    """Detect number boxes, optionally on every `stride`-th frame only.

    A full 90-minute broadcast is 270k frames, far more than a sampled labeling
    set needs, so `stride` trades coverage for wall time: skipped frames are
    grabbed without decoding and simply carry no boxes. `offsets` still spans
    every frame, so the cache stays indexable exactly as before -- but a skipped
    frame is therefore indistinguishable from a genuinely empty one, which is why
    the stride is recorded in the npz.
    """
    from rfdetr import RFDETRLarge

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"loading {checkpoint}", flush=True)
    model = RFDETRLarge(pretrain_weights=str(checkpoint))

    offsets = [0]
    boxes: list[np.ndarray] = []
    confidence: list[float] = []
    scored = 0
    try:
        for frame_index in range(total):
            if frame_index % stride:
                # grab() advances without decoding -- the point of striding
                if not capture.grab():
                    break
                offsets.append(len(boxes))
                continue
            ok, frame_bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb, threshold=threshold, shape=shape)
            keep = detections.class_id == CHECKPOINT_JERSEY_CLASS_ID
            frame_boxes = detections.xyxy[keep]
            frame_scores = detections.confidence[keep]
            for box, score in zip(frame_boxes, frame_scores):
                boxes.append(np.asarray(box, dtype=float))
                confidence.append(float(score))
            offsets.append(len(boxes))
            scored += 1
            if scored % 25 == 0:
                print(
                    f"  [{frame_index+1}/{total}] {scored} scored, "
                    f"{len(boxes)} boxes so far",
                    flush=True,
                )
    finally:
        capture.release()

    # offsets must describe every frame in the video, so a consumer can index any frame
    # without bounds-checking against a short cache.
    while len(offsets) < total + 1:
        offsets.append(len(boxes))

    return {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "boxes": (
            np.stack(boxes) if boxes else np.zeros((0, 4), dtype=float)
        ),
        "confidence": np.asarray(confidence, dtype=float),
        "class_id": np.full(len(boxes), NUMBER_CLASS_ID, dtype=np.int64),
        "stride": np.asarray(stride),
        "scored_frames": np.asarray(scored),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="score every Nth frame only; skipped frames carry no boxes "
        "(default 1 = every frame, the original behaviour)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = detect_all_frames(
        args.video.resolve(),
        args.checkpoint,
        args.threshold,
        INFERENCE_SHAPE,
        args.stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema=np.asarray(1),
        detector_id=np.asarray(str(args.checkpoint)),
        threshold=np.asarray(args.threshold),
        total_frames=np.asarray(len(cache["offsets"]) - 1),
        **cache,
    )
    frames = len(cache["offsets"]) - 1
    scored = int(cache["scored_frames"])
    print(
        f"\n{len(cache['boxes'])} number boxes over {scored} scored frames "
        f"({len(cache['boxes'])/max(scored,1):.2f}/frame) "
        f"spanning {frames} frames at stride {args.stride} -> {args.output}"
    )


if __name__ == "__main__":
    main()
