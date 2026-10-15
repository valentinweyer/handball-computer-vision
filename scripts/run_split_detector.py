"""Run the local split detector (person roles + jersey numbers) over a whole clip.

The sibling fine-tuning repo replaced its single four-class RF-DETR with two
models: person-role-v1 finds Goalkeeper/Player/Referee on the full frame, and
jersey-number-v3 finds numbers on a padded square crop of each person -- so a
number can only ever be reported on somebody the detector already found, and
never on the bench, the crowd, or an advertising board.

This module owns how that pair is invoked: the checkpoints, the two inference
resolutions, the crop geometry, and the host/NMS rules. scripts/
compare_split_detectors.py benchmarks it against the old combined checkpoint and
imports from here, not the other way round.

Runs with the SIBLING repo's interpreter, the only one with `rfdetr` (same
constraint as scripts/cache_number_detections.py):

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" \
        scripts/run_split_detector.py --clip FelixClaar

Writes two things per clip, both under runs/split_detector/:

  * <clip>.npz -- every box, in the schema and the project-wide class numbering
    (goalkeeper 1, player 2, referee 3, number 4) that
    render_raw_team_classification.load_detection_cache already reads. Every
    class the detector produced is stored, so a later question never forces a
    re-detection.
  * <clip>.mp4 -- the same boxes drawn on the clip, to actually look at.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SIBLING = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
# The crop geometry, host-coverage rule and cross-crop NMS are the sibling
# repo's: crop_bounds is documented there as "used for both training and
# inference", so a local copy would be a second source of truth for the one
# number that must match training exactly.
sys.path.insert(0, str(SIBLING / "scripts"))
from expansion.number_crops import crop_bounds, intersection_xywh  # noqa: E402
from expansion.number_refresh import NumberDetector, _deduplicate  # noqa: E402
from expansion.role_refresh import RoleDetector  # noqa: E402

ROLE_CHECKPOINT = SIBLING / "runs/person-role-v1-industrial-s42/checkpoint_best_total.pth"
NUMBER_CHECKPOINT = SIBLING / "runs/jn-v3-industrial-s42/checkpoint_best_total.pth"

ROLE_RESOLUTION = 704
# The number model was trained at 704 but only ever sees person crops, which on
# 1080p broadcast are ~210px. Feeding those to a 704 input is a 3.4x upscale that
# invents texture the sensor never recorded, and it measurably manufactures false
# positives: on the held-out crop test split (294 crops, 299 labelled boxes) 448
# scores F1 0.927 / precision 0.924 against 704's 0.918 / 0.903, at 2.1x the
# speed. The role model is the opposite case -- it sees whole 1080p frames, and
# dropping it below 704 loses distant players (F1 0.957 -> 0.948 at 576).
NUMBER_RESOLUTION = 448

# This project's class numbering, as written by cache_number_detections.py.
GOALKEEPER, PLAYER, REFEREE, NUMBER = 1, 2, 3, 4
ROLE_TO_PROJECT = {0: GOALKEEPER, 1: PLAYER, 2: REFEREE}
# Numbers are looked for only inside people who wear them; a referee box hosts
# nothing. Mirrors HOST_CATEGORY_IDS in the sibling's number_refresh.
HOST_CLASSES = (GOALKEEPER, PLAYER)

CROP_PADDING = 0.15   # must equal the dataset build's --padding
HOST_COVERAGE = 0.5   # fraction of a number box that must fall on its host
NMS_IOU = 0.5         # de-duplicate numbers found in overlapping crops

CLASS_COLOR = {GOALKEEPER: (255, 176, 0), PLAYER: (0, 200, 255),
               REFEREE: (255, 0, 200), NUMBER: (0, 255, 80)}
CLASS_NAME = {GOALKEEPER: "GK", PLAYER: "P", REFEREE: "R", NUMBER: "#"}


def build_model(checkpoint: Path, classes: int, resolution: int, device: str,
                half: bool):
    """An RF-DETR built at the resolution it will actually be asked for.

    The sibling repo's detector adapters take a `resolution` but pass it only to
    `predict(shape=...)`, leaving the module itself built at 704. That is
    invisible in fp32, but `inference()` traces and pins the module's own
    resolution, so an fp16 model built at 704 rejects a 448 predict outright.
    Building at the target resolution first is what makes the two settings
    composable; fp16 itself is free, and measured bit-identical on both held-out
    splits (same TP/FP/FN, median IoU within 0.001) for ~1.9x.
    """
    from rfdetr import RFDETRLarge
    import torch

    model = RFDETRLarge(pretrain_weights=str(checkpoint), num_classes=classes,
                        device=device, resolution=resolution)
    if half:
        model.inference(compile=False, dtype=torch.float16)
    return model


def load_detectors(device: str = "cuda", half: bool = True,
                   number_resolution: int = NUMBER_RESOLUTION):
    role = RoleDetector(ROLE_CHECKPOINT, device, ROLE_RESOLUTION)
    role.model = build_model(ROLE_CHECKPOINT, 3, ROLE_RESOLUTION, device, half)
    number = NumberDetector(NUMBER_CHECKPOINT, device, number_resolution)
    number.model = build_model(NUMBER_CHECKPOINT, 1, number_resolution, device, half)
    return role, number


def split_detections(role_detector, number_detector, image: Image.Image,
                     threshold: float, batch_size: int) -> tuple[list[dict], int]:
    """Roles from the full frame, then numbers from a crop of each person.

    Returns boxes in frame coordinates plus how many crops were scored -- the
    second stage costs one forward pass per person, so that count is the price
    of the accuracy the split buys.
    """
    roles = []
    for row in role_detector.predict([image], threshold)[0]:
        roles.append({"class_id": ROLE_TO_PROJECT[int(row["category_id"]) - 1],
                      "bbox": row["bbox"], "score": row["score"]})

    crops, context = [], []
    for host in (row for row in roles if row["class_id"] in HOST_CLASSES):
        bounds = crop_bounds(host["bbox"], image.width, image.height, CROP_PADDING)
        if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
            continue
        crops.append(image.crop(bounds))
        context.append((bounds, host["bbox"]))

    numbers: list[dict] = []
    for start in range(0, len(crops), batch_size):
        chunk = crops[start:start + batch_size]
        for (bounds, host_box), predictions in zip(
            context[start:start + batch_size],
            number_detector.predict(chunk, threshold),
        ):
            for row in predictions:
                x, y, width, height = row["bbox"]
                global_box = [x + bounds[0], y + bounds[1], width, height]
                area = width * height
                if area <= 0:
                    continue
                # A number near a crop edge may really belong to a neighbouring
                # player; keep it only if it sits on THIS host.
                if intersection_xywh(global_box, host_box) / area >= HOST_COVERAGE:
                    numbers.append({"bbox": global_box, "score": row["score"]})

    deduped = [{"class_id": NUMBER, "bbox": row["bbox"], "score": row["score"]}
               for row in _deduplicate(numbers, NMS_IOU)]
    return roles + deduped, len(crops)


def draw(image: Image.Image, rows: list[dict], title: str) -> Image.Image:
    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    for row in rows:
        x, y, width, height = row["bbox"]
        color = CLASS_COLOR[row["class_id"]]
        # Numbers are small; a thin box on a thick one keeps both readable.
        pen.rectangle([x, y, x + width, y + height], outline=color,
                      width=1 if row["class_id"] == NUMBER else 3)
        pen.text((x + 2, max(0, y - 12)),
                 f"{CLASS_NAME[row['class_id']]}{row['score']:.2f}", fill=color)
    pen.rectangle([0, 0, canvas.width, 24], fill=(0, 0, 0))
    pen.text((8, 7), title, fill=(255, 255, 255))
    return canvas


def clip_frame_rate(clip: str, default: float = 25.0) -> float:
    """The source clip's fps, so the render plays at the speed it was shot."""
    for candidate in (ROOT / f"data/raw/{clip}.mp4",
                      ROOT / f"data/raw/{clip}_cached.mp4"):
        if candidate.is_file():
            capture = cv2.VideoCapture(str(candidate))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if fps > 0:
                return float(fps)
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="FelixClaar",
                        help="directory under data/cache/frames")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N frames (0 = the whole clip)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="crops per number-model pass; 16 covers a frame's "
                             "~12 people in one call")
    parser.add_argument("--number-resolution", type=int, default=NUMBER_RESOLUTION)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-video", action="store_true",
                        help="write only the detection npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/split_detector")
    args = parser.parse_args()

    frame_paths = sorted((ROOT / "data/cache/frames" / args.clip).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no cached frames for clip {args.clip!r}")
    if args.limit:
        frame_paths = frame_paths[:args.limit]

    role, number = load_detectors(args.device, not args.no_fp16,
                                  args.number_resolution)
    args.output.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        with Image.open(frame_paths[0]) as probe:
            size = probe.size
        writer = cv2.VideoWriter(str(args.output / f"{args.clip}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 clip_frame_rate(args.clip), size)

    offsets, boxes, scores, class_ids = [0], [], [], []
    crops_total = 0
    started = time.perf_counter()
    for index, path in enumerate(frame_paths):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        rows, crop_count = split_detections(role, number, image,
                                            args.threshold, args.batch_size)
        crops_total += crop_count
        for row in rows:
            x, y, width, height = row["bbox"]
            boxes.append([x, y, x + width, y + height])
            scores.append(row["score"])
            class_ids.append(row["class_id"])
        offsets.append(len(boxes))
        if writer is not None:
            overlay = draw(image, rows, f"{args.clip}  {path.stem}  "
                                        f"{len(rows)} boxes")
            writer.write(cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR))
        if (index + 1) % 100 == 0 or index + 1 == len(frame_paths):
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"{index + 1}/{len(frame_paths)} frames  "
                  f"{rate:.1f} fps  {len(boxes)} boxes", flush=True)
    elapsed = time.perf_counter() - started
    if writer is not None:
        writer.release()

    class_array = np.asarray(class_ids, dtype=int)
    np.savez(
        args.output / f"{args.clip}.npz",
        schema=np.asarray(1),
        detector_id=np.asarray(f"split:{ROLE_CHECKPOINT.name}+{NUMBER_CHECKPOINT.name}"),
        threshold=np.asarray(args.threshold),
        role_resolution=np.asarray(ROLE_RESOLUTION),
        number_resolution=np.asarray(args.number_resolution),
        fp16=np.asarray(not args.no_fp16),
        stride=np.asarray(1),
        scored_frames=np.asarray(len(frame_paths)),
        total_frames=np.asarray(len(frame_paths)),
        offsets=np.asarray(offsets, dtype=int),
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        confidence=np.asarray(scores, dtype=float),
        class_id=class_array,
    )

    counted = {name: int((class_array == class_id).sum()) for class_id, name in
               ((GOALKEEPER, "goalkeeper"), (PLAYER, "player"),
                (REFEREE, "referee"), (NUMBER, "number"))}
    frames = len(frame_paths)
    print(f"\n{frames} frames in {elapsed:.1f}s "
          f"({frames / elapsed:.1f} fps, {elapsed / frames * 1000:.1f} ms/frame)")
    print(f"boxes {counted}  ({crops_total / frames:.1f} crops/frame)")
    print(f"numbers per frame: {counted['number'] / frames:.2f}")
    print(f"-> {args.output / f'{args.clip}.npz'}")
    if writer is not None:
        print(f"-> {args.output / f'{args.clip}.mp4'}")


if __name__ == "__main__":
    main()
