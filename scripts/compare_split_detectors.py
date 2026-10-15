"""A/B the new split detectors against the old combined checkpoint on real clip frames.

The sibling fine-tuning repo replaced one four-class RF-DETR ("Goalkeeper, Player,
Referee, jersey number", all found in a single full-frame pass) with two models:

  * person-role-v1 -- three classes on the full frame, and
  * jersey-number-v3 -- ONE class, run on a padded square crop of each
    Goalkeeper/Player box rather than on the frame.

That second model is the structural change: numbers are no longer detected at
1/3 of frame scale among 10k-pixel-wide advertising boards, they are detected
inside a ~190px crop that is already known to contain a person. The sibling
repo measured both on its own held-out split; this script asks the different
question -- what changes on this project's real broadcast frames, which are not
in any split, at the operating point this project actually uses.

Runs with the SIBLING repo's interpreter, which is the only one with `rfdetr`
(same constraint as scripts/cache_number_detections.py):

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" \
        scripts/compare_split_detectors.py --clip Melsungen_window --frames 12

Read-only against both repos except for its own output directory under runs/.
Boxes are reported in this project's class numbering (goalkeeper 1, player 2,
referee 3, number 4), matching cache_number_detections.py, so the two detector
generations are directly comparable and neither leaks its own class indices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIBLING = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
sys.path.insert(0, str(ROOT))          # so `scripts.` resolves when run by path
sys.path.insert(0, str(SIBLING / "scripts"))
from expansion.number_crops import intersection_xywh  # noqa: E402

# The split detector's own module owns how that pair is invoked -- checkpoints,
# resolutions, crop geometry, host and NMS rules. This script is a consumer of
# it, so it imports rather than restating any of that.
from scripts.run_split_detector import (  # noqa: E402
    GOALKEEPER, HOST_CLASSES, HOST_COVERAGE, NUMBER, NUMBER_RESOLUTION,
    PLAYER, REFEREE, ROLE_CHECKPOINT, ROLE_RESOLUTION, NUMBER_CHECKPOINT,
    build_model, draw, load_detectors, split_detections,
)

COMBINED_CHECKPOINT = SIBLING / "runs/large-handball-v4-w-numbers-no-ball/checkpoint_best_ema.pth"
COMBINED_TO_PROJECT = {0: GOALKEEPER, 1: PLAYER, 2: REFEREE, 3: NUMBER}


def combined_detections(model, image: Image.Image, threshold: float) -> list[dict]:
    """Every box the old four-class checkpoint finds in one full-frame pass."""
    result = model.predict(
        image, threshold=threshold, shape=(ROLE_RESOLUTION, ROLE_RESOLUTION),
        include_source_image=False
    )
    rows = []
    for box, score, class_id in zip(result.xyxy, result.confidence, result.class_id):
        project_class = COMBINED_TO_PROJECT.get(int(class_id))
        if project_class is None:  # the no-object sentinel index
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 > x1 and y2 > y1:
            rows.append({"class_id": project_class,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(score)})
    return rows


def hosted_numbers(rows: list[dict]) -> tuple[int, int]:
    """Split number boxes into those sitting on a detected person, and the rest.

    The split detector can only ever produce the first kind -- it never looks
    anywhere but inside a person crop. The combined detector searched the whole
    frame, so its extra boxes land on the bench, the crowd and the advertising
    boards. Comparing raw number counts therefore compares two different
    questions; comparing hosted counts compares the same one, and the unhosted
    count is the junk this project already pays a player-overlap filter to drop.
    """
    hosts = [row["bbox"] for row in rows if row["class_id"] in HOST_CLASSES]
    hosted = 0
    for row in rows:
        if row["class_id"] != NUMBER:
            continue
        area = row["bbox"][2] * row["bbox"][3]
        if area > 0 and any(
            intersection_xywh(row["bbox"], host) / area >= HOST_COVERAGE
            for host in hosts
        ):
            hosted += 1
    total = sum(row["class_id"] == NUMBER for row in rows)
    return hosted, total - hosted


def counts(rows: list[dict]) -> dict[str, int]:
    hosted, unhosted = hosted_numbers(rows)
    return {name: sum(row["class_id"] == class_id for row in rows)
            for class_id, name in
            ((GOALKEEPER, "goalkeeper"), (PLAYER, "player"),
             (REFEREE, "referee"), (NUMBER, "number"))
            } | {"number_hosted": hosted, "number_unhosted": unhosted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="Melsungen_window",
                        help="directory under data/cache/frames")
    parser.add_argument("--frames", type=int, default=12,
                        help="how many frames to sample evenly across the clip")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="matches cache_number_detections.py's default")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="crops per number-model forward pass; 16 covers a "
                             "whole frame's ~12 people in one call")
    parser.add_argument("--number-resolution", type=int, default=NUMBER_RESOLUTION,
                        help="input size for the crop model (must divide by 64)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="run fp32, reproducing the pre-optimisation timings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/split_detector_compare")
    args = parser.parse_args()

    frame_dir = ROOT / "data/cache/frames" / args.clip
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no cached frames in {frame_dir}")
    picks = [frame_paths[i] for i in
             np.linspace(0, len(frame_paths) - 1, args.frames, dtype=int)]

    half = not args.no_fp16
    print(f"loading combined {COMBINED_CHECKPOINT.name} @{ROLE_RESOLUTION}", flush=True)
    combined = build_model(COMBINED_CHECKPOINT, 4, ROLE_RESOLUTION, args.device, half)
    print(f"loading role     {ROLE_CHECKPOINT.name} @{ROLE_RESOLUTION}", flush=True)
    print(f"loading number   {NUMBER_CHECKPOINT.name} @{args.number_resolution}", flush=True)
    role, number = load_detectors(args.device, half, args.number_resolution)

    render_dir = args.output / args.clip
    render_dir.mkdir(parents=True, exist_ok=True)
    per_frame = []
    for position, path in enumerate(picks, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")

        start = time.perf_counter()
        old_rows = combined_detections(combined, image, args.threshold)
        old_seconds = time.perf_counter() - start

        start = time.perf_counter()
        new_rows, crop_count = split_detections(
            role, number, image, args.threshold, args.batch_size)
        new_seconds = time.perf_counter() - start

        side_by_side = Image.new("RGB", (image.width, image.height * 2))
        side_by_side.paste(draw(image, old_rows, f"OLD combined  {path.name}"), (0, 0))
        side_by_side.paste(
            draw(image, new_rows, f"NEW split ({crop_count} crops)  {path.name}"),
            (0, image.height))
        side_by_side.save(render_dir / f"{path.stem}.jpg", quality=88)

        old_counts, new_counts = counts(old_rows), counts(new_rows)
        # Every box both generations produced is persisted, not just the counts:
        # this pass costs GPU minutes, and the interesting questions (which
        # numbers did one find that the other missed, how tight are the boxes)
        # all need the geometry, which a recount cannot recover.
        per_frame.append({"frame": path.name,
                          "old": old_counts, "new": new_counts,
                          "crops_scored": crop_count,
                          "old_seconds": old_seconds, "new_seconds": new_seconds,
                          "old_boxes": old_rows, "new_boxes": new_rows})
        print(f"[{position}/{len(picks)}] {path.name}  "
              f"old {old_counts}  {old_seconds:.2f}s | "
              f"new {new_counts}  {new_seconds:.2f}s ({crop_count} crops)",
              flush=True)

    summary = {
        "clip": args.clip,
        "frames_scored": len(per_frame),
        "threshold": args.threshold,
        "role_resolution": ROLE_RESOLUTION,
        "number_resolution": args.number_resolution,
        "fp16": not args.no_fp16,
        "combined_checkpoint": str(COMBINED_CHECKPOINT),
        "role_checkpoint": str(ROLE_CHECKPOINT),
        "number_checkpoint": str(NUMBER_CHECKPOINT),
        "totals": {
            generation: {name: sum(row[generation][name] for row in per_frame)
                         for name in ("goalkeeper", "player", "referee", "number",
                                      "number_hosted", "number_unhosted")}
            for generation in ("old", "new")
        },
        "seconds_per_frame": {
            generation: sum(row[f"{generation}_seconds"] for row in per_frame) / len(per_frame)
            for generation in ("old", "new")
        },
        "crops_per_frame": sum(row["crops_scored"] for row in per_frame) / len(per_frame),
        "per_frame": per_frame,
    }
    report = args.output / f"{args.clip}.json"
    report.write_text(json.dumps(summary, indent=2))
    print(f"\ntotals  old {summary['totals']['old']}")
    print(f"totals  new {summary['totals']['new']}")
    print(f"seconds/frame {summary['seconds_per_frame']}")
    print(f"renders -> {render_dir}\nreport  -> {report}")


if __name__ == "__main__":
    main()
