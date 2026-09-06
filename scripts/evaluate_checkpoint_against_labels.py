"""One-off diagnostic: how much does defective ground-truth box geometry cost the
already-trained jersey-number detector, independent of retraining?

Runs the trained RF-DETR checkpoint on the images behind the 209-sample jersey-number
audit (see build_jersey_audit_set.py / label_jersey_numbers.py review), and for each
audited ground-truth box, finds the model's own best-matching prediction on that image.
Buckets match quality (IoU) by the human's box_status verdict (complete/untagged,
partial, too_loose, not_number) and by readable/unreadable.

Why this instead of retraining: the audit only covers 209 of ~4545 jersey-number
annotations, far too few to retrain on and trust the result -- any AP delta would be
noise. This needs no training at all: it re-scores an already-trained model's existing
predictions against the human-corrected readability, using the model's own predicted
box as an independent reference for "where the number actually is". If boxes tagged
too_loose/partial have systematically lower matched-IoU than clean ones *despite* the
model clearly detecting something in the right place (nonzero overlap, real
confidence), that's direct evidence the official 0.329 AP is partly deflated by label
geometry, not model quality -- and that fixing geometry alone (no new images, no
resolution) would recover real score.

Not part of the reusable scripts.* package: this is a one-off cross-repo diagnostic
that only runs under the sibling repo's own venv (this repo's venv has no `rfdetr`
package):

    /home/valentinweyer/projects/rfdetr-handball-finetune/.venv/bin/python3 \
        scripts/evaluate_checkpoint_against_labels.py [--limit N]

Read-only against both repos: loads a checkpoint and images, writes one report file
under this repo's runs/ (ignored tree).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import cv2

REPO_ROOT = Path("/home/valentinweyer/projects/handball-computer-vision")
CHECKPOINT = Path(
    "/home/valentinweyer/projects/rfdetr-handball-finetune/runs/"
    "large-handball-v4-w-numbers-no-ball/checkpoint_best_ema.pth"
)
SOURCE_ROOT = Path(
    "/home/valentinweyer/projects/rfdetr-handball-finetune/datasets/handball-v4-w-numbers"
)
AUDIT_DATASET = REPO_ROOT / "runs/jersey_audit/dataset.json"
LABELS_PATH = REPO_ROOT / "data/annotations/jersey/jersey_audit_labels.json"
OUTPUT_PATH = REPO_ROOT / "runs/jersey_audit/checkpoint_vs_labels.json"

JERSEY_NUMBER_CLASS_ID = 3  # rfdetr-handball-finetune/exports/classes.json
INFERENCE_SHAPE = (704, 704)  # matches the checkpoint's training resolution
CONFIDENCE_THRESHOLD = 0.1  # low: we want weak-but-real detections for matching, not a usable operating point


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def match_sample(gt_box: list[float], pred_boxes, pred_scores) -> dict:
    best_iou, best_confidence = 0.0, None
    for pred_box, score in zip(pred_boxes, pred_scores):
        iou = box_iou(gt_box, pred_box.tolist())
        if iou > best_iou:
            best_iou, best_confidence = iou, float(score)
    nearest_center_distance = None
    if len(pred_boxes) and best_iou == 0.0:
        nearest_center_distance = min(
            center_distance(gt_box, pred_box.tolist()) for pred_box in pred_boxes
        )
    return {
        "best_iou": best_iou,
        "matched_confidence": best_confidence,
        "nearest_center_distance": nearest_center_distance,
        "n_predictions_on_image": int(len(pred_boxes)),
    }


def run(limit: int | None) -> list[dict]:
    from rfdetr import RFDETRLarge

    dataset = json.loads(AUDIT_DATASET.read_text())
    labels = json.loads(LABELS_PATH.read_text())["labels"]
    coco_samples = [s for s in dataset["samples"] if s.get("source_kind") == "coco"]

    by_image: dict[tuple[str, str], list[dict]] = {}
    for sample in coco_samples:
        key = (sample["source_split"], sample["source_file_name"])
        by_image.setdefault(key, []).append(sample)
    image_keys = list(by_image)[:limit] if limit else list(by_image)

    print(f"loading checkpoint {CHECKPOINT} ...", flush=True)
    model = RFDETRLarge(pretrain_weights=str(CHECKPOINT))

    results = []
    started = time.monotonic()
    for position, (split, file_name) in enumerate(image_keys, start=1):
        image_path = SOURCE_ROOT / split / file_name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        detections = model.predict(image_rgb, threshold=CONFIDENCE_THRESHOLD, shape=INFERENCE_SHAPE)
        jersey_mask = detections.class_id == JERSEY_NUMBER_CLASS_ID
        pred_boxes = detections.xyxy[jersey_mask]
        pred_scores = detections.confidence[jersey_mask]

        for sample in by_image[(split, file_name)]:
            label = labels.get(str(sample["index"]))
            if label is None:
                continue
            results.append({
                "index": sample["index"],
                "height_band": sample["height_band"],
                "readable_status": label["status"],
                "box_status": label.get("box_status") or "untagged",
                **match_sample(sample["box"], pred_boxes, pred_scores),
            })

        if position % 20 == 0 or position == len(image_keys):
            print(f"[{position}/{len(image_keys)}] images ({time.monotonic()-started:.1f}s elapsed)", flush=True)

    return results


def print_summary(results: list[dict]) -> None:
    by_box_status = defaultdict(list)
    for r in results:
        if r["readable_status"] == "readable":
            by_box_status[r["box_status"]].append(r)

    print("\n=== readable ground truth, matched-IoU by box_status ===")
    print(f"{'box_status':<12}{'n':>5}{'mean IoU':>10}{'median IoU':>12}{'>0.5 IoU':>10}{'any overlap':>13}")
    for status, rows in sorted(by_box_status.items(), key=lambda kv: -len(kv[1])):
        ious = [row["best_iou"] for row in rows]
        n = len(ious)
        print(
            f"{status:<12}{n:>5}{statistics.mean(ious):>10.3f}{statistics.median(ious):>12.3f}"
            f"{100*sum(i>0.5 for i in ious)/n:>9.1f}%{100*sum(i>0 for i in ious)/n:>12.1f}%"
        )

    print("\n=== zero-overlap readable samples: how far away was the nearest prediction? ===")
    for status, rows in sorted(by_box_status.items(), key=lambda kv: -len(kv[1])):
        zero = [row for row in rows if row["best_iou"] == 0.0]
        if not zero:
            continue
        with_pred = [row for row in zero if row["nearest_center_distance"] is not None]
        no_pred = len(zero) - len(with_pred)
        dist_note = (
            f"nearest center dist mean={statistics.mean(r['nearest_center_distance'] for r in with_pred):.1f}px"
            if with_pred else ""
        )
        print(f"  {status:<12} {len(zero)} zero-overlap ({no_pred} had no jersey-number prediction on the image at all) {dist_note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(args.limit)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n{len(results)} matched samples -> {OUTPUT_PATH}")
    print_summary(results)


if __name__ == "__main__":
    main()
