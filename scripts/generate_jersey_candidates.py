"""SAM3 jersey-number region proposals over the images sampled for the audit set.

Runs the recipe already validated in `notebooks/SAM3_jersey_numbers.ipynb` (cell 22):
text-prompted SAM3 ("jersey number"), NMS, a dynamically SAM3-detected "scoreboard"
distractor filter (scoreboard has no ground-truth class in this dataset, so it must be
detected), and a ground-truth referee-box filter (referee *is* a real COCO class here,
so the existing annotation is more reliable than re-detecting it).

Every surviving SAM3 box is matched by IoU against this image's existing jersey-number
COCO annotations. A box with no match is a genuinely new proposal -- a candidate for a
number the source dataset missed. An existing annotation with no matching SAM3 box is
the opposite signal: a recall gap, or a number small/occluded enough that SAM3 also
can't find it.

This only reads the RF-DETR training COCO export; nothing under it is modified. Reads
runs/jersey_audit/dataset.json (see build_jersey_audit_set.py) to know which images to
run on, and writes runs/jersey_audit/sam3_candidates.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from PIL import Image
import torch
import torchvision

from scripts.build_jersey_audit_set import DEFAULT_SOURCE_ROOT, SPLITS
from scripts.label_jersey_numbers import utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DATASET = ROOT / "runs/jersey_audit/dataset.json"
DEFAULT_OUTPUT = ROOT / "runs/jersey_audit/sam3_candidates.json"

CATEGORY_NAME = "jersey number"
REFEREE_CATEGORY_NAME = "Referee"
JERSEY_NUMBER_PROMPT = "jersey number"
SCOREBOARD_PROMPT = "scoreboard"

CONFIDENCE_THRESHOLD = 0.4
SCOREBOARD_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.3


def is_center_inside_any(box: list[float], target_boxes: list[list[float]]) -> bool:
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return any(
        tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2
        for tx1, ty1, tx2, ty2 in target_boxes
    )


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


def load_coco_index(source_root: Path) -> dict[str, dict[int, dict]]:
    """split -> {image_id: {file_name, jersey: [(annotation_id, box)], referee: [box]}}."""
    index: dict[str, dict[int, dict]] = {}
    for split in SPLITS:
        coco_path = source_root / split / "_annotations.coco.json"
        if not coco_path.is_file():
            continue
        coco = json.loads(coco_path.read_text())
        category_id_by_name = {item["name"]: item["id"] for item in coco["categories"]}
        jersey_id = category_id_by_name.get(CATEGORY_NAME)
        referee_id = category_id_by_name.get(REFEREE_CATEGORY_NAME)
        by_image = {
            image["id"]: {"file_name": image["file_name"], "jersey": [], "referee": []}
            for image in coco["images"]
        }
        for annotation in coco["annotations"]:
            entry = by_image.get(annotation["image_id"])
            if entry is None:
                continue
            x, y, w, h = annotation["bbox"]
            box = [x, y, x + w, y + h]
            if jersey_id is not None and annotation["category_id"] == jersey_id:
                entry["jersey"].append((annotation["id"], box))
            elif referee_id is not None and annotation["category_id"] == referee_id:
                entry["referee"].append(box)
        index[split] = by_image
    return index


def distinct_images(audit_dataset: dict) -> list[tuple[str, int]]:
    seen = set()
    ordered = []
    for sample in audit_dataset["samples"]:
        key = (sample["source_split"], sample["source_image_id"])
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def sample_index_for(audit_dataset: dict) -> dict[tuple[str, int], int]:
    """(split, annotation_id) -> audit sample index, for linking recall misses back."""
    return {
        (sample["source_split"], sample["source_annotation_id"]): sample["index"]
        for sample in audit_dataset["samples"]
    }


def propose_for_image(processor, image: Image.Image, referee_boxes: list[list[float]]) -> list[dict]:
    state = processor.set_image(image)

    state = processor.set_text_prompt(prompt=SCOREBOARD_PROMPT, state=state)
    scoreboard_scores = state["scores"].to(torch.float32).cpu().numpy()
    scoreboard_boxes = state["boxes"].to(torch.float32).cpu().numpy()
    scoreboard_boxes = [
        box.tolist()
        for box, score in zip(scoreboard_boxes, scoreboard_scores)
        if score > SCOREBOARD_CONFIDENCE
    ]

    state = processor.set_text_prompt(prompt=JERSEY_NUMBER_PROMPT, state=state)
    boxes = state["boxes"].to(torch.float32).cpu()
    scores = state["scores"].to(torch.float32).cpu()

    keep = scores > CONFIDENCE_THRESHOLD
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=NMS_THRESHOLD)
        boxes, scores = boxes[keep_indices], scores[keep_indices]

    proposals = []
    for box, score in zip(boxes.numpy().tolist(), scores.numpy().tolist()):
        if is_center_inside_any(box, scoreboard_boxes):
            continue
        if is_center_inside_any(box, referee_boxes):
            continue
        proposals.append({"box": box, "confidence": float(score)})
    return proposals


def _patch_sam3_fused_mlp_dtype() -> None:
    """sam3/perflib/fused.py's `addmm_act` hardcodes a bf16 cast to use a fused
    aten._addmm_activation kernel for MLP fc1+activation, but nothing casts the result
    back -- so fc2 (an ordinary float32 nn.Linear, confirmed by inspecting every
    parameter's dtype after a fresh build_sam3_image_model()) receives a bf16 input and
    raises "mat1 and mat2 must have the same dtype". Reproduces identically on CPU, so
    it is this checkout's model code, not GB10/cuda-capability-12.1 hardware.

    Patched at runtime on `sam3.model.vitdet`'s own module namespace (where
    `from sam3.perflib.fused import addmm_act` bound the name Mlp.forward looks up at
    call time) rather than editing the vendored file on disk. Keeps the fused kernel's
    speed for fc1 and only adds the missing cast back to the caller's dtype.
    """
    from sam3.model import vitdet

    original_addmm_act = vitdet.addmm_act

    def addmm_act_restoring_dtype(activation, linear, mat1):
        return original_addmm_act(activation, linear, mat1).to(mat1.dtype)

    vitdet.addmm_act = addmm_act_restoring_dtype


def _import_sam3():
    """`sam3` isn't pip-installed in this venv; the package root sits one directory
    below the vendored clone at repo_root/sam3/sam3, so plain `import sam3` silently
    resolves to the outer clone dir as an empty namespace package instead. Only this
    function needs the extra path, so the fixup stays local rather than leaking into
    every other script's sys.path."""
    import sys
    vendor_root = str(ROOT / "sam3")
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    _patch_sam3_fused_mlp_dtype()
    return Sam3Processor, build_sam3_image_model


def match_proposals_to_image(
    proposals: list[dict],
    entry: dict,
    split: str,
    image_id: int,
    linked_index: dict[tuple[str, int], int],
) -> tuple[list[dict], set[tuple[str, int]]]:
    """Match one image's SAM3 proposals against its existing jersey annotations.

    Returns the per-proposal candidate rows and the set of (split, annotation_id)
    pairs matched -- which spans every jersey annotation on this image, not just ones
    in the audit sample (see the recovery-count note in build_report).
    """
    candidates = []
    matched_annotation_ids: set[tuple[str, int]] = set()
    for proposal in proposals:
        best_annotation_id, best_iou = None, 0.0
        for annotation_id, existing_box in entry["jersey"]:
            iou = box_iou(proposal["box"], existing_box)
            if iou > best_iou:
                best_annotation_id, best_iou = annotation_id, iou
        matched = best_iou >= IOU_MATCH_THRESHOLD
        if matched:
            matched_annotation_ids.add((split, best_annotation_id))
        candidates.append({
            "split": split,
            "file_name": entry["file_name"],
            "image_id": image_id,
            "box": proposal["box"],
            "confidence": proposal["confidence"],
            "matched_annotation_id": best_annotation_id if matched else None,
            "matched_sample_index": (
                linked_index.get((split, best_annotation_id)) if matched else None
            ),
            "match_iou": best_iou if matched else None,
        })
    return candidates, matched_annotation_ids


def build_report(
    audit_dataset_path: Path,
    source_root: Path,
    audit_dataset: dict,
    images_to_run: list[tuple[str, int]],
    candidates: list[dict],
    matched_annotation_ids: set[tuple[str, int]],
) -> dict:
    recall_misses = [
        {
            "sample_index": sample["index"],
            "split": sample["source_split"],
            "file_name": sample["source_file_name"],
            "annotation_id": sample["source_annotation_id"],
            "height_band": sample["height_band"],
        }
        for sample in audit_dataset["samples"]
        if (sample["source_split"], sample["source_annotation_id"]) not in matched_annotation_ids
    ]

    # `matched_annotation_ids` spans every jersey-number annotation in these images,
    # not just the ones sampled into the audit set (each image typically carries
    # several unsampled numbers too) -- recovery against the audit set specifically is
    # sample count minus misses, not the raw size of that broader match set.
    existing_annotations_recovered = len(audit_dataset["samples"]) - len(recall_misses)
    new_candidates = sum(c["matched_annotation_id"] is None for c in candidates)
    return {
        "schema_version": 1,
        "source_dataset": str(audit_dataset_path),
        "source_root": str(source_root),
        "created_at": utc_now(),
        "config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "scoreboard_confidence": SCOREBOARD_CONFIDENCE,
            "nms_threshold": NMS_THRESHOLD,
            "iou_match_threshold": IOU_MATCH_THRESHOLD,
        },
        "images_processed": len(images_to_run),
        "total_candidates": len(candidates),
        "new_candidates": new_candidates,
        "existing_annotations": len(audit_dataset["samples"]),
        "existing_annotations_recovered": existing_annotations_recovered,
        "existing_annotations_missed_by_sam3": len(recall_misses),
        "candidates": candidates,
        "recall_misses": recall_misses,
    }


def run(audit_dataset_path: Path, source_root: Path, output_path: Path, device: str) -> dict:
    Sam3Processor, build_sam3_image_model = _import_sam3()

    audit_dataset = json.loads(audit_dataset_path.read_text())
    coco_index = load_coco_index(source_root)
    images_to_run = distinct_images(audit_dataset)
    linked_index = sample_index_for(audit_dataset)

    print(f"loading SAM3 image model on {device} ...", flush=True)
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)

    all_candidates = []
    all_matched_annotation_ids: set[tuple[str, int]] = set()
    started = time.monotonic()
    for position, (split, image_id) in enumerate(images_to_run, start=1):
        entry = coco_index[split][image_id]
        image_path = source_root / split / entry["file_name"]
        image = Image.open(image_path).convert("RGB")

        proposals = propose_for_image(processor, image, entry["referee"])
        candidates, matched_annotation_ids = match_proposals_to_image(
            proposals, entry, split, image_id, linked_index
        )
        all_candidates.extend(candidates)
        all_matched_annotation_ids |= matched_annotation_ids

        print(
            f"[{position}/{len(images_to_run)}] {split}/{entry['file_name']}: "
            f"{len(proposals)} proposals "
            f"({time.monotonic() - started:.1f}s elapsed)",
            flush=True,
        )

    report = build_report(
        audit_dataset_path, source_root, audit_dataset, images_to_run,
        all_candidates, all_matched_annotation_ids,
    )
    write_json_atomic(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dataset", type=Path, default=DEFAULT_AUDIT_DATASET)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.audit_dataset, args.source_root, args.output, args.device)
    print(
        f"\n{report['total_candidates']} SAM3 candidates over {report['images_processed']} images "
        f"({report['new_candidates']} new) -> {args.output}"
    )
    print(
        f"existing annotations recovered: {report['existing_annotations_recovered']}"
        f"/{report['existing_annotations']} "
        f"({report['existing_annotations_missed_by_sam3']} missed by SAM3)"
    )


if __name__ == "__main__":
    main()
