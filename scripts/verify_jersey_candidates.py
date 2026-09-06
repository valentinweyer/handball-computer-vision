"""Combine SAM3 region proposals with Qwen reads into one reviewable candidate set.

Extends runs/jersey_audit/dataset.json (see build_jersey_audit_set.py) with the new,
previously-unannotated boxes SAM3 proposed (see generate_jersey_candidates.py), renders
their crops in the identical geometry, then runs Qwen (the context variant, which
measured best in the FelixClaar benchmark) over every sample -- both the original
COCO-derived ones and the new SAM3-only ones.

Each sample's hidden `predictions` bundle (shown behind the existing reviewer's
"Reveal predictions" toggle, so it can't anchor a human reviewer) ends up carrying:

  - qwen_context / qwen_parse_status: Qwen's read of the padded context crop.
  - sam3_recovered (COCO-derived samples only): whether any SAM3 proposal matched this
    exact annotation at runtime (see generate_jersey_candidates.py's IoU match).
  - sam3_confidence (SAM3-only samples): the proposing box's SAM3 confidence.
  - agreement: a short label combining SAM3 presence and Qwen's read -- see
    `compute_agreement`. This is a triage hint, not a verdict; review still decides.

This computes no ground truth and makes no accept/reject decision -- it only attaches
two independent machine signals to each box so a human review pass can be spent
adjudicating pre-scored candidates instead of starting from bare pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from scripts.benchmark_qwen_jersey_ocr import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    VARIANTS,
    parse_prediction,
    request_prediction,
)
from scripts.build_jersey_audit_set import DEFAULT_SOURCE_ROOT, render_crop_pair
from scripts.label_jersey_numbers import utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "runs/jersey_audit/dataset.json"
DEFAULT_SAM3_CANDIDATES = ROOT / "runs/jersey_audit/sam3_candidates.json"
QWEN_VARIANT = "context"


def append_new_sam3_candidates(
    dataset: dict, sam3_report: dict, source_root: Path, output_dir: Path
) -> dict:
    """Add SAM3-only proposals (no existing-annotation match) as new samples.

    Idempotent against a dataset that already has some appended: re-running after a
    partial Qwen pass must not duplicate or renumber existing sam3_new samples.
    """
    existing_sam3_keys = {
        (s["source_split"], tuple(round(v, 1) for v in s["box"]))
        for s in dataset["samples"]
        if s.get("source_kind") == "sam3_new"
    }
    new_candidates = [
        c for c in sam3_report["candidates"]
        if c["matched_annotation_id"] is None
        and (c["split"], tuple(round(v, 1) for v in c["box"])) not in existing_sam3_keys
    ]
    if not new_candidates:
        return dataset

    crops_dir, contexts_dir = output_dir / "crops", output_dir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)
    next_index = max((s["index"] for s in dataset["samples"]), default=-1) + 1
    image_cache: dict[tuple[str, str], object] = {}

    for candidate in new_candidates:
        cache_key = (candidate["split"], candidate["file_name"])
        image = image_cache.get(cache_key)
        if image is None:
            image_path = source_root / candidate["split"] / candidate["file_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            image_cache[cache_key] = image

        crop_info = render_crop_pair(image, candidate["box"], next_index, crops_dir, contexts_dir)
        dataset["samples"].append({
            "index": next_index,
            "frame": 0,
            "box": candidate["box"],
            "predictions": {},
            "sample_id": f"{Path(candidate['file_name']).stem}-sam3-{next_index}",
            **crop_info,
            "source_kind": "sam3_new",
            "source_split": candidate["split"],
            "source_file_name": candidate["file_name"],
            "sam3_confidence": candidate["confidence"],
        })
        next_index += 1

    return dataset


def compute_agreement(sample: dict, sam3_recovered_indices: set[int]) -> dict:
    """Attach sam3_recovered/agreement to one sample's predictions in place."""
    predictions = sample["predictions"]
    qwen_reads = bool(predictions.get("qwen_context"))

    if sample["source_kind"] == "coco":
        sam3_recovered = sample["index"] in sam3_recovered_indices
        predictions["sam3_recovered"] = sam3_recovered
        if sam3_recovered and qwen_reads:
            agreement = "sam3+qwen agree"
        elif sam3_recovered and not qwen_reads:
            agreement = "sam3 only, qwen abstains"
        elif not sam3_recovered and qwen_reads:
            agreement = "qwen only, sam3 missed"
        else:
            agreement = "neither: sam3 missed, qwen abstains"
    else:  # sam3_new
        agreement = "sam3+qwen agree" if qwen_reads else "sam3 only, qwen abstains"

    predictions["agreement"] = agreement
    return predictions


def run(dataset_path: Path, sam3_report_path: Path, source_root: Path, base_url: str, model: str) -> dict:
    dataset = json.loads(dataset_path.read_text())
    sam3_report = json.loads(sam3_report_path.read_text())

    dataset = append_new_sam3_candidates(dataset, sam3_report, source_root, dataset_path.parent)

    variant = VARIANTS[QWEN_VARIANT]
    samples = dataset["samples"]
    pending = [s for s in samples if "qwen_context" not in s["predictions"]]
    started = time.monotonic()
    for position, sample in enumerate(pending, start=1):
        image_path = dataset_path.parent / sample[variant["path_key"]]
        content, _usage = request_prediction(
            base_url, model, image_path, variant["prompt"], timeout=120, max_tokens=16,
        )
        prediction, parse_status = parse_prediction(content)
        sample["predictions"]["qwen_context"] = prediction
        sample["predictions"]["qwen_parse_status"] = parse_status
        write_json_atomic(dataset_path, dataset)
        print(
            f"[{position}/{len(pending)}] sample={sample['index']} "
            f"({sample['source_kind']}) qwen={prediction or '<abstain>'} "
            f"parse={parse_status} ({time.monotonic() - started:.1f}s elapsed)",
            flush=True,
        )

    sam3_recovered_indices = {
        c["matched_sample_index"]
        for c in sam3_report["candidates"]
        if c["matched_sample_index"] is not None
    }
    for sample in samples:
        compute_agreement(sample, sam3_recovered_indices)

    dataset["qwen_model"] = model
    dataset["qwen_variant"] = QWEN_VARIANT
    dataset["verified_at"] = utc_now()
    write_json_atomic(dataset_path, dataset)

    agreement_counts: dict[str, int] = {}
    for sample in samples:
        label = sample["predictions"]["agreement"]
        agreement_counts[label] = agreement_counts.get(label, 0) + 1
    return {"total_samples": len(samples), "agreement_counts": agreement_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sam3-candidates", type=Path, default=DEFAULT_SAM3_CANDIDATES)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.dataset, args.sam3_candidates, args.source_root, args.base_url, args.model)
    print(f"\n{summary['total_samples']} samples verified")
    for label, count in sorted(summary["agreement_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {label}")


if __name__ == "__main__":
    main()
