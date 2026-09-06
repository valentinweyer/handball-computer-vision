"""Build a jersey-number readability audit set from the RF-DETR training COCO export.

Samples existing `jersey number` annotations from the training dataset in
`rfdetr-handball-finetune`, stratified by box height (the dominant unknown: median
jersey number in that export is ~9x18px) and by source clip, and renders tight +
padded-context crops in the exact schema `label_jersey_numbers.py` already serves and
scores. The existing browser reviewer and Qwen benchmark run against this dataset
unchanged.

This script only reads the source COCO export; nothing under it is modified.

Example:
    python -m scripts.build_jersey_audit_set --output-dir runs/jersey_audit
    python -m scripts.label_jersey_numbers review --dataset-dir runs/jersey_audit
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import cv2

from scripts.label_jersey_numbers import clipped_box, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    "/home/valentinweyer/projects/rfdetr-handball-finetune/datasets/handball-v4-w-numbers"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs/jersey_audit"
CATEGORY_NAME = "jersey number"
SPLITS = ("train", "valid", "test")

# (label, low inclusive, high exclusive) on annotation bbox height in pixels.
# Cut for the 640x640 RF-DETR training export, where number boxes are ~9x18px.
HEIGHT_BANDS = (
    ("<16", 0.0, 16.0),
    ("16-19", 16.0, 20.0),
    ("20-23", 20.0, 24.0),
    (">=24", 24.0, math.inf),
)
HEIGHT_BAND_ORDER = {label: position for position, (label, _, _) in enumerate(HEIGHT_BANDS)}

# Cut for 1920x1080 production, a different regime: measured over 27305 cached
# number boxes on the 1080p clips, heights run p05=14 p25=20 p50=25 p75=30 p95=40
# (max 154). Under HEIGHT_BANDS above, more than half of those land in the single
# ">=24" bucket, which defeats stratification -- hence a separate set on the real
# quantiles. Kept alongside rather than replacing, so runs/jersey_audit stays
# interpretable under the bands it was actually sampled with.
HEIGHT_BANDS_1080P = (
    ("<18", 0.0, 18.0),
    ("18-21", 18.0, 22.0),
    ("22-25", 22.0, 26.0),
    ("26-30", 26.0, 31.0),
    ("31-40", 31.0, 41.0),
    (">=41", 41.0, math.inf),
)
HEIGHT_BAND_1080P_ORDER = {
    label: position for position, (label, _, _) in enumerate(HEIGHT_BANDS_1080P)
}


def band_label(height: float, bands=HEIGHT_BANDS) -> str:
    for label, low, high in bands:
        if low <= height < high:
            return label
    raise AssertionError(f"unbanded height: {height}")


def height_band_label(height: float) -> str:
    return band_label(height, HEIGHT_BANDS)


def source_clip(file_name: str) -> str:
    """Group a Roboflow export filename by originating clip, collapsing frame indices.

    ``Barsa_Kielce_mp4-0009_jpg.rf.<hash>.jpg`` -> ``Barsa_Kielce_mp4``.
    ``image1032_jpg.rf.<hash>.jpg`` -> ``image_stills`` (these have no frame-sequence
    provenance and were found to dominate the class at 72% of annotations).
    """
    stem = file_name.split("_jpg")[0]
    prefix, separator, suffix = stem.rpartition("-")
    if separator and suffix.isdigit():
        stem = prefix
    return "image_stills" if stem.startswith("image") else stem


def load_jersey_annotations(source_root: Path) -> list[dict]:
    records = []
    for split in SPLITS:
        coco_path = source_root / split / "_annotations.coco.json"
        if not coco_path.is_file():
            continue
        coco = json.loads(coco_path.read_text())
        category_id = next(
            (item["id"] for item in coco["categories"] if item["name"] == CATEGORY_NAME),
            None,
        )
        if category_id is None:
            continue
        images = {image["id"]: image for image in coco["images"]}
        for annotation in coco["annotations"]:
            if annotation["category_id"] != category_id:
                continue
            image = images[annotation["image_id"]]
            x, y, w, h = annotation["bbox"]
            records.append({
                "annotation_id": annotation["id"],
                "image_id": annotation["image_id"],
                "split": split,
                "file_name": image["file_name"],
                "box": [x, y, x + w, y + h],
                "box_height": h,
                "source_clip": source_clip(image["file_name"]),
            })
    return records


def _round_robin_by_clip(candidates: list[dict], quota: int, rng: random.Random) -> list[dict]:
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        by_clip[record["source_clip"]].append(record)
    for group in by_clip.values():
        rng.shuffle(group)
    clip_order = sorted(by_clip)
    rng.shuffle(clip_order)

    picked = []
    while len(picked) < quota:
        progressed = False
        for clip in clip_order:
            group = by_clip[clip]
            if group:
                picked.append(group.pop())
                progressed = True
                if len(picked) == quota:
                    break
        if not progressed:
            break
    return picked


def stratified_sample(
    records: list[dict],
    per_band: int,
    seed: int,
    bands=HEIGHT_BANDS,
    band_quota: dict[str, int] | None = None,
) -> list[dict]:
    """Sample up to `per_band` records per height band, round-robined across clips.

    `band_quota` overrides `per_band` for named bands. Uniform quotas assume every
    band is worth equal labeling effort, which is false when a band is both a small
    share of production and dominated by detector false positives.
    """
    rng = random.Random(seed)
    order = {label: position for position, (label, _, _) in enumerate(bands)}
    by_band: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_band[band_label(record["box_height"], bands)].append(record)

    selected = []
    for label, _, _ in bands:
        quota = per_band if band_quota is None else band_quota.get(label, per_band)
        candidates = by_band.get(label, [])
        selected.extend(_round_robin_by_clip(candidates, min(quota, len(candidates)), rng))

    selected.sort(key=lambda r: (
        order[band_label(r["box_height"], bands)],
        r["source_clip"],
        r["annotation_id"],
    ))
    return selected


def render_crop_pair(image, box: list[float], index: int, crops_dir: Path, contexts_dir: Path) -> dict:
    """Write the tight OCR crop and the padded, red-boxed context crop for one box.

    Shared by build_jersey_audit_set.py (existing COCO annotations) and
    generate_jersey_candidates.py (new SAM3-only proposals) so both produce crops in
    the identical geometry `label_jersey_numbers.py`'s reviewer and Qwen already expect
    -- 80% of the longer box side as context padding, red boundary of the exact labeled
    box, scaled line thickness. See scripts/label_jersey_numbers.py:build_dataset for
    the original of this geometry.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clipped_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box clips to an empty crop: {box}")
    crop = image[y1:y2, x1:x2]
    crop_name = f"number_{index:04d}.jpg"
    if not cv2.imwrite(str(crops_dir / crop_name), crop):
        raise RuntimeError(f"could not write crop {crop_name}")

    pad = max(24, round(max(x2 - x1, y2 - y1) * 0.8))
    cx1, cy1, cx2, cy2 = clipped_box(box, width, height, pad)
    context = image[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(
        context,
        (x1 - cx1, y1 - cy1),
        (x2 - cx1 - 1, y2 - cy1 - 1),
        (0, 0, 255),
        max(1, round(max(context.shape[:2]) / 180)),
    )
    context_name = f"number_{index:04d}.jpg"
    if not cv2.imwrite(str(contexts_dir / context_name), context):
        raise RuntimeError(f"could not write context {context_name}")

    return {
        "crop_path": f"crops/{crop_name}",
        "context_path": f"contexts/{context_name}",
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
    }


def render_dataset(records: list[dict], source_root: Path, output_dir: Path) -> dict:
    crops_dir = output_dir / "crops"
    contexts_dir = output_dir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    image_cache: dict[tuple[str, str], object] = {}
    for index, record in enumerate(records):
        cache_key = (record["split"], record["file_name"])
        image = image_cache.get(cache_key)
        if image is None:
            image_path = source_root / record["split"] / record["file_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            image_cache[cache_key] = image

        crop_info = render_crop_pair(image, record["box"], index, crops_dir, contexts_dir)

        rendered.append({
            "index": index,
            "frame": 0,
            "box": record["box"],
            "predictions": {},
            "sample_id": f"{Path(record['file_name']).stem}-a{record['annotation_id']}",
            **crop_info,
            "source_kind": "coco",
            "source_split": record["split"],
            "source_file_name": record["file_name"],
            "source_image_id": record["image_id"],
            "source_annotation_id": record["annotation_id"],
            "source_clip": record["source_clip"],
            "height_band": height_band_label(record["box_height"]),
        })

    return {
        "schema_version": 1,
        "video": f"coco:{source_root.name}",
        "source_root": str(source_root),
        "created_at": utc_now(),
        "sample_count": len(rendered),
        "samples": rendered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-band", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jersey_annotations(args.source_root)
    if not records:
        raise ValueError(f"no jersey-number annotations found under {args.source_root}")

    selected = stratified_sample(records, args.per_band, args.seed)
    output_dir = args.output_dir.resolve()
    dataset = render_dataset(selected, args.source_root, output_dir)
    write_json_atomic(output_dir / "dataset.json", dataset)

    band_counts = Counter(height_band_label(r["box_height"]) for r in selected)
    print(f"built {len(selected)} samples -> {output_dir / 'dataset.json'}")
    for label, _, _ in HEIGHT_BANDS:
        print(f"  {label:>6}: {band_counts.get(label, 0)}")


if __name__ == "__main__":
    main()
