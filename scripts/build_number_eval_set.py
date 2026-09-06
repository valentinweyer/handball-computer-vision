"""Build a 1080p jersey-number evaluation set sampled from production detections.

This exists separately from `build_jersey_audit_set.py` because that set is drawn
from the COCO export RF-DETR trains on, at 640x640, where number boxes have median
width 11px. Production runs at 1920x1080, where the median is 26-30px -- a 2.7x
different regime. A reader benchmarked on the 640 set says nothing about the one we
actually deploy, so this samples the production detector's own cached output on
1080p clips instead, which is what guarantees the regime matches.

Sampling is stratified by box height using `HEIGHT_BANDS_1080P` (the 640-regime
bands collapse over half of production into one bucket) and round-robined across
clips, so no single venue or kit dominates -- the weakness of the existing 1080p
sets, which are all one clip.

Crops are rendered through the shared `render_crop_pair` geometry, so the emitted
dataset works unchanged in `label_jersey_numbers.py`'s reviewer and in
`benchmark_qwen_jersey_ocr.py`'s tight/context variants.

Example:
    python -m scripts.build_number_eval_set \
        --clip source/BHC-FAG.mp4:outputs/number_cache/.BHC-FAG_numbers_v1.npz \
        --clip data/raw/FelixClaar.mp4:outputs/number_cache/.FelixClaar_numbers_v1.npz \
        --per-band-per-clip 20 --output-dir runs/number_eval_1080p
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.build_jersey_audit_set import (
    HEIGHT_BANDS_1080P,
    band_label,
    render_crop_pair,
    stratified_sample,
)
from handball_cv.jersey.identity import NUMBER_CLASS_ID
from scripts.label_jersey_numbers import (
    PLAYER_CLASS_IDS,
    max_player_containment,
    utc_now,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "runs/number_eval_1080p"
EXPECTED_SIZE = (1920, 1080)

# Fraction of the normal per-band quota to spend on bands that are poor value to
# label. ">=41" is only ~5% of production boxes and a genuine number that tall is
# rare, so its detections are dominated by sponsor logos and arm patches; uniform
# stratification would spend a sixth of the labeling budget there.
BAND_QUOTA_SCALE = {">=41": 0.5}


def parse_clip_spec(spec: str) -> tuple[Path, Path]:
    """Split a ``video.mp4:cache.npz`` argument into its two paths."""
    video, separator, cache = spec.rpartition(":")
    if not separator:
        raise ValueError(f"expected '<video>:<cache>', got {spec!r}")
    return Path(video), Path(cache)


def clip_name(video: Path) -> str:
    return video.stem


def records_from_cache(
    video: Path,
    cache_path: Path,
    min_confidence: float,
    min_player_containment: float = 0.9,
) -> tuple[list[dict], int]:
    """Flatten one cached detection npz into per-box records for sampling.

    Only number boxes a player box substantially contains are kept: the detector
    fires on advertising hoardings, the scoreboard and backdrop lettering, and none
    of those are jersey numbers the reader will ever be asked about in earnest.
    Referees are excluded from the containing classes on the same reasoning.

    `offsets` spans every frame of the video, so a frame the detector skipped
    (see `cache_number_detections.py --stride`) simply contributes no records.

    Returns the kept records and how many number boxes the containment filter
    dropped, so the caller can report detector precision rather than hide it.
    """
    cache = np.load(cache_path, allow_pickle=True)
    offsets = np.asarray(cache["offsets"], dtype=np.int64)
    boxes = np.asarray(cache["boxes"], dtype=float).reshape(-1, 4)
    confidence = np.asarray(cache["confidence"], dtype=float)
    class_ids = np.asarray(cache["class_id"], dtype=np.int64)
    name = clip_name(video)

    records, dropped = [], 0
    for frame_index in range(len(offsets) - 1):
        start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
        frame_classes = class_ids[start:end]
        player_boxes = boxes[start:end][np.isin(frame_classes, PLAYER_CLASS_IDS)]
        for local, position in enumerate(range(start, end)):
            if frame_classes[local] != NUMBER_CLASS_ID:
                continue
            box = boxes[position]
            height = float(box[3] - box[1])
            width = float(box[2] - box[0])
            if height <= 0 or width <= 0 or confidence[position] < min_confidence:
                continue
            containment = max_player_containment(box, player_boxes)
            if containment < min_player_containment:
                dropped += 1
                continue
            records.append({
                "source_clip": name,
                "video": str(video),
                "frame": frame_index,
                "box": [float(v) for v in box],
                "box_height": height,
                "box_width": width,
                "confidence": float(confidence[position]),
                "player_containment": containment,
                "annotation_id": f"{name}:{frame_index}:{position}",
            })
    return records, dropped


def render_selection(
    selected: list[dict], crops_dir: Path, contexts_dir: Path
) -> list[dict]:
    """Decode each needed frame once and render its crops, clip by clip.

    Frames are visited in ascending order per clip so the capture only ever seeks
    forward, which matters on 90-minute broadcasts.
    """
    by_video: dict[str, list[tuple[int, dict]]] = {}
    for index, record in enumerate(selected):
        by_video.setdefault(record["video"], []).append((index, record))

    rendered: dict[int, dict] = {}
    for video, entries in by_video.items():
        entries.sort(key=lambda item: item[1]["frame"])
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {video}")
        try:
            for index, record in entries:
                capture.set(cv2.CAP_PROP_POS_FRAMES, record["frame"])
                ok, frame_bgr = capture.read()
                if not ok:
                    print(f"  skip {record['annotation_id']}: frame unreadable", flush=True)
                    continue
                size = (frame_bgr.shape[1], frame_bgr.shape[0])
                if size != EXPECTED_SIZE:
                    raise ValueError(
                        f"{video} is {size[0]}x{size[1]}, not 1920x1080 -- this set is "
                        "deliberately single-regime; exclude the clip instead"
                    )
                rendered[index] = render_crop_pair(
                    frame_bgr, record["box"], index, crops_dir, contexts_dir
                )
        finally:
            capture.release()

    samples = []
    for index, record in enumerate(selected):
        crop = rendered.get(index)
        if crop is None:
            continue
        samples.append({
            "index": len(samples),
            "frame": record["frame"],
            "box": record["box"],
            "predictions": {},
            "sample_id": record["annotation_id"],
            "source_kind": "production_detection_1080p",
            "source_clip": record["source_clip"],
            "source_video": record["video"],
            "detector_confidence": record["confidence"],
            "player_containment": record["player_containment"],
            "height_band": band_label(record["box_height"], HEIGHT_BANDS_1080P),
            **crop,
        })
    return samples


def renumber_crops(samples: list[dict], crops_dir: Path, contexts_dir: Path) -> None:
    """Rename crop files so filename index matches the final sample index.

    Unreadable frames leave gaps in the render pass; the reviewer and the Qwen
    harness both address samples by list index, so the two must not drift apart.
    """
    for sample in samples:
        for key, directory in (("crop_path", crops_dir), ("context_path", contexts_dir)):
            old = directory / Path(sample[key]).name
            new_name = f"number_{sample['index']:04d}.jpg"
            new = directory / new_name
            if old != new:
                old.replace(new)
            sample[key] = f"{directory.name}/{new_name}"


def summarize(samples: list[dict]) -> None:
    widths = np.array([s["crop_width"] for s in samples], dtype=float)
    heights = np.array([s["crop_height"] for s in samples], dtype=float)
    print(f"\n{len(samples)} samples")
    print("  per clip: ", dict(Counter(s["source_clip"] for s in samples)))
    print("  per band: ", dict(Counter(s["height_band"] for s in samples)))
    if len(widths):
        q = lambda a, p: float(np.percentile(a, p))
        print(
            f"  crop w p10/p50/p90 = {q(widths,10):.0f}/{q(widths,50):.0f}/{q(widths,90):.0f}"
            f"   h = {q(heights,10):.0f}/{q(heights,50):.0f}/{q(heights,90):.0f}"
        )
        print("  (production reference: w p50 ~26-30, h p50 ~25)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="VIDEO:CACHE",
        help="1080p video and its cached number-detection npz; repeat per clip",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-band-per-clip", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument(
        "--min-player-containment",
        type=float,
        default=0.9,
        help="fraction of the number box that must fall inside some player box",
    )
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_clip_spec(spec) for spec in args.clip]

    records, total_dropped = [], 0
    for video, cache_path in specs:
        clip_records, dropped = records_from_cache(
            video, cache_path, args.min_confidence, args.min_player_containment
        )
        kept, seen = len(clip_records), len(clip_records) + dropped
        print(f"{clip_name(video)}: {kept} on-player numbers of {seen} detected "
              f"({kept/max(seen,1):.0%} on a player) from {cache_path.name}")
        records.extend(clip_records)
        total_dropped += dropped
    print(f"\ncontainment filter dropped {total_dropped} number boxes not on a player")
    if not records:
        raise SystemExit("no boxes survived the confidence filter")

    per_band = args.per_band_per_clip * len(specs)
    band_quota = {
        label: max(1, round(per_band * scale)) for label, scale in BAND_QUOTA_SCALE.items()
    }
    selected = stratified_sample(
        records, per_band, args.seed, bands=HEIGHT_BANDS_1080P, band_quota=band_quota
    )
    print(f"\nselected {len(selected)} of {len(records)} boxes "
          f"({args.per_band_per_clip}/band/clip over {len(HEIGHT_BANDS_1080P)} bands; "
          f"reduced quota: {band_quota})")

    crops_dir = args.output_dir / "crops"
    contexts_dir = args.output_dir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    samples = render_selection(selected, crops_dir, contexts_dir)
    renumber_crops(samples, crops_dir, contexts_dir)

    write_json_atomic(args.output_dir / "dataset.json", {
        "schema_version": 1,
        "video": "multi:1080p_production",
        "source_kind": "production_detection_1080p",
        "clips": [str(video) for video, _ in specs],
        "caches": [str(cache) for _, cache in specs],
        # The top band is unbounded; JSON has no Infinity, so it serialises as null.
        "height_bands": [
            [label, low, None if high == float("inf") else high]
            for label, low, high in HEIGHT_BANDS_1080P
        ],
        "per_band_per_clip": args.per_band_per_clip,
        "min_confidence": args.min_confidence,
        "min_player_containment": args.min_player_containment,
        "seed": args.seed,
        "created_at": utc_now(),
        "sample_count": len(samples),
        "samples": samples,
    })
    summarize(samples)
    print(f"\n-> {args.output_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
