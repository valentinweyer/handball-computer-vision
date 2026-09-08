"""Does the number crop need padding? Measure it instead of assuming.

`NUMBER_CROP_PAD = 0`: the reader is handed the detector's box exactly. The
evaluation set only carries the two extremes -- the tight crop and a 0.8x padded
context crop -- and PARSeq scores 0.62 on the first and 0.04 on the second, so
"padding is bad" was concluded from one enormous step. Nothing between has been
tried.

There is a specific reason to expect a small pad to help. On the 262 labelled
two-digit crops, the failure the readers share is **digit loss**: every docTR
architecture unanimously misreads 22 as 2, 93 as 3, 17 as 7, 10 as 1. Digit loss
is a horizontal problem, so horizontal-only padding is measured separately --
vertical padding adds shoulders and shirt folds without ever recovering a digit.

Pads are fractions of the box's own size, not pixels: these boxes range from 14
to 154 px wide, and a fixed pixel pad would be a different crop at each end.

Example:
    python -m scripts.sweep_number_crop_padding \
        --dataset runs/number_eval_1080p/dataset.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json \
        --output runs/number_eval_1080p/crop_padding_sweep.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re

import cv2
import numpy as np

from scripts.label_jersey_numbers import score_model, write_json_atomic


# (horizontal, vertical) pad as a fraction of box width / height.
DEFAULT_PADS = (
    (0.00, 0.00),   # what ships today
    (0.10, 0.00), (0.20, 0.00), (0.35, 0.00), (0.50, 0.00),   # horizontal only
    (0.10, 0.10), (0.20, 0.20), (0.35, 0.35),                 # symmetric
)
VALID_NUMBER = re.compile(r"[1-9][0-9]?")
ASPECT_BANDS = ((0.00, 0.70), (0.70, 0.85), (0.85, 1.00), (1.00, 99.0))


def to_number(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped if VALID_NUMBER.fullmatch(stripped) else ""


def pad_box(box, px_frac: float, py_frac: float, width: int, height: int):
    x1, y1, x2, y2 = (float(v) for v in box)
    dx = (x2 - x1) * px_frac
    dy = (y2 - y1) * py_frac
    return (
        max(0, int(round(x1 - dx))), max(0, int(round(y1 - dy))),
        min(width, int(round(x2 + dx))), min(height, int(round(y2 + dy))),
    )


def collect(samples, video_path: Path, pads) -> dict:
    """One video read per frame; every pad variant cropped from it."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    crops = {pad: {} for pad in pads}
    for sample in sorted(samples, key=lambda s: s["frame"]):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame"]))
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        for pad in pads:
            x1, y1, x2, y2 = pad_box(sample["box"], pad[0], pad[1], width, height)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                crops[pad][sample["index"]] = frame_rgb[y1:y2, x1:x2]
    capture.release()
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arch", default="parseq")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from doctr.models import recognition_predictor

    dataset = json.loads(args.dataset.read_text())
    labels = json.loads(args.labels.read_text())["labels"]
    readable = {
        int(i): v["value"] for i, v in labels.items()
        if v.get("status") == "readable" and v.get("value")
    }
    unreadable = {int(i) for i, v in labels.items() if v.get("status") == "unreadable"}
    wanted = set(readable) | unreadable
    samples = {s["index"]: s for s in dataset["samples"] if s["index"] in wanted}
    videos = {Path(p).stem: Path(p) for p in dataset["clips"]}

    pads = list(DEFAULT_PADS)
    by_pad: dict = {pad: {} for pad in pads}
    by_clip = defaultdict(list)
    for sample in samples.values():
        by_clip[sample["source_clip"]].append(sample)
    for clip, clip_samples in sorted(by_clip.items()):
        video = videos.get(clip)
        if video is None:
            print(f"{clip}: not in the dataset manifest -- skipped")
            continue
        for pad, crops in collect(clip_samples, video, pads).items():
            by_pad[pad].update(crops)
        print(f"  cropped {clip}", flush=True)

    predictor = recognition_predictor(args.arch, pretrained=True).eval()
    if args.device != "cpu" and torch.cuda.is_available():
        predictor = predictor.cuda()

    report = {
        "schema_version": 1, "arch": args.arch,
        "dataset": str(args.dataset), "labels": str(args.labels),
        "pads": [], "note": "pads are fractions of the box's own width/height",
    }
    print(f"\n{'pad (h,v)':>12}{'n':>6}{'cov':>7}{'acc':>7}{'sel':>7}{'abst':>7}"
          f"   accuracy by crop aspect ratio (w/h)")
    for pad in pads:
        indices = sorted(by_pad[pad])
        images = [by_pad[pad][i] for i in indices]
        with torch.inference_mode():
            results = predictor(images)
        raw = {i: (r[0], float(r[1])) for i, r in zip(indices, results)}
        predictions = {
            i: (to_number(t) if c >= 0.5 else "") for i, (t, c) in raw.items()
        }
        scored = score_model(readable, predictions, unreadable)

        bands = {}
        for low, high in ASPECT_BANDS:
            rows = [
                i for i in indices if i in readable and len(readable[i]) == 2
                and low <= by_pad[pad][i].shape[1] / by_pad[pad][i].shape[0] < high
            ]
            if rows:
                bands[f"{low:.2f}-{high:.2f}"] = {
                    "n": len(rows),
                    "accuracy": sum(predictions[i] == readable[i] for i in rows) / len(rows),
                }
        report["pads"].append({
            "pad": list(pad), "scores": scored, "by_aspect_band": bands,
            "raw": {str(i): {"text": t, "confidence": c} for i, (t, c) in raw.items()},
        })
        band_text = "  ".join(
            f"{k} {v['accuracy']:.2f}" for k, v in bands.items()
        )
        print(f"{str(pad):>12}{len(indices):>6}{scored['coverage']:>7.2f}"
              f"{scored['accuracy']:>7.2f}{scored['selective_accuracy']:>7.2f}"
              f"{scored['unreadable_abstention_rate']:>7.2f}   {band_text}", flush=True)

    if args.output:
        write_json_atomic(args.output, report)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
