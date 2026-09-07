"""Benchmark docTR's pretrained scene-text recognisers as jersey-number readers.

These are off-the-shelf models with no domain training, scored on the same crops
and labels as the Qwen and EasyOCR arms so the three are directly comparable.

Two things make them worth measuring rather than assuming:

  - EasyOCR's recogniser is already a CRNN+CTC (1.4M params, 96 characters), so
    "CTC over digits" is not an untested architecture -- it is the production
    floor at 0.37 accuracy. Its weakness is plausibly its training data rather
    than its shape, which is exactly what a better-trained recogniser tests.
  - They need the **tight** crop. Measured on the 1080p set, `parseq` scores 0.62
    accuracy on `crop_path` and **0.04** on `context_path`: a recogniser trained on
    cropped text lines is out of distribution on a padded scene. This is the
    opposite of the VLM, which needs the context crop (0.30 tight vs 0.70 context)
    to know what it is looking at. Each reader must be given its own best input or
    the comparison is meaningless.

Abstention comes from the recogniser's own confidence, thresholded -- the analogue
of the VLM returning NONE -- plus the same jersey-number validity rule the rest of
the pipeline applies (1-99, no leading zero).

Example:
    python -m scripts.benchmark_doctr_readers \
        --dataset runs/number_eval_1080p/dataset.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json \
        --arch parseq --arch vitstr_small --arch crnn_vgg16_bn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2

from scripts.label_jersey_numbers import score_model, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHS = ("parseq", "vitstr_small", "crnn_vgg16_bn")
DEFAULT_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)
# Recognisers emit free text; only a legal jersey number counts as an answer. This
# is the same rule is_valid_number applies downstream, so a reader is not credited
# for an output the pipeline would discard anyway.
VALID_NUMBER = re.compile(r"[1-9][0-9]?")


def to_number(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped if VALID_NUMBER.fullmatch(stripped) else ""


def load_truth(labels_path: Path) -> tuple[dict[int, str], set[int]]:
    labels = json.loads(labels_path.read_text())["labels"]
    readable = {
        int(i): v["value"] for i, v in labels.items()
        if v.get("status") == "readable" and v.get("value")
    }
    unreadable = {int(i) for i, v in labels.items() if v.get("status") == "unreadable"}
    return readable, unreadable


def read_crops(dataset_path: Path, indices, path_key: str) -> list:
    samples = {x["index"]: x for x in json.loads(dataset_path.read_text())["samples"]}
    images = []
    for index in indices:
        image = cv2.imread(str(dataset_path.parent / samples[index][path_key]))
        if image is None:
            raise FileNotFoundError(samples[index][path_key])
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images


def score_at_thresholds(raw, readable, unreadable, thresholds) -> dict:
    scored = {}
    for threshold in thresholds:
        predictions = {
            index: (to_number(text) if confidence >= threshold else "")
            for index, (text, confidence) in raw.items()
        }
        scored[f"{threshold:.1f}"] = score_model(readable, predictions, unreadable)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arch", action="append", default=None)
    parser.add_argument(
        "--path-key",
        default="crop_path",
        choices=("crop_path", "context_path"),
        help="tight crop by default; these models are trained on cropped text lines",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from doctr.models import recognition_predictor

    archs = args.arch or list(DEFAULT_ARCHS)
    readable, unreadable = load_truth(args.labels)
    indices = sorted(set(readable) | unreadable)
    crops = read_crops(args.dataset, indices, args.path_key)

    report = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "labels": str(args.labels),
        "path_key": args.path_key,
        "readable_labels": len(readable),
        "unreadable_labels": len(unreadable),
        "models": {},
    }
    for arch in archs:
        predictor = recognition_predictor(arch, pretrained=True).eval()
        if torch.cuda.is_available():
            predictor = predictor.cuda()
        started = time.monotonic()
        with torch.inference_mode():
            results = predictor(crops)
        elapsed_ms = (time.monotonic() - started) / max(len(crops), 1) * 1000
        raw = {i: (r[0], float(r[1])) for i, r in zip(indices, results)}
        report["models"][arch] = {
            "ms_per_crop": elapsed_ms,
            "by_confidence_threshold": score_at_thresholds(
                raw, readable, unreadable, DEFAULT_THRESHOLDS
            ),
            "raw": {str(i): {"text": t, "confidence": c} for i, (t, c) in raw.items()},
        }
        best = report["models"][arch]["by_confidence_threshold"]["0.5"]
        print(
            f"{arch:<18} {elapsed_ms:>5.1f} ms/crop   @0.5: "
            f"cov {best['coverage']:.2f}  acc {best['accuracy']:.2f}  "
            f"sel {best['selective_accuracy']:.2f}  "
            f"abst {best['unreadable_abstention_rate']:.2f}",
            flush=True,
        )

    if args.output:
        write_json_atomic(args.output, report)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
