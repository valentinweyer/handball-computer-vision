"""Score the sports-jersey fine-tuned PARSeq against the scene-text one.

Same crops, same labels, same scoring as `benchmark_doctr_readers.py`, so the
number lands next to the 0.62 that three off-the-shelf scene-text recognisers
converged on. The question is narrow: those three share a ~6% floor of
confidently-wrong reads and an oracle of only 0.71, which says the failure is
the *domain* rather than the model family. These weights are the same PARSeq
architecture trained on sports jerseys, so they test that claim directly --
without collecting or labelling anything, and without touching the evaluation
set's role as a held-out benchmark.

Example:
    PYTHONPATH=parseq-upstream python -m scripts.benchmark_jersey_parseq \
        --dataset runs/number_eval_1080p/dataset.json \
        --labels data/annotations/jersey/number_eval_1080p_labels.json \
        --checkpoint models/jersey_parseq/parseq_soccernet.ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scripts.benchmark_doctr_readers import (
    DEFAULT_THRESHOLDS, load_truth, read_crops, score_at_thresholds,
)
from scripts.label_jersey_numbers import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--path-key", default="crop_path",
                        choices=("crop_path", "context_path"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from handball_cv.jersey.parseq_backend import (
        load_jersey_parseq, read_crops as run_reader,
    )

    readable, unreadable = load_truth(args.labels)
    indices = sorted(set(readable) | unreadable)
    crops = read_crops(args.dataset, indices, args.path_key)

    report = {
        "schema_version": 1, "dataset": str(args.dataset), "labels": str(args.labels),
        "path_key": args.path_key, "readable_labels": len(readable),
        "unreadable_labels": len(unreadable), "models": {},
    }
    for checkpoint in args.checkpoint:
        model, transform = load_jersey_parseq(checkpoint, args.device)
        started = time.monotonic()
        results = run_reader(model, transform, crops)
        elapsed_ms = (time.monotonic() - started) / max(len(crops), 1) * 1000
        raw = {i: (t, c) for i, (t, c) in zip(indices, results)}
        report["models"][checkpoint.stem] = {
            "checkpoint": str(checkpoint),
            "ms_per_crop": elapsed_ms,
            "by_confidence_threshold": score_at_thresholds(
                raw, readable, unreadable, DEFAULT_THRESHOLDS
            ),
            "raw": {str(i): {"text": t, "confidence": c} for i, (t, c) in raw.items()},
        }
        for threshold, s in report["models"][checkpoint.stem][
            "by_confidence_threshold"
        ].items():
            print(f"{checkpoint.stem:<20} @{threshold}  cov {s['coverage']:.2f}  "
                  f"acc {s['accuracy']:.2f}  sel {s['selective_accuracy']:.2f}  "
                  f"abst {s['unreadable_abstention_rate']:.2f}", flush=True)
        print(f"{'':<20} {elapsed_ms:.1f} ms/crop\n")

    if args.output:
        write_json_atomic(args.output, report)
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
