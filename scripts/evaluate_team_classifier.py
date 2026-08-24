"""Reproducible held-out evaluation for per-video team discovery.

Labeled montage crops are excluded from unsupervised fitting. Cluster ids are
mapped to the two human team names only for scoring.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from handball_cv.teams.model import TeamModel, jersey_color_features, torso_boxes


EXCLUDED_LABELS = {"GK", "OTHER", "MIXED", "UNCLEAR", "SKIP"}


def iter_frames(run: dict, frame_dir: Path | None):
    if frame_dir is not None:
        for path in sorted(frame_dir.glob("*.jpg")):
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame
        return
    yield from sv.get_video_frames_generator(str(run["source"]))


def collect(run: dict, labels: dict, stride: int, frame_dir: Path | None):
    boxes = run["boxes"]
    is_goalkeeper = run.get(
        "is_goalkeeper", np.zeros(boxes.shape[1], dtype=bool)
    )
    fit_crops, test_crops, ground_truth, columns = [], [], [], []

    for source_frame, frame_bgr in enumerate(iter_frames(run, frame_dir)):
        row_index = source_frame - 1
        if row_index < 0 or row_index >= len(boxes):
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        row = boxes[row_index]
        valid_columns = np.nonzero(
            np.isfinite(row).all(axis=1) & ~is_goalkeeper
        )[0]

        if row_index % stride == 0:
            for column in valid_columns:
                key = (row_index, int(column))
                if key in labels:
                    continue
                crop = sv.crop_image(
                    frame_rgb, torso_boxes(row[column][None])[0]
                )
                if crop.size:
                    fit_crops.append(crop)

        for column in valid_columns:
            key = (row_index, int(column))
            if key not in labels:
                continue
            crop = sv.crop_image(
                frame_rgb, torso_boxes(row[column][None])[0]
            )
            if crop.size:
                test_crops.append(crop)
                ground_truth.append(labels[key])
                columns.append(int(column))

    return fit_crops, test_crops, ground_truth, columns


def score(prediction, ground_truth, columns):
    names = sorted(set(ground_truth))
    if len(names) != 2:
        raise RuntimeError(f"expected exactly two team labels, got {names}")
    mappings = ({0: names[0], 1: names[1]}, {0: names[1], 1: names[0]})
    mapping = max(
        mappings,
        key=lambda item: sum(
            item[int(predicted)] == truth
            for predicted, truth in zip(prediction, ground_truth)
        ),
    )
    mapped = np.array([mapping[int(value)] for value in prediction])
    crop_accuracy = float(np.mean(mapped == np.asarray(ground_truth)))

    columns_array = np.asarray(columns)
    track_correct = []
    for column in sorted(set(columns)):
        values = mapped[columns_array == column]
        labels, counts = np.unique(values, return_counts=True)
        track_prediction = labels[counts.argmax()]
        truth = ground_truth[columns.index(column)]
        track_correct.append(track_prediction == truth)
    return 100.0 * crop_accuracy, 100.0 * float(np.mean(track_correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.run, allow_pickle=True) as data:
        run = {key: data[key] for key in data.files}
    raw = json.loads(args.labels.read_text())["labels"]
    labels = {}
    for key, value in raw.items():
        if value in EXCLUDED_LABELS:
            continue
        frame, column = key.split("_")
        labels[(int(frame), int(column))] = value

    fit_crops, test_crops, truth, columns = collect(
        run, labels, args.stride, args.frame_dir
    )
    model = TeamModel.fit_from_crops(
        fit_crops, seed=0, device=args.device
    )
    prediction, confidence = model.predict_crops(test_crops)
    crop_accuracy, track_accuracy = score(prediction, truth, columns)

    for _ in range(5):
        jersey_color_features(test_crops)
    started = time.perf_counter()
    for _ in range(100):
        jersey_color_features(test_crops)
    color_seconds = (time.perf_counter() - started) / 100.0

    print(f"fit_crops={len(fit_crops)}")
    print(f"labels_scored={len(test_crops)}")
    print(f"tracks_scored={len(set(columns))}")
    print(f"crop_accuracy={crop_accuracy:.2f}%")
    print(f"track_accuracy={track_accuracy:.2f}%")
    print(f"mean_effective_confidence={np.mean(confidence):.3f}")
    print(f"rejected_crops={(confidence == 0).sum()}")
    print(
        f"color_cost={1000 * color_seconds:.3f} ms/batch "
        f"({1000 * color_seconds / len(test_crops):.4f} ms/crop)"
    )
    print(f"visual_color_fit_agreement={model.visual_color_agreement:.3f}")


if __name__ == "__main__":
    main()
