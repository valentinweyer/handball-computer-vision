"""The TorchScript detector's decoding must reproduce rfdetr's PostProcess + predict threshold."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from handball_cv.detection.rfdetr_torchscript import (  # noqa: E402
    CHECKPOINT_TO_PROJECT_CLASS, NUM_SELECT, decode_detections,
)


def logit(p):
    return float(np.log(p / (1 - p)))


def test_scores_labels_and_boxes():
    logits = torch.full((3, 5), -20.0)
    logits[0, 1] = logit(0.9)   # query 0, class 1
    logits[2, 3] = logit(0.6)   # query 2, class 3
    logits[1, 0] = logit(0.2)   # below the threshold
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1], [0.9, 0.95, 0.4, 0.2]])
    xyxy, scores, labels = decode_detections(logits, boxes, width=200, height=100, threshold=0.3)
    assert scores == pytest.approx([0.9, 0.6], abs=1e-6)
    assert labels.tolist() == [1, 3]
    assert xyxy[0] == pytest.approx([80, 30, 120, 70], abs=1e-4)
    # x2 = 1.1 * 200 and y2 = 1.05 * 100 are clamped to the frame, as rfdetr clamps.
    assert xyxy[1] == pytest.approx([140, 85, 200, 100], abs=1e-4)


def test_threshold_is_strict():
    logits = torch.full((1, 2), -20.0)
    logits[0, 0] = logit(0.5)
    _, scores, _ = decode_detections(logits, torch.tensor([[0.5, 0.5, 0.1, 0.1]]), 10, 10,
                                     threshold=float(torch.tensor(logit(0.5)).sigmoid()))
    assert len(scores) == 0


def test_ties_keep_the_lower_flattened_index_and_only_the_top_k_survive():
    queries = NUM_SELECT + 20
    logits = torch.full((queries, 2), logit(0.7))  # every pair ties
    boxes = torch.rand(queries, 4) * 0.5 + 0.25
    _, scores, labels = decode_detections(logits, boxes, 100, 100, threshold=0.3)
    assert len(scores) == NUM_SELECT
    assert labels[:4].tolist() == [0, 1, 0, 1]  # flattened order: query 0 class 0, class 1, ...


def test_only_trained_classes_map_to_project_ids():
    assert CHECKPOINT_TO_PROJECT_CLASS == {0: 1, 1: 2, 2: 3, 3: 4}
