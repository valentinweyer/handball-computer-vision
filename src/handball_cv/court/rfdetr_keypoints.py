"""Offline inference for the RF-DETR court-keypoint model (Roboflow version 4).

Version 4 of ``keypointv333-uwois-xprdi`` is an ``rfdetr-keypoint-preview``
model, a type the pinned ``inference`` 0.62.0 cannot load, which is why the
pipeline ran the YOLO11s-pose version 3 instead. Run directly from its ONNX
export (fetched read-only from Roboflow's ``/ort`` endpoint; the PyTorch file
is not downloadable on this plan), it is the better model on the export's own
held-out test images, decisively so on the middle-court framing:

=====================  ===========  ===========
test images            v4 RF-DETR   v3 YOLO
=====================  ===========  ===========
71 goal-view: recall      0.97         0.84
  median px (1280x720)    9.1          9.5
  court error cm           39           56
10 middle-court: recall   0.94         0.63
  median px               30.5         51.3
  court error cm           75          653
=====================  ===========  ===========

(``scripts/vlm_court_keypoints.py score``; court error is where the hand-labelled
points land through the predicted homography, against a 20-36 cm hand floor.)
Its remaining error on middle-court frames is mostly horizontal, consistent with
a 16:9 frame squashed into a 576 px square.

The ONNX graph returns, per query, a box, class logits and a keypoint block of
37 slots per keypoint class, each ``(x, y, visibility logit, ..., precision
Cholesky terms)`` with x/y normalised to the input frame. Two layouts occur:
Roboflow's export puts its synthetic background class first, so ``Court`` is
logit 1 and the block is ``(2 * 37, 8)``; a local ``rfdetr`` export trained on
the YOLO-pose download (``rfdetr-handball-finetune``, ``train_keypoints.py``)
has ``Court`` at logit 0 and a ``(37, 8)`` block. In both, ``Court`` is the last
keypoint group, which is how :func:`decode_court_keypoints` tells them apart. The slot order is the export's, so slots go
through :data:`~handball_cv.court.keypoints.KEYPOINT_TO_VERTEX` like version 3's;
Roboflow's ``keypoints_metadata`` names agree with that mapping on all 37 slots
for both versions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .keypoints import KEYPOINT_COUNT

__all__ = ["OfflineCourtKeypoints", "decode_court_keypoints"]

_MEANS = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STDS = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def decode_court_keypoints(
    logits: np.ndarray, keypoints: np.ndarray, width: int, height: int
) -> tuple[list[tuple[int, float, float, float]], float]:
    """All 37 slots of the best ``Court`` query, in pixels.

    Args:
        logits: ``(queries, classes)`` class logits for one image.
        keypoints: ``(queries, groups * 37, D)`` keypoint block for one image;
            ``Court`` is the last group and its logit index (see module docstring).
        width, height: the frame the coordinates should come back in.

    Returns:
        ``[(slot, x, y, visibility), ...]`` for every slot -- the tuple
        ``CourtTracker.update`` consumes, unfiltered so a caller can cache the
        whole output and choose a threshold later -- and the court score.
    """
    groups = keypoints.shape[1] // KEYPOINT_COUNT
    if groups * KEYPOINT_COUNT != keypoints.shape[1] or groups > logits.shape[1]:
        raise ValueError(f"unexpected court-keypoint layout: logits {logits.shape}, keypoints {keypoints.shape}")
    court = groups - 1
    query = int(np.argmax(logits[:, court]))
    block = keypoints[query].reshape(groups, KEYPOINT_COUNT, -1)[court]
    visibility = _sigmoid(block[:, 2].astype(np.float64))
    points = [
        (slot, float(block[slot, 0] * width), float(block[slot, 1] * height), float(visibility[slot]))
        for slot in range(KEYPOINT_COUNT)
    ]
    return points, float(_sigmoid(np.float64(logits[query, court])))


class OfflineCourtKeypoints:
    """Run the exported RF-DETR court-keypoint model without network access."""

    def __init__(self, weights: Path, device: str = "cuda") -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(weights)
        available = set(ort.get_available_providers())
        providers = []
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(weights), providers=providers, sess_options=options)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.height = int(model_input.shape[2])
        self.width = int(model_input.shape[3])

    def infer(self, frame_rgb: np.ndarray) -> tuple[list[tuple[int, float, float, float]], float]:
        """Keypoints for one RGB frame; see :func:`decode_court_keypoints`."""
        height, width = frame_rgb.shape[:2]
        image = (frame_rgb.astype(np.float32) / 255.0 - _MEANS) / _STDS
        # "Stretch to" the square input, as the model was trained.
        image = cv2.resize(image, (self.width, self.height))
        image = np.ascontiguousarray(image.transpose(2, 0, 1)[None], np.float32)
        _, logits, keypoints = self.session.run(None, {self.input_name: image})
        return decode_court_keypoints(logits[0], keypoints[0], width, height)
