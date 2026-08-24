"""Minimal, network-free inference for a cached RF-DETR ONNX detector."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OfflineRFDETR:
    """Run an exported Roboflow RF-DETR model without resolving remote assets."""

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
        self.session = ort.InferenceSession(
            str(weights), providers=providers, sess_options=options
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.height = int(model_input.shape[2])
        self.width = int(model_input.shape[3])

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        z = np.exp(-np.abs(values))
        return np.where(values >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    def infer_detections(
        self,
        frame_rgb: np.ndarray,
        confidence: float = 0.5,
        max_detections: int = 300,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``xyxy``, confidence and zero-indexed class IDs."""
        original_height, original_width = frame_rgb.shape[:2]
        image = frame_rgb.astype(np.float32) / 255.0
        means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        stds = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - means) / stds
        image = cv2.resize(image, (self.width, self.height))
        image = np.ascontiguousarray(image.transpose(2, 0, 1)[None], np.float32)

        boxes_cxcywh, logits = self.session.run(None, {self.input_name: image})
        scores_flat = self._sigmoid(logits[0].astype(np.float32)).reshape(-1)
        if len(scores_flat) > max_detections:
            indices = np.argpartition(-scores_flat, max_detections)[:max_detections]
            indices = indices[np.argsort(-scores_flat[indices])]
        else:
            indices = np.argsort(-scores_flat)
        scores = scores_flat[indices]
        keep = scores > confidence
        indices, scores = indices[keep], scores[keep]

        class_count = logits.shape[-1]
        query_indices = indices // class_count
        class_ids = (indices % class_count).astype(np.int64)

        # This export contains Roboflow's synthetic background class at index 0.
        keep = class_ids != 0
        query_indices, scores, class_ids = (
            query_indices[keep], scores[keep], class_ids[keep] - 1
        )
        selected = boxes_cxcywh[0, query_indices].astype(np.float32)
        centers, sizes = selected[:, :2], selected[:, 2:]
        boxes = np.concatenate(
            [centers - sizes * 0.5, centers + sizes * 0.5], axis=1
        )
        boxes *= np.asarray(
            [original_width, original_height, original_width, original_height],
            dtype=np.float32,
        )
        np.clip(
            boxes,
            [0, 0, 0, 0],
            [original_width, original_height, original_width, original_height],
            out=boxes,
        )
        return boxes, scores.astype(np.float32), class_ids
