"""Run the RF-DETR Large player/number detector from its TorchScript export.

`scripts/export_detector.py` traces the fine-tuned checkpoint in the sibling
venv; this runs that trace here with plain torch, so detection no longer needs
a second interpreter. Pre- and post-processing reproduce `rfdetr`'s `predict()`
step for step:

- `to_tensor` (/255), bilinear resize to the square input without antialias,
  then ImageNet normalisation -- resize *before* normalise, as rfdetr does;
- sigmoid over every query x class logit, a stable descending sort, the top
  300, labels as `index % classes` and queries as `index // classes`;
- `cxcywh -> xyxy`, scaled to the frame and clamped to it;
- keep scores strictly above the threshold.

TF32 follows rfdetr by default. Loading an rfdetr model switches TF32 on for
matmuls and convolutions (`set_float32_matmul_precision("high")`), so every
detection cache this project made ran with it, and matching it is what keeps
the trace close to them. It is set only around each call and restored after.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["CHECKPOINT_TO_PROJECT_CLASS", "TorchScriptDetector", "decode_detections"]

_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)
NUM_SELECT = 300

# The fine-tuned checkpoint's class index -> this project's class ids (goalkeeper
# 1, player 2, referee 3, number 4), the same map scripts/cache_number_detections.py
# applies. Any other logit slot (rfdetr's extra no-object slot) is dropped.
CHECKPOINT_TO_PROJECT_CLASS = {0: 1, 1: 2, 2: 3, 3: 4}


def decode_detections(logits, boxes_cxcywh, width: int, height: int, threshold: float):
    """rfdetr PostProcess plus predict's threshold, for one image's raw outputs.

    `logits` is `(queries, classes)`, `boxes_cxcywh` `(queries, 4)` normalised;
    both torch tensors. Returns `(xyxy, scores, checkpoint_labels)` as numpy.
    """
    import torch

    classes = logits.shape[-1]
    probability = logits.float().sigmoid().reshape(-1)
    order = torch.argsort(probability, descending=True, stable=True)[:min(NUM_SELECT, probability.numel())]
    scores = probability[order]
    labels = order % classes
    queries = order // classes
    cx, cy, w, h = boxes_cxcywh.float()[queries].unbind(-1)
    xyxy = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
    scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype, device=xyxy.device)
    xyxy = (xyxy * scale).clamp_min(0.0).clamp(max=scale)
    keep = scores > threshold
    return xyxy[keep].cpu().numpy(), scores[keep].cpu().numpy(), labels[keep].cpu().numpy()


class TorchScriptDetector:
    """`infer(frame_rgb, threshold) -> (xyxy, confidence, project_class_id)`."""

    def __init__(self, weights: Path, device: str = "cuda", resolution: int = 704,
                 tf32: bool = True) -> None:
        import torch

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights}: export it with scripts/export_detector.py in the finetune venv")
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        self.resolution = resolution
        self.tf32 = tf32
        self.mean = torch.tensor(_MEANS, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STDS, device=self.device).view(1, 3, 1, 1)

    def infer(self, frame_rgb: np.ndarray, threshold: float = 0.3):
        torch = self.torch
        height, width = frame_rgb.shape[:2]
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), \
                    torch.backends.cudnn.flags(enabled=True, allow_tf32=self.tf32):
                image = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(self.device)
                image = image.permute(2, 0, 1)[None].float() / 255.0
                image = torch.nn.functional.interpolate(
                    image, size=(self.resolution, self.resolution), mode="bilinear",
                    align_corners=False, antialias=False)
                image = (image - self.mean) / self.std
                outputs = self.model(image)
                boxes, logits = outputs[0][0], outputs[1][0]
                xyxy, scores, labels = decode_detections(logits, boxes, width, height, threshold)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        project = np.array([CHECKPOINT_TO_PROJECT_CLASS.get(int(l), -1) for l in labels], dtype=np.int64)
        keep = project > 0
        return xyxy[keep].astype(np.float64), scores[keep].astype(np.float64), project[keep]
