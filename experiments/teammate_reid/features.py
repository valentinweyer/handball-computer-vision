"""Frozen PRTReID features for the prototype: the runtime global embedding plus
fixed horizontal stripes of the backbone feature map.

`global` is exactly what the runtime re-ID consumes (`PRTReIDBackend`,
`feature_kind="global"`). The stripes are what a trainable head can use to
weight body regions differently from the SoccerNet-trained global pooling: the
1920-channel HRNet map is average-pooled inside four fixed bands of the crop,
weighted by PRTReID's own learned foreground attention. No keypoints are used;
the bands are fixed fractions of the player box (head/shoulders, torso,
thighs, lower legs/feet).
"""
from __future__ import annotations

import numpy as np
import torch

from handball_cv.embeddings.prtreid import GLOBAL, PRTReIDBackend

STRIPES = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.75), (0.75, 1.0))
STRIPE_NAMES = ("head_shoulders", "torso", "thighs", "lower_legs_feet")


class StripeEncoder:
    def __init__(self, prtreid_root, checkpoint, device: str = "cuda"):
        self.backend = PRTReIDBackend(prtreid_root, checkpoint, device, feature_kind="global")

    @torch.inference_mode()
    def encode(self, crops_rgb: list, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """(N, 256) global embeddings and (N, 4, 1920) stripe features."""
        b = self.backend
        globals_, stripes = [], []
        for start in range(0, len(crops_rgb), batch_size):
            images = torch.stack([b._array_tensor(c) for c in crops_rgb[start:start + batch_size]])
            images = images.to(b.device)
            output = b.model((images - b.mean) / b.std)
            globals_.append(output[0][GLOBAL].float().cpu().numpy())
            fmap = output[6].float()                                  # B, C, H, W
            attention = output[7]["foreg"].float().clamp(min=0)       # B, H, W
            height = fmap.shape[2]
            pooled = []
            for lo, hi in STRIPES:
                rows = slice(int(round(lo * height)), int(round(hi * height)))
                weight = attention[:, rows][:, None]                  # B, 1, h, W
                pooled.append((fmap[:, :, rows] * weight).sum((2, 3))
                              / weight.sum((2, 3)).clamp(min=1e-6))
            stripes.append(torch.stack(pooled, 1).cpu().numpy())
        return (np.concatenate(globals_).astype(np.float32),
                np.concatenate(stripes).astype(np.float16))
