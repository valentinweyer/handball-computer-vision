"""Top-down COCO-17 body keypoints for boxes the project's detector produced."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

DEFAULT_CHECKPOINT = "usyd-community/vitpose-base-simple"

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KEYPOINT_COUNT = len(KEYPOINT_NAMES)

# COCO limb topology, as (start, end) indices into KEYPOINT_NAMES.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


class ViTPoseEstimator:
    """Describe the pose of supplied person boxes, one frame at a time.

    Top-down by construction: this never looks for people, it only describes
    boxes it is handed, so the project's RF-DETR detector stays the single
    source of who is on the frame. A box in, a pose out, in the same order --
    which keeps the result attachable to whatever produced the boxes without
    a second association step.
    """

    def __init__(
        self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = (
            VitPoseForPoseEstimation.from_pretrained(checkpoint)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def estimate(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray
    ) -> np.ndarray:
        """Return ``(N, 17, 3)`` of x, y, score, row-aligned with ``boxes_xyxy``."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, KEYPOINT_COUNT, 3), dtype=np.float32)

        # The processor speaks COCO xywh; every cache in this project is xyxy.
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]

        inputs = self.processor(
            frame_rgb, boxes=[boxes_xywh], return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        poses = self.processor.post_process_pose_estimation(
            outputs, boxes=[boxes_xywh]
        )[0]

        keypoints = np.zeros((len(boxes), KEYPOINT_COUNT, 3), dtype=np.float32)
        for index, pose in enumerate(poses):
            keypoints[index, :, :2] = pose["keypoints"].cpu().numpy()
            keypoints[index, :, 2] = pose["scores"].cpu().numpy()
        return keypoints
