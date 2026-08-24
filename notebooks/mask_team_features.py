"""Guarded MCByte mask evidence for per-frame jersey classification."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from team_model import torso_boxes


MIN_MASK_AVERAGE_CONFIDENCE = 0.60
MIN_MASK_BOX_COVERAGE = 0.80
MIN_MASK_BOX_FILL = 0.08
MIN_SPATIAL_ASSIGNMENT_SCORE = 0.25
MIN_SPATIAL_ASSIGNMENT_MARGIN = 0.02
MIN_SAFE_TORSO_COVERAGE = 0.15
MIN_SAFE_RETENTION = 0.50
MIN_SAFE_PIXELS = 80
MIN_RESIZED_SAFE_PIXELS = 24


@dataclass(frozen=True)
class MaskEvidence:
    """One detector crop's spatially validated mask-interior evidence."""

    valid: bool
    reason: str
    crop_mask: np.ndarray | None = None
    mask_index: int | None = None
    mask_tracker_id: int | None = None
    tracker_agrees: bool | None = None
    mask_confidence: float = 0.0
    box_coverage: float = 0.0
    box_fill: float = 0.0
    assignment_score: float = 0.0
    assignment_margin: float = 0.0
    torso_coverage: float = 0.0
    retained_fraction: float = 0.0
    selected_pixels: int = 0
    quality: float = 0.0


def _rounded_box(box, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    return (
        int(np.clip(x1, 0, width)),
        int(np.clip(y1, 0, height)),
        int(np.clip(x2, 0, width)),
        int(np.clip(y2, 0, height)),
    )


def _mask_box_geometry(boxes_xyxy, masks):
    """Return mask-in-box coverage, fill, and their geometric-mean score."""
    masks = np.asarray(masks, dtype=bool)
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    coverage = np.zeros((len(boxes), len(masks)), dtype=float)
    fill = np.zeros_like(coverage)
    if not len(boxes) or not len(masks):
        return coverage, fill, coverage.copy()
    mask_area = masks.reshape(len(masks), -1).sum(axis=1).astype(float)
    for detection_index, box in enumerate(boxes):
        x1, y1, x2, y2 = _rounded_box(box, masks.shape[1:])
        box_area = max((x2 - x1) * (y2 - y1), 1)
        if x2 <= x1 or y2 <= y1:
            continue
        intersection = masks[:, y1:y2, x1:x2].reshape(len(masks), -1).sum(axis=1)
        coverage[detection_index] = np.divide(
            intersection,
            mask_area,
            out=np.zeros(len(masks), dtype=float),
            where=mask_area > 0,
        )
        fill[detection_index] = intersection / box_area
    return coverage, fill, np.sqrt(coverage * fill)


def _spatial_assignments(boxes_xyxy, masks):
    coverage, fill, score = _mask_box_geometry(boxes_xyxy, masks)
    assignments = np.full(len(boxes_xyxy), -1, dtype=int)
    margins = np.zeros(len(boxes_xyxy), dtype=float)
    if score.size == 0:
        return assignments, coverage, fill, score, margins
    detection_rows, mask_columns = linear_sum_assignment(-score)
    assignments[detection_rows] = mask_columns
    for detection_index, mask_index in zip(detection_rows, mask_columns):
        alternatives = np.delete(score[detection_index], mask_index)
        second = float(alternatives.max()) if len(alternatives) else 0.0
        margins[detection_index] = float(score[detection_index, mask_index] - second)
    return assignments, coverage, fill, score, margins


def guarded_torso_masks(
    boxes_xyxy,
    tracker_ids,
    mask_output,
    *,
    require_tracker_agreement: bool = True,
):
    """Build reliable torso pixels without trusting a propagated ID blindly.

    Current detector boxes are assigned one-to-one to current-frame masks by
    geometry. The track-ID mapping is a separate consistency check. Boundaries
    of both the target and nearby masks are removed before color extraction.
    """
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    tracker_ids = np.asarray(tracker_ids, dtype=int).reshape(-1)
    if len(boxes) != len(tracker_ids):
        raise ValueError("expected one tracker ID per box")
    if (
        mask_output is None
        or mask_output.masks is None
        or len(mask_output.masks) == 0
    ):
        return [MaskEvidence(False, "no_mask_output") for _ in boxes]

    all_masks = np.asarray(mask_output.masks, dtype=bool)
    if all_masks.ndim != 3:
        raise ValueError(f"expected masks with shape (K,H,W), got {all_masks.shape}")
    row_to_tracker = {
        int(row): int(tracker_id)
        for tracker_id, row in mask_output.tracklet_mask_dict.items()
        if 0 <= int(row) < len(all_masks)
    }
    confidences = mask_output.mask_avg_prob_dict or {}
    candidate_rows = np.array(
        [
            row for row in sorted(row_to_tracker)
            if float(confidences.get(row_to_tracker[row], 0.0))
            >= MIN_MASK_AVERAGE_CONFIDENCE
            and all_masks[row].any()
        ],
        dtype=int,
    )
    if len(candidate_rows) == 0:
        return [MaskEvidence(False, "no_confident_mask") for _ in boxes]

    masks = all_masks[candidate_rows]
    assignments, coverage, fill, score, margins = _spatial_assignments(boxes, masks)
    torso_regions = torso_boxes(boxes)
    results = []
    for detection_index, (box, torso, tracker_id) in enumerate(
        zip(boxes, torso_regions, tracker_ids)
    ):
        local_row = int(assignments[detection_index])
        if local_row < 0:
            results.append(MaskEvidence(False, "unassigned"))
            continue
        mask_index = int(candidate_rows[local_row])
        mask_tracker_id = row_to_tracker[mask_index]
        mapped_row = mask_output.tracklet_mask_dict.get(int(tracker_id))
        tracker_agrees = mapped_row is not None and int(mapped_row) == mask_index
        mask_confidence = float(confidences.get(mask_tracker_id, 0.0))
        box_coverage = float(coverage[detection_index, local_row])
        box_fill = float(fill[detection_index, local_row])
        assignment_score = float(score[detection_index, local_row])
        assignment_margin = float(margins[detection_index])

        reason = None
        if box_coverage < MIN_MASK_BOX_COVERAGE:
            reason = "low_box_coverage"
        elif box_fill < MIN_MASK_BOX_FILL:
            reason = "low_box_fill"
        elif assignment_score < MIN_SPATIAL_ASSIGNMENT_SCORE:
            reason = "weak_spatial_assignment"
        elif assignment_margin < MIN_SPATIAL_ASSIGNMENT_MARGIN:
            reason = "ambiguous_spatial_assignment"
        elif require_tracker_agreement and not tracker_agrees:
            reason = "tracker_mask_disagreement"

        target = sv.crop_image(all_masks[mask_index].astype(np.uint8), torso).astype(bool)
        if target.size == 0 or not target.any():
            reason = reason or "empty_target_torso"
            safe = np.zeros_like(target, dtype=bool)
        else:
            radius = int(np.clip(round(min(target.shape) * 0.025), 1, 5))
            target_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            safe = cv2.erode(target.astype(np.uint8), target_kernel).astype(bool)
            neighbor = np.zeros_like(target, dtype=np.uint8)
            for other_index in candidate_rows:
                if int(other_index) == mask_index:
                    continue
                other = sv.crop_image(
                    all_masks[int(other_index)].astype(np.uint8), torso
                )
                if other.shape == neighbor.shape and other.any():
                    neighbor |= other
            if neighbor.any():
                neighbor_radius = min(5, radius + 1)
                neighbor_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * neighbor_radius + 1, 2 * neighbor_radius + 1),
                )
                unsafe_boundary = cv2.dilate(neighbor, neighbor_kernel).astype(bool)
                safe &= ~unsafe_boundary

        selected_pixels = int(safe.sum())
        torso_pixels = max(int(safe.size), 1)
        target_pixels = max(int(target.sum()), 1)
        torso_coverage = selected_pixels / torso_pixels
        retained_fraction = selected_pixels / target_pixels
        resized_pixels = int(cv2.resize(
            safe.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST
        ).sum()) if safe.size else 0
        if reason is None and selected_pixels < MIN_SAFE_PIXELS:
            reason = "too_few_pixels"
        elif reason is None and resized_pixels < MIN_RESIZED_SAFE_PIXELS:
            reason = "too_few_resized_pixels"
        elif reason is None and torso_coverage < MIN_SAFE_TORSO_COVERAGE:
            reason = "low_torso_coverage"
        elif reason is None and retained_fraction < MIN_SAFE_RETENTION:
            reason = "low_safe_retention"

        coverage_quality = float(np.clip(torso_coverage / 0.45, 0.0, 1.0))
        retention_quality = float(np.clip(retained_fraction / 0.80, 0.0, 1.0))
        spatial_quality = float(np.clip(assignment_score / 0.55, 0.0, 1.0))
        quality = float(
            (mask_confidence * coverage_quality * retention_quality * spatial_quality)
            ** 0.25
        )
        results.append(MaskEvidence(
            valid=reason is None,
            reason=reason or "accepted",
            crop_mask=safe,
            mask_index=mask_index,
            mask_tracker_id=mask_tracker_id,
            tracker_agrees=tracker_agrees,
            mask_confidence=mask_confidence,
            box_coverage=box_coverage,
            box_fill=box_fill,
            assignment_score=assignment_score,
            assignment_margin=assignment_margin,
            torso_coverage=torso_coverage,
            retained_fraction=retained_fraction,
            selected_pixels=selected_pixels,
            quality=quality if reason is None else 0.0,
        ))
    return results
