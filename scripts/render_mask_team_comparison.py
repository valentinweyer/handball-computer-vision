"""Compare raw torso color with guarded MCByte-mask color on overlap frames.

Team classification remains frame-local. MCByte provides propagated masks only;
its track history never supplies, freezes, or votes on a team label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.teams.masks import MaskEvidence, guarded_torso_masks
from scripts.render_raw_team_classification import (
    FIELD_PLAYER_CLASS_ID,
    GOALKEEPER_CLASS_ID,
    TEAM_BGR,
    draw_text,
    frame_detections,
    load_detection_cache,
)
from handball_cv.teams.model import (
    MAX_TORSO_CONTAMINATION,
    MIN_TEAM_VOTE_CONFIDENCE,
    TeamModel,
    crop_quality,
    jersey_color_features,
    masked_jersey_color_features,
    torso_boxes,
    torso_contamination,
)


OVERLAP_THRESHOLD = 0.05
MIN_OBSERVATION_QUALITY = 0.40
ABSTAIN_BGR = (45, 45, 235)
MASK_TINT = 0.45
CODE_BGR = {"A": (255, 190, 0), "B": (0, 120, 255)}
MAX_CONTACT_EVENTS = 80


class _ColorOnlyClassifier:
    """Placeholder that prevents loading SigLIP for a color-only experiment."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-mask-dir", type=Path)
    parser.add_argument("--reference-track-labels", type=Path)
    parser.add_argument("--reference-min-iou", type=float, default=0.65)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def load_manifest(path):
    document = json.loads(path.read_text())
    labels = {}
    samples = []
    for sample in document["samples"]:
        code = (sample.get("annotation") or {}).get("code")
        if code not in ("A", "B"):
            continue
        flat_index = int(sample["flat_detection_index"])
        labels[flat_index] = code
        samples.append(sample)
    return labels, samples


def align_team_codes(model, manifest_path, samples):
    crops = []
    truth = []
    for sample in samples:
        crop_path = manifest_path.parent / sample["torso_path"]
        image = cv2.imread(str(crop_path))
        if image is None:
            continue
        crops.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        truth.append(sample["annotation"]["code"])
    if not crops:
        return {0: "A", 1: "B"}, 0.0, 0
    teams, _confidence = model._color_prediction(jersey_color_features(crops))
    candidates = ({0: "A", 1: "B"}, {0: "B", 1: "A"})
    scores = [
        sum(mapping[int(team)] == code for team, code in zip(teams, truth))
        for mapping in candidates
    ]
    best = int(np.argmax(scores))
    return candidates[best], scores[best] / len(truth), len(truth)


def load_reference_tracks(path):
    if path is None:
        return {}
    document = json.loads(path.read_text())
    result = {}
    for track_id, details in document.get("tracks", {}).items():
        if details.get("code") not in ("A", "B"):
            continue
        result[int(track_id)] = {
            "code": details["code"],
            "excluded": set(int(x) for x in details.get("excluded_samples", [])),
        }
    return result


def reference_ground_truth(
    frame_index, field_boxes, mask_dir, reference_tracks, min_iou,
):
    output = [None] * len(field_boxes)
    if mask_dir is None or not reference_tracks or len(field_boxes) == 0:
        return output
    path = mask_dir / f"{frame_index:05d}.npz"
    if not path.is_file():
        return output
    with np.load(path, allow_pickle=False) as data:
        label_image = data["label"]
    reference_boxes = []
    reference_ids = []
    for track_id, details in reference_tracks.items():
        if frame_index in details["excluded"]:
            continue
        y, x = np.nonzero(label_image == track_id)
        if len(x) == 0:
            continue
        reference_boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        reference_ids.append(track_id)
    if not reference_boxes:
        return output
    iou = sv.box_iou_batch(
        np.asarray(field_boxes, dtype=float),
        np.asarray(reference_boxes, dtype=float),
    )
    detection_rows, reference_columns = linear_sum_assignment(-iou)
    for detection_index, reference_index in zip(detection_rows, reference_columns):
        overlap = float(iou[detection_index, reference_index])
        if overlap < min_iou:
            continue
        track_id = reference_ids[reference_index]
        output[detection_index] = {
            "code": reference_tracks[track_id]["code"],
            "track_id": int(track_id),
            "iou": overlap,
        }
    return output


def tracker_ids_for_people(people, tracked):
    result = np.full(len(people), -1, dtype=int)
    if len(people) == 0 or len(tracked) == 0:
        return result
    flat_to_id = {
        int(flat_index): int(tracker_id)
        for flat_index, tracker_id in zip(
            tracked.data.get("flat_detection_index", []), tracked.tracker_id
        )
    }
    for index, flat_index in enumerate(people.data["flat_detection_index"]):
        result[index] = flat_to_id.get(int(flat_index), -1)
    return result


def effective_quality(raw_crop_quality, contamination, mask_evidence=None):
    if mask_evidence is not None:
        return raw_crop_quality * mask_evidence.quality
    overlap_weight = np.clip(
        1.0 - contamination / MAX_TORSO_CONTAMINATION, 0.0, 1.0
    )
    return raw_crop_quality * float(overlap_weight)


def tint_mask(annotated, torso, crop_mask, color):
    if crop_mask is None or crop_mask.size == 0:
        return
    height, width = annotated.shape[:2]
    x1, y1, x2, y2 = np.rint(torso).astype(int)
    x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
    y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
    roi = annotated[y1:y2, x1:x2]
    if roi.shape[:2] != crop_mask.shape:
        return
    selected = crop_mask.astype(bool)
    if not selected.any():
        return
    overlay = np.asarray(color, dtype=np.float32)
    roi[selected] = np.clip(
        roi[selected] * (1.0 - MASK_TINT) + overlay * MASK_TINT, 0, 255
    ).astype(np.uint8)


def draw_header(frame, source, frame_index, fps, counts, accepted_total):
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, min(height, round(88 * scale))), (8, 12, 18), -1)
    cv2.addWeighted(shade, 0.84, frame, 0.16, 0, frame)
    cv2.putText(
        frame, f"{source.name}   {frame_index / fps:05.2f}s",
        (round(18 * scale), round(31 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
        0.72 * scale, (245, 245, 245), max(1, round(1.5 * scale)), cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "RAW BOX COLOR -> GUARDED MASK COLOR ON OVERLAP | FRAME-LOCAL TEAM "
        "PREDICTION | NO TEMPORAL TEAM VOTE",
        (round(18 * scale), round(65 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * scale, (190, 205, 215), max(1, round(scale)), cv2.LINE_AA,
    )
    text = (
        f"overlap {counts['overlap']}   mask accepted {counts['accepted']}   "
        f"changed {counts['changed']}   abstained {counts['abstained']}   "
        f"accepted total {accepted_total}"
    )
    cv2.putText(
        frame, text,
        (max(round(18 * scale), width - round(690 * scale)), round(31 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (235, 235, 235),
        max(1, round(scale)), cv2.LINE_AA,
    )


def make_event_tile(frame, box, torso, evidence, title, color):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    pad_x, pad_y = (x2 - x1) * 0.45, (y2 - y1) * 0.25
    crop_box = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y])
    cx1, cy1, cx2, cy2 = np.rint(crop_box).astype(int)
    cx1, cx2 = int(np.clip(cx1, 0, width)), int(np.clip(cx2, 0, width))
    cy1, cy2 = int(np.clip(cy1, 0, height)), int(np.clip(cy2, 0, height))
    view = frame[cy1:cy2, cx1:cx2].copy()
    if view.size == 0:
        return None
    if evidence.valid:
        local_torso = np.asarray(torso) - np.array([cx1, cy1, cx1, cy1])
        tint_mask(view, local_torso, evidence.crop_mask, color)
    tile_w, tile_h = 480, 310
    content_h = tile_h - 62
    ratio = min(tile_w / view.shape[1], content_h / view.shape[0])
    resized = cv2.resize(
        view, (max(1, round(view.shape[1] * ratio)), max(1, round(view.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((tile_h, tile_w, 3), 18, dtype=np.uint8)
    ox = (tile_w - resized.shape[1]) // 2
    oy = 62 + (content_h - resized.shape[0]) // 2
    tile[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    cv2.putText(
        tile, title[:72], (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
        (245, 245, 245), 1, cv2.LINE_AA,
    )
    detail = (
        f"mask={evidence.reason} cov={evidence.torso_coverage:.0%} "
        f"keep={evidence.retained_fraction:.0%} spatial={evidence.assignment_score:.2f}"
    )
    cv2.putText(
        tile, detail[:82], (9, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
        color, 1, cv2.LINE_AA,
    )
    return tile


def save_contact_sheet(events, path):
    if not events:
        return None
    events = sorted(events, key=lambda item: item[0], reverse=True)[:24]
    tiles = [item[1] for item in events if item[1] is not None]
    if not tiles:
        return None
    columns = 4
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * 310, columns * 480, 3), 10, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 310:(row + 1) * 310, column * 480:(column + 1) * 480] = tile
    cv2.imwrite(str(path), sheet)
    return path


def accuracy(rows, prediction_key, truth_key):
    usable = [row for row in rows if row.get(truth_key) in ("A", "B") and row.get(prediction_key) in ("A", "B")]
    if not usable:
        return None
    return sum(row[prediction_key] == row[truth_key] for row in usable) / len(usable)


def comparison_metrics(rows, truth_key):
    labeled = [
        row for row in rows
        if row["contamination"] >= OVERLAP_THRESHOLD and row.get(truth_key) in ("A", "B")
    ]
    paired = [row for row in labeled if row.get("mask_code") in ("A", "B")]
    corrected = sum(
        row["raw_code"] != row[truth_key] and row["mask_code"] == row[truth_key]
        for row in paired
    )
    regressed = sum(
        row["raw_code"] == row[truth_key] and row["mask_code"] != row[truth_key]
        for row in paired
    )
    qualified = [row for row in labeled if row["mask_usable"]]
    qualified_predictions = [
        row["mask_code"] if row["mask_usable"] else row["raw_code"]
        for row in labeled
    ]
    qualified_accuracy = (
        sum(
            prediction == row[truth_key]
            for prediction, row in zip(qualified_predictions, labeled)
        ) / len(labeled)
        if labeled else None
    )
    qualified_corrected = sum(
        row["raw_code"] != row[truth_key]
        and row["mask_code"] == row[truth_key]
        for row in qualified
    )
    qualified_regressed = sum(
        row["raw_code"] == row[truth_key]
        and row["mask_code"] != row[truth_key]
        for row in qualified
    )
    result = {
        "overlap_labels": len(labeled),
        "mask_accepted": len(paired),
        "raw_accuracy": accuracy(labeled, "raw_code", truth_key),
        "mask_accuracy_when_accepted": accuracy(paired, "mask_code", truth_key),
        "chosen_accuracy": accuracy(labeled, "chosen_code", truth_key),
        "corrected": corrected,
        "regressed": regressed,
        "qualified_mask_selected": len(qualified),
        "qualified_chosen_accuracy": qualified_accuracy,
        "qualified_corrected": qualified_corrected,
        "qualified_regressed": qualified_regressed,
        "unchanged_correct": sum(
            row["raw_code"] == row[truth_key] and row.get("mask_code") == row[truth_key]
            for row in paired
        ),
        "unchanged_wrong": sum(
            row["raw_code"] != row[truth_key]
            and row.get("mask_code") not in (None, row[truth_key])
            for row in paired
        ),
    }
    if truth_key == "reference_code":
        by_track = defaultdict(list)
        for row in labeled:
            by_track[row["reference_track_id"]].append(row)
        result["per_track"] = {
            str(track_id): {
                "count": len(track_rows),
                "raw_accuracy": accuracy(track_rows, "raw_code", truth_key),
                "chosen_accuracy": accuracy(track_rows, "chosen_code", truth_key),
            }
            for track_id, track_rows in sorted(by_track.items())
        }
        raw_values = [item["raw_accuracy"] for item in result["per_track"].values()]
        chosen_values = [item["chosen_accuracy"] for item in result["per_track"].values()]
        result["raw_macro_track_accuracy"] = float(np.mean(raw_values)) if raw_values else None
        result["chosen_macro_track_accuracy"] = float(np.mean(chosen_values)) if chosen_values else None
    return result


def overlap_bins(rows):
    definitions = (("0.05-0.25", 0.05, 0.25), ("0.25-0.50", 0.25, 0.50), ("0.50+", 0.50, np.inf))
    result = {}
    for name, lower, upper in definitions:
        selected = [row for row in rows if lower <= row["contamination"] < upper]
        result[name] = {
            "detections": len(selected),
            "mask_accepted": sum(row["mask_valid"] for row in selected),
            "label_changed": sum(row["mask_changed"] for row in selected),
            "raw_usable": sum(row["raw_usable"] for row in selected),
            "mask_usable": sum(row["mask_usable"] for row in selected),
        }
    return result


def render(args):
    source = args.video.resolve()
    cache = load_detection_cache(args.detections)
    model = TeamModel.load(
        args.team_model, classifier=_ColorOnlyClassifier(), device=args.device
    )
    manual_labels, alignment_samples = load_manifest(args.manifest)
    team_codes, alignment_accuracy, alignment_count = align_team_codes(
        model, args.manifest, alignment_samples
    )
    reference_tracks = load_reference_tracks(args.reference_track_labels)
    info = sv.VideoInfo.from_video_path(str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = args.output.with_name(f"{args.output.stem}_preview.jpg")
    contact_path = args.output.with_name(f"{args.output.stem}_overlaps.jpg")
    rows_path = args.output.with_name(f"{args.output.stem}_rows.jsonl")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        enable_mask_manager=True,
        mask_config=McByteMaskConfig(
            device=args.device,
            mask_creation_bbox_overlap_threshold=0.20,
        ),
        minimum_mask_average_confidence=0.6,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {args.output}")

    all_rows = []
    events = []
    reason_counts = Counter()
    accepted_total = 0
    best_preview = (-1, None)
    frames_written = 0
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=info.total_frames,
        desc=f"mask team A/B {source.stem}",
    )):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = frame_detections(cache, frame_index)
        start = int(cache["offsets"][frame_index])
        detections.data["flat_detection_index"] = np.arange(
            start, start + len(detections), dtype=np.int64
        )
        person_indices = np.flatnonzero(np.isin(
            detections.class_id, [FIELD_PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID]
        ))
        people = detections[person_indices]
        tracked = tracker.update(people, frame=frame_rgb)
        people_tracker_ids = tracker_ids_for_people(people, tracked)
        evidence = guarded_torso_masks(
            people.xyxy, people_tracker_ids, tracker._last_mask_output
        )
        field_slots = np.flatnonzero(people.class_id == FIELD_PLAYER_CLASS_ID)
        field_boxes = people.xyxy[field_slots]
        field_flat = people.data["flat_detection_index"][field_slots]
        field_evidence = [evidence[int(slot)] for slot in field_slots]
        crops = model._crop_boxes(frame_rgb, field_boxes)
        raw_team, raw_confidence = model._color_prediction(
            jersey_color_features(crops)
        )
        crop_scores = np.asarray([crop_quality(crop).score for crop in crops])
        contamination = torso_contamination(field_boxes, people.xyxy)

        mask_team = np.full(len(field_boxes), -1, dtype=int)
        mask_confidence = np.zeros(len(field_boxes), dtype=float)
        valid_slots = [
            index for index, item in enumerate(field_evidence)
            if item.valid and contamination[index] >= OVERLAP_THRESHOLD
        ]
        if valid_slots:
            masked_features = masked_jersey_color_features(
                [crops[index] for index in valid_slots],
                [field_evidence[index].crop_mask for index in valid_slots],
            )
            predicted_team, predicted_confidence = model._color_prediction(masked_features)
            mask_team[valid_slots] = predicted_team
            mask_confidence[valid_slots] = predicted_confidence

        reference = reference_ground_truth(
            frame_index, field_boxes, args.reference_mask_dir,
            reference_tracks, args.reference_min_iou,
        )
        torsos = torso_boxes(field_boxes)
        annotated = frame_bgr.copy()
        counts = {"overlap": 0, "accepted": 0, "changed": 0, "abstained": 0}
        scale = max(info.width / 1920.0, 1.0)
        line = max(2, round(3 * scale))

        for slot, (box, torso, flat_index, item) in enumerate(zip(
            field_boxes, torsos, field_flat, field_evidence
        )):
            raw_code = team_codes[int(raw_team[slot])]
            is_overlap = bool(contamination[slot] >= OVERLAP_THRESHOLD)
            mask_code = (
                team_codes[int(mask_team[slot])] if mask_team[slot] >= 0 else None
            )
            chosen_code = mask_code if is_overlap and item.valid else raw_code
            raw_quality = effective_quality(crop_scores[slot], contamination[slot])
            mask_quality = effective_quality(crop_scores[slot], contamination[slot], item)
            raw_usable = bool(
                raw_confidence[slot] >= MIN_TEAM_VOTE_CONFIDENCE
                and raw_quality >= MIN_OBSERVATION_QUALITY
            )
            mask_usable = bool(
                mask_code is not None
                and mask_confidence[slot] >= MIN_TEAM_VOTE_CONFIDENCE
                and mask_quality >= MIN_OBSERVATION_QUALITY
            )
            manual_code = manual_labels.get(int(flat_index))
            reference_item = reference[slot]
            row = {
                "frame_index": int(frame_index),
                "flat_detection_index": int(flat_index),
                "tracker_id": int(people_tracker_ids[field_slots[slot]]),
                "bbox_xyxy": [float(x) for x in box],
                "contamination": float(contamination[slot]),
                "crop_quality": float(crop_scores[slot]),
                "raw_team": int(raw_team[slot]),
                "raw_code": raw_code,
                "raw_confidence": float(raw_confidence[slot]),
                "raw_effective_quality": float(raw_quality),
                "raw_usable": raw_usable,
                "mask_valid": bool(item.valid and is_overlap),
                "mask_reason": item.reason if is_overlap else "not_overlapped",
                "mask_team": int(mask_team[slot]) if mask_team[slot] >= 0 else None,
                "mask_code": mask_code,
                "mask_confidence": float(mask_confidence[slot]),
                "mask_effective_quality": float(mask_quality),
                "mask_usable": mask_usable,
                "mask_torso_coverage": float(item.torso_coverage),
                "mask_retained_fraction": float(item.retained_fraction),
                "mask_assignment_score": float(item.assignment_score),
                "mask_assignment_margin": float(item.assignment_margin),
                "mask_tracker_agrees": item.tracker_agrees,
                "mask_changed": bool(mask_code is not None and mask_code != raw_code),
                "chosen_code": chosen_code,
                "manual_code": manual_code,
                "reference_code": reference_item["code"] if reference_item else None,
                "reference_track_id": reference_item["track_id"] if reference_item else None,
                "reference_iou": reference_item["iou"] if reference_item else None,
            }
            all_rows.append(row)

            x1, y1, x2, y2 = np.rint(box).astype(int)
            tx1, ty1, tx2, ty2 = np.rint(torso).astype(int)
            raw_color = CODE_BGR[raw_code]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), raw_color, line)
            if is_overlap:
                counts["overlap"] += 1
                reason_counts[item.reason] += 1
                if item.valid:
                    counts["accepted"] += 1
                    accepted_total += 1
                    mask_color = CODE_BGR[mask_code]
                    if mask_code != raw_code:
                        counts["changed"] += 1
                    tint_mask(annotated, torso, item.crop_mask, mask_color)
                    cv2.rectangle(
                        annotated, (tx1, ty1), (tx2, ty2), mask_color,
                        max(1, round(2 * scale)),
                    )
                    label = (
                        f"BOX {raw_code} {raw_confidence[slot]:.0%} -> "
                        f"MASK {mask_code} {mask_confidence[slot]:.0%} | "
                        f"ov={contamination[slot]:.0%} cov={item.torso_coverage:.0%}"
                    )
                    label_color = mask_color
                else:
                    counts["abstained"] += 1
                    label = (
                        f"BOX {raw_code} {raw_confidence[slot]:.0%} -> "
                        f"MASK ABSTAIN ({item.reason}) | ov={contamination[slot]:.0%}"
                    )
                    label_color = ABSTAIN_BGR
                ground_truth = manual_code or (reference_item["code"] if reference_item else None)
                title = (
                    f"f{frame_index} ov={contamination[slot]:.0%} | "
                    f"BOX {raw_code} {raw_confidence[slot]:.0%} -> "
                    f"MASK {mask_code or 'X'} {mask_confidence[slot]:.0%}"
                    + (f" | GT {ground_truth}" if ground_truth else "")
                )
                priority = (
                    (1000 if manual_code else 0)
                    + (500 if reference_item else 0)
                    + (100 if mask_code is not None and mask_code != raw_code else 0)
                    + contamination[slot]
                )
                tile = make_event_tile(
                    frame_bgr, box, torso, item, title,
                    CODE_BGR[mask_code] if item.valid else ABSTAIN_BGR,
                )
                events.append((priority, tile))
                if len(events) > MAX_CONTACT_EVENTS:
                    events.sort(key=lambda event: event[0], reverse=True)
                    del events[MAX_CONTACT_EVENTS:]
            else:
                label = None
            if label is not None:
                draw_text(
                    annotated, label,
                    (max(0, x1), max(round(108 * scale), y1 - round(5 * scale))),
                    label_color, scale,
                )

        draw_header(
            annotated, source, frame_index, info.fps, counts, accepted_total
        )
        writer.write(annotated)
        frames_written += 1
        score = counts["accepted"] * 3 + counts["changed"] * 5 + counts["overlap"]
        if score > best_preview[0]:
            best_preview = (score, annotated.copy())

    writer.release()
    if best_preview[1] is not None:
        preview = best_preview[1]
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(
                preview, (1600, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)
    saved_contact = save_contact_sheet(events, contact_path)
    with rows_path.open("w") as output_file:
        for row in all_rows:
            output_file.write(json.dumps(row) + "\n")

    overlaps = [row for row in all_rows if row["contamination"] >= OVERLAP_THRESHOLD]
    result = {
        "source": str(source),
        "output": str(args.output),
        "preview": str(preview_path),
        "contact_sheet": str(saved_contact) if saved_contact else None,
        "rows": str(rows_path),
        "frames": frames_written,
        "experiment": "raw_color_vs_guarded_mcbyte_mask_color",
        "team_prediction_is_temporal": False,
        "tracker_supplies_team_label": False,
        "mask_creation_overlap_threshold": 0.20,
        "team_code_mapping": {str(key): value for key, value in team_codes.items()},
        "mapping_label_accuracy": alignment_accuracy,
        "mapping_label_count": alignment_count,
        "field_detections": len(all_rows),
        "overlap_detections": len(overlaps),
        "mask_accepted_on_overlap": sum(row["mask_valid"] for row in overlaps),
        "mask_acceptance_rate": (
            sum(row["mask_valid"] for row in overlaps) / len(overlaps)
            if overlaps else None
        ),
        "mask_changed_label": sum(row["mask_changed"] for row in overlaps),
        "raw_unusable_but_mask_usable": sum(
            not row["raw_usable"] and row["mask_usable"] for row in overlaps
        ),
        "mask_reasons": dict(reason_counts),
        "overlap_bins": overlap_bins(all_rows),
        "manual_ground_truth": comparison_metrics(all_rows, "manual_code"),
        "reference_ground_truth": comparison_metrics(all_rows, "reference_code"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    if (args.reference_mask_dir is None) != (args.reference_track_labels is None):
        raise SystemExit(
            "--reference-mask-dir and --reference-track-labels must be supplied together"
        )
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
