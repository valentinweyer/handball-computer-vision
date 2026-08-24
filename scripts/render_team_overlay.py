"""Clean, broadcast-style team overlay: MCByte tracking plus reversible team labels.

Same pipeline as `render_mcbyte_team_correction.py` (MCByte tracker,
`IdentityManager`, the tracker/team-error-decoupling logic in
`identity_manager.py`), but the overlay style matches the notebook's original
"Full video team clustering" cell: a translucent team-colored mask fill plus a
team-colored box border, no per-player debug text. Useful for visually
reviewing team-label quality without the dense diagnostic readout.

Masks are pulled directly from MCByte's `tracklet_mask_dict` (the tracker's own
belief about which mask belongs to which tracklet), not through the guarded
spatial-assignment/tracker-agreement checks in `mask_team_features.py`. That
guard exists because an unearned mask must not become a confident *team-color*
observation; here the mask only decides which pixels to tint, so an occasional
wrong mask is a cosmetic glitch, not a label error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.tracking.identity import TEAM_SWITCH_OBSERVATIONS, IdentityManager
from scripts.render_raw_team_classification import frame_detections, load_detection_cache
from handball_cv.teams.model import MIN_STABLE_TEAM_CONFIDENCE, TeamModel

TEAM_BGR = {0: (255, 190, 0), 1: (0, 120, 255)}
# Two different "not a stable team color" states, kept visually distinct:
# a player who has never had a qualified observation yet (bootstrapping,
# expected and uninteresting) versus an established, previously-stable
# player whose label is under genuine active opposition right now (the
# signal worth noticing -- often a crossing or a suspected tracker swap).
NEW_UNCERTAIN_BGR = (0, 220, 255)
CONTESTED_BGR = (0, 0, 220)
GOALKEEPER_BGR = (180, 180, 180)
GOALKEEPER_CLASS_ID = 1
MASK_FILL_ALPHA = 0.45
BOX_THICKNESS = 2
LEGEND = [
    (TEAM_BGR[0], "Team A"),
    (TEAM_BGR[1], "Team B"),
    (NEW_UNCERTAIN_BGR, "New / uncertain"),
    (CONTESTED_BGR, "Contested"),
    (GOALKEEPER_BGR, "GK"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-masks", action="store_true")
    return parser.parse_args()


def _player_color(player) -> tuple[int, int, int]:
    if player.is_goalkeeper:
        return GOALKEEPER_BGR
    if player.team_is_provisional:
        return NEW_UNCERTAIN_BGR
    if player.team_confidence < MIN_STABLE_TEAM_CONFIDENCE:
        return CONTESTED_BGR
    return TEAM_BGR[player.voted_team_id]


def _tracklet_masks(mask_output, tracker_ids, frame_shape) -> list[np.ndarray | None]:
    """One boolean (H, W) mask per tracker_id, or None if MCByte has none for it."""
    if mask_output is None or mask_output.masks is None or len(mask_output.masks) == 0:
        return [None] * len(tracker_ids)
    masks = np.asarray(mask_output.masks, dtype=bool)
    row_by_tracker = mask_output.tracklet_mask_dict
    out = []
    for tracker_id in tracker_ids:
        row = row_by_tracker.get(int(tracker_id))
        if row is None or not (0 <= int(row) < len(masks)):
            out.append(None)
            continue
        out.append(masks[int(row)])
    return out


def draw_legend(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    x, y = round(18 * scale), height - round(18 * scale)
    chip = round(14 * scale)
    for color, label in reversed(LEGEND):
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), color, -1)
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), (20, 20, 20), 1)
        (text_w, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, round(scale))
        )
        cv2.putText(
            frame, label, (x + chip + round(6 * scale), y - round(2 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
            max(1, round(scale)), cv2.LINE_AA,
        )
        y -= chip + round(10 * scale)


def render(source, cache_path, model_path, output_path, device, enable_masks):
    source = source.resolve()
    cache = load_detection_cache(cache_path)
    model = TeamModel.load(model_path, device=device)
    info = sv.VideoInfo.from_video_path(str(source))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_name(f"{output_path.stem}_preview.jpg")

    if enable_masks and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        enable_mask_manager=enable_masks,
        mask_config=McByteMaskConfig(device=device) if enable_masks else None,
        minimum_mask_average_confidence=0.6,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    identity = IdentityManager(
        model,
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        team_switch_observations=TEAM_SWITCH_OBSERVATIONS,
    )
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")

    preview = None
    frames_written = 0
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=info.total_frames,
        desc=f"team overlay {source.stem}",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = frame_detections(cache, frame_index)
        tracked = tracker.update(detections, frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        player_ids = identity.update(frame_index, frame_rgb, tracked)
        alive = getattr(tracker, "tracked_objects", sv.Detections.empty())
        alive_ids = (
            alive.tracker_id
            if getattr(alive, "tracker_id", None) is not None
            else np.empty(0, dtype=int)
        )
        identity.retire_missing(frame_index, alive_ids)

        annotated = frame_bgr.copy()
        masks = _tracklet_masks(
            getattr(tracker, "_last_mask_output", None),
            tracked.tracker_id, frame_bgr.shape,
        ) if enable_masks else [None] * len(tracked)

        fill = np.zeros_like(frame_bgr)
        any_fill = False
        for box, mask, player_id in zip(tracked.xyxy, masks, player_ids):
            player = identity.players.get(int(player_id))
            if player is None:
                player = next(
                    item for item in reversed(identity.retired)
                    if item.player_id == int(player_id)
                )
            color = _player_color(player)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            if mask is not None and mask.any():
                fill[mask] = color
                any_fill = True
            else:
                cv2.rectangle(fill, (x1, y1), (x2, y2), color, -1)
                any_fill = True
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        if any_fill:
            painted = fill.any(axis=2)
            annotated[painted] = cv2.addWeighted(
                annotated, 1 - MASK_FILL_ALPHA, fill, MASK_FILL_ALPHA, 0,
            )[painted]

        draw_legend(annotated)
        writer.write(annotated)
        frames_written += 1
        if frame_index == info.total_frames // 2:
            preview = annotated.copy()

    writer.release()
    if preview is not None:
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(
                preview, (1600, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)

    everyone = list(identity.players.values()) + identity.retired
    result = {
        "source": str(source),
        "output": str(output_path),
        "preview": str(preview_path),
        "frames": frames_written,
        "mcbyte_masks": enable_masks,
        "label_changes_total": sum(player.team_switches for player in everyone),
        **identity.summary(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    result = render(
        args.video, args.detections, args.team_model, args.output,
        args.device, not args.no_masks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
