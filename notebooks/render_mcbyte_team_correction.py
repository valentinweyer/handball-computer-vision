"""Render MCByte tracks with continuously monitored, reversible team labels."""
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

from identity_manager import (
    TEAM_OBSERVATION_INTERVAL,
    TEAM_SWITCH_MIN_QUALITY,
    TEAM_SWITCH_OBSERVATIONS,
    IdentityManager,
)
from render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)
from team_model import MIN_STABLE_TEAM_CONFIDENCE, TeamModel, torso_boxes

TEAM_BGR = {0: (255, 190, 0), 1: (0, 120, 255)}
# A brand-new player still bootstrapping evidence looks different from an
# established player whose label is under genuine active opposition right
# now -- the second case is the interesting one to notice, not the first.
NEW_UNCERTAIN_BGR = (0, 220, 255)
CONTESTED_BGR = (0, 0, 220)
GOALKEEPER_BGR = (180, 180, 180)
GOALKEEPER_CLASS_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-masks", action="store_true")
    return parser.parse_args()


def draw_text(frame, text, origin, color, scale):
    font_scale = 0.48 * scale
    thickness = max(1, round(1.5 * scale))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        frame, (x, y - height - 7), (x + width + 8, y + baseline + 3),
        (16, 16, 16), -1,
    )
    cv2.rectangle(
        frame, (x, y - height - 7), (x + width + 8, y + baseline + 3),
        color, max(1, round(scale)),
    )
    cv2.putText(
        frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA,
    )


def draw_header(
    frame, source, frame_index, fps, visible, switches, id_switches, masks,
):
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    header_height = min(height, round(86 * scale))
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, header_height), (8, 12, 18), -1)
    cv2.addWeighted(shade, 0.84, frame, 0.16, 0, frame)
    cv2.putText(
        frame, f"{source.name}   {frame_index / fps:05.2f}s",
        (round(18 * scale), round(31 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
        0.72 * scale, (245, 245, 245), max(1, round(1.5 * scale)),
        cv2.LINE_AA,
    )
    detail = (
        f"MCBYTE ({'MASKS' if masks else 'BOXES'}) | REVERSIBLE TEAM | "
        f"sample/{TEAM_OBSERVATION_INTERVAL}f | switch after "
        f"{TEAM_SWITCH_OBSERVATIONS} qualified opposite reads "
        f"(q>={TEAM_SWITCH_MIN_QUALITY:.2f})"
    )
    cv2.putText(
        frame, detail, (round(18 * scale), round(64 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (190, 205, 215),
        max(1, round(scale)), cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"visible {visible}   team switches {switches}"
        f"   suspected id switches {id_switches}",
        (max(round(18 * scale), width - round(560 * scale)), round(31 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.47 * scale, (235, 235, 235),
        max(1, round(scale)), cv2.LINE_AA,
    )


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
        desc=f"MCByte corrected team {source.stem}",
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
        scale = max(info.width / 1920.0, 1.0)
        line = max(2, round(3 * scale))
        torsos = torso_boxes(tracked.xyxy)
        for box, torso, player_id in zip(tracked.xyxy, torsos, player_ids):
            player = identity.players.get(int(player_id))
            if player is None:
                player = next(
                    item for item in reversed(identity.retired)
                    if item.player_id == int(player_id)
                )
            if player.is_goalkeeper:
                color = GOALKEEPER_BGR
                label = f"P{player_id} | GK"
            else:
                team = player.voted_team_id
                stable = player.team_confidence >= MIN_STABLE_TEAM_CONFIDENCE
                if stable:
                    color = TEAM_BGR[team]
                elif player.team_is_provisional:
                    color = NEW_UNCERTAIN_BGR
                else:
                    color = CONTESTED_BGR
                pending = ""
                if player.pending_team_id is not None:
                    pending = (
                        f" | pending {'A' if player.pending_team_id == 0 else 'B'} "
                        f"{player.pending_team_observations}/"
                        f"{player.team_switch_observations}"
                    )
                label = (
                    f"P{player_id} | {'A' if team == 0 else 'B'} "
                    f"{player.team_confidence:.0%} | obs={player.team_observations}"
                    f"{pending} | switches={player.team_switches}"
                )
            x1, y1, x2, y2 = np.rint(box).astype(int)
            tx1, ty1, tx2, ty2 = np.rint(torso).astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line)
            cv2.rectangle(
                annotated, (tx1, ty1), (tx2, ty2), color,
                max(1, round(1.5 * scale)),
            )
            draw_text(
                annotated, label,
                (max(0, x1), max(round(105 * scale), y1 - round(5 * scale))),
                color, scale,
            )

        # A flip on a settled label more likely means McByte changed person than
        # that the colour read was wrong; count it apart from plain corrections
        # rather than folding both into one `player.team_switches` total.
        switches = sum(
            1 for event in identity.events if event["type"] == "team_switch"
        )
        id_switches = sum(
            1 for event in identity.events
            if event["type"] == "suspected_id_switch"
        )
        draw_header(
            annotated, source, frame_index, info.fps, len(tracked), switches,
            id_switches, enable_masks,
        )
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
        "team_switch_events": [
            event for event in identity.events
            if event["type"] in ("team_switch", "suspected_id_switch")
        ],
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
