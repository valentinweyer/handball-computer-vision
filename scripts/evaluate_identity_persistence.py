"""Does a stable player_id survive a whole match? Unsupervised identity-drift diagnostic.

The product requirement is that one player keeps one identity for the length of a match
(~60 min). Every identity measurement in this repo so far used clips of ~10 seconds of
real time, which cannot show drift: `evaluate_tracker_identity.py` measures fragments and
id switches properly but needs hand-verified per-pixel reference labels, which exist only
for Han-Ber.

This needs no ground truth. It records, per frame, how many distinct player_ids
IdentityManager has ever allocated, and how many are alive right now. The *shape* of the
cumulative curve is the diagnostic and it extrapolates:

  * plateaus after the on-court cast is established -> identities are stable, and a
    60-minute match is mostly more of the same;
  * keeps climbing roughly linearly with time -> every minute manufactures new
    identities, and the same slope over 60 minutes is a different order of problem.

A rising curve is not proof of a bug on its own -- substitutions and players entering
frame legitimately create new ids -- so allocations are also reported per minute against
the count of ids alive at once, which is bounded by how many players are actually on
screen. Sustained allocation far above that ceiling is drift, not roster change.

    python -m scripts.evaluate_identity_persistence data/raw/BHC-FAG.mp4 \
        --detections outputs/team_confidence_v2/.BHC-FAG_detections_v1.npz \
        --team-model outputs/team_confidence_v2/.BHC-FAG_team.pkl \
        --output runs/identity_persistence/BHC-FAG.json
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

from handball_cv.teams.model import TeamModel
from handball_cv.tracking.identity import TEAM_SWITCH_OBSERVATIONS, IdentityManager
from scripts.label_jersey_numbers import utc_now, write_json_atomic
from scripts.render_full_pipeline import GOALKEEPER_CLASS_ID
from scripts.render_raw_team_classification import frame_detections, load_detection_cache


def summarize(samples: list[dict], fps: float) -> dict:
    """Allocation rate over time, and the late-window rate used for extrapolation."""
    if not samples:
        return {}
    total = samples[-1]["allocated"]
    duration_s = samples[-1]["frame"] / fps
    peak_alive = max(s["alive"] for s in samples)

    # The first seconds always allocate quickly -- that's the cast appearing, not drift.
    # Rate over the last two-thirds is the part that would repeat for 60 minutes.
    tail = [s for s in samples if s["frame"] / fps >= duration_s / 3]
    tail_span_s = (tail[-1]["frame"] - tail[0]["frame"]) / fps if len(tail) > 1 else 0.0
    tail_allocated = tail[-1]["allocated"] - tail[0]["allocated"] if len(tail) > 1 else 0
    tail_rate_per_min = (tail_allocated / tail_span_s * 60) if tail_span_s > 0 else 0.0

    return {
        "duration_seconds": round(duration_s, 1),
        "total_ids_allocated": total,
        "peak_ids_alive_at_once": peak_alive,
        "ids_allocated_per_minute_overall": round(total / duration_s * 60, 1) if duration_s else 0.0,
        "ids_allocated_per_minute_steady_state": round(tail_rate_per_min, 1),
        "projected_ids_at_60_min": round(peak_alive + tail_rate_per_min * 60),
    }


def run(args: argparse.Namespace) -> dict:
    source = args.video.resolve()
    cache = load_detection_cache(args.detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    info = sv.VideoInfo.from_video_path(str(source))

    enable_masks = not args.no_masks
    if enable_masks and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        enable_mask_manager=enable_masks,
        mask_config=McByteMaskConfig(device=args.device) if enable_masks else None,
        minimum_mask_average_confidence=0.6,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    identity = IdentityManager(
        team_model,
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        team_switch_observations=TEAM_SWITCH_OBSERVATIONS,
    )

    ever_seen: set[int] = set()
    samples: list[dict] = []
    first_seen_frame: dict[int, int] = {}

    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=min(info.total_frames, len(cache["offsets"]) - 1),
        desc=f"identity persistence {source.stem}",
    )):
        if frame_index >= len(cache["offsets"]) - 1:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = frame_detections(cache, frame_index)
        tracked = tracker.update(detections, frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        player_ids = identity.update(frame_index, frame_rgb, tracked)
        alive = getattr(tracker, "tracked_objects", sv.Detections.empty())
        alive_ids = (
            alive.tracker_id if getattr(alive, "tracker_id", None) is not None
            else np.empty(0, dtype=int)
        )
        identity.retire_missing(frame_index, alive_ids)

        for pid in np.asarray(player_ids, dtype=int).tolist():
            if pid not in ever_seen:
                ever_seen.add(pid)
                first_seen_frame[pid] = frame_index

        if frame_index % args.sample_every == 0:
            samples.append({
                "frame": frame_index,
                "allocated": len(ever_seen),
                "alive": int(len(set(np.asarray(player_ids, dtype=int).tolist()))),
            })

    report = {
        "schema_version": 1,
        "video": str(source),
        "fps": info.fps,
        "created_at": utc_now(),
        "summary": summarize(samples, info.fps),
        "first_seen_frame": {str(k): v for k, v in sorted(first_seen_frame.items())},
        "samples": samples,
    }
    write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--sample-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["summary"], indent=2))
    samples = report["samples"]
    fps = report["fps"]
    print("\nallocation curve (10s buckets):")
    bucket = 0
    for s in samples:
        t = s["frame"] / fps
        if t >= bucket:
            print(f"  t={bucket:>4.0f}s  allocated={s['allocated']:>4}  alive={s['alive']:>3}")
            bucket += 10


if __name__ == "__main__":
    main()
