"""Can an appearance embedding tell two TEAMMATES apart?

This is the question that decides the identity strategy. Cross-team identity
swaps are already catchable from team colour (98.6% on clean crops). Same-team
swaps are not, and nothing in the pipeline currently detects them -- so either
an appearance embedding separates teammates, or jersey numbers are the only
remaining route and the OCR investment is forced.

A first attempt embedded the full detection box with PRTReID and found no
usable signal: different players sat ~0.05 apart in cosine distance while one
player varied by ~0.04. The diagnosis was that a player's box in a scrum is
full of background and neighbouring players, so the embedding describes the
crowd. This measures that directly by comparing the same embedding on full
boxes against mask-isolated pixels.

Scoring uses only tracklets a human labelled CLEAN, so each tracklet really is
one person and a between-tracklet distance really is a between-person distance.
Pairs are reported split by team, because the same-team number is the one that
matters -- a good cross-team score proves only that the model can see shirt
colour, which we already knew.

Usage:
    python -m scripts.evaluate_reid_separability data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --team-model outputs/team_comparison/.FelixClaar_team.pkl \\
        --clean 1 2 5 8 11 12 13 14 15 16
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.embeddings.prtreid import PRTReIDBackend
from handball_cv.teams.model import TeamModel
from handball_cv.tracking.identity import IdentityManager, TEAM_SWITCH_OBSERVATIONS
from handball_cv.tracking.tracklets import normalise
from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRTREID_ROOT = ROOT / "prtreid-upstream"
DEFAULT_PRTREID_CKPT = ROOT / "models/prtreid/prtreid-soccernet-baseline.pth.tar"
MIN_MASK_PIXELS = 60
MIN_BOX_W, MIN_BOX_H = 12, 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument(
        "--clean", type=int, nargs="+", required=True,
        help="tracker_ids a human labelled clean (one player throughout)",
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prtreid-root", type=Path, default=DEFAULT_PRTREID_ROOT)
    parser.add_argument("--prtreid-checkpoint", type=Path, default=DEFAULT_PRTREID_CKPT)
    return parser.parse_args()


def collect(args) -> tuple[dict, dict, dict]:
    """-> (full_box_crops, masked_crops, team_by_tracklet), keyed by tracker_id."""
    cache = load_detection_cache(args.detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    info = sv.VideoInfo.from_video_path(str(args.video))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps, lost_track_buffer=30, track_activation_threshold=0.7,
        enable_mask_manager=True, mask_config=McByteMaskConfig(device=args.device),
        minimum_mask_average_confidence=0.6, minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    identity = IdentityManager(
        team_model, goalkeeper_class_id=1,
        team_switch_observations=TEAM_SWITCH_OBSERVATIONS,
    )
    wanted = set(args.clean)
    full, masked = defaultdict(list), defaultdict(list)
    team_votes = defaultdict(lambda: defaultdict(int))

    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(args.video)),
        total=info.total_frames, desc="collect",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(frame_detections(cache, frame_index), frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        player_ids = identity.update(frame_index, frame_rgb, tracked)
        alive = getattr(tracker, "tracked_objects", sv.Detections.empty())
        identity.retire_missing(
            frame_index,
            alive.tracker_id if getattr(alive, "tracker_id", None) is not None
            else np.empty(0, dtype=int),
        )
        # Record each tracklet's team so pairs can be split same/cross team.
        for tracker_id, player_id in zip(tracked.tracker_id, player_ids):
            player = identity.players.get(int(player_id))
            if player is not None and not player.team_is_provisional:
                team_votes[int(tracker_id)][player.voted_team_id] += 1

        mask_output = getattr(tracker, "_last_mask_output", None)
        if mask_output is None or mask_output.masks is None or not len(mask_output.masks):
            continue
        masks = np.asarray(mask_output.masks, dtype=bool)
        height, width = frame_rgb.shape[:2]
        for box, tracker_id in zip(tracked.xyxy, tracked.tracker_id):
            tracker_id = int(tracker_id)
            if tracker_id not in wanted or len(full[tracker_id]) >= args.samples:
                continue
            row = mask_output.tracklet_mask_dict.get(tracker_id)
            if row is None or not (0 <= int(row) < len(masks)):
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                continue
            crop = frame_rgb[y1:y2, x1:x2]
            mask = masks[int(row)][y1:y2, x1:x2]
            if mask.sum() < MIN_MASK_PIXELS:
                continue
            isolated = crop.copy()
            isolated[~mask] = 0
            full[tracker_id].append(crop)
            masked[tracker_id].append(isolated)

    teams = {
        tracker_id: max(votes, key=votes.get)
        for tracker_id, votes in team_votes.items() if votes
    }
    return full, masked, teams


def separability(backend, store: dict, teams: dict, label: str) -> dict:
    """Within-person scatter vs between-person distance, split by team."""
    embeddings = {
        tracker_id: normalise(backend.encode_images(crops))
        for tracker_id, crops in store.items() if len(crops) >= 8
    }
    centroids = {}
    within = []
    for tracker_id, values in embeddings.items():
        centroid = values.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        centroids[tracker_id] = centroid
        within.append(float(np.mean(1.0 - values @ centroid)))

    same_team, cross_team = [], []
    ids = sorted(centroids)
    for index, a in enumerate(ids):
        for b in ids[index + 1:]:
            distance = float(1.0 - np.dot(centroids[a], centroids[b]))
            if a in teams and b in teams:
                (same_team if teams[a] == teams[b] else cross_team).append(distance)
    mean_within = float(np.mean(within)) if within else 0.0
    result = {
        "tracklets": len(embeddings),
        "within": mean_within,
        "same_team_between": float(np.mean(same_team)) if same_team else float("nan"),
        "cross_team_between": float(np.mean(cross_team)) if cross_team else float("nan"),
        "same_team_pairs": len(same_team),
        "cross_team_pairs": len(cross_team),
        "same_team_min": float(np.min(same_team)) if same_team else float("nan"),
    }
    result["same_team_ratio"] = result["same_team_between"] / max(mean_within, 1e-9)
    result["cross_team_ratio"] = result["cross_team_between"] / max(mean_within, 1e-9)
    print(
        f"{label:>9}: n={result['tracklets']:2d} within={mean_within:.4f} | "
        f"same-team between={result['same_team_between']:.4f} "
        f"(ratio {result['same_team_ratio']:.2f}, {result['same_team_pairs']} pairs) | "
        f"cross-team between={result['cross_team_between']:.4f} "
        f"(ratio {result['cross_team_ratio']:.2f})"
    )
    return result


def main() -> None:
    args = parse_args()
    full, masked, teams = collect(args)
    print(
        f"\ncollected {sum(len(v) for v in full.values())} crops "
        f"over {len(full)} clean tracklets; teams resolved for {len(teams)}\n"
    )
    backend = PRTReIDBackend(
        source_root=args.prtreid_root, checkpoint=args.prtreid_checkpoint,
        device=args.device, feature_kind="global",
    )
    results = {
        "full_box": separability(backend, full, teams, "full-box"),
        "masked": separability(backend, masked, teams, "masked"),
    }
    delta = results["masked"]["same_team_ratio"] - results["full_box"]["same_team_ratio"]
    print(f"\nmasking changes the SAME-TEAM ratio by {delta:+.2f}")
    print(
        "ratio > ~1.5 means teammates are separable; ~1.0 means one player's own "
        "variation is as large as the gap to a different teammate"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"video": str(args.video), "teams": teams, **results}, indent=2
        ))


if __name__ == "__main__":
    main()
