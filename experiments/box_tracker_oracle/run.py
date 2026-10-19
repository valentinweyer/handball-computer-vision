"""Score box-tracker variants on both detection arms and list their errors.

Usage:
    CUDA_VISIBLE_DEVICES= python -m experiments.box_tracker_oracle.run \
        --clips felix han bhc --arms oracle detector --variants sort colour_gate ...
    python -m experiments.box_tracker_oracle.run --calibrate han --feature colour

`--calibrate` prints same-identity vs overlapping-other-identity appearance
distances from the oracle arm, so a gate is chosen on one clip and applied
unchanged to the others.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from trackers import ByteTrackTracker, SORTTracker

from experiments.box_tracker_oracle.appearance_sort import AppearanceSORT
from experiments.box_tracker_oracle.common import OUT, Clip, features
from scripts.evaluate_tracker_identity import iou_matrix, run_tracker, score


def wrong_runs(matches: list, keep: set) -> list[dict]:
    """Contiguous spans where a tracker id sits on a non-dominant identity."""
    per_track = defaultdict(list)
    for frame, track, ref in sorted(matches):
        if ref is not None and ref in keep:
            per_track[track].append((frame, ref))
    runs = []
    for track, entries in per_track.items():
        dominant = Counter(r for _, r in entries).most_common(1)[0][0]
        current = None
        for frame, ref in entries:
            if ref != dominant and current and current["ref"] == ref:
                current["last"], current["frames"] = frame, current["frames"] + 1
            elif ref != dominant:
                current = {"track": int(track), "dominant": int(dominant), "ref": int(ref),
                           "first": int(frame), "last": int(frame), "frames": 1}
                runs.append(current)
            else:
                current = None
    return sorted(runs, key=lambda r: -r["frames"])


def factory(name: str, clip: Clip, arm: str, gates: dict):
    fps = clip.fps
    if name == "sort":
        return lambda: SORTTracker(frame_rate=fps, lost_track_buffer=30,
                                   track_activation_threshold=0.7)
    if name == "sort_strict":
        # "Break, don't swap": demand more overlap and forget lost tracks
        # sooner, so contact ends a track instead of handing it over. Fixed
        # before running, not tuned.
        return lambda: SORTTracker(frame_rate=fps, lost_track_buffer=10,
                                   track_activation_threshold=0.7,
                                   minimum_iou_threshold=0.5)
    if name == "bytetrack":
        return lambda: ByteTrackTracker(frame_rate=fps, lost_track_buffer=30,
                                        track_activation_threshold=0.7)
    feature, mode, *strict = name.split("_")      # e.g. colour_gate, colour_gate_strict
    feats = features(clip, arm, feature)
    sort_args = (dict(lost_track_buffer=10, minimum_iou_threshold=0.5) if strict
                 else dict(lost_track_buffer=30))
    return lambda: AppearanceSORT(feats, mode, gates[feature], frame_rate=fps,
                                  track_activation_threshold=0.7, **sort_args)


def evaluate(clip: Clip, arm: str, name: str, gates: dict) -> dict:
    matches = run_tracker(factory(name, clip, arm, gates), clip.reference,
                          clip.video, clip.detections_for(arm), "cpu")
    metrics = score(matches, clip.keep)
    players = sum(sum(i in clip.keep for i in t) for t in clip.reference.values())
    metrics["recall_pct"] = 100 * metrics["matched_detections"] / players
    metrics["correct_pct"] = 100 * (
        metrics["matched_detections"] - metrics["wrong_identity_frames"]) / players
    metrics["wrong_runs"] = wrong_runs(matches, clip.keep)
    return metrics


def calibrate(clip: Clip, feature: str, lag: int = 5) -> dict:
    """Appearance distance: same identity `lag` frames apart vs. overlapping
    different identities in the same frame (the pairs association confuses)."""
    feats = features(clip, "oracle", feature)
    boxes = clip.oracle_boxes
    by_frame = {}
    for f, frame_feats in enumerate(feats):
        ids = sorted(boxes.get(f, {}))
        by_frame[f] = dict(zip(ids, frame_feats))
    same, other = [], []
    for f, current in by_frame.items():
        earlier = by_frame.get(f - lag, {})
        same += [1 - float(v @ earlier[i]) for i, v in current.items() if i in earlier]
        ids = sorted(current)
        if len(ids) > 1:
            xyxy = np.stack([boxes[f][i] for i in ids])
            overlap = iou_matrix(xyxy, xyxy)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if overlap[a, b] > 0.1:
                        other.append(1 - float(current[ids[a]] @ current[ids[b]]))
    q = lambda xs, p: float(np.percentile(xs, p)) if xs else float("nan")
    return {"same_n": len(same), "same_q50": q(same, 50), "same_q95": q(same, 95),
            "same_q99": q(same, 99), "other_n": len(other), "other_q05": q(other, 5),
            "other_q25": q(other, 25), "other_q50": q(other, 50),
            "other_below_same_q99_pct": 100 * float(np.mean(np.array(other) <= q(same, 99)))
            if other else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["felix", "han", "bhc"])
    parser.add_argument("--arms", nargs="+", default=["oracle", "detector"])
    parser.add_argument("--variants", nargs="+", default=["sort"])
    parser.add_argument("--gate", action="append", default=[],
                        help="feature=distance, e.g. colour=0.35")
    parser.add_argument("--calibrate", default=None)
    parser.add_argument("--feature", default="colour")
    parser.add_argument("--tag", default="results")
    args = parser.parse_args()

    if args.calibrate:
        print(json.dumps(calibrate(Clip(args.calibrate), args.feature), indent=2))
        return
    gates = {k: float(v) for k, v in (g.split("=") for g in args.gate)}
    results = {}
    for clip_name in args.clips:
        clip = Clip(clip_name)
        for arm in args.arms:
            for name in args.variants:
                m = evaluate(clip, arm, name, gates)
                results[f"{clip_name}/{arm}/{name}"] = m
                top = ", ".join(f"{r['dominant']}->{r['ref']}@{r['first']}({r['frames']})"
                                for r in m["wrong_runs"] if r["frames"] >= 3)
                print(f"{clip_name:5} {arm:8} {name:14} correct={m['correct_pct']:5.1f} "
                      f"recall={m['recall_pct']:5.1f} wrong={m['wrong_identity_pct']:5.1f} "
                      f"sw={m['id_switches']:3d} frags={m['mean_fragments_per_reference_id']:.2f} "
                      f"| {top}", flush=True)
    out = OUT / f"{args.tag}.json"
    out.write_text(json.dumps({"gates": gates, "results": results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
