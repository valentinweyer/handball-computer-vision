"""How hard is it to stitch a fragmenting box tracker's tracklets back together?

`sort_strict` matches SAM2's identity accuracy on oracle boxes but leaves 2-3.5
fragments per player. This links them offline with the simplest plausible
rule and scores the result, as a lower bound on what the identity layer could
recover:

  * a tracklet that ends at frame t may continue as one that starts within
    `max_gap` frames;
  * the start must lie within `max_jump` box heights of the ended tracklet's
    last box centre, extrapolated with its last-5-frame velocity;
  * optionally (`--colour`), the two tracklets' mean kit-colour features must
    be within the Han-Ber4-calibrated gate;
  * ends and starts are paired one-to-one by Hungarian assignment on distance.

Not the production identity layer (`PlayerRegistry`, re-ID); a proxy.

    python -m experiments.box_tracker_oracle.stitch --arm oracle
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers import SORTTracker

from experiments.box_tracker_oracle.common import Clip, features
from experiments.box_tracker_oracle.run import wrong_runs
from scripts.evaluate_tracker_identity import MATCH_IOU, iou_matrix, score


def track(clip: Clip, arm: str):
    """-> {tid: [(frame, xyxy, colour_feature)]}, plus scorer matches."""
    detections_for = clip.detections_for(arm)
    feats = features(clip, arm, "colour")
    tracker = SORTTracker(frame_rate=clip.fps, lost_track_buffer=10,
                          track_activation_threshold=0.7, minimum_iou_threshold=0.5)
    tracklets = defaultdict(list)
    count = len(feats)
    for f in range(count):
        out = tracker.update(detections_for(f))
        for row in np.flatnonzero(out.tracker_id >= 0):
            tracklets[int(out.tracker_id[row])].append((f, out.xyxy[row], feats[f][row]))
    return tracklets


def link(tracklets: dict, max_gap: int, max_jump: float, colour_gate: float | None) -> dict:
    """-> {tid: root tid} after linking ends to starts."""
    info = {}
    for tid, rows in tracklets.items():
        frames = np.array([r[0] for r in rows])
        boxes = np.stack([r[1] for r in rows])
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        tail = slice(max(len(rows) - 5, 0), len(rows))
        span = max(frames[tail][-1] - frames[tail][0], 1)
        velocity = (centres[tail][-1] - centres[tail][0]) / span
        colour = np.mean([r[2] for r in rows], axis=0)
        info[tid] = dict(start=frames[0], end=frames[-1], first=centres[0], last=centres[-1],
                         velocity=velocity, height=float(np.median(boxes[:, 3] - boxes[:, 1])),
                         colour=colour / max(np.linalg.norm(colour), 1e-9))
    ends = sorted(info, key=lambda t: info[t]["end"])
    starts = sorted(info, key=lambda t: info[t]["start"])
    cost = np.full((len(ends), len(starts)), np.inf)
    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            gap = info[b]["start"] - info[a]["end"]
            if a == b or not 0 < gap <= max_gap:
                continue
            predicted = info[a]["last"] + info[a]["velocity"] * gap
            jump = np.linalg.norm(info[b]["first"] - predicted) / max(info[a]["height"], 1.0)
            if jump > max_jump:
                continue
            if colour_gate is not None and 1 - float(info[a]["colour"] @ info[b]["colour"]) > colour_gate:
                continue
            cost[i, j] = jump
    finite = np.where(np.isfinite(cost), cost, 1e6)
    parent = {}
    if len(ends) and len(starts):
        for i, j in zip(*linear_sum_assignment(finite)):
            if np.isfinite(cost[i, j]):
                parent[starts[j]] = ends[i]
    root = {}
    for tid in info:
        r = tid
        while r in parent:
            r = parent[r]
        root[tid] = r
    return root


def matches_for(clip: Clip, tracklets: dict, root: dict) -> list:
    """Scorer triples, matched to the reference exactly as run_tracker does."""
    by_frame = defaultdict(list)
    for tid, rows in tracklets.items():
        for f, box, _ in rows:
            by_frame[f].append((root[tid], box))
    matches = []
    for f, items in by_frame.items():
        truth = clip.reference.get(f, {})
        if not truth:
            continue
        ids = sorted(truth)
        scores = iou_matrix(np.stack([b for _, b in items]), np.stack([truth[i] for i in ids]))
        pairs = sorted(((scores[r, c], r, c) for r in range(scores.shape[0])
                        for c in range(scores.shape[1]) if scores[r, c] >= MATCH_IOU), reverse=True)
        used_r, used_c, assigned = set(), set(), {}
        for _, r, c in pairs:
            if r in used_r or c in used_c:
                continue
            used_r.add(r); used_c.add(c); assigned[r] = ids[c]
        matches += [(f, tid, assigned.get(r)) for r, (tid, _) in enumerate(items)]
    return matches


def report(clip, matches, label):
    m = score(matches, clip.keep)
    players = sum(sum(i in clip.keep for i in t) for t in clip.reference.values())
    correct = 100 * (m["matched_detections"] - m["wrong_identity_frames"]) / players
    top = ", ".join(f"{r['dominant']}->{r['ref']}@{r['first']}({r['frames']})"
                    for r in wrong_runs(matches, clip.keep) if r["frames"] >= 3)
    print(f"{clip.name:5} {label:22} correct={correct:5.1f} wrong={m['wrong_identity_pct']:5.1f} "
          f"frags={m['mean_fragments_per_reference_id']:.2f} | {top}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["felix", "han", "bhc"])
    parser.add_argument("--arm", default="oracle")
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--max-jump", type=float, default=0.75)
    parser.add_argument("--colour-gate", type=float, default=0.28)
    args = parser.parse_args()
    for name in args.clips:
        clip = Clip(name)
        tracklets = track(clip, args.arm)
        identity = {t: t for t in tracklets}
        report(clip, matches_for(clip, tracklets, identity), f"{args.arm} unstitched")
        for label, gate in (("motion", None), ("motion+colour", args.colour_gate)):
            root = link(tracklets, args.max_gap, args.max_jump, gate)
            report(clip, matches_for(clip, tracklets, root), f"{args.arm} {label}")


if __name__ == "__main__":
    main()
