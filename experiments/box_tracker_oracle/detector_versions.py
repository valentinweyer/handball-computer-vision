"""Compare detection caches against the reference boxes on the frames both scored.

The oracle study's detector arm uses the `_v1` caches from the first tracker
comparison. `scripts.process_video` has since written `data/interim/score_*`
caches with the current detector (every `stride`-th frame). Recall here is
reference-box recall at IoU >= 0.5 over verified player identities, on the
frames both caches scored, so the two detectors are compared like for like.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.box_tracker_oracle.common import MAIN, Clip
from scripts.evaluate_tracker_identity import MATCH_IOU, iou_matrix

INTERIM = {"felix": "score_FelixClaar", "han": "score_Han-Ber4_cached",
           "bhc": "score_BHC-FAG_window_cached"}
PERSON = (1, 2)   # goalkeeper, field player


def load(path):
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    return out


def scored_frames(cache, count):
    stride = int(cache["stride"]) if "stride" in cache else 1
    return set(range(0, count, stride))


def boxes_at(cache, frame, threshold):
    start, end = int(cache["offsets"][frame]), int(cache["offsets"][frame + 1])
    keep = np.isin(cache["class_id"][start:end], PERSON) & (cache["confidence"][start:end] >= threshold)
    return cache["boxes"][start:end][keep]


def recall(clip, cache, frames, threshold):
    hit = total = 0
    for f in frames:
        truth = {i: b for i, b in clip.reference.get(f, {}).items() if i in clip.keep}
        if not truth:
            continue
        ref = np.stack(list(truth.values()))
        det = boxes_at(cache, f, threshold)
        total += len(ref)
        if len(det):
            hit += int((iou_matrix(ref, det).max(axis=1) >= MATCH_IOU).sum())
    return hit, total


def contact_separation(clip, cache, frames, threshold, contact_iou=0.2):
    """Of reference player pairs in contact (box IoU > contact_iou), the share
    where one-to-one matching gives each player its own detection at IoU >= 0.5.

    This is what the oracle study found box trackers lack: in contact the
    detector tends to emit one box that matches both players.
    """
    from scipy.optimize import linear_sum_assignment

    separated = pairs = 0
    for f in frames:
        truth = {i: b for i, b in clip.reference.get(f, {}).items() if i in clip.keep}
        if len(truth) < 2:
            continue
        ids = sorted(truth)
        ref = np.stack([truth[i] for i in ids])
        det = boxes_at(cache, f, threshold)
        overlap = iou_matrix(ref, ref)
        own = np.zeros(len(ids), bool)
        if len(det):
            scores = iou_matrix(ref, det)
            rows, cols = linear_sum_assignment(scores, maximize=True)
            own[rows[scores[rows, cols] >= MATCH_IOU]] = True
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if overlap[a, b] > contact_iou:
                    pairs += 1
                    separated += int(own[a] and own[b])
    return separated, pairs


def main():
    report = {}
    for name, interim in INTERIM.items():
        clip = Clip(name)
        old = load(clip.detection_cache)
        new = load(MAIN / "data/interim" / interim / "detections.npz")
        count = min(len(old["offsets"]), len(new["offsets"])) - 1
        frames = sorted(scored_frames(old, count) & scored_frames(new, count))
        row = {"frames": len(frames), "new_stride": int(new.get("stride", 1)),
               "new_detector": str(new.get("detector_id", ""))}
        for threshold in (0.3, 0.5, 0.7):
            h_old, t = recall(clip, old, frames, threshold)
            h_new, _ = recall(clip, new, frames, threshold)
            s_old, pairs = contact_separation(clip, old, frames, threshold)
            s_new, _ = contact_separation(clip, new, frames, threshold)
            row[f"conf{threshold}"] = {
                "recall_v1": round(100 * h_old / t, 1), "recall_current": round(100 * h_new / t, 1),
                "refs": t, "contact_pairs": pairs,
                "separated_v1": round(100 * s_old / max(pairs, 1), 1),
                "separated_current": round(100 * s_new / max(pairs, 1), 1)}
        report[name] = row
        print(name, json.dumps(row))


if __name__ == "__main__":
    main()
