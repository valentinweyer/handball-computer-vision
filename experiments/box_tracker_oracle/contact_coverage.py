"""How much of a match is player contact? Sizes a contact-triggered SAM2 hybrid.

Every swap in the oracle study sat at a contact: two player boxes overlapping.
A hybrid would run a box tracker everywhere and SAM2 only on players near a
contact. Its saving depends on how often that is, measured here from existing
detection caches (current detector, every 5th frame for the full-game
windows), restricted to frames with the court in view where available.

Per scored frame: person boxes (goalkeeper, field player) at confidence >= 0.5;
a player is "in contact" if its box overlaps another's with IoU > `--iou`.
Reported:

  * frames_any: share of frames with at least one contact;
  * players_now: share of player boxes in contact on that frame;
  * frames_window / players_window: the same after widening each contact by
    +/- `--window` frames, since SAM2 has to be running before the contact
    starts and until it resolves. For players, the window is carried by the
    nearest box in the neighbouring frames (no identities needed).

    python -m experiments.box_tracker_oracle.contact_coverage
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.box_tracker_oracle.common import MAIN, OUT
from scripts.evaluate_tracker_identity import iou_matrix

PERSON = (1, 2)


def frame_boxes(cache, f, min_conf):
    start, end = int(cache["offsets"][f]), int(cache["offsets"][f + 1])
    keep = (np.isin(cache["class_id"][start:end], PERSON)
            & (cache["confidence"][start:end] >= min_conf))
    return cache["boxes"][start:end][keep]


def coverage(cache, in_view, iou, window, min_conf):
    stride = int(cache["stride"]) if "stride" in cache.files else 1
    count = len(cache["offsets"]) - 1
    frames = [f for f in range(0, count, stride) if in_view is None or in_view[f]]
    boxes = {f: frame_boxes(cache, f, min_conf) for f in frames}
    contact = {}
    for f in frames:
        b = boxes[f]
        if len(b) < 2:
            contact[f] = np.zeros(len(b), bool)
            continue
        overlap = iou_matrix(b, b)
        np.fill_diagonal(overlap, 0)
        contact[f] = (overlap > iou).any(axis=1)
    steps = max(window // stride, 0)
    index = {f: i for i, f in enumerate(frames)}
    frames_any = frames_window = players_now = players_window = total = 0
    for f in frames:
        b, c = boxes[f], contact[f]
        total += len(b)
        players_now += int(c.sum())
        frames_any += int(c.any())
        neighbours = [frames[j] for j in range(max(index[f] - steps, 0),
                                               min(index[f] + steps + 1, len(frames)))]
        frames_window += int(any(contact[g].any() for g in neighbours))
        # a player is in a window if it is near a box in contact on a nearby frame
        flagged = c.copy()
        centres = (b[:, :2] + b[:, 2:]) / 2 if len(b) else np.zeros((0, 2))
        heights = (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
        for g in neighbours:
            if g == f or not contact[g].any() or not len(b):
                continue
            other = boxes[g][contact[g]]
            oc = (other[:, :2] + other[:, 2:]) / 2
            distance = np.linalg.norm(centres[:, None] - oc[None], axis=2)
            flagged |= (distance < 0.5 * heights[:, None]).any(axis=1)
        players_window += int(flagged.sum())
    n = max(len(frames), 1)
    return {"frames": len(frames), "stride": stride,
            "frames_any_pct": round(100 * frames_any / n, 1),
            "frames_window_pct": round(100 * frames_window / n, 1),
            "players_now_pct": round(100 * players_now / max(total, 1), 1),
            "players_window_pct": round(100 * players_window / max(total, 1), 1),
            "players_per_frame": round(total / n, 1)}


def sources():
    interim = MAIN / "data/interim"
    for work in sorted(interim.iterdir()):
        det = work / "detections.npz"
        if not det.is_file() or not work.name.endswith("_10min"):
            continue
        view = work / "court_in_view.npy"
        yield work.name, det, (np.load(view) if view.is_file() else None)
    for name in ("felix", "han", "bhc"):
        yield f"eval:{name}", OUT / "detections/detector-expanded-v1" / f"{name}.npz", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-conf", type=float, default=0.5)
    args = parser.parse_args()
    report = {}
    for name, path, in_view in sources():
        with np.load(path, allow_pickle=False) as cache:
            cache = {k: cache[k] for k in cache.files}
        cache = _NpzView(cache)
        row = coverage(cache, in_view, args.iou, args.window, args.min_conf)
        report[name] = row
        print(f"{name:42} {json.dumps(row)}", flush=True)
    out = OUT / f"contact_coverage_iou{args.iou}_w{args.window}.json"
    out.write_text(json.dumps({"args": vars(args), "clips": report}, indent=1))
    print("wrote", out)


class _NpzView(dict):
    @property
    def files(self):
        return list(self)


if __name__ == "__main__":
    main()
