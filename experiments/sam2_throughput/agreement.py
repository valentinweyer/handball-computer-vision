"""Where do two SAM2 runs of an unreferenced clip disagree on identity?

No reference exists for the review clips, so the full-rate run stands in for
one: each candidate track is matched per frame to the full-rate box it
overlaps (IoU >= 0.5), and frames where it sits on a different full-rate
track than its dominant one are disagreements -- the frames to look at.
Symmetric counts, so a candidate that is better than full rate in places
shows up too; the pixels decide which run is right.

    python -m experiments.sam2_throughput.agreement Hannover
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np

from experiments.sam2_throughput.common import OUT
from scripts.evaluate_tracker_identity import iou_matrix


def load(path):
    with np.load(path) as d:
        by_frame = defaultdict(dict)
        for f, t, b in zip(d["frame_index"], d["tracker_id"], d["boxes"]):
            by_frame[int(f)][int(t)] = b
    return by_frame


def disagreements(candidate, baseline):
    pairs = []          # (frame, candidate track, baseline track)
    for f, tracks in candidate.items():
        base = baseline.get(f, {})
        if not tracks or not base:
            continue
        ct, bt = sorted(tracks), sorted(base)
        scores = iou_matrix(np.stack([tracks[t] for t in ct]), np.stack([base[t] for t in bt]))
        used = set()
        for score, r, c in sorted(((scores[r, c], r, c) for r in range(len(ct))
                                   for c in range(len(bt)) if scores[r, c] >= 0.5), reverse=True):
            if r in used or ("c", c) in used:
                continue
            used.update({r, ("c", c)})
            pairs.append((f, ct[r], bt[c]))
    per = defaultdict(list)
    for f, c, b in pairs:
        per[c].append((f, b))
    runs, matched, wrong = [], len(pairs), 0
    for c, entries in per.items():
        dominant = Counter(b for _, b in entries).most_common(1)[0][0]
        current = None
        for f, b in sorted(entries):
            if b == dominant:
                current = None
                continue
            wrong += 1
            if current and current["other"] == b and f - current["last"] <= 2:
                current["last"], current["frames"] = f, current["frames"] + 1
            else:
                current = {"track": c, "dominant": dominant, "other": b, "first": f, "last": f,
                           "frames": 1}
                runs.append(current)
    return matched, wrong, sorted(runs, key=lambda r: r["first"])


def main():
    clip = sys.argv[1]
    base = load(OUT / "review" / f"{clip}_s1" / "replay.npz")
    cand = load(OUT / "review" / f"{clip}_s2" / "replay.npz")
    report = {}
    for label, a, b in (("stride2_vs_full", cand, base), ("full_vs_stride2", base, cand)):
        matched, wrong, runs = disagreements(a, b)
        report[label] = {"matched": matched, "disagreeing": wrong,
                         "pct": round(100 * wrong / max(matched, 1), 2),
                         "tracks": len({t for f in a.values() for t in f}),
                         "runs_3plus": [r for r in runs if r["frames"] >= 3]}
        print(f"{clip} {label}: {wrong}/{matched} matched boxes disagree "
              f"({report[label]['pct']}%), tracks={report[label]['tracks']}")
        for r in report[label]["runs_3plus"]:
            print(f"   track {r['track']} (usually {r['dominant']}) on {r['other']} "
                  f"f{r['first']}-{r['last']} ({r['frames']} frames)")
    (OUT / "review" / f"{clip}_agreement.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
