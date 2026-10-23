"""Where do the half-rate and full-rate SAM2 runs disagree?

For each clip: the persistent wrong-identity runs of each arm, and the
reference detections one arm matches and the other misses (grouped into
frame spans per reference id), so every difference can be checked against
the pixels rather than read off an aggregate.

    python -m experiments.sam2_throughput.diff full stride2
"""
from __future__ import annotations

import sys
from collections import defaultdict

from experiments.box_tracker_oracle.common import Clip
from experiments.box_tracker_oracle.run import wrong_runs
from experiments.sam2_throughput.common import OUT
from scripts.evaluate_tracker_identity import Sam2ReplayTracker, run_tracker


def spans(frames):
    frames = sorted(frames)
    out, start = [], None
    for a, b in zip(frames, frames[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            out.append((start, a))
            start = None
    return out


def main():
    base, other = sys.argv[1:3] if len(sys.argv) > 2 else ("full", "stride2")
    for name in ("felix", "han", "bhc"):
        clip = Clip(name)
        matched = {}
        for arm in (base, other):
            matches = run_tracker(lambda: Sam2ReplayTracker(OUT / arm / f"{name}.npz"),
                                  clip.reference, clip.video, clip.detection_cache, "cpu")
            runs = [r for r in wrong_runs(matches, clip.keep) if r["frames"] >= 1]
            print(f"{name} {arm:8} wrong runs:",
                  ", ".join(f"track{r['track']} {r['dominant']}->{r['ref']}@{r['first']}-{r['last']}"
                            f"({r['frames']})" for r in runs) or "none")
            matched[arm] = {(f, r) for f, _, r in matches if r is not None and r in clip.keep}
        for label, a, b in ((f"only {base}", base, other), (f"only {other}", other, base)):
            per_id = defaultdict(list)
            for f, r in matched[a] - matched[b]:
                per_id[r].append(f)
            text = "; ".join(f"id{r}: " + ",".join(f"{s}-{e}" for s, e in spans(fs))
                             for r, fs in sorted(per_id.items()))
            print(f"{name} matched {label}: {text or 'none'}")
        print()


if __name__ == "__main__":
    main()
