"""Score SAM2 throughput-experiment dumps and compare them with the full-rate run.

    python -m experiments.sam2_throughput.score full stride2
"""
from __future__ import annotations

import json
import sys

from experiments.sam2_throughput.common import OUT, score_dump


def main():
    arms = sys.argv[1:] or ["full", "stride2"]
    results = {}
    for arm in arms:
        for clip in ("felix", "han", "bhc"):
            dump = OUT / arm / f"{clip}.npz"
            if not dump.exists():
                continue
            m = score_dump(clip, dump)
            results[f"{arm}/{clip}"] = m
            print(f"{arm:9} {clip:5} correct={m['correct_pct']:6.2f} recall={m['recall_pct']:6.2f} "
                  f"wrong={m['wrong_identity_pct']:5.2f} wrong_frames={m['wrong_identity_frames']:4d} "
                  f"mixed={m['mixed_tracklets']:2d} switches={m['id_switches']:3d} "
                  f"frags={m['mean_fragments_per_reference_id']:.2f}", flush=True)
    (OUT / f"scores_{'_'.join(arms)}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
