"""Measure mask-propagation identity drift without any human labels.

Seed-once propagation looked better at holding identity than every
detection-association tracker we scored (14/14 clean on Han-Ber, 10/13 on
FelixClaar). But that judgement came from sheets sampling ~12 frames per
identity, while the trackers were scored per frame -- a swap lasting under
~20 frames, or one that reverted between samples, would be invisible in the
sheets and fully counted against a tracker. The comparison flatters
propagation by an unknown amount.

This removes the asymmetry. Two propagation runs are seeded at *different*
frames and compared over the frames they share. At the later run's seed frame
that run is, by construction, freshly derived from detections, so the identity
correspondence between the runs can be established there. Walking forward, any
frame where that correspondence breaks means at least one run has drifted.

It cannot say *which* run drifted, so the disagreement rate is a lower bound on
drift -- but it is measured at per-frame resolution, on the same footing as the
tracker numbers, and needs no ground truth at all.

Usage:
    python -m scripts.evaluate_propagation_drift \\
        --run-a source/.FelixClaar_ref_masks \\
        --run-b source/.FelixClaar_ref_masks_seed60 \\
        --anchor 60
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_run(directory: Path) -> dict:
    """-> {frame_index: {identity: mask}} as boolean arrays."""
    frames = {}
    for path in sorted(glob.glob(str(directory / "*.npz"))):
        stem = Path(path).stem
        if not stem.isdigit():
            continue
        frames[int(stem)] = path
    return frames


def masks_at(path: str) -> dict:
    label = np.load(path)["label"]
    out = {}
    for value in np.unique(label):
        if value == 0:
            continue
        mask = label == value
        if mask.sum() < 50:
            continue
        out[int(value)] = mask
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    if not intersection:
        return 0.0
    return float(intersection / np.logical_or(a, b).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument(
        "--anchor", type=int, required=True,
        help="frame where the two runs' identities are put in correspondence "
             "(use run B's seed frame, where B is freshly detector-derived)",
    )
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_a, run_b = load_run(args.run_a), load_run(args.run_b)
    shared = sorted(set(run_a) & set(run_b))
    shared = [f for f in shared if f >= args.anchor]
    if args.anchor not in run_a or args.anchor not in run_b:
        raise SystemExit(f"anchor frame {args.anchor} missing from one run")

    # Correspondence at the anchor, greedy best-IoU one-to-one.
    a_anchor, b_anchor = masks_at(run_a[args.anchor]), masks_at(run_b[args.anchor])
    pairs = sorted(
        ((iou(ma, mb), ia, ib) for ia, ma in a_anchor.items()
         for ib, mb in b_anchor.items()),
        reverse=True,
    )
    mapping, used_a, used_b = {}, set(), set()
    for score, ia, ib in pairs:
        if score < args.min_iou or ia in used_a or ib in used_b:
            continue
        used_a.add(ia); used_b.add(ib)
        mapping[ia] = ib
    print(f"anchor frame {args.anchor}: matched {len(mapping)} identities "
          f"(run A had {len(a_anchor)}, run B had {len(b_anchor)})")

    agree = defaultdict(int)
    disagree = defaultdict(int)
    absent = defaultdict(int)
    first_break = {}
    for frame_index in shared:
        a_masks, b_masks = masks_at(run_a[frame_index]), masks_at(run_b[frame_index])
        for ia, ib in mapping.items():
            ma, mb = a_masks.get(ia), b_masks.get(ib)
            if ma is None or mb is None:
                absent[ia] += 1
                continue
            # Does A's identity still overlap the SAME B identity best?
            best_ib, best_score = None, 0.0
            for candidate, mask in b_masks.items():
                score = iou(ma, mask)
                if score > best_score:
                    best_ib, best_score = candidate, score
            if best_score < args.min_iou:
                absent[ia] += 1
            elif best_ib == ib:
                agree[ia] += 1
            else:
                disagree[ia] += 1
                first_break.setdefault(ia, frame_index)

    print(f"\ncompared frames {shared[0]}-{shared[-1]} ({len(shared)} frames)\n")
    print(f'{"A id":>5} {"B id":>5} {"agree":>6} {"disagree":>9} {"unmatched":>10} {"first break":>12}')
    total_agree = total_disagree = 0
    for ia in sorted(mapping):
        total_agree += agree[ia]; total_disagree += disagree[ia]
        print(f'{ia:>5} {mapping[ia]:>5} {agree[ia]:>6} {disagree[ia]:>9} '
              f'{absent[ia]:>10} {first_break.get(ia, "-"):>12}')
    scored = total_agree + total_disagree
    rate = 100.0 * total_disagree / max(scored, 1)
    print(f'\ncomparable observations: {scored}')
    print(f'disagreements:           {total_disagree}  ({rate:.1f}%)')
    print(f'identities that ever break: {len(first_break)} of {len(mapping)}')
    print("\nThis is a LOWER bound on drift: it counts only disagreements, and")
    print("cannot attribute them to run A or run B.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "anchor": args.anchor, "mapping": {str(k): v for k, v in mapping.items()},
            "agree": dict(agree), "disagree": dict(disagree), "absent": dict(absent),
            "first_break": {str(k): v for k, v in first_break.items()},
            "disagreement_pct": rate,
        }, indent=2))


if __name__ == "__main__":
    main()
