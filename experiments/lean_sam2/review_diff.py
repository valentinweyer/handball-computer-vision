"""Cut side-by-side clips wherever a candidate tracker and SAM2 disagree on who is who.

SAM2 is the bar, so only disagreements need a human: where both trackers keep
the same players under one ID each, there is nothing to judge. Both runs come
from experiments/cutie/render_review.py (replay.npz + review.mp4).

Per frame, candidate boxes are matched one-to-one to SAM2 boxes (IoU >= 0.3).
A disagreement is one tracker keeping an ID while the other tracker's ID on
that player changes from one held ID to another (each held >= HOLD matched
frames). Both directions are searched: the side whose ID changes is usually
the one to look at, but which side is wrong is for the reviewer to say.

Kinds:
  swap    the new ID was already on someone else just before -- a handover
  new_id  the new ID first appears there -- the player was dropped and re-added

Writes <output>/NN_<clip>_f<frame>.mp4 (candidate left, SAM2 right) and
index.md with one line per clip to fill in: better / worse / same.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment

HOLD = 10
MATCH_IOU = 0.3
PAD = 25                      # frames either side of the change (1 s at 25 fps)


def load(run):
    data = np.load(run / 'replay.npz')
    frames = defaultdict(list)
    for f, t, box in zip(data['frame_index'], data['tracker_id'], data['boxes']):
        frames[int(f)].append((int(t), box.astype(float)))
    return frames


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def matched_pairs(a, b):
    pairs = []
    for f in sorted(set(a) & set(b)):
        ta, ba = zip(*a[f])
        tb, bb = zip(*b[f])
        s = iou(np.array(ba), np.array(bb))
        rows, cols = linear_sum_assignment(-s)
        pairs += [(f, ta[r], tb[c]) for r, c in zip(rows, cols) if s[r, c] >= MATCH_IOU]
    return pairs


def changes(pairs, other_ids_by_frame, changed_side):
    """A keeps one ID while B's ID on that player changes between held runs."""
    by_track = defaultdict(list)
    for f, ta, tb in pairs:
        by_track[ta].append((f, tb))
    first_seen = {}
    for f in sorted(other_ids_by_frame):
        for tid, _ in other_ids_by_frame[f]:
            first_seen.setdefault(tid, f)
    found = []
    for ta, seq in by_track.items():
        runs = []                                   # [id, first, last, matched frames]
        for f, tb in seq:
            if runs and runs[-1][0] == tb:
                runs[-1][2], runs[-1][3] = f, runs[-1][3] + 1
            else:
                runs.append([tb, f, f, 1])
        held = [r for r in runs if r[3] >= HOLD]
        for before, after in zip(held, held[1:]):
            if before[0] == after[0]:
                continue
            kind = 'swap' if first_seen[after[0]] < after[1] - HOLD else 'new_id'
            found.append({'frame': after[1], 'last_old': before[2], 'kind': kind,
                          'changed_side': changed_side, 'steady_id': ta,
                          'old_id': before[0], 'new_id': after[0]})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--sam2', type=Path, required=True)
    parser.add_argument('--clip', required=True, help='short name used in file names')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=25.0)
    args = parser.parse_args()
    cand, sam2 = load(args.candidate), load(args.sam2)
    pairs = matched_pairs(cand, sam2)
    events = (changes(pairs, sam2, 'sam2')
              + changes([(f, b, a) for f, a, b in pairs], cand, 'candidate'))
    events.sort(key=lambda e: e['frame'])
    # One handover is usually seen from both sides, and a flicker produces a
    # change and its reversal; events within PAD frames share one clip.
    clusters = []
    for e in events:
        if clusters and e['frame'] - clusters[-1][-1]['frame'] <= PAD:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    args.output.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, cluster in enumerate(clusters, 1):
        first, last = cluster[0]['frame'], cluster[-1]['frame']
        # From the last frame the old ID was matched, but at most 3 s back:
        # a player can go unmatched for a long stretch before the change.
        start = max(0, max(min(e['last_old'] for e in cluster), first - 3 * PAD) - PAD)
        end = last + PAD
        name = f'{n:02d}_{args.clip}_f{first}.mp4'
        trim = f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,scale=960:-2'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                        '-i', str(args.candidate / 'review.mp4'), '-i', str(args.sam2 / 'review.mp4'),
                        '-filter_complex', f'[0:v]{trim}[l];[1:v]{trim}[r];[l][r]hstack',
                        '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
                        str(args.output / name)], check=True)
        what = []
        for e in cluster:
            side = 'left (candidate)' if e['changed_side'] == 'candidate' else 'right (SAM2)'
            what.append(f'f{e["frame"]} {side} {e["old_id"]} -> {e["new_id"]} ({e["kind"]})')
            e['clip'] = name
        lines.append(f'- [ ] {name} (frames {start}-{end}): ' + '; '.join(what)
                     + '. Candidate is: better / worse / same')
    (args.output / 'events.json').write_text(json.dumps(events, indent=1))
    (args.output / 'index.md').write_text(
        f'# {args.clip}: candidate (left) vs SAM2 (right)\n\n' + '\n'.join(lines) + '\n')
    print(f'{len(events)} disagreements in {len(clusters)} clips -> {args.output}/index.md')


if __name__ == '__main__':
    main()
