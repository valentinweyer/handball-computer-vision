"""Half-frame-rate SAM2: prepare inputs, then expand the result back to full rate.

`prepare`: a JPEG folder holding every `stride`-th frame (symlinks, renumbered
from 0) and a detection cache for those frames, so the unchanged production
runner propagates at 25/stride fps. Run it with `--check-every 10/stride`
to keep detector checkpoints at the same real-time interval.

`expand`: map subsampled frame i back to frame stride*i and fill the skipped
frames by linear interpolation of each player's box between its two
neighbouring processed frames (only when the player is present in both).

    python -m experiments.sam2_throughput.half_rate prepare felix --stride 2
    python -m experiments.sam2_throughput.half_rate expand felix --stride 2
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.sam2_throughput.common import CASES, MAIN, OUT


def paths(clip: str, stride: int):
    base = OUT / f"stride{stride}"
    return base / "frames" / CASES[clip][0], base / f"{clip}_detections.npz", \
        base / f"{clip}_sub.npz", base / f"{clip}.npz"


def prepare(clip: str, stride: int) -> None:
    frames_dir, det_path, _, _ = paths(clip, stride)
    source = MAIN / "data/cache/frames" / CASES[clip][0]
    files = sorted(source.glob("*.jpg"), key=lambda p: int(p.stem))[::stride]
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        link = frames_dir / f"{i:05d}.jpg"
        if not link.exists():
            link.symlink_to(f)
    with np.load(MAIN / CASES[clip][1], allow_pickle=False) as d:
        cache = {k: d[k] for k in d.files}
    offsets = cache["offsets"]
    rows, new_offsets = [], [0]
    for f in range(0, len(files) * stride, stride):
        start, end = int(offsets[f]), int(offsets[f + 1])
        rows.append(np.arange(start, end))
        new_offsets.append(new_offsets[-1] + end - start)
    take = np.concatenate(rows)
    np.savez_compressed(det_path, offsets=np.array(new_offsets), boxes=cache["boxes"][take],
                        confidence=cache["confidence"][take], class_id=cache["class_id"][take])
    print(f"{clip}: {len(files)} frames -> {frames_dir}, detections -> {det_path}")


def expand(clip: str, stride: int) -> None:
    _, _, sub_path, out_path = paths(clip, stride)
    with np.load(sub_path) as d:
        frame, tid, boxes = d["frame_index"] * stride, d["tracker_id"], d["boxes"]
        source = str(d["source"])
    by_frame = {}
    for f, t, b in zip(frame, tid, boxes):
        by_frame.setdefault(int(f), {})[int(t)] = b
    keyed = sorted(by_frame)
    rows_f, rows_t, rows_b = [], [], []
    for a, b in zip(keyed, keyed[1:] + [None]):
        for t, box in by_frame[a].items():
            rows_f.append(a); rows_t.append(t); rows_b.append(box)
        if b is None:
            continue
        for f in range(a + 1, b):
            w = (f - a) / (b - a)
            for t, box in by_frame[a].items():
                if t in by_frame[b]:
                    rows_f.append(f); rows_t.append(t)
                    rows_b.append((1 - w) * box + w * by_frame[b][t])
    np.savez_compressed(out_path, frame_index=np.array(rows_f), tracker_id=np.array(rows_t),
                        boxes=np.array(rows_b, float).reshape(-1, 4), source=source)
    print(f"{clip}: expanded {len(frame)} rows -> {len(rows_f)} rows, {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "expand"])
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    for clip in args.clips:
        (prepare if args.action == "prepare" else expand)(clip, args.stride)


if __name__ == "__main__":
    main()
