"""Side-by-side frames: full-rate (top) vs half-rate (bottom) SAM2 boxes.

Tracker boxes are drawn with their track ids; reference identities in
`--refs` are outlined (thin, white) and labelled "R<id>". Crops to the region
around the referenced players.

    python -m experiments.sam2_throughput.render_compare felix 136 140 143 146 \
        --refs 1 8 11 --out felix_140.jpg
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments.box_tracker_oracle.common import Clip
from experiments.sam2_throughput.common import OUT

COLOURS = [(0, 0, 255), (0, 200, 0), (255, 150, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def boxes_at(dump, frame):
    m = dump["frame_index"] == frame
    return dump["tracker_id"][m], dump["boxes"][m]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip")
    p.add_argument("frames", type=int, nargs="+")
    p.add_argument("--refs", type=int, nargs="+", required=True)
    p.add_argument("--arms", nargs=2, default=["full", "stride2"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    clip = Clip(a.clip)
    dumps = {arm: dict(np.load(OUT / arm / f"{a.clip}.npz")) for arm in a.arms}
    cap = cv2.VideoCapture(str(clip.video))
    images, i = {}, 0
    while i <= max(a.frames):
        ok, img = cap.read()
        if not ok:
            break
        if i in a.frames:
            images[i] = img
        i += 1
    rows = []
    for arm in a.arms:
        tiles = []
        for f in a.frames:
            img = images[f].copy()
            refs = {r: b for r, b in clip.oracle_boxes.get(f, {}).items() if r in a.refs}
            for r, b in refs.items():
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(img, f"R{r}", (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tids, bxs = boxes_at(dumps[arm], f)
            for t, b in zip(tids, bxs):
                x1, y1, x2, y2 = map(int, b)
                c = COLOURS[int(t) % len(COLOURS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
                cv2.putText(img, f"T{t}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3)
            if refs:
                allb = np.stack(list(refs.values()))
                x1, y1 = np.maximum(allb[:, :2].min(0) - [200, 120], 0).astype(int)
                x2 = int(min(allb[:, 2].max() + 200, img.shape[1]))
                y2 = int(min(allb[:, 3].max() + 120, img.shape[0]))
                img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
            cv2.putText(img, f"{arm} f{f}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(img)
        h = max(t.shape[0] for t in tiles)
        rows.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT)
                               for t in tiles]))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    legend = np.zeros((40, w, 3), np.uint8)
    cv2.putText(legend, "coloured T<n> = tracker boxes (top: full rate, bottom: half rate); "
                "thin white R<n> = reference", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(a.out, np.vstack([legend] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
