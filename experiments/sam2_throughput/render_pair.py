"""Two review runs of an unreferenced clip, same frames, one row each.

Tracks named in `--focus` (per run, `run:track`) are drawn thick and
labelled; every other track thin. Full frames, downscaled.

    python -m experiments.sam2_throughput.render_pair Hannover 296 304 312 320 328 \
        --focus s1:7 s1:15 s2:7 s2:15 --out hannover.jpg
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments.sam2_throughput.common import MAIN, OUT

COLOURS = [(0, 0, 255), (0, 200, 0), (255, 150, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip")
    p.add_argument("frames", type=int, nargs="+")
    p.add_argument("--focus", nargs="+", default=[])
    p.add_argument("--crop", type=int, nargs=4, default=None, help="x1 y1 x2 y2 in source pixels")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    focus = {(r, int(t)) for r, t in (f.split(":") for f in a.focus)}
    rows = []
    for run in ("s1", "s2"):
        with np.load(OUT / "review" / f"{a.clip}_{run}" / "replay.npz") as d:
            fi, ti, bx = d["frame_index"], d["tracker_id"], d["boxes"]
        tiles = []
        for f in a.frames:
            img = cv2.imread(str(MAIN / "data/cache/frames" / a.clip / f"{f:05d}.jpg"))
            for t, b in zip(ti[fi == f], bx[fi == f]):
                x1, y1, x2, y2 = map(int, b)
                key = (run, int(t))
                c = COLOURS[int(t) % len(COLOURS)]
                thick = 6 if key in focus else 1
                cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
                if key in focus:
                    cv2.putText(img, f"T{t}", (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.6, c, 4)
            if a.crop:
                x1, y1, x2, y2 = a.crop
                img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
            label = "full rate" if run == "s1" else "stride 2"
            cv2.putText(img, f"{label} f{f}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            tiles.append(img)
        rows.append(np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT) for t in tiles]))
    legend = np.zeros((40, rows[0].shape[1], 3), np.uint8)
    cv2.putText(legend, "thick + T<n> = tracks being compared (top: full rate, bottom: stride 2); "
                "thin = other tracks", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imwrite(a.out, np.vstack([legend] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
