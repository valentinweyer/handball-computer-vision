"""Render reference masks of two identities around an error event, for visual
classification (same team or not, real crossing or reference error).

    python -m experiments.box_tracker_oracle.render_event felix 1 6 121 out.jpg
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from experiments.box_tracker_oracle.common import Clip

COLOURS = ((0, 0, 255), (255, 200, 0))   # BGR: red = first id, cyan = second


def render(clip_name: str, a: int, b: int, frame: int, out: str, offsets=(-15, -5, 0, 5, 15)):
    clip = Clip(clip_name)
    wanted = {frame + o for o in offsets}
    tiles = []
    cap = cv2.VideoCapture(str(clip.video))
    index = 0
    frames = {}
    while index <= max(wanted):
        ok, image = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = image
        index += 1
    for f in sorted(frames):
        image = frames[f].copy()
        path = clip.reference_dir / f"{f:05d}.npz"
        boxes = []
        if path.exists():
            label = np.load(path)["label"]
            for identity, colour in zip((a, b), COLOURS):
                mask = (label == identity).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, contours, -1, colour, 3)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        if boxes:
            x1 = max(min(b[0] for b in boxes) - 150, 0)
            y1 = max(min(b[1] for b in boxes) - 100, 0)
            x2 = min(max(b[2] for b in boxes) + 150, image.shape[1])
            y2 = min(max(b[3] for b in boxes) + 100, image.shape[0])
            image = image[y1:y2, x1:x2]
        image = cv2.resize(image, (420, int(420 * image.shape[0] / image.shape[1])))
        cv2.putText(image, f"f{f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(image)
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    legend = np.zeros((50, sum(t.shape[1] for t in tiles), 3), np.uint8)
    cv2.putText(legend, f"{clip_name} REFERENCE masks: red = id {a}, cyan = id {b}",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(out, np.vstack([legend, np.hstack(tiles)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    clip_name, a, b, frame, out = sys.argv[1:6]
    render(clip_name, int(a), int(b), int(frame), out)
