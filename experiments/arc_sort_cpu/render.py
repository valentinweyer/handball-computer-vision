"""Small contact sheets of native tracker outputs against the trusted masks."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.arc_sort_cpu.run import frames
from experiments.box_tracker_oracle.common import Clip, MAIN


def load_mot(path):
    result = defaultdict(list)
    for line in path.read_text().splitlines():
        f, tid, x, y, w, h, *_ = map(float, line.split(","))
        result[int(f) - 1].append((int(tid), np.array([x, y, x + w, y + h])))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    clip = Clip("felix")
    stages = ["arc_osnet_localconf", "arc_final_localconf", "sam2_stored"]
    labels = ["ARC online (confidence 0.5)", "ARC + offline split/merge", "Stored SAM2"]
    tracked, dominant = {}, {}
    for stage in stages:
        matches = json.loads((root / f"felix_{stage}_matches.json").read_text())
        counts = defaultdict(Counter)
        for f, tid, rid in sorted(matches):
            if rid in clip.keep:
                counts[tid][rid] += 1
        dominant[stage] = {t: c.most_common(1)[0][0] for t, c in counts.items()}
        if stage == "sam2_stored":
            data = np.load(MAIN / "runs/sam2_reprompt/FelixClaar_policy_reprompt.npz")
            by_frame = defaultdict(list)
            for f, tid, box in zip(data["frame_index"], data["tracker_id"], data["boxes"]):
                by_frame[int(f)].append((int(tid), box))
            tracked[stage] = by_frame
        else:
            tracked[stage] = load_mot(root / stage / "felix.txt")
    # First sequence: contact ambiguity. Second: a new offline wrong merge.
    for name, ids, wanted in [("contact", (6, 7), (80, 95, 105, 125)),
                              ("offline_merge", (1, 4), (145, 160, 185, 205))]:
        selected = {i: frame for i, frame in enumerate(frames(clip.video)) if i in wanted}
        boxes = [clip.reference[f][rid] for f in wanted for rid in ids if rid in clip.reference[f]]
        x1 = max(int(min(b[0] for b in boxes)) - 75, 0)
        y1 = max(int(min(b[1] for b in boxes)) - 65, 0)
        x2 = min(int(max(b[2] for b in boxes)) + 75, selected[wanted[0]].shape[1])
        y2 = min(int(max(b[3] for b in boxes)) + 65, selected[wanted[0]].shape[0])
        rows = []
        for f in wanted:
            panes = []
            for stage, label in zip(stages, labels):
                image = selected[f].copy()
                reference = np.load(clip.reference_dir / f"{f:05d}.npz")["label"]
                for rid, colour in zip(ids, [(0, 70, 255), (255, 210, 0)]):
                    mask = (reference == rid).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, colour, 2)
                for tid, box in tracked[stage].get(f, []):
                    a, b, c, d = map(int, box)
                    if c < x1 or a > x2 or d < y1 or b > y2:
                        continue
                    cv2.rectangle(image, (a, b), (c, d), (90, 255, 100), 2)
                    text = f"T{tid} -> R{dominant[stage].get(tid, '?')}"
                    cv2.putText(image, text, (a, max(20, b - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 4)
                    cv2.putText(image, text, (a, max(20, b - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (90, 255, 100), 2)
                image = image[y1:y2, x1:x2]
                image = cv2.resize(image, (500, round(500 * image.shape[0] / image.shape[1])))
                head = np.zeros((42, 500, 3), np.uint8)
                cv2.putText(head, f"{label} / f{f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (255, 255, 255), 1)
                panes.append(np.vstack([head, image]))
            rows.append(np.hstack(panes))
        legend = np.zeros((60, 1500, 3), np.uint8)
        cv2.putText(legend, f"Reference outlines: red R{ids[0]}, cyan R{ids[1]}. Green: tracker boxes.",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(legend, "R on green labels = dominant reference identity over the clip; diagnostic mapping, not jersey number.",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(root / f"felix_{name}.jpg"), np.vstack([legend, *rows]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
