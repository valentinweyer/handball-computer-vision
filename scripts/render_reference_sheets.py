"""Render one verification sheet per identity in a reference label-map set.

A propagated reference (`scripts/build_identity_reference.py`, or the existing
SAM2 Han-Ber masks) is only ground truth once a human has confirmed each id
follows one player for its whole life. This renders the sheets for that check
and writes the answers template alongside them.

Non-mask pixels are dimmed rather than cropped away. A bounding box around a
player in a scrum routinely contains a second player, so a bare crop leaves the
identity genuinely ambiguous -- dimming makes it unmistakable which body the id
refers to while keeping enough context to judge.

Usage:
    python -m scripts.render_reference_sheets data/raw/FelixClaar.mp4 \\
        --masks source/.FelixClaar_ref_masks \\
        --out runs/tracklet_labels/FelixClaar_reference
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

TILE_HEIGHT = 280
LABEL_STRIP = 26
COLUMNS = 6
DIM = 0.35
MIN_MASK_PIXELS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--frame-offset", type=int, default=0,
        help="added to each mask filename stem to get the video frame index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mask_paths = {
        int(Path(p).stem) + args.frame_offset: p
        for p in sorted(glob.glob(str(args.masks / "*.npz")))
        if Path(p).stem.isdigit()
    }
    if not mask_paths:
        raise SystemExit(f"no label maps found in {args.masks}")

    per_identity = {}
    for frame_index, path in sorted(mask_paths.items()):
        label = np.load(path)["label"]
        for value in np.unique(label):
            if value == 0:
                continue
            per_identity.setdefault(int(value), []).append(frame_index)

    wanted = {}
    for identity, frames in per_identity.items():
        picks = np.linspace(0, len(frames) - 1, args.samples).round().astype(int)
        for index in dict.fromkeys(picks.tolist()):
            wanted.setdefault(frames[index], []).append(identity)

    tiles, captions = {}, {}
    total = sv.VideoInfo.from_video_path(str(args.video)).total_frames
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(args.video)), total=total, desc="crops",
    )):
        if frame_index not in wanted:
            continue
        label = np.load(mask_paths[frame_index])["label"]
        for identity in wanted[frame_index]:
            ys, xs = np.where(label == identity)
            if len(xs) < MIN_MASK_PIXELS:
                continue
            pad = 6
            height, width = frame_bgr.shape[:2]
            x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
            x2, y2 = min(width, xs.max() + pad), min(height, ys.max() + pad)
            crop = frame_bgr[y1:y2, x1:x2].copy()
            if crop.size == 0 or crop.shape[0] < 8:
                continue
            outside = label[y1:y2, x1:x2] != identity
            crop[outside] = (crop[outside] * DIM).astype(crop.dtype)
            scale = TILE_HEIGHT / crop.shape[0]
            crop = cv2.resize(
                crop, (max(1, int(crop.shape[1] * scale)), TILE_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            tiles.setdefault(identity, []).append(crop)
            captions.setdefault(identity, []).append(
                f"f{frame_index}  {x2 - x1}x{y2 - y1}"
            )

    manifest = {}
    for identity in sorted(tiles):
        images, labels = tiles[identity], captions[identity]
        columns = min(COLUMNS, len(images))
        rows = (len(images) + columns - 1) // columns
        cell_width = max(i.shape[1] for i in images) + 8
        cell_height = TILE_HEIGHT + LABEL_STRIP
        sheet = np.full(
            (rows * cell_height + 34, columns * cell_width, 3), 28, dtype=np.uint8
        )
        frames = per_identity[identity]
        cv2.putText(
            sheet,
            f"reference id {identity}   frames {frames[0]}-{frames[-1]}   n={len(frames)}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        for index, (image, caption) in enumerate(zip(images, labels)):
            row, column = divmod(index, columns)
            top, left = 34 + row * cell_height, column * cell_width
            sheet[top:top + TILE_HEIGHT, left:left + image.shape[1]] = image
            cv2.putText(
                sheet, caption, (left + 4, top + TILE_HEIGHT + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(args.out / f"reference_id_{identity:02d}.jpg"), sheet)
        manifest[identity] = {
            "first": frames[0], "last": frames[-1], "frames": len(frames)
        }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        f"# Reference identities for {args.video.name}.",
        "# One line per identity. C = clean (one player throughout),",
        "# M <frame> = mixed (changes hands near <frame>), X = unclear.",
        "# Use X freely on motion-blurred ones -- a smaller trustworthy",
        "# reference is worth more than a larger uncertain one.",
        "# sheet = reference_id_<NN>.jpg   non-mask pixels are dimmed",
    ]
    for identity in sorted(manifest):
        info = manifest[identity]
        lines.append(
            f"{identity}    # frames {info['first']}-{info['last']} (n={info['frames']})"
        )
    (args.out / "answers.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(manifest)} sheets to {args.out}")
    print(f"fill in {args.out / 'answers.txt'}")


if __name__ == "__main__":
    main()
