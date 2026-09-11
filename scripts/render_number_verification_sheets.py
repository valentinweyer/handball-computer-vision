"""Sheets for checking a run's jersey-number claims against the footage.

Every number a run asserts is unverified until somebody reads the shirt. This
renders one sheet per identity so that pass can happen, from the geometry cache
a render already wrote -- no tracker, no reader, no GPU.

Two decisions make the sheets worth trusting:

**Blind by default.** Captions carry the frame index and nothing else. Seeing
`#15` printed under a crop is exactly the suggestion that turns a check into a
confirmation, so the run's own verdict is withheld until `--show-verdict`, which
exists for reviewing a finished pass rather than making one.

**Mask-dimmed, not cropped.** A box around a player in a scrum routinely
contains a second player, and the number visible in it may be the neighbour's.
Dimming everything outside the identity's own mask keeps that unmistakable while
leaving the context needed to judge. Same reasoning as
`render_reference_sheets.py`, which does this for identity continuity.

Frames are chosen where the number detector actually fired for that identity,
largest box first: those are the crops the claim rests on, and the ones where a
human has any chance of reading the shirt. `--samples` caps how many.

Usage:
    python -m scripts.render_number_verification_sheets \\
        data/raw/Melsungen_window_cached.mp4 \\
        --run runs/full_pipeline/Melsungen_geo.json \\
        --out runs/number_truth/Melsungen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.tracking.sam2_driver import ensure_frame_cache

ROOT = Path(__file__).resolve().parents[1]
TILE_HEIGHT = 360
LABEL_STRIP = 24
COLUMNS = 5
DIM = 0.35
PAD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run", required=True, type=Path, help="a run summary JSON")
    parser.add_argument("--geometry-cache", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--frame-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--show-verdict", action="store_true",
        help="print the run's own number on each sheet. Off by default so the "
             "pass is blind; turn it on only to review a finished one",
    )
    parser.add_argument(
        "--number-crops", action="store_true",
        help="tile the number boxes themselves at high magnification instead of "
             "whole players. The player sheet says which body the id follows; "
             "this one is where the digits are actually legible, so a real pass "
             "needs both -- a number read off the wrong body is still wrong",
    )
    parser.add_argument(
        "--span", action="store_true",
        help="sample evenly across each identity's whole life instead of only "
             "frames where the number detector fired. What answers 'is this one "
             "person throughout' and 'was there ever a legible number', neither "
             "of which the read frames can address -- an identity with no reads "
             "has none to show",
    )
    parser.add_argument(
        "--all-identities", action="store_true",
        help="also sheet identities the run resolved no number for, which is "
             "what recall needs -- precision only needs the resolved ones",
    )
    return parser.parse_args()


def tile(frame_bgr, mask, box) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
    x2, y2 = min(width, x2 + PAD), min(height, y2 + PAD)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame_bgr[y1:y2, x1:x2].copy()
    outside = ~mask[y1:y2, x1:x2]
    crop[outside] = (crop[outside] * DIM).astype(crop.dtype)
    scale = TILE_HEIGHT / crop.shape[0]
    return cv2.resize(
        crop, (max(1, int(crop.shape[1] * scale)), TILE_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def number_tile(frame_bgr, box, height: int) -> np.ndarray | None:
    """The number box alone, padded a little and blown up.

    Padded because the detector's box is tight by design (NUMBER_CROP_PAD = 0,
    measured best for the reader) and a human reading it wants the digit edges.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    x1, y1 = max(0, int(round(x1 - dx))), max(0, int(round(y1 - dy)))
    x2, y2 = min(w, int(round(x2 + dx))), min(h, int(round(y2 + dy)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    scale = height / crop.shape[0]
    return cv2.resize(
        crop, (max(1, int(crop.shape[1] * scale)), height),
        interpolation=cv2.INTER_CUBIC,
    )


def main() -> None:
    args = parse_args()
    run = json.loads(args.run.read_text())
    cache_path = args.geometry_cache or Path(run["geometry_cache"])
    cache = GeometryCache.load(cache_path)
    resolved = {int(k): v for k, v in run["numbers_resolved"].items()}
    teams = {int(k): v for k, v in run["player_teams"].items()}

    # Frames where the number detector fired for each identity: the evidence the
    # claim rests on, and the only frames where the shirt is plausibly legible.
    reads: dict = {}
    boxes_by_read: dict = {}
    for row in run["number_reads"]:
        identity = int(row["player_id"])
        reads.setdefault(identity, []).append(int(row["frame"]))
        boxes_by_read[(identity, int(row["frame"]))] = row["box"]

    live_frames: dict = {}
    for frame_idx, ids in run["frame_players"].items():
        for identity in ids:
            live_frames.setdefault(int(identity), []).append(int(frame_idx))

    wanted = sorted(resolved) if not args.all_identities else sorted(teams)
    frame_files = ensure_frame_cache(
        args.video.resolve(),
        args.frame_cache_dir or (ROOT / "data/cache/frames" / args.video.stem),
    )

    # Rank each identity's candidate frames by how large it is on screen -- a
    # bigger silhouette is a more readable shirt.
    picks: dict = {}
    for identity in wanted:
        candidates = sorted(set(reads.get(identity, [])))
        # An identity the detector never fired on has no read frames at all, and
        # those are the ones recall is about -- fall back to its life so the
        # question "was the number ever legible" can be asked of it.
        if args.span or not candidates:
            live = sorted(live_frames.get(identity, []))
            if live:
                step = np.linspace(0, len(live) - 1, args.samples).round().astype(int)
                picks[identity] = [live[i] for i in dict.fromkeys(step.tolist())]
                continue
        sized = []
        for frame_idx in candidates:
            geo = cache.frame(frame_idx)
            rows = np.nonzero(geo.player_ids == identity)[0]
            if not len(rows):
                continue
            box = geo.boxes[rows[0]]
            sized.append(((box[2] - box[0]) * (box[3] - box[1]), frame_idx))
        sized.sort(reverse=True)
        picks[identity] = [f for _area, f in sized[:args.samples]]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for identity in wanted:
        images, captions = [], []
        for frame_idx in sorted(picks.get(identity, [])):
            geo = cache.frame(frame_idx)
            rows = np.nonzero(geo.player_ids == identity)[0]
            if not len(rows):
                continue
            row = rows[0]
            frame_bgr = cv2.imread(str(frame_files[frame_idx]))
            if args.number_crops:
                box = boxes_by_read.get((identity, frame_idx))
                image = None if box is None else number_tile(
                    frame_bgr, box, TILE_HEIGHT
                )
            else:
                image = tile(frame_bgr, geo.masks[row], geo.boxes[row])
            if image is None:
                continue
            images.append(image)
            captions.append(f"f{frame_idx}")
        if not images:
            continue

        columns = min(COLUMNS, len(images))
        rows_n = (len(images) + columns - 1) // columns
        cell_w = max(i.shape[1] for i in images) + 8
        cell_h = TILE_HEIGHT + LABEL_STRIP
        sheet = np.full((rows_n * cell_h + 34, columns * cell_w, 3), 28, np.uint8)
        header = f"identity p{identity}   team {teams.get(identity, '?')}   n={len(images)}"
        if args.show_verdict:
            header += f"   run says #{resolved.get(identity, '-')}"
        cv2.putText(sheet, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2, cv2.LINE_AA)
        for index, (image, caption) in enumerate(zip(images, captions)):
            r, c = divmod(index, columns)
            top, left = 34 + r * cell_h, c * cell_w
            sheet[top:top + TILE_HEIGHT, left:left + image.shape[1]] = image
            cv2.putText(sheet, caption, (left + 4, top + TILE_HEIGHT + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        stem = "digits" if args.number_crops else "number"
        cv2.imwrite(str(args.out / f"{stem}_p{identity:02d}.jpg"), sheet)
        manifest[identity] = {"tiles": len(images), "frames": sorted(picks[identity])}

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    answers = args.out / "answers.txt"
    if not answers.exists():
        lines = [
            f"# Jersey numbers for {args.video.name}, read off the sheets.",
            "# One line per identity: `p<id> <number>`, or",
            "#   X  -- cannot read it; the claim is unverifiable from these crops",
            "#   ?  -- a guess, do not score it",
            "# Bias to X. A smaller trustworthy reference beats a larger uncertain one.",
            "",
        ]
        lines += [f"p{identity}" for identity in sorted(manifest)]
        answers.write_text("\n".join(lines) + "\n")
    print(f"{len(manifest)} sheets -> {args.out}")


if __name__ == "__main__":
    main()
