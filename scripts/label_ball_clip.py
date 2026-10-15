#!/usr/bin/env python
"""Label the ball on one clip's frames, using the sibling repo's labeller unchanged.

    python -m scripts.label_ball_clip --clip BHC-FAG_window_cached --every 5

Opens the same page `rfdetr-handball-finetune/scripts/label_ball.py` serves --
same keys, same preloading, same atomic writes, the tool that already produced
1,050 labels at ~5.4 s/frame. This module only points it at different files.

Why a launcher instead of a second labeller
-------------------------------------------
A fork would be a second thing to fix. The sibling's tool reads its paths from
four module-level constants (`FRAME_DIR`, `MATCHES`, `LABELS`, `CACHE`) and uses
them *inside* its functions, so rebinding them before `main()` redirects the
whole tool with no change to its code.

Why not just add these frames to its dataset
--------------------------------------------
The obvious route -- drop the frames in `datasets/expansion-v1/full/` and add a
match to its `matches.json` -- would break the sibling repo. Its
`build_ball_dataset.py` does `writers[match["split"]].add(...)` against writers
built only for train/valid/test, so a match carrying any other split raises
KeyError and the next dataset build dies. Staging here keeps its dataset exactly
as it is, and keeps these labels where this project's other protected labels
live, in `data/annotations/ball/`.

What these labels are for
-------------------------
Recall. The sibling's 1,050 labelled frames are isolated samples -- median 2547
frames apart, no two adjacent -- which can score a single-frame detector and
nothing else, because SAM2 propagation needs the frames in between. Measuring
what the *tracker* finds needs labels on a clip that can actually be tracked.

Two properties make the result a recall measurement rather than another
precision one, and both are worth preserving if this is ever re-run:

* **Frames are chosen by index** (`--every`), never by where the tracker looked.
  Sampling the tracker's own output can only re-measure what it already found.
* **"No ball visible" is an answer, not a skip.** Roughly a third of broadcast
  frames genuinely have no visible ball -- 341 of the sibling's 1,050 -- and
  treating those as unlabelled would turn every correct abstention into a miss.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SIBLING = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
FRAME_ROOT = ROOT / "data/cache/frames"
OUT_DIR = ROOT / "data/annotations/ball"

# The labeller splits a frame's stem on the LAST "-f" to recover its match id, so
# the id must not contain that substring or the frame index parses wrong.
SPLIT_NAME = "recall"


def stage(clip: str, every: int, match_id: str) -> tuple[Path, Path, list[int]]:
    """Write the sampled frames as PNGs named the way the labeller expects."""
    from PIL import Image

    source = FRAME_ROOT / clip
    files = sorted(source.glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no frames in {source}")
    chosen = files[::every]

    frame_dir = OUT_DIR / f"{clip}_label_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    width = height = None
    for path in chosen:
        image = Image.open(path)
        width, height = image.size
        target = frame_dir / f"{match_id}-f{int(path.stem):06d}.png"
        if not target.exists():
            image.convert("RGB").save(target, "PNG")

    matches = OUT_DIR / f"{clip}_label_matches.json"
    matches.write_text(json.dumps({"matches": [{
        "id": match_id, "title": clip, "split": SPLIT_NAME,
        "width": width, "height": height,
    }]}, indent=1))
    return frame_dir, matches, [int(p.stem) for p in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--every", type=int, default=5,
                        help="label every Nth frame (5 gives 100 from a 500-frame clip)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not (SIBLING / "scripts/label_ball.py").is_file():
        raise SystemExit(
            f"needs the sibling repo's labeller at {SIBLING}/scripts/label_ball.py; "
            "set RFDETR_FINETUNE_DIR if it lives elsewhere"
        )

    # Stripped to letters and digits so the id cannot contain "-f", which the
    # labeller splits on to recover the frame index.
    match_id = re.sub(r"[^a-z0-9]", "", args.clip.lower()) + "recall"

    frame_dir, matches, frames = stage(args.clip, args.every, match_id)
    labels = OUT_DIR / f"{args.clip}_recall_labels.json"

    sys.path.insert(0, str(SIBLING / "scripts"))
    import label_ball

    label_ball.FRAME_DIR = frame_dir
    label_ball.MATCHES = matches
    label_ball.LABELS = labels
    label_ball.CACHE = OUT_DIR / f"{args.clip}_label_cache"

    print(f"clip {args.clip}: every {args.every}th frame -> {len(frames)} to label")
    print(f"labels    -> {labels}")
    print(f"staged in -> {frame_dir}")

    sys.argv = ["label_ball.py", "--split", SPLIT_NAME,
                "--host", args.host, "--port", str(args.port)]
    label_ball.main()


if __name__ == "__main__":
    main()
