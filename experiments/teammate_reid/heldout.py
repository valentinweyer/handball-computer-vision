"""Held-out match windows for the smart re-ID transfer test.

Kiel-Lemgo chose the prototype's method and thresholds, so it cannot test them.
These windows come from matches whose teams appear nowhere in the prototype's
data (Kiel, Lemgo, BHC, FAG, Melsungen, Berlin). The detector-unseen `eval/`
matches are listed first.

The window rule was fixed before any of them was processed: 10 minutes from
66:00 of the broadcast file, at 25 fps. This is the same rule as
`Kiel_Lemgo_66m00_10min.mp4`. Frames at 66-76 min were checked only for live
play.

    python -m experiments.teammate_reid.heldout cut
    python -m scripts.process_video data/raw/<window>.mp4     # from the main checkout

`process_video` runs unmodified, with production defaults (SAM2), so the
bundles are also the production baseline that Stage 2 needs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from handball_cv.video import VideoFrames, ffmpeg_exe

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
SOURCES = Path("/home/valentinweyer/projects/rfdetr-handball-finetune/datasets/videos")
START_S = 66 * 60
MINUTES = 10
FPS = 25

# (window name, match = split unit, source, detector exposure)
WINDOWS = [
    ("Eisenach_Hamburg_66m00_10min", "Eisenach-Hamburg",
     "eval/2026-06-07_1284788_Eisenach_Hamburg.mp4", "detector-unseen"),
    ("VfL_GM_Loewen_66m00_10min", "Gummersbach-Loewen",
     "eval/2026-06-07_1284786_VfL_GM_Loewen.mp4", "detector-unseen"),
    # Only if the pooled sample from the two above is too thin. Note that RNL-SGF
    # shares the Loewen (RNL) players with the match above.
    ("Ludwigshafen_Wetzlar_66m00_10min", "Ludwigshafen-Wetzlar",
     "train/Ludwigshafen-Wetzlar.mp4", "detector-train"),
    ("RNL_SGF_66m00_10min", "RNL-SGF", "train/RNL-SGF.mp4", "detector-train"),
]
MANIFEST = REPO / "runs/smart-reid-shadow/heldout_windows.json"


def cut(name: str, source: Path, out: Path) -> dict:
    if not out.is_file():
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(START_S), "-i", str(source),
                        "-t", str(MINUTES * 60), "-vf", f"fps={FPS}", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True)
    frames = VideoFrames(out)
    return {"window": name, "video": str(out), "source": str(source),
            "source_bytes": source.stat().st_size, "start_s": START_S, "minutes": MINUTES,
            "frames": len(frames), "fps": frames.fps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("cut",))
    parser.add_argument("--count", type=int, default=2, help="how many windows from the top of WINDOWS")
    args = parser.parse_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {"windows": {}}
    manifest["rule"] = f"{MINUTES} min from {START_S // 60}:00 of the broadcast, {FPS} fps (fixed before processing)"
    for name, match, rel, exposure in WINDOWS[:args.count]:
        started = time.monotonic()
        entry = cut(name, SOURCES / rel, REPO / "data/raw" / f"{name}.mp4")
        entry.update(match=match, detector_exposure=exposure, cut_s=round(time.monotonic() - started, 1))
        manifest["windows"][name] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
