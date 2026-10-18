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
    # Fresh test for rules locked after Stage 1 (the two above are development data
    # now). RNL-SGF shares the Loewen (RNL) players with the match above.
    ("Ludwigshafen_Wetzlar_67m00_10min", "Ludwigshafen-Wetzlar",
     "train/Ludwigshafen-Wetzlar.mp4", "detector-train"),
    ("RNL_SGF_66m00_10min", "RNL-SGF", "train/RNL-SGF.mp4", "detector-train"),
]
# Development bank (2026-10-19): `train/` matches, only the detector trained on them.
# Their teams overlap earlier development data only, never the test or sealed sets.
# Tuning and leave-one-match-out checks run here in seconds once processed.
DEV_WINDOWS = [
    ("BAL_THW_66m00_10min", "Balingen-Kiel", "train/BAL-THW.mp4", "detector-train"),
    ("BER_BHC_66m00_10min", "Berlin-BHC", "train/BER-BHC.mp4", "detector-train"),
    ("TBV_SCM_66m00_10min", "Lemgo-Magdeburg", "train/TBV-SCM.mp4", "detector-train"),
    ("FAG_LEI_66m00_10min", "Goeppingen-Leipzig", "train/FAG-LEI.mp4", "detector-train"),
    ("DRE_MTM_66m00_10min", "Dresden-Melsungen", "train/DRE-MTM.mp4", "detector-train"),
    ("Vinnhorst_Potsdam_66m00_10min", "Vinnhorst-Potsdam",
     "train/sl_1435862_2026-08-21_1435862_tus_vinnhorst-1__vfl_potsdam_720p24_4001kbps.mp4", "detector-train"),
]
# Sealed test bank: every team is new to all development data. Processed now, mined
# only after a rule is locked; nothing in them may inform a method or threshold choice.
SEALED_WINDOWS = [
    ("Altenholz_Luebeck_66m00_10min", "Altenholz-Luebeck", "train/Altenholz-Lübeck.mp4", "detector-train"),
    ("Pforzheim_Bietigheim_66m00_10min", "Pforzheim-Bietigheim", "train/Pfortzheim-BBM.mp4", "detector-train"),
]
WINDOW_SETS = {"test": WINDOWS, "dev": DEV_WINDOWS, "sealed": SEALED_WINDOWS}
# The rule's one allowed nudge: step forward in whole minutes to the first live play.
# Ludwigshafen-Wetzlar is in the half-time break at 66:00; the second-half throw-off
# is at 67:00 (checked 66:00-70:30 for live play only).
START_OVERRIDE_S = {"Ludwigshafen_Wetzlar_67m00_10min": 67 * 60}
MANIFEST = REPO / "runs/smart-reid-shadow/heldout_windows.json"


def cut(name: str, source: Path, out: Path) -> dict:
    start = START_OVERRIDE_S.get(name, START_S)
    if not out.is_file():
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source),
                        "-t", str(MINUTES * 60), "-vf", f"fps={FPS}", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True)
    frames = VideoFrames(out)
    return {"window": name, "video": str(out), "source": str(source),
            "source_bytes": source.stat().st_size, "start_s": start, "minutes": MINUTES,
            "frames": len(frames), "fps": frames.fps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("cut",))
    parser.add_argument("--count", type=int, default=None, help="how many windows from the top of the set")
    parser.add_argument("--set", dest="window_set", choices=tuple(WINDOW_SETS), default="test")
    args = parser.parse_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {"windows": {}}
    manifest["rule"] = f"{MINUTES} min from {START_S // 60}:00 of the broadcast, {FPS} fps (fixed before processing)"
    for name, match, rel, exposure in WINDOW_SETS[args.window_set][:args.count]:
        started = time.monotonic()
        entry = cut(name, SOURCES / rel, REPO / "data/raw" / f"{name}.mp4")
        entry.update(match=match, detector_exposure=exposure, role=args.window_set, cut_s=round(time.monotonic() - started, 1))
        manifest["windows"][name] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
