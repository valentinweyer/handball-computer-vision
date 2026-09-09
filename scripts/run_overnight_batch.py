"""Extract a window from each full match and run the whole pipeline over it.

Qualitative, not scored: these windows have no ground truth. The point is to see
the current defaults behave on unseen footage at a length that exposes what a
60-second clip cannot -- identities surviving substitutions, numbers holding
across possessions, how fragmentation accumulates.

Cost is dominated entirely by the SAM2 render at ~1.07 s/frame; detection is
0.036 s/frame and the team fit is minutes, so the only lever on total runtime is
frame count. Windows are emitted at 25 fps (the rate FelixClaar, Han-Ber4 and
the Melsungen window already use), halving the frames a 50 fps broadcast would
otherwise cost.

Every stage skips work that already exists, so an interrupted batch resumes by
being run again.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / (".venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/"
                 "ffmpeg-linux-aarch64-v7.0.2")
RFDETR_PYTHON = Path("/home/valentinweyer/projects/rfdetr-handball-finetune/.venv/bin/python3")
FPS = 25


def run(command, **kwargs):
    print(f"    $ {' '.join(str(c) for c in command[:6])} ...", flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def cut_window(source: Path, start_frame: int, minutes: int, out: Path) -> None:
    """A `minutes`-long window starting at `start_frame` of a 50 fps source."""
    if out.is_file():
        print(f"    window exists: {out.name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-y", "-ss", f"{start_frame / 50:.3f}", "-i", source,
         "-t", str(minutes * 60), "-vf", f"fps={FPS}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect(window: Path, out: Path, threshold: float) -> None:
    """Dense RF-DETR pass. The full-match caches are stride-100 and unusable here."""
    if out.is_file():
        print(f"    detections exist: {out.name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run([RFDETR_PYTHON, "-m", "scripts.cache_number_detections", window,
         "--output", out, "--threshold", str(threshold)], cwd=ROOT)


def fit_team_model(window: Path, detections: Path, out: Path, device: str) -> None:
    """Team discovery is per-video, so every window needs its own model."""
    if out.is_file():
        print(f"    team model exists: {out.name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import numpy as np;"
        "from handball_cv.teams.model import TeamModel;"
        "from scripts.render_raw_team_classification import person_detections;"
        "import scripts.render_full_pipeline as R;"
        f"cache=R.load_detection_cache(__import__('pathlib').Path(r'{detections}'));"
        "state={'i':-1};\n"
        "def detect_fn(frame_rgb):\n"
        "    state['i']+=1\n"
        "    return person_detections(cache, state['i'])\n"
        f"m=TeamModel.fit_from_video(r'{window}', detect_fn, device=r'{device}');"
        f"m.save(r'{out}')"
    )
    run([sys.executable, "-c", code], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="JSON: [{name, source, start_frame}]")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--work", type=Path, default=ROOT / "data/interim/overnight")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/overnight")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render", action="store_true",
                        help="also run the render; omit to only prepare inputs")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in plan:
        name = entry["name"]
        started = time.monotonic()
        print(f"\n=== {name} ===", flush=True)
        window = args.work / f"{name}_{args.minutes}min.mp4"
        detections = args.work / f".{name}_{args.minutes}min_det.npz"
        team = args.work / f".{name}_{args.minutes}min_team.pkl"

        cut_window(Path(entry["source"]), int(entry["start_frame"]), args.minutes, window)
        detect(window, detections, args.threshold)
        fit_team_model(window, detections, team, args.device)
        print(f"    prepared in {(time.monotonic() - started) / 60:.1f} min", flush=True)

        if args.render:
            run([sys.executable, "-m", "scripts.render_full_pipeline", window,
                 "--detections", detections, "--number-detections", detections,
                 "--team-model", team, "--device", args.device,
                 "--output", args.out / f"{name}.mp4"], cwd=ROOT)
            print(f"    {name} done in {(time.monotonic() - started) / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
