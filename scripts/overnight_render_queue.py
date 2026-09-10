"""Pick real-play windows from prepared 10-minute clips and render them, unattended.

Runs standalone: waits for the preparation pass to finish, chooses each window
from the *dense* detection cache rather than the stride-100 full-match ones (the
sparse caches picked pre-game lineups), renders one clip at a time, transcodes
for the browser, and keeps going until a wall-clock deadline.

Serial by design. SAM2's `init_state` allocates the whole clip up front at
12.58 MB/frame, so a 3,000-frame window costs ~35 GiB of the machine's 121 GiB.
Two at once would fit; three would not, and a failure at 4am costs the night.

Every failure is caught and logged so one bad clip cannot take the queue down.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / (".venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/"
                 "ffmpeg-linux-aarch64-v7.0.2")
PERSON_CLASS_IDS = (1, 2)


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def next_occurrence(now: datetime, hhmm: str) -> datetime:
    """The next time the clock reads `hhmm`.

    The deadline used to be compared as a string against `now.strftime("%H:%M")`,
    which reads correctly only when the queue starts before it in the same day.
    Launched at 22:55 with the 07:30 default, `"22:55" >= "07:30"` is true and
    the queue exits having rendered nothing -- the exact case it exists for.
    """
    hour, minute = (int(part) for part in hhmm.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def wait_for(pid: int) -> None:
    if pid <= 0:
        return
    log(f"waiting for preparation pid {pid}")
    while Path(f"/proc/{pid}").exists():
        time.sleep(30)
    log("preparation finished")


def person_counts(cache_path: Path) -> np.ndarray:
    """People per frame, from a dense cache."""
    with np.load(cache_path, allow_pickle=False) as cache:
        offsets = np.asarray(cache["offsets"])
        class_id = np.asarray(cache["class_id"])
    return np.array([
        int(np.isin(class_id[offsets[f]:offsets[f + 1]], PERSON_CLASS_IDS).sum())
        for f in range(len(offsets) - 1)
    ])


def best_windows(counts: np.ndarray, length: int, how_many: int, gap: int) -> list:
    """Starts of the most play-like windows, non-overlapping.

    Seven-a-side plus two keepers is 14 on court. A timeout or a replay shows a
    different count and a less steady one, so score on distance from 14, how
    often the frame is nearly empty, and variance -- the same rule that moved the
    earlier picks off pre-game lineups.
    """
    scores = []
    for start in range(0, len(counts) - length, 25):
        window = counts[start:start + length]
        scores.append((
            -abs(np.median(window) - 14) - 5 * np.mean(window < 8) - 0.3 * np.std(window),
            start,
        ))
    scores.sort(reverse=True)
    chosen: list[int] = []
    for _score, start in scores:
        if all(abs(start - taken) >= gap for taken in chosen):
            chosen.append(start)
        if len(chosen) == how_many:
            break
    return chosen


def run(command, cwd=ROOT) -> None:
    subprocess.run([str(c) for c in command], check=True, cwd=cwd,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def render_one(name, window_video, window_detections, team_model, out_dir) -> bool:
    output = out_dir / f"{name}.mp4"
    if output.with_suffix(".json").is_file():
        log(f"{name}: already rendered")
        return True
    started = time.monotonic()
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.render_full_pipeline", str(window_video),
             "--detections", str(window_detections),
             "--number-detections", str(window_detections),
             "--team-model", str(team_model), "--output", str(output)],
            check=True, cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        log(f"{name}: RENDER FAILED ({error})")
        return False
    log(f"{name}: rendered in {(time.monotonic() - started) / 60:.1f} min")
    try:
        run([FFMPEG, "-y", "-i", output, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             out_dir / f"{name}_h264.mp4"])
    except subprocess.CalledProcessError as error:
        log(f"{name}: transcode failed ({error}); raw mp4 is still there")
    return True


def write_index(out_dir: Path) -> None:
    rows = []
    for summary in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        name = summary.stem
        if not (out_dir / f"{name}_h264.mp4").is_file():
            continue
        rows.append(
            f"<h2>{name}</h2><p>{data.get('frames', '?')} frames &middot; "
            f"{data.get('players', '?')} identities &middot; "
            f"{data.get('fragmented_players', '?')} fragmented &middot; "
            f"{len(data.get('numbers_resolved', {}))} numbered &middot; "
            f"reader {data.get('reader', '?')}</p>"
            f'<video src="{name}_h264.mp4" controls preload=metadata></video>'
        )
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Overnight renders</title>"
        "<style>body{font:14px system-ui;background:#111;color:#eee;margin:24px}"
        "h2{margin:26px 0 4px;font-size:15px}video{width:min(100%,1200px);"
        "background:#000}p{color:#aaa;margin:2px 0 8px}</style>"
        "<h1>Overnight renders</h1>" + "".join(rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "data/interim/overnight")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/overnight")
    parser.add_argument("--minutes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--per-match", type=int, default=3)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--deadline", default="07:30",
                        help="stop starting new renders after this local time")
    args = parser.parse_args()

    deadline = next_occurrence(datetime.now(), args.deadline)
    wait_for(args.wait_pid)
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"will not start renders after {deadline:%a %H:%M}")
    length = args.minutes * 60 * args.fps

    # Round 1 takes each match's best window, so if the night is cut short there
    # is one clip from every venue rather than three from one.
    queue: list[tuple] = []
    for rank in range(args.per_match):
        for video in sorted(args.work.glob(f"*_10min.mp4")):
            stem = video.stem.replace("_10min", "")
            detections = args.work / f".{stem}_10min_det.npz"
            team = args.work / f".{stem}_10min_team.pkl"
            if not (detections.is_file() and team.is_file()):
                log(f"{stem}: preparation incomplete, skipping")
                continue
            queue.append((stem, rank, video, detections, team))

    starts: dict[str, list[int]] = {}
    for stem, rank, video, detections, team in queue:
        if datetime.now() >= deadline:
            log(f"deadline {deadline:%a %H:%M} reached; not starting more")
            break
        if stem not in starts:
            counts = person_counts(detections)
            starts[stem] = best_windows(counts, length, args.per_match, gap=length)
            log(f"{stem}: windows at frames {starts[stem]} "
                f"(median people {np.median(counts):.0f})")
        if rank >= len(starts[stem]):
            continue
        start = starts[stem][rank]
        name = f"{stem}_{args.minutes}min_{rank + 1}"
        clip = args.work / f"{name}.mp4"
        clip_detections = args.work / f".{name}_det.npz"
        if not clip.is_file():
            try:
                run([sys.executable, "-m", "scripts.extract_clip_window", video,
                     "--detections", detections, "--start", start,
                     "--stop", start + length, "--step", 1,
                     "--output-video", clip, "--output-detections", clip_detections])
            except subprocess.CalledProcessError as error:
                log(f"{name}: extraction failed ({error})")
                continue
        log(f"{name}: rendering frames {start}-{start + length}")
        render_one(name, clip, clip_detections, team, args.out)
        write_index(args.out)

    write_index(args.out)
    log("queue finished")


if __name__ == "__main__":
    main()
