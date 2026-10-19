"""Stage 2 shadow video: the production render with smart re-ID's accepted names drawn on top.

Production is rendered unchanged (`scripts.render_result`, into the stage-2 folder, not
the bundle). This pass then draws, only on the frames of the segments the matcher
accepted, a magenta box and the proposed number with "smart" beside the production
label. Proposals stay shadow output: nothing is written back to any bundle. A short
highlight reel cuts just those segments (with context) out of the overlay video.

    python -m experiments.teammate_reid.shadow_video Kiel-Lemgo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import cv2
from tqdm import tqdm

from experiments.teammate_reid.stage2 import OUT, SPECS
from handball_cv.pipeline.result import load
from handball_cv.visualization.video_writer import FfmpegWriter

MAGENTA = (255, 0, 255)
CONTEXT_S = 2.0
CAPTION = "magenta = smart re-ID proposal (shadow, not in production)"


def accepted_segments(match: str) -> list:
    """The matcher's accepted proposals for one match, with track id and frame range."""
    pop = {s["review_id"]: s for s in json.loads((OUT / "population.json").read_text())["population"]}
    rows = json.loads((OUT / "proposals_SEALED.json").read_text())["rows"]
    out = []
    for r in rows:
        if r["match"] == match and r["decision"] == "accepted":
            s = pop[r["review_id"]]
            number = r["proposal"].split("#")[-1]
            out.append({"review_id": r["review_id"], "track_id": s["track_id"], "start": s["start"],
                        "end": s["end"], "number": number})
    return sorted(out, key=lambda s: s["start"])


def overlay(match: str, base, dest) -> None:
    result = load(SPECS[match].bundle)
    segs = accepted_segments(match)
    tracks = result.tracks
    boxes = {}
    for s in segs:
        rows = tracks[(tracks.track_id == s["track_id"]) & tracks.frame.between(s["start"], s["end"])]
        for r in rows.itertuples():
            boxes.setdefault(int(r.frame), []).append((r.x1, r.y1, r.x2, r.y2, s))
    src = result.manifest["source"]
    w, h, fps = int(src["width"]), int(src["height"]), float(src["fps"])
    cap = cv2.VideoCapture(str(base))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap_w = cv2.getTextSize(CAPTION, font, 0.8, 2)[0][0]   # bottom-right: the legend is bottom-left
    with FfmpegWriter(dest, w, h, fps) as writer:
        for frame in tqdm(range(int(src["frames"])), desc=f"overlay {match}"):
            ok, img = cap.read()
            if not ok:
                break
            for x1, y1, x2, y2, s in boxes.get(frame, []):
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, MAGENTA, 3)
                text = f"#{s['number']} smart"
                (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
                tl = (p1[0], max(0, p1[1] - th - 12))
                cv2.rectangle(img, tl, (tl[0] + tw + 10, tl[1] + th + 12), MAGENTA, -1)
                cv2.putText(img, text, (tl[0] + 5, tl[1] + th + 5), font, 0.9, (255, 255, 255), 2)
            cv2.putText(img, CAPTION, (w - cap_w - 24, h - 24), font, 0.8, MAGENTA, 2)
            writer.write(img)
    cap.release()


def highlights(match: str, video, dest) -> None:
    """Concatenate each accepted segment with context, merging overlaps."""
    fps = float(load(SPECS[match].bundle).manifest["source"]["fps"])
    spans = []
    for s in accepted_segments(match):
        a, b = max(0.0, s["start"] / fps - CONTEXT_S), s["end"] / fps + CONTEXT_S
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    parts = "".join(f"[0:v]trim={a:.2f}:{b:.2f},setpts=PTS-STARTPTS[v{i}];" for i, (a, b) in enumerate(spans))
    graph = parts + "".join(f"[v{i}]" for i in range(len(spans))) + f"concat=n={len(spans)}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", graph,
                    "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-pix_fmt", "yuv420p", str(dest)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("match", choices=tuple(SPECS))
    args = parser.parse_args()
    folder = OUT / "video"
    folder.mkdir(exist_ok=True)
    base = folder / f"{args.match}_production.mp4"
    if not base.is_file():
        subprocess.run([sys.executable, "-m", "scripts.render_result", str(SPECS[args.match].bundle),
                        "--output", str(base)], check=True)
    smart = folder / f"{args.match}_smart.mp4"
    overlay(args.match, base, smart)
    highlights(args.match, smart, folder / f"{args.match}_smart_highlights.mp4")
    print(f"wrote {smart} and highlights")


if __name__ == "__main__":
    main()
