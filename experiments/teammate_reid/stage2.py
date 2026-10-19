"""Stage 2: incremental coverage on baseline-unresolved segments (the three split-v1 test matches).

Order, so the matcher can never shape the labels:

1. `population` -- fix the population before any matcher runs. A local segment is one
   tracker id's uninterrupted presence, cut at identity breaks and >1 s gaps (exactly
   `mining.split_fragments`). It is in the population when it is not a goalkeeper, the
   production bundle shows a number on < 50% of its rows (baseline-unresolved), and at
   least 3 clean crops can be sampled from it (`mining.sample_crops`, the matcher's own
   evidence floor). Shorter or crop-poor unresolved segments are listed as
   `insufficient_evidence` with durations, never labelled or scored.
2. `review` -- blind review material per segment: a keyframe sheet (whole-frame context
   before, during and after, plus zoomed crops) and a short mp4. Only the target box is
   drawn, in one neutral colour; no IDs, team colours, numbers or proposals.
3. Tier-1 labels (assistant, blind) go to `annotations_assistant_v1.json` and are
   frozen (hashed) before step 4. Tier 2 is the user's blind check of a compact queue.
4. `propose` -- the split-v1 lock's matcher on the population; written separately.

    python -m experiments.teammate_reid.stage2 population
    python -m experiments.teammate_reid.stage2 review
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments.teammate_reid.mining import load_video, mine_video, sample_crops
from experiments.teammate_reid.run import HELDOUT, VIDEOS

OUT = Path("/home/valentinweyer/projects/handball-computer-vision/runs/smart-reid-shadow/stage2")
SPECS = {"Kiel-Lemgo": VIDEOS[0], "Eisenach-Hamburg": HELDOUT[0], "Gummersbach-Loewen": HELDOUT[1]}
SHOWN_MAX = 0.5          # baseline shows a number on fewer rows than this -> unresolved
MIN_CROPS = 3            # the matcher's evidence floor
FPS = 25.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    segments, excluded, inputs = [], [], {}
    for match, spec in SPECS.items():
        loaded = load_video(spec)
        frags, _ = mine_video(loaded)
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        tbf = {f: g for f, g in tracks.groupby("frame")}
        nbf = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        det = pd.read_parquet(spec.bundle / "detections.parquet")
        refs = {f: list(g.itertuples()) for f, g in det[det["class_name"] == "referee"].groupby("frame")}
        src = loaded["manifest"]["source"]
        inputs[match] = {k: v["sha256"] for k, v in loaded["inputs"].items()}
        for f in frags:
            rows = by_track[f.track_id].loc[f.start:f.end]
            record = {"segment": f"{match}|{f.key}", "match": match, "video": spec.name, "key": f.key,
                      "track_id": f.track_id, "start": f.start, "end": f.end,
                      "duration_s": round(len(rows) / FPS, 2), "opened_by": f.opened_by, "closed_by": f.closed_by}
            if rows["is_goalkeeper"].any():
                continue
            shown = float(rows["number"].notna().mean())
            if shown >= SHOWN_MAX:
                continue                     # baseline-resolved: not in the population
            crops, _ = sample_crops(f, tbf, nbf, set(det["frame"].unique()), src["width"], src["height"],
                                    every=10, cap=60, referees_by_frame=refs)
            record.update(baseline_shown=round(shown, 3), clean_crops=len(crops),
                          own_reads=dict(Counter(f.reads)) if f.reads else {})
            (segments if len(crops) >= MIN_CROPS else excluded).append(record)
    for i, s in enumerate(sorted(segments, key=lambda s: (s["match"], s["start"], s["track_id"]))):
        s["review_id"] = f"S{i:03d}"
    report = {"definition": {"shown_max": SHOWN_MAX, "min_crops": MIN_CROPS, "goalkeepers": "excluded",
                             "segments": "mining.split_fragments (identity breaks, >1 s gaps)"},
              "inputs": inputs, "population": segments, "insufficient_evidence": excluded,
              "summary": {m: {"segments": sum(s["match"] == m for s in segments),
                              "seconds": round(sum(s["duration_s"] for s in segments if s["match"] == m), 1),
                              "insufficient": sum(s["match"] == m for s in excluded),
                              "insufficient_seconds": round(sum(s["duration_s"] for s in excluded if s["match"] == m), 1)}
                          for m in SPECS}}
    (OUT / "population.json").write_text(json.dumps(report, indent=1, default=float))
    return report


BOX = (0, 255, 255)      # one neutral colour for the target; nothing else is drawn


def review() -> None:
    """Blind review sheets and clips; reads only population.json and the bundles."""
    pop = json.loads((OUT / "population.json").read_text())["population"]
    rdir = OUT / "review"
    rdir.mkdir(exist_ok=True)
    for match, spec in SPECS.items():
        segs = [s for s in pop if s["match"] == match]
        if not segs:
            continue
        tracks = pd.read_parquet(spec.bundle / "tracks.parquet", columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        box_of = {(int(r.frame), int(r.track_id)): (r.x1, r.y1, r.x2, r.y2) for r in tracks.itertuples()}
        manifest = json.loads((spec.bundle / "manifest.json").read_text())
        cap = cv2.VideoCapture(str(spec.bundle / manifest["files"]["video"]))
        want: dict = {}
        plans = {}
        for s in segs:
            span = list(range(s["start"], s["end"] + 1))
            inside = [span[i] for i in np.linspace(0, len(span) - 1, 8).round().astype(int)]
            before = [max(0, s["start"] - 50), max(0, s["start"] - 12)]
            after = [s["end"] + 12, s["end"] + 50]
            clip = list(range(max(0, s["start"] - 50), s["end"] + 51, 2))[:450]   # <= 36 s at half rate
            plans[s["review_id"]] = (s, before + inside + after, clip)
            for f in before + inside + after + clip:
                want.setdefault(f, []).append(s["review_id"])
        frames: dict = {}
        writers = {}
        for f in range(max(want) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if f not in want:
                continue
            for rid in want[f]:
                s, keys, clip = plans[rid]
                box = box_of.get((f, s["track_id"]))
                if f in keys:
                    frames.setdefault(rid, {})[f] = (frame.copy(), box)
                if f in clip:
                    small = cv2.resize(frame, (960, 540))
                    if box is not None:
                        x1, y1, x2, y2 = (int(v / 2) for v in box)
                        cv2.rectangle(small, (x1, y1), (x2, y2), BOX, 2)
                    cv2.putText(small, f"{rid}  frame {f}", (8, 24), 0, 0.7, (255, 255, 255), 2)
                    if rid not in writers:
                        writers[rid] = cv2.VideoWriter(str(rdir / f"{rid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                       12.5, (960, 540))
                    writers[rid].write(small)
        for w in writers.values():
            w.release()
        for rid, (s, keys, _) in plans.items():
            _sheet(rdir / f"{rid}.jpg", rid, s, keys, frames.get(rid, {}))


def _sheet(path: Path, rid: str, s: dict, keys: list, frames: dict) -> None:
    W = 1600
    tiles, zooms = [], []
    for f in keys:
        if f not in frames:
            continue
        frame, box = frames[f]
        ctx = frame.copy()
        tag = "before" if f < s["start"] else "after" if f > s["end"] else "in"
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(ctx, (x1, y1), (x2, y2), BOX, 4)
            if tag == "in":
                bw, bh = x2 - x1, y2 - y1
                zx1, zy1 = max(0, x1 - bw // 2), max(0, y1 - bh // 4)
                zx2, zy2 = min(frame.shape[1], x2 + bw // 2), min(frame.shape[0], y2 + bh // 4)
                z = frame[zy1:zy2, zx1:zx2]
                if z.size:
                    z = cv2.resize(z, (max(1, int(z.shape[1] * 300 / z.shape[0])), 300))
                    cv2.putText(z, str(f), (4, 292), 0, 0.5, (255, 255, 255), 1)
                    zooms.append(z)
        t = cv2.resize(ctx, (384, 216))
        cv2.putText(t, f"{tag} f{f}", (6, 22), 0, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    head = np.full((44, W, 3), 25, np.uint8)
    cv2.putText(head, f"{rid}  ({s['duration_s']:.1f} s)  target = yellow box. Who is it? Number if readable anywhere,"
                      f" else 'unverifiable'; 'mixed' if the box changes person.", (8, 30), 0, 0.6, (255, 255, 255), 1)
    grid = [np.hstack(tiles[i:i + 4] + [np.zeros((216, 384, 3), np.uint8)] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    rows = [head] + [np.hstack([g, np.zeros((216, W - g.shape[1], 3), np.uint8)]) for g in grid]
    if zooms:
        z = np.hstack(zooms)
        z = z[:, :W] if z.shape[1] >= W else np.hstack([z, np.zeros((300, W - z.shape[1], 3), np.uint8)])
        rows.append(z)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("population", "review"))
    args = parser.parse_args()
    if args.command == "population":
        print(json.dumps(population()["summary"], indent=1))
    else:
        review()


if __name__ == "__main__":
    main()
