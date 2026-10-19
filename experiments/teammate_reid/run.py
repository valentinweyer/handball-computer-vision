"""Teammate re-ID prototype: mine -> extract -> evaluate/train. Offline, shadow only.

    PYTHONPATH=src:. .venv/bin/python -m experiments.teammate_reid.run mine
    PYTHONPATH=src:. .venv/bin/python -m experiments.teammate_reid.run extract
    PYTHONPATH=src:. .venv/bin/python -m experiments.teammate_reid.run evaluate

Every stage reads the repository's existing bundles and run summaries
read-only and writes only under `--out`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from experiments.teammate_reid.heldout import DEV_WINDOWS, SEALED_WINDOWS, WINDOWS
from experiments.teammate_reid.masking import MODES, apply_mask, crop_with_digits
from experiments.teammate_reid.mining import VideoSpec, load_video, mine_video, sample_crops

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
DEFAULT_OUT = REPO / "runs/smart-reid-prototype-2026-10-18/results"

# match = the leakage unit: two videos of one match never sit on both sides of a split.
VIDEOS = [
    VideoSpec("kiel_lemgo_10min", "Kiel-Lemgo",
              REPO / "runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4_backfill_fix",
              REPO / "data/interim/Kiel_Lemgo_66m00_10min_expanded_validation_v4/track.json"),
    VideoSpec("bhc_fag", "BHC-FAG", REPO / "runs/bundles/BHC-FAG",
              REPO / "data/interim/BHC-FAG/track.json"),
    VideoSpec("bhc_fag_window", "BHC-FAG", REPO / "runs/bundles/score_BHC-FAG_window_cached",
              REPO / "data/interim/score_BHC-FAG_window_cached/track.json"),
    VideoSpec("melsungen_window", "Melsungen-Berlin", REPO / "runs/bundles/Melsungen_window",
              REPO / "runs/perf_speedups/baseline/melsungen.json"),
    VideoSpec("felixclaar", "FelixClaar", REPO / "runs/bundles/score_FelixClaar",
              REPO / "data/interim/score_FelixClaar/track.json"),
    VideoSpec("han_ber4", "Han-Ber4", REPO / "runs/bundles/score_Han-Ber4_cached",
              REPO / "data/interim/score_Han-Ber4_cached/track.json"),
]

# Held-out windows (experiments.teammate_reid.heldout): production `process_video`
# bundles of matches whose teams the prototype never saw. Mined with exactly the
# rules above; scored only by `transfer.evaluate` at the locked thresholds.
def _specs(windows: list) -> list:
    return [VideoSpec(name.lower(), match, REPO / "runs/bundles" / name, REPO / "data/interim" / name / "track.json")
            for name, match, _source, _exposure in windows]


HELDOUT = _specs(WINDOWS)
VIDEO_SETS = {"prototype": VIDEOS, "heldout": HELDOUT, "fresh": _specs(WINDOWS[2:]), "dev": _specs(DEV_WINDOWS),
              "sealed": _specs(SEALED_WINDOWS)}


def cmd_mine(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    fragments, crops, summaries, inputs = [], [], [], {}
    for spec in args.videos:
        if not spec.bundle.is_dir():
            print(f"skipped (no bundle): {spec.name}")
            continue
        loaded = load_video(spec)
        frags, summary = mine_video(loaded)
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        tracks_by_frame = {f: g for f, g in tracks.groupby("frame")}
        numbers_by_frame = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        detections = pd.read_parquet(spec.bundle / "detections.parquet")
        detection_frames = set(detections["frame"].unique().tolist())
        referees = detections[detections["class_name"] == "referee"]
        referees_by_frame = {f: list(g.itertuples()) for f, g in referees.groupby("frame")}
        src = loaded["manifest"]["source"]
        dropped = Counter()
        for f in frags:
            if f.status != "accepted":
                continue
            records, drop = sample_crops(f, tracks_by_frame, numbers_by_frame, detection_frames,
                                         src["width"], src["height"], every=args.every, cap=args.cap,
                                         referees_by_frame=referees_by_frame)
            crops += records
            dropped += drop
        summary["crops"] = sum(r["video"] == spec.name for r in crops)
        summary["crops_dropped"] = dict(dropped)
        summary["crops_with_digit_box"] = sum(bool(r["digit_boxes"]) for r in crops
                                              if r["video"] == spec.name)
        summaries.append(summary)
        fragments += [f.to_json() for f in frags]
        inputs[spec.name] = loaded["inputs"]
        print(json.dumps(summary))
    for i, r in enumerate(crops):
        r["crop_id"] = i
    (args.out / "fragments.json").write_text(json.dumps(fragments, indent=1))
    (args.out / "crops.json").write_text(json.dumps(crops))
    (args.out / "mining_summary.json").write_text(json.dumps(
        {"videos": summaries, "inputs": inputs,
         "params": {"every": args.every, "cap": args.cap}}, indent=2))


class FrameReader:
    """Sequential decode of the frames a video needs, in ascending order."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.pos = 0

    def read(self, index: int) -> np.ndarray:
        if index < self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index
        while self.pos < index:
            self.cap.grab()
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        if not ok:
            raise RuntimeError(f"frame {index} unreadable")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def cmd_extract(args) -> None:
    from experiments.teammate_reid.features import StripeEncoder
    crops = json.loads((args.out / "crops.json").read_text())
    encoder = StripeEncoder(REPO / "prtreid-upstream",
                            REPO / "models/prtreid/prtreid-soccernet-baseline.pth.tar", args.device)
    pixels = args.out / "crops"
    pixels.mkdir(exist_ok=True)
    by_video: dict = {}
    for r in crops:
        by_video.setdefault(r["video"], []).append(r)
    feats = {m: {"global": np.zeros((len(crops), 256), np.float32),
                 "stripes": np.zeros((len(crops), 4, 1920), np.float16)} for m in MODES}
    mask_info = []
    started = time.time()
    for spec in args.videos:
        rows = sorted(by_video.get(spec.name, ()), key=lambda r: r["frame"])
        if not rows:
            continue
        manifest = json.loads((spec.bundle / "manifest.json").read_text())
        reader = FrameReader(spec.bundle / manifest["files"]["video"])
        batch = {m: [] for m in MODES}
        ids = []
        for r in rows:
            frame = reader.read(r["frame"])
            crop, local = crop_with_digits(frame, r["box"], r["digit_boxes"])
            for m in MODES:
                masked, info = apply_mask(crop, local, m, seed_key=str(r["crop_id"]))
                batch[m].append(masked)
                if m == "digits":
                    mask_info.append({"crop_id": r["crop_id"], **info})
            cv2.imwrite(str(pixels / f"{r['crop_id']}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            ids.append(r["crop_id"])
            if len(ids) == 256 or r is rows[-1]:
                for m in MODES:
                    g, s = encoder.encode(batch[m])
                    feats[m]["global"][ids] = g
                    feats[m]["stripes"][ids] = s
                    batch[m] = []
                ids = []
        print(spec.name, len(rows), f"{time.time() - started:.0f}s")
    np.savez(args.out / "features.npz", **{f"{m}_{k}": v for m, d in feats.items() for k, v in d.items()})
    (args.out / "mask_info.json").write_text(json.dumps(mask_info))
    (args.out / "extract_timing.json").write_text(json.dumps(
        {"seconds": time.time() - started, "crops": len(crops), "modes": list(MODES)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--set", dest="video_set", choices=tuple(VIDEO_SETS), default="prototype",
                        help="which videos mine/extract read")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mine")
    p.add_argument("--every", type=int, default=10, help="min frames between sampled crops")
    p.add_argument("--cap", type=int, default=60, help="max crops per fragment")
    p = sub.add_parser("extract")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("sheets")
    p = sub.add_parser("evaluate")
    p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.videos = VIDEO_SETS[args.video_set]
    if args.cmd == "sheets":
        from experiments.teammate_reid.sheets import render_sheets
        render_sheets(args.out)
    elif args.cmd == "evaluate":
        from experiments.teammate_reid.benchmark import run_benchmark
        run_benchmark(args.out, device=args.device)
    else:
        {"mine": cmd_mine, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    main()
