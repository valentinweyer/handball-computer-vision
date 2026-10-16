"""Build a result bundle from an existing tracking run.

Converts what `render_full_pipeline --tracker sam2` already wrote -- the run
JSON, its geometry cache, the detection cache and the team model -- into a
bundle (`handball_cv.pipeline.result`), adding the court: keypoints from the
court model and camera motion, both decoded straight from the video, then the
tracked court fit. Keypoints and motion are cached under `--work`, so a rebuild
after a change to the assembly costs seconds.

    python -m scripts.bundle_from_run runs/perf_speedups/baseline/melsungen.json \\
        --detections outputs/team_dataset/.Melsungen_window_detections_v1.npz \\
        --team-model outputs/team_dataset/.Melsungen_window_team.pkl \\
        --out runs/bundles/Melsungen_window
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from tqdm import tqdm

from handball_cv.court.sequence import fit_court, infer_keypoints, measure_motion
from handball_cv.jersey.identity import OCR_EVERY_N_FRAMES
from handball_cv.pipeline import tables
from handball_cv.pipeline.result import sha256, write_bundle
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.video import VideoFrames
from scripts.render_court_view import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]


def frames_rgb(video: VideoFrames, desc: str):
    for index in tqdm(range(len(video)), desc=desc):
        yield video.rgb(index)


def cached(path: Path, compute):
    if path.is_file():
        print(f"[cache] {path}")
        return json.loads(path.read_text())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="run JSON written by render_full_pipeline --tracker sam2")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--team-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, default=None,
                        help="defaults to the run JSON's geometry_cache")
    parser.add_argument("--court-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--work", type=Path, default=None,
                        help="keypoint and motion caches; defaults to data/interim/<video stem>")
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES,
                        help="the cadence the run read numbers at")
    args = parser.parse_args()

    run = json.loads(args.run.read_text())
    if run.get("tracker") != "sam2":
        raise SystemExit(f"{args.run}: only sam2 runs carry a geometry cache")
    video_path = Path(run["source"])
    geometry_path = args.geometry or Path(run["geometry_cache"])
    work = args.work or ROOT / "data/interim" / video_path.stem
    weights_tag = args.court_weights.parent.name

    video = VideoFrames(video_path)
    geometry = GeometryCache.load(geometry_path)
    print(f"{video_path.name}: {len(video)} frames, {video.width}x{video.height} @ {video.fps:g} fps")

    def run_keypoints():
        from handball_cv.court.rfdetr_keypoints import load_court_keypoints
        return infer_keypoints(load_court_keypoints(args.court_weights),
                               frames_rgb(video, "court keypoints"))

    keypoints = cached(work / f"keypoints_{weights_tag}.json", run_keypoints)
    motion = cached(work / "motion.json", lambda: measure_motion(frames_rgb(video, "camera motion")))
    fits = fit_court(keypoints, motion, (video.width, video.height))

    with np.load(args.detections, allow_pickle=False) as data:
        detection_cache = {k: data[k] for k in data.files}

    tracks = tables.tracks_table(run, geometry, fits, video.fps, args.ocr_every)
    built = {
        "tracks": tracks,
        "players": tables.players_table(run, tracks),
        "court": tables.court_table(fits),
        "detections": tables.detections_table(detection_cache),
        "reads": tables.reads_table(run),
        "events": tables.events_table(run),
        "keypoints": tables.keypoints_table(keypoints),
    }
    info = {
        "source": {"path": str(video_path), "sha256": sha256(video_path), "fps": video.fps,
                   "width": video.width, "height": video.height, "frames": len(video)},
        "coordinates": {"image": "pixels, origin top-left",
                        "court": "centimetres; x along the 4000 cm length, y across the 2000 cm "
                                 "width, origin at a court corner (sports IHF template)"},
        "models": {
            "detector": str(detection_cache.get("detector_id", "")),
            "tracker": run.get("tracker"), "reid_embedding": run.get("reid_embedding"),
            "reader": run.get("reader"), "court_keypoints": str(args.court_weights),
        },
        "parameters": {
            "ocr_every": args.ocr_every,
            "detection_threshold": float(detection_cache["threshold"]) if "threshold" in detection_cache else None,
            "detection_stride": int(detection_cache["stride"]) if "stride" in detection_cache else 1,
        },
        "provenance": {"built_by": "scripts.bundle_from_run", "from_run": str(args.run),
                       "geometry": str(geometry_path), "detections": str(args.detections),
                       "git_commit": git_commit()},
    }
    write_bundle(args.out, built,
                 files={"video": video_path, "masks": geometry_path, "team_model": args.team_model},
                 info=info)
    print(f"bundle -> {args.out}")
    for name, frame in built.items():
        print(f"  {name:<11} {len(frame):>7} rows")


if __name__ == "__main__":
    main()
