"""Join a tracking run, its detections and the court fit into a result bundle."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from handball_cv.court.sequence import fit_court
from handball_cv.pipeline import tables
from handball_cv.pipeline.result import sha256, write_bundle
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.video import VideoFrames

__all__ = ["build_bundle"]

ROOT = Path(__file__).resolve().parents[3]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_bundle(run: dict, video: VideoFrames, video_path: Path, geometry_path: Path,
                 detection_cache: dict, team_model_path: Path, keypoints: list, motion: list,
                 out: Path, *, ocr_every: int, court_weights: Path, provenance: dict,
                 parameters: dict | None = None) -> Path:
    """Join a tracking run, its detections and the court into a bundle at `out`."""
    run = tables.with_backed_links(run)
    geometry = GeometryCache.load(geometry_path)
    fits = fit_court(keypoints, motion, (video.width, video.height))
    team_confidence = None
    if run.get("extras") and Path(run["extras"]).is_file():
        with np.load(run["extras"]) as extras:
            team_confidence = extras["team_confidence"]
    tracks = tables.tracks_table(run, geometry, fits, video.fps, ocr_every, team_confidence)
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
            "reader": run.get("reader"), "court_keypoints": str(court_weights),
        },
        "parameters": {
            "ocr_every": ocr_every,
            "frame_source": run.get("frame_source", "jpeg"),
            "detection_threshold": float(detection_cache["threshold"]) if "threshold" in detection_cache else None,
            "detection_stride": int(detection_cache["stride"]) if "stride" in detection_cache else 1,
            **(parameters or {}),
        },
        "provenance": {**provenance, "git_commit": git_commit()},
    }
    write_bundle(out, built,
                 files={"video": video_path, "masks": geometry_path, "team_model": team_model_path},
                 info=info)
    print(f"bundle -> {out}")
    for name, frame in built.items():
        print(f"  {name:<11} {len(frame):>7} rows")
    return Path(out)
