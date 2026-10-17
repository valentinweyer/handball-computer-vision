"""Extract everything from a handball video into a result bundle, in this venv alone.

    python -m scripts.process_video data/raw/<match>.mp4
    python -m scripts.render_result runs/bundles/<match>

Nothing has to exist beforehand. Stages run in order, each writing its output
to `--work`; a stage whose output exists is skipped, so an interrupted run
resumes where it stopped, and `--force STAGE` reruns that stage and every one
after it.

  prepare   resample to 25 fps if the source is faster (everything here was
            tuned and scored at ~25 fps), count frames
  scan      one decode for the per-frame models: RF-DETR detections on every
            5th frame (the only frames tracking and number reading look at),
            court keypoints and camera motion on every frame
  teams     per-video two-team model, from crops spread over the whole video
  track     SAM2 + periodic reprompting, re-ID, jersey numbers -- reading the
            video directly, writing no frames and no video. Paused wherever
            the court fit sees no court (a timeout huddle), restarted after
  bundle    join it all into Parquet tables under `--out`

No frames are written to disk at any stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from handball_cv.jersey.identity import OCR_EVERY_N_FRAMES
from handball_cv.pipeline.scan import detection_stride_guard, scan_video
from handball_cv.tracking.sam2_driver import CHECK_EVERY_DEFAULT
from handball_cv.video import VideoFrames, ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "scan", "teams", "track", "bundle")
TARGET_FPS = 25
MAX_NATIVE_FPS = 30
DETECT_STRIDE = 5
DETECT_THRESHOLD = 0.3


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def stage_outputs(work: Path) -> dict:
    return {
        "prepare": [work / "meta.json"],
        "scan": [work / "detections.npz", work / "keypoints.json", work / "motion.json"],
        "teams": [work / "team.pkl"],
        # The reads cache is listed so a forced track pass reads again instead of
        # replaying the previous pass's reads.
        "track": [work / "track.json", work / "track_reads.json", work / "track_geometry.npz",
                  work / "track_extras.npz", work / "court_in_view.npy"],
        "bundle": [],  # always rebuilt: seconds, and the inputs may have changed
    }


def clear_from(stage: str, outputs: dict) -> None:
    """Delete `stage`'s outputs and every later stage's: they were built on them."""
    for later in STAGES[STAGES.index(stage):]:
        for path in outputs[later]:
            path.unlink(missing_ok=True)


def prepare(source: Path, work: Path) -> dict:
    probe = VideoFrames(source, count=1)
    video = source
    if probe.fps > MAX_NATIVE_FPS:
        video = work / "video.mp4"
        log(f"resampling {probe.fps:g} -> {TARGET_FPS} fps into {video}")
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                        "-vf", f"fps={TARGET_FPS}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(video)], check=True)
    counted = VideoFrames(video)
    return {"source": str(source), "video": str(video), "frames": len(counted),
            "fps": counted.fps, "width": counted.width, "height": counted.height,
            "source_fps": probe.fps}


def save_detections(path: Path, cache: dict) -> None:
    np.savez_compressed(path, **cache)


def save_court_in_view(work: Path, meta: dict) -> Path:
    """Per frame, whether the court fit sees a court at all -- what pauses tracking."""
    from handball_cv.court.sequence import fit_court

    keypoints = json.loads((work / "keypoints.json").read_text())["keypoints"]
    motion = json.loads((work / "motion.json").read_text())
    fits = fit_court(keypoints, motion, (meta["width"], meta["height"]))
    path = work / "court_in_view.npy"
    np.save(path, np.array([fit.court_in_view for fit in fits], dtype=bool))
    return path


def main() -> None:
    from scripts.export_detector import DEFAULT_OUTPUT as DETECTOR_WEIGHTS
    from scripts.render_court_view import DEFAULT_WEIGHTS as COURT_WEIGHTS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="bundle; defaults to runs/bundles/<stem>")
    parser.add_argument("--work", type=Path, default=None, help="defaults to data/interim/<stem>")
    parser.add_argument("--force", choices=STAGES, default=None,
                        help="rerun this stage and every stage after it")
    parser.add_argument("--detector", type=Path, default=DETECTOR_WEIGHTS)
    parser.add_argument("--court-weights", type=Path, default=COURT_WEIGHTS)
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY_DEFAULT)
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = args.video.resolve()
    work = (args.work or ROOT / "data/interim" / source.stem).resolve()
    out = (args.out or ROOT / "runs/bundles" / source.stem).resolve()
    work.mkdir(parents=True, exist_ok=True)
    detection_stride_guard(DETECT_STRIDE, args.check_every, args.ocr_every)
    outputs = stage_outputs(work)
    if args.force:
        clear_from(args.force, outputs)
    timings_path = work / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else {}

    def done(stage: str) -> bool:
        return bool(outputs[stage]) and all(p.is_file() for p in outputs[stage])

    def timed(stage: str, started: float) -> None:
        timings[stage] = round(time.monotonic() - started, 1)
        timings_path.write_text(json.dumps(timings, indent=2))
        log(f"{stage} done in {timings[stage]:.0f} s")

    # ── prepare ──
    if not done("prepare"):
        started = time.monotonic()
        meta = prepare(source, work)
        (work / "meta.json").write_text(json.dumps(meta, indent=2))
        timed("prepare", started)
    meta = json.loads((work / "meta.json").read_text())
    video_path = Path(meta["video"])
    log(f"{video_path.name}: {meta['frames']} frames, {meta['width']}x{meta['height']} @ {meta['fps']:g} fps")

    # ── scan ──
    if not done("scan"):
        started = time.monotonic()
        import torch
        from handball_cv.court.rfdetr_keypoints import load_court_keypoints
        from handball_cv.detection.rfdetr_torchscript import TorchScriptDetector

        need_detections = not (work / "detections.npz").is_file()
        need_court = not ((work / "keypoints.json").is_file() and (work / "motion.json").is_file())
        detector = TorchScriptDetector(args.detector, device=args.device) if need_detections else None
        court_model = load_court_keypoints(args.court_weights, device=args.device) if need_court else None
        scanned = scan_video(
            VideoFrames(video_path, count=meta["frames"]), detector=detector, court_model=court_model,
            motion=need_court, detect_stride=DETECT_STRIDE, threshold=DETECT_THRESHOLD,
            detector_id=str(args.detector), desc="scan",
        )
        if need_detections:
            save_detections(work / "detections.npz", scanned["detections"])
        if need_court:
            (work / "keypoints.json").write_text(json.dumps(
                {"model": str(args.court_weights), "keypoints": scanned["keypoints"]}))
            (work / "motion.json").write_text(json.dumps(scanned["motion"]))
        del detector, court_model, scanned
        torch.cuda.empty_cache()
        timed("scan", started)

    with np.load(work / "detections.npz", allow_pickle=False) as data:
        detections = {k: data[k] for k in data.files}
    if int(detections["stride"]) != DETECT_STRIDE:
        detection_stride_guard(int(detections["stride"]), args.check_every, args.ocr_every)

    # ── teams ──
    if not done("teams"):
        started = time.monotonic()
        from handball_cv.teams.model import TeamModel

        model = TeamModel.fit_from_detection_cache(video_path, detections, device=args.device,
                                                   frame_count=meta["frames"])
        model.save(work / "team.pkl")
        timed("teams", started)

    # ── track ──
    if not done("track"):
        started = time.monotonic()
        # Each stage's GPU models are freed by running the tracker in its own process,
        # with the same entry point and flags a SAM2 render uses.
        command = [sys.executable, "-m", "scripts.render_full_pipeline", str(video_path),
                   "--tracker", "sam2", "--detections", str(work / "detections.npz"),
                   "--number-detections", str(work / "detections.npz"),
                   "--team-model", str(work / "team.pkl"), "--output", str(work / "track.mp4"),
                   "--frame-source", "video", "--no-video", "--device", args.device,
                   "--court-in-view", str(save_court_in_view(work, meta)),
                   "--check-every", str(args.check_every), "--ocr-every", str(args.ocr_every)]
        env = {**os.environ, "SAM2_UPSTREAM_DIR": os.environ.get("SAM2_UPSTREAM_DIR", str(ROOT / "sam2-upstream"))}
        with (work / "track.log").open("w") as log_file:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise SystemExit(f"tracking failed; see {work / 'track.log'}")
        timed("track", started)

    # ── bundle ──
    started = time.monotonic()
    from scripts.bundle_from_run import build_bundle

    run = json.loads((work / "track.json").read_text())
    court = json.loads((work / "keypoints.json").read_text())
    build_bundle(
        run, VideoFrames(video_path, count=meta["frames"]), video_path, Path(run["geometry_cache"]),
        detections, work / "team.pkl", court["keypoints"], json.loads((work / "motion.json").read_text()),
        out, ocr_every=args.ocr_every, court_weights=Path(court["model"]),
        provenance={"built_by": "scripts.process_video", "source": str(source), "work": str(work),
                    "timings_s": timings},
        parameters={"check_every": args.check_every, "source_fps": meta["source_fps"]},
    )
    timed("bundle", started)
    log(f"done: {out}  (render: python -m scripts.render_result {out})")


if __name__ == "__main__":
    main()
