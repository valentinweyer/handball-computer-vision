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
  numbers   only with --sheet-numbers: the number boxes found again on sheets
            of each frame's player boxes (handball_cv.detection.number_sheets)
            -- best with --detect-stride 1 --ocr-every 1, which reads every frame
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
from handball_cv.pipeline.models import default_model, model_label, same_model
from handball_cv.pipeline.scan import detection_stride_guard, scan_video
from handball_cv.tracking.sam2_driver import CHECK_EVERY_DEFAULT
from handball_cv.video import VideoFrames, ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "scan", "teams", "numbers", "track", "bundle")
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
        # Kept when the stage is off, so --force numbers still clears a previous one.
        "numbers": [work / "numbers.npz"],
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
        # Written aside and renamed over: a finished bundle hard-links this file,
        # and ffmpeg overwriting it in place would rewrite that bundle's video
        # under tables made from the old one.
        partial = video.with_suffix(".partial.mp4")
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                        "-vf", f"fps={TARGET_FPS}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(partial)], check=True)
        os.replace(partial, video)
    counted = VideoFrames(video)
    return {"source": str(source), "source_bytes": source.stat().st_size,
            "video": str(video), "frames": len(counted),
            "fps": counted.fps, "width": counted.width, "height": counted.height,
            "source_fps": probe.fps}


def keep_hint(flag: str, stored: str) -> str:
    """How to keep a stage's output made by `stored`: its file, or a file of its version."""
    if Path(stored).is_absolute():
        return f"keep them with {flag} {stored}"
    return f"keep them with {flag} pointing at {stored}'s file"


def resume_conflicts(work: Path, source: Path, *, detector: Path, court_weights: Path,
                     track_settings: dict, detect_stride: int = DETECT_STRIDE,
                     sheet_model: Path | None = None) -> list[str]:
    """Why `work`'s finished stages do not belong to this invocation, if they don't.

    A stage counts as done when its files exist, which says nothing about what
    made them: a reused `--work`, or two videos sharing a stem, once bundled one
    video's tracks under the other's name. Each check reads what the stage
    recorded, and a value a stage never recorded is not held against it.
    """
    conflicts = []
    meta_path = work / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (Path(meta["source"]).resolve() != source.resolve()
                or meta.get("source_bytes", source.stat().st_size) != source.stat().st_size):
            conflicts.append(f"{work} was prepared from {meta['source']}, not {source}: "
                             "rerun with --force prepare, or use another --work")
    detections_path = work / "detections.npz"
    if detections_path.is_file():
        with np.load(detections_path, allow_pickle=False) as data:
            stored = str(data["detector_id"]) if "detector_id" in data.files else ""
            stride = int(data["stride"]) if "stride" in data.files else detect_stride
        if stored and not same_model(stored, detector, "player_detector"):
            conflicts.append(f"{work} holds detections from {stored}, not {detector}: "
                             f"rerun with --force scan, or {keep_hint('--detector', stored)}")
        if stride != detect_stride:
            conflicts.append(f"{work} holds detections made every {stride} frames, not every "
                             f"{detect_stride}: rerun with --force scan, or use another --work")
    keypoints_path = work / "keypoints.json"
    if keypoints_path.is_file():
        stored = json.loads(keypoints_path.read_text()).get("model")
        if stored and not same_model(stored, court_weights, "court_keypoints"):
            conflicts.append(f"{work} holds court keypoints from {stored}, not {court_weights}: "
                             f"rerun with --force scan, or {keep_hint('--court-weights', stored)}")
    numbers_path = work / "numbers.npz"
    if sheet_model is not None and numbers_path.is_file():
        with np.load(numbers_path, allow_pickle=False) as data:
            stored = str(data["detector_id"])
        made_by = stored.rpartition(" + sheet numbers ")[2]
        if not same_model(made_by, sheet_model, "number_sheets"):
            conflicts.append(f"{work} holds sheet numbers from {stored}, not {sheet_model}: "
                             "rerun with --force numbers, or use another --work")
    track_path = work / "track.json"
    if track_path.is_file():
        run = json.loads(track_path.read_text())
        # Each setting was always recorded, at its default before the flag existed.
        stored = {"sam2_stride": 1, "small_number_crops": False,
                  "number_detections": str(work / "detections.npz"),
                  **{k: run[k] for k in track_settings if k in run}}
        for key, wanted in track_settings.items():
            if key in stored and stored[key] != wanted:
                flag = "--" + key.replace("_", "-")
                conflicts.append(f"{work} holds a track pass made with {flag} {stored[key]}: "
                                 f"rerun with --force track, or use another --work, for {wanted}")
    return conflicts


def weights(path: Path, role: str) -> Path:
    """`path`, or a stop naming the default version it should hold."""
    if not Path(path).is_file():
        ref = default_model(role)
        raise SystemExit(f"model file not found: {path}. The default is {ref.name} "
                         f"(its {ref.file}), expected at {ref.path}; or pass another file")
    return Path(path)


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="bundle; defaults to runs/bundles/<stem>")
    parser.add_argument("--work", type=Path, default=None, help="defaults to runs/interim/<stem>")
    parser.add_argument("--force", choices=STAGES, default=None,
                        help="rerun this stage and every stage after it")
    parser.add_argument("--detector", type=Path, default=default_model("player_detector").path)
    parser.add_argument("--court-weights", type=Path, default=default_model("court_keypoints").path)
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY_DEFAULT)
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES)
    parser.add_argument("--sam2-stride", type=int, default=1,
                        help="track: propagate SAM2 on every Nth frame (plus checkpoints and "
                             "read frames), interpolating the rest. 2 is ~1.6-1.9x faster "
                             "at unchanged quality")
    parser.add_argument("--small-number-crops", action="store_true",
                        help="track: also read number crops refused for their area alone, under "
                             "a stricter confidence rule; mostly matters at 720p")
    parser.add_argument("--detect-stride", type=int, default=DETECT_STRIDE,
                        help="scan: run the detector (players and number boxes) on every Nth "
                             "frame; --check-every and --ocr-every must be multiples of it")
    parser.add_argument("--sheet-numbers", action="store_true",
                        help="numbers: find number boxes again on sheets of each frame's player "
                             "boxes and read those; best with --detect-stride 1 --ocr-every 1, "
                             "where it identifies clearly more of the visible players")
    parser.add_argument("--sheet-model", type=Path, default=default_model("number_sheets").path,
                        help="numbers: the sheet model's TorchScript export")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = args.video.resolve()
    work = (args.work or ROOT / "runs/interim" / source.stem).resolve()
    out = (args.out or ROOT / "runs/bundles" / source.stem).resolve()
    work.mkdir(parents=True, exist_ok=True)
    detection_stride_guard(args.detect_stride, args.check_every, args.ocr_every)
    outputs = stage_outputs(work)
    if args.force:
        clear_from(args.force, outputs)
    number_cache = work / ("numbers.npz" if args.sheet_numbers else "detections.npz")
    conflicts = resume_conflicts(
        work, source, detector=args.detector, court_weights=args.court_weights,
        detect_stride=args.detect_stride, sheet_model=args.sheet_model if args.sheet_numbers else None,
        track_settings={"sam2_stride": args.sam2_stride, "check_every": args.check_every,
                        "ocr_every": args.ocr_every, "small_number_crops": args.small_number_crops,
                        "number_detections": str(number_cache)},
    )
    if conflicts:
        raise SystemExit("cannot resume:\n  " + "\n  ".join(conflicts))
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
        detector = (TorchScriptDetector(weights(args.detector, "player_detector"), device=args.device)
                    if need_detections else None)
        court_model = (load_court_keypoints(weights(args.court_weights, "court_keypoints"), device=args.device)
                       if need_court else None)
        scanned = scan_video(
            VideoFrames(video_path, count=meta["frames"]), detector=detector, court_model=court_model,
            motion=need_court, detect_stride=args.detect_stride, threshold=DETECT_THRESHOLD,
            detector_id=model_label(args.detector, "player_detector"), desc="scan",
        )
        if need_detections:
            save_detections(work / "detections.npz", scanned["detections"])
        if need_court:
            (work / "keypoints.json").write_text(json.dumps(
                {"model": model_label(args.court_weights, "court_keypoints"), "keypoints": scanned["keypoints"]}))
            (work / "motion.json").write_text(json.dumps(scanned["motion"]))
        del detector, court_model, scanned
        torch.cuda.empty_cache()
        timed("scan", started)

    with np.load(work / "detections.npz", allow_pickle=False) as data:
        detections = {k: data[k] for k in data.files}
    detection_stride_guard(int(detections["stride"]), args.check_every, args.ocr_every)

    # ── teams ──
    if not done("teams"):
        started = time.monotonic()
        from handball_cv.teams.model import TeamModel

        model = TeamModel.fit_from_detection_cache(video_path, detections, device=args.device,
                                                   frame_count=meta["frames"])
        model.save(work / "team.pkl")
        timed("teams", started)

    # ── numbers ──
    if args.sheet_numbers and not done("numbers"):
        started = time.monotonic()
        import torch
        from tqdm import tqdm
        from handball_cv.detection.number_sheets import SheetNumberDetector, replace_number_rows

        frames = VideoFrames(video_path, count=meta["frames"])
        sheets = replace_number_rows(
            detections, frames.rgb, SheetNumberDetector(weights(args.sheet_model, "number_sheets"), device=args.device),
            frames=tqdm(range(meta["frames"]), desc="numbers"),
            detector_id=model_label(args.sheet_model.resolve(), "number_sheets"))
        save_detections(work / "numbers.npz", sheets)
        del sheets
        torch.cuda.empty_cache()
        timed("numbers", started)
    if args.sheet_numbers:
        # The bundle's detections show the number boxes that were read.
        with np.load(number_cache, allow_pickle=False) as data:
            detections = {k: data[k] for k in data.files}

    # ── track ──
    if not done("track"):
        started = time.monotonic()
        # Each stage's GPU models are freed by running the tracker in its own process.
        command = [sys.executable, "-m", "handball_cv.pipeline.track", str(video_path),
                   "--detections", str(work / "detections.npz"),
                   "--number-detections", str(number_cache),
                   "--team-model", str(work / "team.pkl"), "--run", str(work / "track.json"),
                   "--device", args.device,
                   "--court-in-view", str(save_court_in_view(work, meta)),
                   "--check-every", str(args.check_every), "--ocr-every", str(args.ocr_every),
                   "--sam2-stride", str(args.sam2_stride)]
        if args.small_number_crops:
            command.append("--small-number-crops")
        env = {**os.environ, "SAM2_UPSTREAM_DIR": os.environ.get("SAM2_UPSTREAM_DIR", str(ROOT / "sam2-upstream"))}
        with (work / "track.log").open("w") as log_file:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise SystemExit(f"tracking failed; see {work / 'track.log'}")
        timed("track", started)

    # ── bundle ──
    started = time.monotonic()
    from handball_cv.pipeline.bundle import build_bundle

    run = json.loads((work / "track.json").read_text())
    court = json.loads((work / "keypoints.json").read_text())
    build_bundle(
        run, VideoFrames(video_path, count=meta["frames"]), video_path, Path(run["geometry_cache"]),
        detections, work / "team.pkl", court["keypoints"], json.loads((work / "motion.json").read_text()),
        out, ocr_every=args.ocr_every, court_weights=Path(court["model"]),
        provenance={"built_by": "scripts.process_video", "source": str(source), "work": str(work),
                    "timings_s": timings},
        parameters={"check_every": args.check_every, "source_fps": meta["source_fps"],
                    "sheet_numbers": args.sheet_numbers},
    )
    timed("bundle", started)
    log(f"done: {out}  (render: python -m scripts.render_result {out})")


if __name__ == "__main__":
    main()
