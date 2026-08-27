"""Extract a subsampled frame window from a source video, plus a matching
cached-detections npz, into the small-clip format the tracking-evaluation
scripts consume (mirrors the existing Han-Ber4_cached.mp4 pattern).

Exists for docs/tracking-evaluation.md §8's third-clip validation: BHC-FAG is
4554 frames at ~50fps, too long and too fast (relative to the 25fps FelixClaar
and Han-Ber4 clips) to use directly. This carves out a fixed frame range at a
fixed subsample stride and writes both a standalone mp4 and a detections npz
sliced from the existing full-video cache, re-indexed to the new frame
numbering.

Usage:
    python -m scripts.extract_clip_window \\
        source/BHC-FAG.mp4 \\
        --detections outputs/team_confidence_v2/.BHC-FAG_detections_v1.npz \\
        --start 3030 --stop 4030 --step 2 \\
        --output-video outputs/team_dataset/BHC-FAG_window_cached.mp4 \\
        --output-detections outputs/team_dataset/.BHC-FAG_window_detections_v1.npz
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import supervision as sv


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_detection_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True, help="exclusive")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-detections", required=True, type=Path)
    args = parser.parse_args()

    selected_frames = list(range(args.start, args.stop, args.step))
    cache = load_detection_cache(args.detections)
    max_valid = len(cache["offsets"]) - 2
    if selected_frames[-1] > max_valid:
        raise ValueError(
            f"frame {selected_frames[-1]} exceeds the detection cache's valid "
            f"range (0..{max_valid})"
        )

    # ── write the subsampled video ───────────────────────────────────────────
    info = sv.VideoInfo.from_video_path(str(args.video))
    out_fps = info.fps / args.step
    out_info = sv.VideoInfo(width=info.width, height=info.height, fps=out_fps,
                             total_frames=len(selected_frames))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    wanted = set(selected_frames)
    frame_by_index: dict[int, np.ndarray] = {}
    for idx, frame in enumerate(sv.get_video_frames_generator(str(args.video))):
        if idx > selected_frames[-1]:
            break
        if idx in wanted:
            frame_by_index[idx] = frame
    if len(frame_by_index) != len(selected_frames):
        missing = wanted - set(frame_by_index)
        raise RuntimeError(f"video ended before frames {sorted(missing)} were read")

    with sv.VideoSink(str(args.output_video), out_info) as sink:
        for idx in selected_frames:
            sink.write_frame(frame_by_index[idx])

    # h264 for portability, matching every other *_cached.mp4 in this repo
    h264_path = args.output_video.with_stem(args.output_video.stem + "_h264")
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(args.output_video),
         "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(h264_path)],
        check=True,
    )
    args.output_video.unlink()
    h264_path.rename(args.output_video)

    # ── slice + re-index the detections cache ────────────────────────────────
    boxes, confidence, class_id, offsets = [], [], [], [0]
    for idx in selected_frames:
        start, end = int(cache["offsets"][idx]), int(cache["offsets"][idx + 1])
        boxes.append(cache["boxes"][start:end])
        confidence.append(cache["confidence"][start:end])
        class_id.append(cache["class_id"][start:end])
        offsets.append(offsets[-1] + (end - start))

    args.output_detections.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_detections,
        schema=cache["schema"],
        detector_id=cache["detector_id"],
        source_size=args.output_video.stat().st_size,
        source_mtime_ns=args.output_video.stat().st_mtime_ns,
        total_frames=len(selected_frames),
        offsets=np.array(offsets, dtype=np.int64),
        boxes=np.concatenate(boxes, axis=0),
        confidence=np.concatenate(confidence, axis=0),
        class_id=np.concatenate(class_id, axis=0),
    )

    print(f"wrote {len(selected_frames)} frames -> {args.output_video} ({out_fps:.2f}fps)")
    print(f"wrote matching detections -> {args.output_detections}")
    print(f"source frames: {selected_frames[0]}..{selected_frames[-1]} step {args.step}")


if __name__ == "__main__":
    main()
