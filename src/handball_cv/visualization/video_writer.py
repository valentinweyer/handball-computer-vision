"""Write frames straight into an H.264 mp4 through an ffmpeg pipe.

`cv2.VideoWriter` with `mp4v` produces MPEG-4 Part 2, which browsers do not
play and which every render then had to transcode -- a second lossy encode on
top of a first, weaker one. Piping raw frames to ffmpeg's x264 encodes once, at
a quality the eye cannot tell from the source (CRF 18), into a file that plays
anywhere.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from handball_cv.video import ffmpeg_exe

__all__ = ["FfmpegWriter"]


class FfmpegWriter:
    """`write(frame_bgr)` per frame, then `close()`; also a context manager."""

    def __init__(self, path: Path, width: int, height: int, fps: float,
                 crf: int = 18, preset: str = "medium") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (int(height), int(width), 3)
        self.frames = 0
        self._process = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
             "-i", "-",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape != self.shape or frame_bgr.dtype != np.uint8:
            raise ValueError(f"expected a {self.shape} uint8 frame, got {frame_bgr.shape} {frame_bgr.dtype}")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError:
            self._fail()
        self.frames += 1

    def close(self) -> Path:
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.wait() != 0:
            self._fail()
        return self.path

    def _fail(self):
        error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
        raise RuntimeError(f"ffmpeg failed writing {self.path}: {error.strip()}")

    def __enter__(self) -> "FfmpegWriter":
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if kind is None:
            self.close()
        else:
            self._process.kill()
            self._process.wait()
