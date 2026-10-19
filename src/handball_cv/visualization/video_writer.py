"""Write frames straight into an H.264 mp4 through an ffmpeg pipe.

`cv2.VideoWriter` with `mp4v` produces MPEG-4 Part 2, which browsers do not
play and which every render then had to transcode -- a second lossy encode on
top of a first, weaker one. Piping raw frames to ffmpeg's x264 encodes once, at
a quality the eye cannot tell from the source (CRF 18), into a file that plays
anywhere.

The pipe itself costs ~3 ms per 1080p frame (6 MB through the kernel), so a
thread feeds it while the caller draws the next frame. `write` therefore keeps
the array it is given: do not modify a frame after handing it over.
"""
from __future__ import annotations

from pathlib import Path
import queue
import subprocess
import threading

import numpy as np

from handball_cv.video import ffmpeg_exe

__all__ = ["FfmpegWriter"]

_QUEUE_FRAMES = 8  # ~50 MB at 1080p; enough to ride out the encoder's lumps


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
        self._queue: queue.Queue = queue.Queue(_QUEUE_FRAMES)
        self._broken = False
        self._feeder = threading.Thread(target=self._feed, name="ffmpeg-feed", daemon=True)
        self._feeder.start()

    def _feed(self) -> None:
        while (frame := self._queue.get()) is not None:
            if self._broken:
                continue  # keep draining, so a writer blocked on a full queue wakes up
            try:
                self._process.stdin.write(memoryview(frame).cast("B"))
            except (BrokenPipeError, ValueError):
                self._broken = True

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape != self.shape or frame_bgr.dtype != np.uint8:
            raise ValueError(f"expected a {self.shape} uint8 frame, got {frame_bgr.shape} {frame_bgr.dtype}")
        if self._broken:
            self._fail()
        self._queue.put(np.ascontiguousarray(frame_bgr))
        self.frames += 1

    def close(self) -> Path:
        self._stop_feeder()
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                self._broken = True
        if self._process.wait() != 0 or self._broken:
            self._fail()
        return self.path

    def _stop_feeder(self) -> None:
        if self._feeder.is_alive():
            self._queue.put(None)
            self._feeder.join()

    def _fail(self):
        self._process.wait()
        error = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
        raise RuntimeError(f"ffmpeg failed writing {self.path}: {error.strip()}")

    def __enter__(self) -> "FfmpegWriter":
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if kind is None:
            self.close()
        else:
            # Kill first: a feeder mid-write then fails fast and drains the rest.
            self._process.kill()
            self._stop_feeder()
            self._process.wait()
