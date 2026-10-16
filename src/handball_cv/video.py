"""Frames decoded straight from a video file, for every stage that reads pixels."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import shutil
import threading

import cv2
import numpy as np

__all__ = ["VideoFrames", "ffmpeg_exe"]


def ffmpeg_exe() -> str:
    """Bare `ffmpeg` if on PATH, else the imageio-ffmpeg bundled static binary.

    This host has no system ffmpeg and no passwordless sudo to install one;
    imageio-ffmpeg ships a working static binary as a plain pip package.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoFrames:
    """RGB frames decoded straight from the video, in place of a JPEG folder.

    The JPEG folder costs ~230 KB a frame on disk (~21 GB for a full match) and
    a second lossy encode between the video and the tracker. Tracking reads
    frames almost strictly in order -- the propagated frame, the repeated
    chunk-boundary frame, the checkpoint frame -- all of them recent, so one
    forward decoder and a short buffer of what it last produced serve it.

    Thread-safe: SAM2's frame prefetch and the consumer's reader both call in.
    An index older than the buffer reopens the video and decodes forward to it,
    which is exact but slow; `backward_seeks` counts those so an access pattern
    that relies on them shows up rather than quietly costing time.
    """

    def __init__(self, video: Path, keep: int = 16, count: int | None = None) -> None:
        self.video = str(video)
        self.keep = keep
        self.backward_seeks = 0
        self._lock = threading.Lock()
        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        capture = self._open()
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count is None:
            # Count by decoding: the container's frame count is only a header
            # claim. A caller that already knows it (a result bundle's manifest)
            # passes `count` and skips a pass that takes minutes on a full match.
            count = 0
            while capture.grab():
                count += 1
        capture.release()
        if count == 0:
            raise ValueError(f"no frames decoded from {self.video}")
        self._count = int(count)
        self._capture = self._open()
        self._next = 0  # index the decoder produces on its next read

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.video)
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {self.video}")
        return capture

    def __len__(self) -> int:
        return self._count

    def rgb(self, idx: int) -> np.ndarray:
        """Frame `idx` as an RGB array. A copy, so a caller may modify it."""
        idx = int(idx)
        if not 0 <= idx < self._count:
            raise IndexError(f"frame {idx} out of range [0, {self._count})")
        with self._lock:
            if idx not in self._buffer:
                if idx < self._next:
                    self.backward_seeks += 1
                    self._capture.release()
                    self._capture = self._open()
                    self._buffer.clear()
                    for _ in range(idx):
                        if not self._capture.grab():
                            raise RuntimeError(f"{self.video} ended before frame {idx}")
                    self._next = idx
                # Frames that would fall out of the buffer before `idx` is reached
                # are only grabbed: decoded by ffmpeg, never converted or stored.
                while self._next < idx - self.keep:
                    if not self._capture.grab():
                        raise RuntimeError(f"{self.video} ended at frame {self._next}")
                    self._next += 1
                while self._next <= idx:
                    ok, frame_bgr = self._capture.read()
                    if not ok:
                        raise RuntimeError(f"{self.video} ended at frame {self._next}")
                    self._buffer[self._next] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self._next += 1
                    while len(self._buffer) > self.keep:
                        self._buffer.popitem(last=False)
            return self._buffer[idx].copy()
