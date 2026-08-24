"""On-disk per-frame mask cache shared by the SAM2 and McByte pipelines.

Both pipelines produce (K, H, W) bool masks during pass 1 but only render in
pass 3, and holding them in RAM does not survive a full match -- at 1080p a
single frame of 12 dense bool masks is ~25 MB. Each frame is instead flattened
to one uint8 label map (0 = background, column + 1 = that player's column) and
compressed. Mostly-zero maps land at ~12 KB/frame.

The label map caps at 255 tracked columns per run, which is far above the
MAX_LIVE_OBJECTS ceilings either pipeline uses.
"""
from pathlib import Path

import numpy as np


class MaskCache:
    """Per-frame mask storage keyed by the caller's player/track column index."""

    def __init__(self, directory: Path, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Columns are renumbered on every run, so a stale map from a previous
        # run would paint the right silhouette in the wrong player's colour.
        for stale in self.directory.glob("*.npz"):
            stale.unlink()

    @classmethod
    def read_only(cls, directory: Path) -> "MaskCache":
        """Open an existing cache for reading only.

        Bypasses __init__'s mkdir + stale-file cleanup, which exists so a
        fresh pipeline run doesn't paint a new player's silhouette in an old
        run's column colour -- exactly the cache a read-only consumer (e.g.
        team_labels.py, reading a reference run's masks after the fact) is
        trying to read, so that cleanup must not run here.
        """
        self = cls.__new__(cls)
        self.directory = Path(directory)
        self.enabled = True
        return self

    def _path(self, frame_idx: int) -> Path:
        return self.directory / f"{frame_idx:05d}.npz"

    def save(self, frame_idx: int, masks: np.ndarray, cols) -> None:
        """Store `masks` (K, H, W) bool, where `cols[i]` is row i's column index.

        Rows whose column is None are dropped -- a track can hold a mask before
        the caller has assigned it a column.
        """
        if not self.enabled or masks is None or len(masks) == 0:
            return
        label = np.zeros(masks.shape[1:], dtype=np.uint8)
        for row, col in enumerate(cols):
            if col is None:
                continue
            label[masks[row]] = int(col) + 1
        np.savez_compressed(self._path(frame_idx), label=label)

    def load(self, frame_idx: int, cols: np.ndarray, shape: tuple):
        """Rebuild an (N, H, W) bool stack for `cols`, or None if unavailable."""
        if not self.enabled or len(cols) == 0:
            return None
        path = self._path(frame_idx)
        if not path.exists():
            return None
        with np.load(path) as data:
            label = data["label"]
        if label.shape != tuple(shape):
            return None
        return np.stack([label == (int(c) + 1) for c in cols])
