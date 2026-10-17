"""The result bundle: everything extracted from one video, in one directory.

A bundle is what processing a video produces and what everything downstream
reads -- a render, a highlight clip, a notebook. It is plain files:

    manifest.json        what is here, how it was made, and each table's version
    <table>.parquet      one per entry in manifest["tables"]
    masks.npz            per player-frame masks (`GeometryCache` format)
    team_model.pkl       the per-video team model
    video.mp4            the video the rows refer to, frame for frame

Each table carries its own schema version, so a later extraction stage (ball,
player actions) adds a table by adding a manifest entry, and a bundle written
before that stage existed still loads. `load` refuses only a table whose
version it does not know -- a silently misread column is worse than an error.
The manifest is written last, so an interrupted write leaves no bundle that
looks complete.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

__all__ = ["BUNDLE_SCHEMA", "TABLES", "Result", "load", "write_bundle", "sha256"]

BUNDLE_SCHEMA = 1

# name -> (schema version, description). Bump a version whenever a column's
# meaning or type changes, not only when columns are removed.
TABLES = {
    "tracks": (1, "One row per tracked player per frame: identity, team, jersey number, "
                  "image box and foot point, court position."),
    "players": (1, "One row per player identity after number-based linking."),
    "court": (1, "One row per frame: the court homography, where it came from, and whether "
                 "a court was in view at all."),
    "detections": (1, "Raw detector output, every class, unfiltered above the cache threshold."),
    "reads": (1, "One row per jersey-number read that reached the vote."),
    "events": (1, "Identity events: re-identification, linking, team switches, retirement."),
    "keypoints": (1, "Court-keypoint model output, all 37 slots per frame, unfiltered."),
}

# Copied into the bundle under these names.
FILES = {"video": "video.mp4", "masks": "masks.npz", "team_model": "team_model.pkl"}


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _place(source: Path, target: Path) -> None:
    """Hard link when the filesystem allows it -- a full match's video is GBs -- else copy."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_bundle(out_dir: Path, tables: dict, *, files: dict, info: dict) -> Path:
    """Write `tables` (name -> DataFrame) and `files` (key in FILES -> path) to `out_dir`.

    `info` goes into the manifest as-is: source video, models, parameters,
    provenance. Tables not listed in `TABLES` are refused here; a new table
    gets its version and description there first.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)  # incomplete until rewritten below

    listed = {}
    for name, frame in tables.items():
        if name not in TABLES:
            raise ValueError(f"unknown table {name!r}; add it to TABLES with a version first")
        version, description = TABLES[name]
        file_name = f"{name}.parquet"
        frame.to_parquet(out_dir / file_name, index=False)
        listed[name] = {"file": file_name, "version": version, "rows": len(frame),
                        "columns": list(frame.columns), "description": description}

    placed = {}
    for key, source in files.items():
        if key not in FILES:
            raise ValueError(f"unknown bundle file {key!r}; expected one of {sorted(FILES)}")
        _place(Path(source), out_dir / FILES[key])
        placed[key] = FILES[key]

    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **info,
        "tables": listed,
        "files": placed,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=_json_default))
    tmp.replace(manifest_path)
    return out_dir


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


class Result:
    """A loaded bundle. Tables load on first access and are pandas DataFrames.

        result = load("runs/bundles/Melsungen_window")
        result.tracks                 # or result.table("tracks")
        result.frame(120)             # RGB frame from the bundle's video
        result.mask(row.mask_row)     # one player-frame's mask
    """

    def __init__(self, path: Path, manifest: dict) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._tables: dict[str, pd.DataFrame] = {}
        self._video = None
        self._masks = None

    @property
    def tables(self) -> list[str]:
        return list(self.manifest["tables"])

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.manifest["tables"]:
            raise KeyError(f"no table {name!r} in {self.path}; it has {self.tables}")
        if name not in self._tables:
            self._tables[name] = pd.read_parquet(self.path / self.manifest["tables"][name]["file"])
        return self._tables[name]

    def __getattr__(self, name: str):
        manifest = self.__dict__.get("manifest")
        if manifest is not None and name in manifest["tables"]:
            return self.table(name)
        raise AttributeError(name)

    @property
    def fps(self) -> float:
        return float(self.manifest["source"]["fps"])

    def frame(self, index: int) -> np.ndarray:
        """RGB frame `index` of the bundle's video."""
        if self._video is None:
            from handball_cv.video import VideoFrames
            self._video = VideoFrames(self.path / self.manifest["files"]["video"],
                                      count=self.manifest["source"]["frames"])
        return self._video.rgb(index)

    def mask(self, mask_row: int) -> np.ndarray:
        """Full-size boolean mask for one `tracks` row, by its `mask_row`."""
        if self._masks is None:
            from handball_cv.tracking.geometry_cache import GeometryCache
            self._masks = GeometryCache.load(self.path / self.manifest["files"]["masks"])
        return self._masks.row_mask(int(mask_row))


def load(path: Path) -> Result:
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json in {path}: not a bundle, or an interrupted write")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bundle_schema") != BUNDLE_SCHEMA:
        raise ValueError(f"{path}: bundle schema {manifest.get('bundle_schema')}, "
                         f"this code reads {BUNDLE_SCHEMA}")
    for name, entry in manifest["tables"].items():
        known = TABLES.get(name)
        if known is not None and entry["version"] != known[0]:
            raise ValueError(f"{path}: table {name!r} is version {entry['version']}, "
                             f"this code reads version {known[0]}")
    return Result(path, manifest)
