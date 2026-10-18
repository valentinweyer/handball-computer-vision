"""Display-only smoothing of already mapped court positions.

Uses a short, symmetric window within one uninterrupted track segment. Raw
bundle measurements, boxes, identities and labels are never changed. The
positive weights cannot overshoot a local coordinate range; using future
frames avoids the trailing effect of a causal moving average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Light smoothing: at 25 fps this is seven frames, centred on the current one.
RADIUS_SECONDS = 0.12
SIGMA_SECONDS = 0.05
MAX_CORRECTION_CM = 30.0
# An implausible one-frame displacement is a boundary, not a path to interpolate.
MAX_STEP_CM_PER_SECOND = 2500.0
BREAK_EVENTS = ("reid", "suspected_id_switch")


def smooth_minimap_positions(tracks: pd.DataFrame, events: pd.DataFrame | None,
                             fps: float) -> np.ndarray:
    """Return (N, 2) court coordinates in the input row order, without mutation.

    Missing measurements stay missing. Windows never cross a frame gap,
    identity change, known identity-break event, or implausible position jump.
    Event-frame positions and segment ends remain raw rather than borrowing
    evidence from another segment. Filtering the full table before selecting
    a render range makes a short render agree with the same full-video frames.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    original = tracks[["court_x", "court_y"]].to_numpy(dtype=float, copy=True)
    output = original.copy()
    if not len(tracks):
        return output
    frames = tracks.frame.to_numpy(dtype=np.int64)
    people = tracks.player_id.to_numpy()
    radius = max(1, int(round(RADIUS_SECONDS * fps)))
    offsets = np.arange(-radius, radius+1, dtype=float)
    weights = np.exp(-0.5 * (offsets / max(SIGMA_SECONDS * fps, 0.5))**2)
    weights /= weights.sum()
    width = len(weights)

    breaks: dict[int, list[int]] = {}
    if events is not None and not events.empty:
        for row in events[events.type.isin(BREAK_EVENTS)].itertuples(index=False):
            if pd.notna(row.player_id):
                breaks.setdefault(int(row.player_id), []).append(int(row.frame))

    for track_id, indices in tracks.groupby("track_id", sort=False).indices.items():
        order = np.asarray(indices)[np.argsort(frames[indices], kind="stable")]
        f = frames[order]
        xy = original[order]
        cuts = np.array(sorted(set(breaks.get(int(track_id), ()))), dtype=np.int64)
        valid = np.isfinite(xy).all(axis=1) & ~np.isin(f, cuts)
        boundary = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            boundary[1:] = (
                ~valid[:-1] | ~valid[1:] | (np.diff(f) != 1)
                | (people[order[1:]] != people[order[:-1]])
                | (np.searchsorted(cuts, f[1:]) != np.searchsorted(cuts, f[:-1]))
                | (np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps > MAX_STEP_CM_PER_SECOND)
            )
        for segment in np.split(np.arange(len(order)), np.flatnonzero(boundary)[1:]):
            if len(segment) < width or not valid[segment].all():
                continue
            points = xy[segment]
            filtered = np.column_stack([
                np.convolve(points[:, axis], weights, mode="valid") for axis in (0, 1)
            ])
            centre = segment[radius:-radius]
            correction = filtered - xy[centre]
            distance = np.linalg.norm(correction, axis=1)
            scale = np.minimum(1., MAX_CORRECTION_CM / np.maximum(distance, 1e-12))
            output[order[centre]] = xy[centre] + correction * scale[:, None]
    return output
