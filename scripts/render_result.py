"""Render a result bundle: team masks and jersey numbers on the video, plus a court minimap.

Reads only the bundle (`handball_cv.pipeline.result`) -- no tracker, no reader,
no model -- so it is seconds per hundred frames and can be re-run for any range,
style or label rule without touching the extraction.

    python -m scripts.render_result runs/bundles/Melsungen_window
    python -m scripts.render_result runs/bundles/Melsungen_window --start 20 --end 35 \\
        --output runs/bundles/Melsungen_window/highlight.mp4

Labels are each player's backfilled number (`tracks.number`, as `--redraw`
draws them), `P<id>` where none is known; `--live-numbers` shows what the run
knew at each frame instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from handball_cv.court.homography import CourtFit
from handball_cv.pipeline.result import load
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.video import prefetch
from handball_cv.visualization.court_minimap import CourtMinimap
from handball_cv.visualization.labels import LabelDrawer
from handball_cv.visualization.minimap_positions import smooth_minimap_positions
from handball_cv.visualization.palette import PALETTE_HEX, color_index_for
from handball_cv.visualization.overlay import (
    DEFAULT_FONT, annotate_frame, draw_legend, load_roster, number_label,
)
from handball_cv.visualization.video_writer import FfmpegWriter


def court_fits(court) -> list:
    """`CourtFit` per frame back from the bundle's court table."""
    fits = []
    h_cols = [f"h{i}" for i in range(9)]
    inv_cols = [f"inv{i}" for i in range(9)]
    for row in court.itertuples(index=False):
        h = np.array([getattr(row, c) for c in h_cols], dtype=np.float64)
        if not np.all(np.isfinite(h)):
            # Bundles from before the column existed: no reason to think it was out of view.
            fits.append(CourtFit(source=row.source, age=int(row.age),
                                 court_in_view=bool(getattr(row, "court_in_view", True))))
            continue
        inv = np.array([getattr(row, c) for c in inv_cols], dtype=np.float64).reshape(3, 3)
        fits.append(CourtFit(homography=h.reshape(3, 3), inverse=inv, inliers=int(row.inliers),
                             source=row.source, age=int(row.age)))
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="defaults to <bundle>/render.mp4")
    parser.add_argument("--start", type=float, default=None, help="seconds")
    parser.add_argument("--end", type=float, default=None, help="seconds, exclusive")
    parser.add_argument("--no-minimap", action="store_true")
    parser.add_argument("--raw-minimap", action="store_true",
                        help="draw original court positions without display smoothing")
    parser.add_argument("--minimap-corner", default="top-right",
                        choices=("top-right", "top-left", "bottom-right", "bottom-left"),
                        help="move it off whatever the broadcast already shows there")
    parser.add_argument("--live-numbers", action="store_true",
                        help="label with what the run knew at each frame, not the backfilled verdict")
    parser.add_argument("--roster", type=Path, default=None, help="as render_full_pipeline --roster")
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    args = parser.parse_args()

    result = load(args.bundle)
    source = result.manifest["source"]
    width, height, fps, total = int(source["width"]), int(source["height"]), float(source["fps"]), int(source["frames"])
    first = 0 if args.start is None else max(0, int(round(args.start * fps)))
    last = total if args.end is None else min(total, int(round(args.end * fps)))
    if first >= last:
        raise SystemExit(f"empty range: frames {first}..{last}")
    output = args.output or args.bundle / "render.mp4"

    tracks = result.tracks
    positions = tracks[["court_x", "court_y"]].to_numpy(dtype=float)
    if not args.no_minimap and not args.raw_minimap and "court" in result.tables:
        positions = smooth_minimap_positions(
            tracks, result.events if "events" in result.tables else None, fps,
        )
    # Derived display values only: leave the loaded measurements and video
    # annotations untouched, and smooth before selecting a clip range.
    tracks = tracks.assign(minimap_x=positions[:, 0], minimap_y=positions[:, 1])
    tracks = tracks[(tracks.frame >= first) & (tracks.frame < last)]
    by_frame = {int(f): rows for f, rows in tracks.groupby("frame", sort=True)}
    masks = GeometryCache.load(args.bundle / result.manifest["files"]["masks"])
    fits = court_fits(result.court) if "court" in result.tables else []
    number_column = "number_live" if args.live_numbers else "number"
    roster = load_roster(args.roster)

    palette = sv.ColorPalette.from_hex(PALETTE_HEX)
    label_drawer = LabelDrawer(args.font, palette, width)
    minimap = None if args.no_minimap or not fits else CourtMinimap(
        round(width * 0.27), [palette.by_idx(i).as_bgr() for i in range(len(PALETTE_HEX))])

    with FfmpegWriter(output, width, height, fps) as writer:
        decoded = prefetch(cv2.cvtColor(result.frame(frame), cv2.COLOR_RGB2BGR)
                           for frame in range(first, last))
        for frame, frame_bgr in tqdm(zip(range(first, last), decoded), total=last - first,
                                     desc=f"render {args.bundle.name}"):
            rows = by_frame.get(frame)
            players = []
            if rows is not None:
                colors = [color_index_for(t, s, g) for t, s, g in
                          zip(rows.team, rows.team_state, rows.is_goalkeeper)]
                numbers = [n if isinstance(n, str) else None for n in rows[number_column]]
                labels = [number_label(n, c, roster, int(p))
                          for n, c, p in zip(numbers, colors, rows.player_id)]
                crops = [masks.row_crop(row) for row in rows.mask_row]
                frame_bgr = annotate_frame(
                    frame_bgr, rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
                    colors, crops, labels, palette, label_drawer,
                )
                players = list(zip(rows.minimap_x, rows.minimap_y, colors, numbers))
            draw_legend(frame_bgr, palette)
            if minimap is not None:
                fit = fits[frame] if frame < len(fits) else None
                minimap.composite(frame_bgr, minimap.draw(fit, (width, height), players),
                                  corner=args.minimap_corner)
            writer.write(frame_bgr)
    print(f"wrote {output} ({last - first} frames)")


if __name__ == "__main__":
    main()
