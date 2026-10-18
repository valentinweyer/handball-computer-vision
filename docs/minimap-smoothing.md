# Minimap position smoothing

Minimap rendering applies light temporal smoothing in court coordinates. It is
a display operation: stored court measurements, the court transform, tracking
boxes, masks, teams, identity links and jersey labels are unchanged.

## Method and boundaries

The renderer smooths the complete track table before choosing a render range.
Each uninterrupted track segment uses a symmetric Gaussian window with radius
0.12 s and sigma 0.05 s (seven frames at 25 fps). Positive weights prevent
coordinate overshoot; the centered window preserves constant-velocity motion
without a trailing offset. Corrections are capped at 30 cm.

Separate segments are used across missing/nonconsecutive frames, missing court
positions, canonical identity changes, known re-ID/suspected-ID-switch events,
and displacements faster than 25 m/s. Event-frame positions and segment ends
stay raw. There is no interpolation across gaps or invented player presence.
Very short tracks remain raw. Camera cuts without a recorded discontinuity are
not explicitly detected by this display pass.

This reduces visual jitter; it does not repair an inaccurate homography or
prove that a smoothed coordinate is closer to ground truth. Large mapping
errors remain visible. The court-view outline still uses the original fit.

## Render and compare

Use a separate output to retain an existing video:

```bash
.venv/bin/python -m scripts.render_result \
  runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4 \
  --start 10 --end 30 --output runs/minimap-smoothing-2026-10-18/sample.mp4
```

Add `--raw-minimap` to draw the original coordinates. `--no-minimap` also skips
smoothing. The full table is processed before selecting a clip, so a clip uses
the same positions as the corresponding frames in a full render.

## Validation on the saved Kiel v4 run

139,478 player-frame rows; 128,273 display coordinates change. Median correction
is 3.61 cm, p95 is 15.26 cm, and the maximum is 30 cm. Missing-position masks
are identical. High-frequency displacement is the distance from the midpoint
of the previous and next consecutive positions:

| Quantile | Original | Smoothed |
| --- | ---: | ---: |
| p50 | 5.61 cm | 1.18 cm |
| p90 | 20.69 cm | 4.62 cm |
| p95 | 35.29 cm | 15.17 cm |
| p99 | 184.89 cm | 184.89 cm |

These are steadiness measurements, not position-accuracy scores. The unchanged
p99 is expected because the pass does not interpolate implausible jumps.

The full suite passes 542 tests, including 15 new cases for stationary jitter,
constant velocity, a sharp turn, bounded corrections, segment ends, gaps,
identity breaks, missing positions, row ordering and input immutability.

Actual renderer samples cover 00:10-00:30 and 03:55-04:15. The latter contains
re-entry/identity events. Raw and smoothed encodes each contain 500 frames.
`runs/minimap-smoothing-2026-10-18/minimap-comparison.mp4` combines both ranges
into a 40-second enlarged side-by-side comparison. The original bundle's
manifest, tracks, players, reads and events retain their SHA-256 hashes.

No full-video tracking rerun is required. Original v3/v4 videos and their
saved bundles are retained; this experiment writes separate samples.
