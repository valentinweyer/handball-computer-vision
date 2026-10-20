# Court in view: confirming the carried court (2026-10-20)

Branch `fix/court-carried-view`. Only the court fit has been re-run so far.
Nothing has been re-tracked.

## Problem

`process_video` pauses tracking when `CourtTracker` decides there is no court on
screen. That decision is made per frame: a frame counts as showing the court
when at least half of its confident keypoints agree on a single court. The court
is then declared out of view when fewer than half of the last 25 frames agreed.

The rule was built for Kiel's lights-off timeout, where the keypoint model
finds points on silhouettes. It also fires on two lower-league broadcasts,
Saarlouis–Münster and Würzburg–Regensburg, where the court is clearly visible.
In mid-court views the model puts the centre-line points (vertices 13–18) on a
sponsor circle or a floor advert. That leaves the correct goal-area points in
the minority, so the frame does not agree, and 13% of frames in both windows
were declared courtless. The model actually finds *more* confident keypoints in
those frames than in fitted ones (14–17 against 12).

The free-RANSAC inlier count cannot separate the two cases: Kiel's timeout also
reaches about 6 chance inliers from 17 hallucinated points. What does separate
them is whether the keypoints sit where the court from the previous frames,
carried forward by the measured camera motion, puts them:

| no-court frames | keypoints within 20 px of the carried court |
| --- | --- |
| Kiel, lights-off timeout | 0 (max 3); 0% of frames reach 4 |
| Saarlouis | median 5–6; 81% of frames reach 4 |
| Würzburg | median 3; 34% of frames reach 4 |

## Change

`CourtTracker(min_confirmed=4)` adds a fallback for frames whose keypoints
mostly disagree. If at least 4 of them lie within `ransac_px` of where the
carried estimate puts them, the frame counts as showing the court and is
fitted from those keypoints, plus the usual weak prior. Passing
`min_confirmed=None` restores the old behaviour.

## Effect: court fit re-run on cached keypoints and motion

Measured with `scripts/compare_court_view.py`. The old rule reproduces every
bundle's `court.parquet` exactly.

| clip | no court, old | no court, new | longest no-court stretch, old → new |
| --- | ---: | ---: | --- |
| Saarlouis–Münster | 13.2% | 0.3% | 8.8 s → 1.2 s |
| Würzburg–Regensburg | 13.3% | 7.5% | 19.6 s → 9.0 s |
| Eisenach–Hamburg | 11.7% | 7.9% | 8.3 s → 8.3 s |
| Ludwigshafen–Wetzlar | 4.7% | 0.2% | 7.9 s → 1.2 s |
| RNL–SGF | 6.3% | 2.3% | 14.0 s → 14.0 s |
| Bielefeld–Aurich | 2.6% | 0.3% | 1.8 s → 1.6 s |
| VfL Gummersbach–Löwen | 4.2% | 2.6% | 4.1 s → 4.1 s |
| Pforzheim–Bietigheim | 69.8% | 64.1% | court model fails in this hall |
| Altenholz–Lübeck | 94.7% | 93.0% | court model fails in this hall |
| **Kiel–Lemgo (lights-off timeout)** | 9.1% | **9.1%** | 25.4 s → 25.4 s |
| BER–BHC, FAG–LEI, TBV–SCM | 8.3%, 12.0%, 5.2% | unchanged | — |
| BAL–THW, Balingen II, BHC–FAG, FelixClaar, Han-Ber4 | ≤ 0.5% | ≤ 0.3% | — |

On no clip does the share of out-of-view frames go up. The real breaks are
unchanged: Kiel's timeout, and the stretches in BER–BHC, FAG–LEI and TBV–SCM.

## Fit quality on rescued frames

Overlays are in `runs/smart-reid-shadow/court_check/` (`*_NEW_*.mp4`).

- **Saarlouis:** the rescued fits sit on the real 6 m and 9 m lines and the
  centre line.
- **Würzburg:** the rescued fits are approximate. The arcs follow the real
  lines, but the fitted centre line is tilted against the real one.
- **Würzburg, remaining 7.5%:** these frames have too few keypoints that agree
  with the carried court. They need the keypoint model itself to improve, for
  example a fine-tune on these halls with labels transferred from their
  well-fitted frames.

## Not yet done

- **Tracking re-run:** tracking has not been re-run under the new rule. The
  measured tracking accuracy (`docs/tracking-evaluation.md` §8) is unaffected:
  those clips are at 0% no-court under both rules.
- **Position accuracy:** court positions in rescued frames inherit the fit
  quality described above. Nothing measures position accuracy against ground
  truth here.
