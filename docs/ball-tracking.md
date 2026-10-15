# Ball detection and tracking

The pipeline had no ball. The detector's checkpoint is named
`large-handball-v4-w-numbers-no-ball`, and `pose-feasibility.md` §3 records
passes as unreachable for exactly that reason. This describes what now finds the
ball, what it measures, and what it still gets wrong.

Code: `src/handball_cv/ball/`. Demo: `scripts/render_ball_demo.py`. Verified
labels: `data/annotations/ball/`.

```bash
python -m scripts.render_ball_demo --clip Melsungen_window \
    --detections outputs/team_dataset/.Melsungen_window_detections_v1.npz
```

## Why not a ball class in the detector

Two independent measurements say retraining the person detector is the wrong
lever.

**The labels are worse than sparse.** The sibling repo's export carries 277 ball
boxes over 1033 train images. It was excluded from training as "too sparse", but
sparsity is the lesser problem: a handball is visible in far more than 27% of
frames, so the other 73% of images teach the model *there is no ball here* —
concentrated on exactly the blurred, ambiguous frames an annotator skipped. The
class is trained to suppress what it is meant to find.

**The resolution is thrown away before inference.** Ball boxes in that export
measure 9px wide by 16px tall, which reads as hopeless until you notice the
export stretches 1920x1080 into 640x640. Player boxes there have median
height/width 3.95 against 2.2–3.0 for an undistorted standing person — the exact
1.78 squash. Undoing it, the ball is **~27x27px at native 1080p**. The detector
then runs at 704x704 on a 16:9 frame and is shown roughly 10x18px. A ball class
would inherit both handicaps.

Motion needs neither. It finds the ball with nothing trained and nothing
labelled.

## The pipeline

### Candidates — `candidates.py`

**Three frames, not two.** Differencing `t` against `t-1` leaves residual where
the ball *was* and where it *is*, and the pair is not separable. The per-pixel
minimum of the differences against `t-1` and `t+1` keeps only what differs from
both, which is where the ball is now.

**A play volume, not a play surface.** The residual on this footage is
overwhelmingly crowd and bench; the court is nearly clean. But a handball in
flight leaves the court region in the image entirely — unlike football, where
the ball stays near the ground. So the court mask (HSV: court hue ~109 at value
>150, while boards and crowd sit below 90) is dilated **upward only**, with a
vertical kernel anchored at its top. Three of the eighteen confirmed tracks are
balls airborne against the crowd, so this is load-bearing, not decorative.

**Person boxes** are excluded, referees included — the tracker ignores referees
because it should not follow them, but the question here is whose limbs throw
off ball-sized residuals, and a referee's do.

**Misregistration.** A few pixels of camera-compensation error along a painted
floor marking produces residual the right size and shape to be a ball. The test
is not how it looks but whether it was already there: if the same patch appears
in the warped previous frame within a few pixels, nothing new happened. Where
the ball actually is, the previous frame holds bare floor.

**Broadcast overlays.** See below — this one cost three rounds to get right.

### Trajectories — `tracking.py`

Candidates are chained into constant-velocity paths. A blob is accepted because
of where its neighbours are, never on its own merits.

**Compensation is pairwise.** Mapping every frame into one shared reference
frame by composing homographies works over dozens of frames and fails silently
over 1498: the composed transform drifts until distances stop meaning anything.
Positions and velocities are carried forward one frame at a time instead, and
nothing is ever composed. `believable_motion` additionally refuses a homography
that moves the image corners more than 250px or changes area by more than 2x,
because RANSAC on eight agreeing points in a textureless frame will happily
return one that does.

**The speed gate applies to every step.** Gating only the opening pair lets a
chain start at ball speed and then decay into a cluster of crowd blobs sitting
within tolerance of each other. This was the dominant false-positive mode before
it was fixed, and `test_ball_trajectory.py` guards it directly.

**Nothing is bridged.** A frame with no candidate within tolerance ends the
track.

### Possession — `tracking.py`

A trajectory ending beside a person hands the ball to that person. From there it
is carried, not thrown, and motion cannot see it. Possession is asserted by
association and then has to keep earning itself: after `max_unconfirmed` frames
with no appearance confirmation it lapses to absent rather than claiming a
holder nobody verified.

Occluded, out of frame and never-acquired are all **absent**, which is a state,
not a gap to be interpolated through.

### Appearance — `appearance.py`

How "in hand" is reached without pose. By the time anyone is holding the ball,
the clip has already shown it in flight, and those trajectories were accepted on
physics rather than appearance — so they are a free, unlabelled supply of
examples of this ball, under this lighting, from this camera. That is the same
anonymous per-video calibration `teams/model.py` uses for kit colour.

Scale comes from the players, not from geometry: the size prior is ball radius
over person-box height, which survives a zoom because both terms move together
and needs no homography. Measured over the 389 ball annotations sharing an image
with a person box, that ratio is **0.0822** (p10–p90 0.058–0.113) — well above
the naive 19cm-ball-over-180cm-player figure, because a detection box tightens
around a crouched or running player rather than spanning a standing one. It is
only the fallback; a clip with enough sightings measures its own.

Below `min_samples` the model is not built at all. A model fitted on three
glimpses would still be consulted and still be believed.

## The scoreboard

Compensating the camera turns every burned-in graphic into a perfect projectile.
The score bug is painted at fixed pixels, so as the arena slides underneath it,
it travels across the compensated frame at exactly the pan speed, dead straight,
for as long as the pan lasts. On the first full-clip run **the four longest
trajectories were all glyphs in the score bug**, the longest 37 frames — longer
and straighter than any real ball. Nothing about the trajectory gives it away.

Two things do, and both were needed:

- **It never goes anywhere.** Its centroid *does* wobble a few pixels a frame as
  the residual reshapes, so a per-step threshold cannot separate it from a slow
  ball. Over a trajectory the gap is stark: on frames 500–800 the score-bug
  chains walked 12–78px of path and finished 6–11px from where they started,
  while the ball covered 423px of path and ended 408px away. `min_image_travel`
  is an excursion, not a path length, so fidgeting cannot accumulate past it.
- **It never changes.** During a whip pan the residual smears far enough to
  clear the excursion bar over five frames, which left three short score-bug
  tracks. `overlay_mask` samples frames from across the clip: the arena differs
  in every one because the camera moved, and the score bug is identical in all
  of them, so the overlay is simply the low-variance pixels. It abstains when
  the mask would cover more than a quarter of the frame, since a locked-off
  sample cannot tell an overlay from a wall.

## Measured, on Melsungen_window (1498 frames)

Every proposed trajectory was read on a contact sheet and judged by eye, at two
zoom levels — against a packed crowd, a ball and a forearm are identical at
300px. Verdicts: `data/annotations/ball/Melsungen_window_track_verdicts.json`.

| | |
|---|---|
| tracks proposed | 21 |
| tracks that are the ball | **18 (85.7%)** |
| flight frames on the ball | **161 / 181 (89.0%)** |
| frames with any ball position | 857 / 1498 (57.2%) |
| — in flight | 177 (11.8%) |
| — held, by association | 680 (45.4%) |
| frames abstained | 641 (42.8%), in 10 runs, longest 177 |
| held frames confirmed by appearance | 382 |

The three false positives are two **bare forearms** — skin-coloured, ball-sized,
smoothly moving, and just outside the person box an extended arm sticks out of —
and one **scrolling LED advertising board**, whose text moves at a constant rate
and so looks exactly like constant-velocity motion.

The appearance model is the clearest evidence that track precision matters
downstream. While the score-bug tracks were still being accepted, it calibrated
on scoreboard pixels and confirmed 4 held frames out of 446. Once they were
rejected it calibrated on 181 real ball sightings and confirmed **382**.

**Recall is not measured.** The 42.8% abstention is where the ball is
unaccounted for, and how much of it is a miss rather than a correct abstention
is unknown. Twelve frames sampled across the longest gap (301–477) show a
defensive set-up with the ball held in a cluster and not visible by eye, which
is consistent with correct abstention but is weak evidence. Establishing recall
needs ground truth this repo does not have.

## Not done

Event detection (passes, throws, goals); any trained model; pose or wrist
anchoring; metric coordinates. Nothing in the detection, tracking,
team-classification or identity code was touched — the ball path reads the
detection caches and nothing else.

## Where to go next

1. **A second pass using the appearance model.** It is now calibrated on real
   ball pixels and is not consulted during chaining at all. Re-scoring accepted
   tracks by colour would likely reject both forearms and the LED board, which
   is all three remaining false positives.
2. **Recall.** Precision is measurable by reading what the pipeline proposed;
   recall needs labels for what it did not. Trajectory-assisted labelling is the
   cheap route — a proposed ballistic path drawn over consecutive frames is
   something a person verifies rather than finds.
3. **A learned model, bootstrapped from this.** The verified trajectories are
   exactly the training data a multi-frame heatmap detector needs, and they cost
   nothing to produce. That is the path to the held and occluded frames, which
   motion cannot reach by construction.
