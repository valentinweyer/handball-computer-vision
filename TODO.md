# Tasks

Open work first, each entry with the evidence that motivates it and a concrete
first step; finished work is under **Done** at the bottom, kept because the
measurement in it is usually the reason the fix looks the way it does. Fuller
detail lives in `docs/team-classification-handoff.md`.

---

## 1. Chunked SAM2 propagation, without resetting identity at the seam

**This is the hard cap on clip length, and it is a memory cap, not a compute
one.** `SAM2VideoPredictor.init_state` allocates the whole video up front:

```python
images = torch.zeros(num_frames, 3, 1024, 1024, dtype=torch.float32)
```

12.58 MB per frame at model resolution, before a single frame is tracked:

```
 1,500 frames   60s @25fps    17.6 GiB   proven (the Melsungen clip)
 3,000 frames    2min         35.2 GiB   comfortable
 9,000 frames    6min        105.5 GiB   at the limit
15,000 frames   10min        175.8 GiB   exceeds the machine
```

The machine has 121 GiB. `offload_video_to_cpu=True` is the usual escape and
**does not help on GB10**, whose memory is unified -- CPU and GPU draw on the
same pool. A second term grows on top: `output_dict_per_obj[obj]
["non_cond_frame_outputs"][frame_idx]` accumulates for every propagated frame
per object, and `clear_non_cond_mem_around_input` defaults to False, so nothing
is pruned as propagation advances.

Compute, by contrast, is flat: ~0.46 s/frame at any length, because the
attention memory bank is bounded (`num_maskmem=7`). Frame 15,000 costs what
frame 100 costs. It was ~1.07 s/frame until the CPU mask work halved it (see
`docs/sam2-speed-research.md`), which only widens the gap: compute fell by half
and the memory cap did not move at all.

### What "seamless" has to mean

Naively restarting per chunk would retire every track at each boundary and let
re-ID re-acquire them -- and re-ID is 0.55 rank-1 within a team, so a boundary
every 3,000 frames would inject a burst of exactly the identity errors the rest
of this work removes. The seam must not go through re-ID at all.

Carry across the boundary:

- **`PlayerRegistry` whole.** It is already tracker-agnostic: identities, team
  votes, embeddings and the retired gallery survive a predictor swap untouched.
- **`NumberVoter` whole.** Keyed by `player_id`, so it needs nothing.
- **The `obj_id -> player_id` mapping.** This is the actual seam. Re-seed the
  new chunk with the *final masks/boxes of the live tracks*, not with fresh
  detections, and assign each new `obj_id` the `player_id` it already had. No
  retire, no revive, no re-ID call, no `suspend()`.
- **Overlap a few frames** so the new state has real content to prompt from and
  SAM2's memory bank refills before the first output frame is kept.

What is genuinely lost is SAM2's own memory bank (7 frames) -- unavoidable, and
small against a 3,000-frame chunk. Expect a brief quality dip in the overlap,
which is why the overlap frames should be discarded from the output rather than
written.

### Verification that would actually prove it

Render one clip that fits in memory both ways -- unchunked, and chunked with a
boundary deliberately placed mid-possession -- and require the run summaries to
match: same `players`, same `numbers_resolved`, and no `reid`/`link`/
`suspected_id_switch` event within the overlap window. Anything else means the
seam is leaking.

---

## 2. The court test is a no-op

`sam2_driver.drive_sam2` passes `court_test_fn=lambda box: True`. A colour-mask
approach was tried and abandoned -- the bench sits inside the mask, so it
excluded 8% of detections and none of the bench. The identified route is
homography from the keypoints in `scripts/run_court_mapping.py`.

Lower value than it first appeared: most of what looked like crowd-tracking was
number boxes being tracked as people (fixed in `6dc4ff1`), and only ~8% of
person detections have feet off the floor.

**Revised upward 2026-09-11 by the recall pass** (item 6). Of 23 identities on
the Melsungen clip, p7 is a goalkeeper-plus-staff track that never holds a
numbered shirt, and p10 starts on a player and ends on a spectator in the
stands. Both are exactly what a court test excludes, both consume one of the 20
`MAX_LIVE_OBJECTS` slots for the whole clip, and neither can ever produce a
number. The 8% figure counts detections; what matters here is that a single
persistent off-court track costs a slot and a whole identity.

**The homography it depends on was fitted to a scrambled correspondence, and
that is now fixed** (see Done, 2026-09-11). Two blockers remain before the court
test itself can be built, and they are the reason this item is still open:

- ~~**The keypoint model does not run locally.**~~ **It does** -- `get_model`
  downloads the weights and runs them locally through onnxruntime; the API key
  is for that download and Roboflow's usage tracking, not for inference. What
  actually failed was version-specific: **version 4** is served as
  `rfdetr-keypoint-preview`, a type the pinned `inference` 0.62.0 has no
  implementation class for, so `get_model` raises `KeyError` after the metadata
  fetch succeeds. **Version 3 loads and runs locally**, and Roboflow reports it
  as the better model besides -- mAP 99.5 / precision 99.96 / recall 100.0
  against 97.0 / 98.6 / 97.9 for version 4, which was trained from scratch
  rather than fine-tuned. `run_court_mapping.py` now pins version 3. No
  retraining and no `inference` upgrade is needed, so the pinned aarch64/GB10
  torch install does not have to be touched.
- **The estimator cannot tell a good fit from a bad one.** `ViewTransformer`
  calls `cv2.findHomography(source, target)` with no method argument, so it is
  plain least squares over every point -- no RANSAC, and the inlier mask is
  discarded. There is no residual check, no degeneracy guard, and the minimum of
  4 points leaves zero redundancy to measure. A court test is an *exclusion*
  test, so a silently-wrong transform removes real players; it needs a reported
  residual and the ability to abstain, per the constraint that abstaining beats
  a confident wrong observation. Note also that `make_court_test` in
  `experiments/sam2_baseline/run_pipeline.py` fails *closed* when the homography
  is unavailable (rejects every new detection) -- the opposite of abstaining,
  and already flagged in `experiments/team_gated_tracking/run_pipeline.py:199`.

Camera motion is settled and matters for the design: the broadcast camera pans
across the court (compare frames 0 and 700 of the Melsungen clip -- the goal
moves from the right edge to the left), so a single per-clip homography is not
an option and per-frame estimates need temporal coherence.

### Measured over the whole Melsungen clip (2026-09-11)

With the corrected mapping and version 3, solved per frame, no smoothing
(`runs/court_mapping/`, keypoints cached in `Melsungen_keypoints.json` so
re-scoring needs no second inference pass):

```
frames                 1500      solved 1486    abstained 14 (no homography)
keypoints / frame      median 12   min 0   max 18      under 6: 20 frames
RANSAC inliers         median  8   min 4   p10  8      at the 4-point minimum: 9
reprojection           median 5.07 px   p90 6.67 px    max 10.54 px
player-frames          17626 inside the rectangle, 1084 outside (5.8%)
```

**A court test built on this today would be unsafe, and the residual will not
tell you when.** Group the solved frames by how many players they exclude:

```
                 n     median inliers   median reprojection
0 excluded      902          9.0              5.42 px
1-2 excluded    488          8.0              4.50 px
3+ excluded      96          7.0              3.61 px
```

Residual moves the *wrong way*: the frames doing the most damage score best on
it. Fewer inliers means fewer constraints, so the fit hugs the surviving points
more tightly while the transform degrades -- overfitting, read as quality. Frame
1337 is the worst case and looks unremarkable by every scalar: 12 keypoints,
6 inliers, 6.8 px. It reports **on court 0/12** -- every player rejected, with
the whole court collapsed into the right edge of the image -- while the footage
plainly shows twelve players around the centre circle.

So the gate cannot be reprojection error. But chasing a better gate turns out to
be the wrong response, because the frames that fail are not a separate
population -- they are the same estimator, run where extrapolation stops being
benign.

### The estimator is under-constrained on essentially every frame

Measured on the inliers, in court centimetres:

```
inlier coverage of the 4000 x 2000 cm court
  x-span   median  900 cm    p90  900    max 2180
  y-span   median 1150 cm    p90 1150    max 2000
  frames whose inliers span under 1/4 of the court length:  91%
```

**No frame in the clip has inliers spanning more than 55% of the court**, and
nine in ten span under a quarter of it. Every homography here is extrapolated
roughly fourfold beyond its evidence. Frames that look right are not
well-conditioned; they are benignly extrapolated. Note the medians are identical
for frames that exclude nobody and frames that exclude more than half their
players (900 cm x-span either way) -- coverage alone does not separate them,
which is why a spread threshold cannot be the gate either.

The same shortage shows up as the jitter, measuring how far the projected court
moves between consecutive solved frames:

```
  p50   32 px      p90  361 px      p99 1374 px      max 4845 px
  moves more than 50 px between adjacent frames: 40.7% of frame pairs
```

A broadcast camera does not move like that. Almost all of it is estimation
noise from re-solving 8 degrees of freedom, from scratch, every frame, against
evidence covering a fifth of the court.

**Both symptoms are one cause.** With the goals in view the extrapolation is
benign, so the overlay looks right and merely jitters. With only the centre
circle and centre line in view the evidence collapses toward the centre line --
five of those landmarks are exactly collinear at x=2000, and the two circle
extremes sit 180 cm off it -- so the fit goes rank-deficient and the projected
court collapses to a line. Frame 1310 is that case: 7 keypoints, 5 inliers, the
whole court rendered as a single green stroke, 1 of 12 players "on court", at a
reprojection error of **1.0 px** -- among the best in the clip, because five
near-collinear points are trivial to fit perfectly.

### Temporal propagation, built and measured (2026-09-11)

`handball_cv.court.homography` carries each frame's estimate forward and
regularises the next fit toward it, and `handball_cv.court.camera_motion`
measures the motion it is carried through. Replayed over the cached keypoints:

```
motion source   prior   jitter p50   p90    >50px   off court   frames rejecting >50%
none (today)     --        23.1     342.4   33.3%      5.0%            17
keypoints       0.05        8.2      37.4    7.9%      8.2%            20
keypoints       0.30        6.8      30.1    3.7%     15.4%            19
optical flow    0.15        6.8      25.4    4.1%      4.4%             6
optical flow    0.30        6.4      23.6    3.3%      5.8%             4
```

**Where the motion estimate comes from decides whether the prior helps or
drags.** Read off the court landmarks, a stronger prior trades jitter for
exclusions -- the classic lag signature -- because a similarity fitted to
collinear centre-line points is a poor model of a panning perspective camera and
the error accumulates. Measured from the pictures instead, a stronger prior
improves every axis at once: at weight 0.15 the jitter p90 falls 13x while
off-court player-frames drop *below* the per-frame baseline and catastrophic
frames go 17 -> 6.

Two earlier numbers need correcting. The per-frame baseline reads 5.0% off-court
here against the 5.8% reported from the render, because the render fitted the
forward and inverse homographies independently -- they are not inverses of each
other, so it was projecting the court onto the image with one transform and
players onto the court with another. Everything above inverts a single fit.

### The middle of the court fails by mis-labelling, not by missing landmarks

The stretches that still fell apart (frames ~460-560 and ~1300-1360) are the
views holding the centre circle with no goal. The obvious reading is that the
keypoint model needs more training there. It is the wrong one, in an instructive
way:

```
                         detections / frame   mutually consistent
whole clip                       12                 8   (67%)
frames 1300-1360                 13                 7   (50%)
frames  460-560                  15                 8   (50%)
```

Those frames carry **more** detections than average, and half of them contradict
the other half. Lowering the confidence threshold makes it worse, not better --
at 0.25 the frames rejecting over half their players go from 6 to 25 -- so the
sub-gate detections are noise rather than recall waiting to be recovered. Read
one out and the problem is plain: in frame 1330 the model reports landmarks at
`(4000,850)` and `(4000,1150)` *left* of the centre circle while `(3400,*)` and
`(3100,*)` fall to its right. The ordering is inverted; they cannot all be true.

The cause is that a handball court is symmetric, and with no goal in frame one
goal area's arc is the other's. A single frame genuinely cannot say which end it
is looking at, so a per-frame detector has to guess. Retraining can sharpen the
guess; it cannot remove the ambiguity, because the information is not in the
frame.

What is dangerous about it is that the wrong answer is *coherent*: mis-labelling
by the court's own mirror yields a set that agrees perfectly with itself, so
RANSAC has two self-consistent stories and no reason to prefer the true one.
Once the mirrored reading holds the majority it simply wins. (Random label noise
RANSAC removes unaided -- verified, and the unit test corrupts by mirror rather
than by shuffle for exactly this reason.)

The previous frame does know which end was in view, so `identity_gate_px`
rejects a landmark sitting further than 400 px from where the carried estimate
places it, before it can vote. Loose on purpose: it is there to catch a landmark
on the wrong half of a 40 m court, not to second-guess localisation.

```
identity gate   jitter p90   off court   frames rejecting >50%   460-560   1300-1360
off                 25.4        4.4%              6                 0          4
400px               25.0        3.6%              0                 0          0
```

**Zero frames now reject more than half their players**, against 17 for the
per-frame estimator, and off-court player-frames are the lowest measured. The
loosest gate tested is also the best, which is the reassuring direction: it
earns its keep by refusing only egregious identity errors.

Known limit: a *fully* mirrored frame defeats it, since with nothing left to
keep, the gate falls back to the ungated set rather than starve the solve.
Abstaining would be the answer there, and that path is still unexercised.

**Still not good enough to switch the court test on.** The degeneracy is gone --
frame 1310 no longer collapses to a line, and its far sideline now lands on the
real one -- but the overlay still carries visible error there, and nothing in
this has been checked against ground truth. Off-court count is a proxy, not a
label: an exclusion may be right (the bench, the crowd) or wrong, and these
numbers cannot tell them apart. Scoring that needs frames where somebody has
said which people were on the court.

Also still open: the `prior_weight` was picked off this one 60-second clip, and
the abstain path (`max_age`, the `CourtFit.source` states) is written but has
only unit tests behind it, never a clip where the camera genuinely loses the
court for a long stretch.

### Why propagation rather than a better gate

Temporal propagation is not polish for the jitter; it is the only way a
centre-only frame can be solved at all. The court is planar and the camera pans,
tilts and zooms about a fixed centre, so consecutive frames are themselves
related by a homography -- chaining is principled rather than a smoothing hack,
and it lets a frame that sees only the centre circle inherit scale and
orientation from frames that saw a goal. Refine the carried estimate with the
current keypoints instead of re-solving from them.

Abstention stays as a backstop, but it cannot be the primary mechanism and must
not key on residual. Gates scored over the 1486 solved frames, against the 36
that reject more than half their players:

```
                             keeps (player-frames)   bad frames admitted
no gate                            100.0%                 36/36
reproj < 5px                        48.6%                 30/36   <- harmful
inliers >= 8                        93.4%                 19/36   <- best simple
inliers>=8 and aniso>=0.20          93.4%                 19/36
x-span >= 2000cm                     3.0%                  0/36   <- rejects all
```

`reproj < 5px` throws away half the clip and still admits five in six bad
frames. The only gate that catches them all keeps 3% of the data, because
almost no frame is well-conditioned on its own -- which is the finding above,
restated.

Artifacts: `runs/court_mapping/Melsungen_court_overlay_h264.mp4` (the overlay
render), `Melsungen_keypoints.json` (per-frame keypoints, so re-scoring needs no
inference pass) and `Melsungen_court_perframe.json`.

---

## 3. Bench occupancy as an identity constraint (blocked on the court test)

A player on the bench cannot be on the pitch at the same instant. Once
court/bench is distinguishable this is a *hard* mutual-exclusion constraint, and
a stronger version of the simultaneity guard the linker already uses. Deliberately
deferred until the court test exists.

---

## 4. Goalkeepers are a third kit forced into a two-cluster model

Found while disproving the Kiel-Lemgo entry (now under Done). The fit correctly
excludes goalkeepers (`exclude_class_ids=(1,)`), but the pipeline's
`person_detections` returns classes 1 and 2, so at predict time keepers *are*
scored and *do* cast team votes -- against two clusters neither of which is their
kit.

Measured over the three 10-minute windows, ~35 keeper crops each:

```
                     split     mean conf   below gate (0.30)
Eisenach-Hamburg     5 / 28       0.273        21/33   64%
Melsungen-Berlin    17 / 21       0.442         8/38   21%
Kiel-Lemgo          28 /  7       0.555         4/35   11%
```

Eisenach's keeper wears **yellow** -- visible in `runs/team_grid/
Eisenach_Hamburg/goalkeepers.png`, and neither maroon nor navy. The gate is
doing its job on the majority of those crops, which is the designed behaviour
(abstaining beats injecting a confident wrong observation). The open question is
the **36% that clear 0.30 anyway** and vote with an essentially arbitrary label.

Cheap options, in order: exclude class 1 from team voting entirely (keepers do
not need a team label for any current consumer); or raise the gate for class 1
only; or fit a third cluster and treat it as "neither". Do not reach for the
third without measuring the first -- two keepers per match is a small
denominator, and the fit deliberately never saw them.

Note the constraint any replacement inherits: raw team classification stays
frame-local, and a tracker must never supply or freeze a team label.

---

## 5. Detector threshold 0.3 -> 0.5

Detector precision measured monotonic with confidence (50% -> 100%) on the 1080p
set, recommending the raise. Documented, **never verified end to end** -- it
would change what the tracker and the reader see on every clip.

---

## 6. Missing ground truth

- ~~**The 60s Melsungen clip has none.**~~ **Precision is now measured: 10/10**
  (2026-09-11). Every number the run resolves was read off the footage and all
  ten are right -- p1 25, p3 19, p11 24, p12 93, p14 10, p16 25, p18 53, p20 15,
  p22 18, p23 6. Sheets, readings and per-call notes in
  `runs/number_truth/Melsungen/`, regenerate with
  `scripts.render_number_verification_sheets`.

  Two things it settles. **p20 is really 15**, so folding did not manufacture
  that number -- its 11 bare `5` reads are partial views of a real 15, which is
  what the fold rule assumes. And the **duplicate 25 is legitimate**: p1 and p16
  are on opposite teams, legal in handball, so arbitration was right to leave
  both alone.

  **Recall was measured next, and the reader is not the bottleneck.** 13 of 23
  identities resolved no number. Only **3 of those 13 had any number read at
  all**, so ten never reached the voter -- the gates are not what is losing them.
  Four were examined frame by frame (`--all-identities --span`):

  - **p7 is not a player.** A goalkeeper in a tracksuit early, sideline staff in
    dark tracksuits later. There is no jersey number to read.
  - **p10 starts as a player** (`11` legible at f243) **and drifts into the
    crowd** around f848 -- later frames are a spectator in a red jacket with a
    high-vis steward beside them.
  - **p2 is the real 18** (f318) **until its re-ID at frame 860 moves the track
    onto 11** (f1023, f1182). `suspend` withheld the inherited 18 and no read
    ever re-vouched it, so the pipeline correctly said nothing. Not a reader
    miss -- an identity error that the number layer handled correctly.
  - **p8 is a genuine player tracked cleanly for all 1499 frames with zero
    reads**, apparently because its back is rarely to the camera. This is the
    only one of the four that is a true visibility limit.

  Of the three with reads, p5 (5 reads) and p6 (20, split `29`/`2`) are thin or
  contested and abstaining is right. So the recall losses divide into tracks on
  people who *have* no number (court test, item 2), tracks that changed person
  (re-ID), and genuine invisibility -- not reader accuracy and not the voter's
  gates. **That raises item 2 above its recorded "lower value than it first
  appeared".**

  Caveat: 4 of 13 were examined individually; the rest are characterised only by
  read counts and lifetimes. Note also that `11` shows up on both p2-late and
  p10-early, which is consistent with one fragmented player rather than two.

  **The pass was careful, not blind.** The sheets print no verdict, but whoever
  reads them in one sitting has usually seen `numbers_resolved` already. That
  makes confirming an existing claim easier than discovering an unexpected
  number, and it is why 10/10 should be read as "no claim is visibly wrong"
  rather than as an independent replication.
- **Backfill raised the cost of being wrong, which raises the value of this.**
  A verdict is now displayed over its whole segment rather than from the frame
  it committed, so a confidently-wrong verdict that never revises is wrong for
  the whole span instead of part of it -- 41.4% of player-frames carried a
  number before, 60.8% after. The guard only catches segments that *changed*
  their mind (p20, 284 frames withheld); a segment that is steadily wrong looks
  exactly like one that is steadily right. Nothing in the pipeline can tell
  them apart without labels. p20's own case is the live example: the fold-rule
  entry under Done measured digit loss at 1.3% of two-digit reads, nothing like
  p20's 11-bare-`5`-against-6-raw-`15` split, so whether folding manufactured
  that number is still unanswerable on this clip. This is an argument for
  scoring the clip, not against backfill.
- **Real Bundesliga rosters.** Two roster simulations were run on invented squad
  lists and both were retracted as worthless. The roster question -- constrain
  reads to numbers that exist in the squad -- cannot be answered honestly without
  them.

---

## 7. Hardcoded absolute paths in tracked files

`/home/valentinweyer/...` remains in three scripts: `build_jersey_audit_set.py`,
`cache_number_detections.py`, `evaluate_checkpoint_against_labels.py`. Those
cannot run on another machine. `.env.example` already defines the override
pattern (`HANDBALL_CV_VIDEO`, `SAM2_UPSTREAM_DIR`); these should use it.

The five notebook copies that also carried absolute paths went with the deletion
with the legacy notebook copies (see Done), and `data/annotations/team/Han-Ber4.json` was made repo-relative in
`c616040`. The remaining tracked annotations under `data/annotations/` still
carry absolute paths in their provenance fields.

---

## 8. Housekeeping

- ~~`pytest -q tests` fails collection on duplicated basenames~~ -- fixed in
  `a0a5a9e` by deleting the superseded root copies. The documented command now
  collects and passes 292 tests in one run.
- `runs/reid_analysis/` is gitignored, so the re-ID discriminability reports live
  on disk only. Regenerate with `scripts/measure_reid_discriminability.py`.

---

# Done

## The court homography was fitted to a scrambled correspondence (2026-09-11)

Every homography this repository has ever built paired each detected court
landmark with an unrelated court position. `run_court_mapping.py` indexed
`config.vertices` with the keypoint model's own slot number:

```python
landmark_indices = np.array([int(kp.class_name) - 1 for kp in confident_kps])
court_landmarks  = np.array(config.vertices)[landmark_indices]
```

The model numbers its 37 landmarks in the order its Roboflow export happened to
use; the `sports` template numbers its 37 vertices in a different one. Nobody
had written down the translation, so the code assumed there wasn't one.

**A homography still comes back.** `cv2.findHomography` cannot detect a
contradictory correspondence -- it returns the least-bad solution to a set of
false claims, with no error and no warning. That is why this survived: the
output looks like a transform and behaves like one, and only a measurement shows
it is meaningless.

Measured over the 892 labelled images of the export, before and after:

```
                                   BEFORE      AFTER
  homography fit residual          922.1 cm     32.9 cm
  leave-one-out error             1701.5 cm     47.8 cm
  images under 50 cm                 0.0 %      88.2 %
  flip_idx agreement (of 37)           4          35
```

A handball court is planar and a broadcast camera is near enough a pinhole, so a
correct correspondence has to fit to within annotation noise -- and ~33 cm on a
40 m court is that noise (the export is 640x640, so one click pixel is already
~6 cm near the centre line). 9.2 m is not a tuning problem; it is only possible
if the pairs are wrong.

**The `flip_idx` column is the independent check.** That is the export's own
left/right mirror table, which the recovery never consults. Applying the
recovered permutation takes it from agreeing on 4 slots to 35. The two
exceptions are slots 19 and 20, which sit on the centre line at the goalpost
offsets: mirroring the court left to right maps each to itself, and the export
swaps them instead -- a vertical flip, and a quirk of the annotation rather than
evidence against the mapping. Recorded as `SELF_MIRRORING_SLOTS` so the next
person meets it as a documented exception.

**How it was recovered.** Seeded only from the centre circle -- the five-point
cross whose centre, left/right and near/far arms are identifiable by eye -- with
all eight orientations of that seed tried so a wrong guess could not bias the
result. From each seed, alternate projecting the labelled points onto the court
and re-solving a one-to-one (Hungarian) assignment until it stops moving. All
eight seeds converged on the same answer, so the basin is global. Slots absent
from the anchor image were recovered by repeating the fit across the export and
pooling votes; the result is a full 37/37 bijection.

**What was built.** `KEYPOINT_TO_VERTEX` in
`src/handball_cv/court/keypoints.py` (a package that existed but was empty),
`scripts/recover_court_keypoint_mapping.py` to re-derive and re-score it from
scratch (`--compare` diffs against the shipped constant), and
`tests/unit/test_court_keypoints.py`. The tests layer three guards: structure
(bijection), the `flip_idx` table (needs nothing on disk, since the export is
gitignored), and the homography residual when the export is present. Each has a
companion asserting the naive identity mapping *fails* it, so a guard that
passes for any mapping cannot go unnoticed.

**The defect had a second half, found by running the model.** A prediction
carries both `class_id` and `class_name`, and they are different numberings:
`class_id` is the slot, `class_name` is the label the annotator typed. Matched
against the labelled export, `class_id` is the slot on 28 of 31 landmarks (the
exceptions are single-observation nearest-neighbour confusions between adjacent
centre-circle points); `int(class_name) - 1` is the slot on 6 of 31. So
`run_court_mapping.py` read the wrong field *and* skipped the translation, and
fixing either alone still leaves a wrong homography. It now reads `class_id`.

**Confirmed end to end on real footage.** Projecting the court template back
onto Melsungen frame 700 with version 3: the old path found **4 RANSAC inliers
of 11** confident keypoints -- exactly the minimum a homography needs, so
nothing agreed -- and drew sidelines crossing diagonally through the middle of
the court. The corrected path finds **8**, and the reprojected 6 m and 9 m lines
land on the painted arcs, with the far sideline and goal line on theirs.

Residual error is concentrated where the keypoints are not: with 11 confident
landmarks all in the left half of that frame, the near sideline and the far
right are extrapolation. That is the conditioning problem the estimator work in
item 2 has to answer, and the reason it needs a residual and an abstain path
rather than better tuning.

**Not fixed:** `experiments/sam2_baseline/run_pipeline.py:167` and
`experiments/team_gated_tracking/run_pipeline.py:180` carry the same wrong
indexing. Both are recorded baselines, so they were left as they ran rather than
retroactively corrected -- but any number either produced from court
coordinates is void.

---

## The fold rule's premise holds, but only at the confidence gate (2026-09-11)

Kept unchanged. Measured on the 323 labelled 1080p crops with the shipping
reader, via `scripts/audit_fold_rule.py`:

```
2-digit jersey read as a single digit      3 of 237
  trailing digit                           3     <- what folding assumes
  leading digit                            0
1-digit jersey read as two digits          1 of 53
  ending in the true digit                 1     <- folding would capture it ('2' -> '32')
```

**The entry's own premise was a weak-reader artefact.** It said the labelled set
holds 5 leading-digit-only reads; with baudm there are none above the gate. So
one-sided folding is not the arbitrary choice it looked like.

**But "the reverse cannot happen" is false as stated, and the gate is what makes
it true.** Ungated there are 6 partial reads, one of them a leading digit; every
leading-digit case sits below `PARSEQ_MIN_CONFIDENCE = 0.5`:

```
gate   reads   2d->1d   trailing   leading
0.00     301        6          4         1
0.25     298        5          4         1
0.50     290        3          3         0     <- what ships
0.90     251        1          1         0
```

That is a coupling nothing recorded: **lowering the reader's confidence gate
would begin admitting the case the fold rule assumes away.** Written into
`_Votes.resolved_counts` so the next person to touch either sees it.

Worth knowing how little the rule now does. Digit loss was severe on docTR --
the FelixClaar `{17: 4, 7: 4}` tie in the docstring is that reader -- and is 1.3%
of two-digit reads on this one. It is kept for being conservative and correct,
not for being load-bearing.

**p20 was not resolved by this, and has since been resolved by footage.** The
1.3% partial-read rate could not explain an 11-against-6 split in either
direction, and the eval crops are a readability-gated 1080p population that does
not transfer to in-game crops anyway. Reading the shirt settled it on
2026-09-11: **p20 really is 15**, so the bare `5` reads are partial views of a
real number and folding manufactured nothing. See item 6.

---

## Crop padding: the shipped value is right, and the old sweep measured the wrong reader (2026-09-11)

`NUMBER_CROP_PAD = 0` stays. Padding degrades the reader the pipeline actually
ships, monotonically, with no useful region between the two extremes:

```
pad (h,v)     docTR 0.622     baudm 0.858 (ships)
(0.00, 0.00)      0.625            0.870   <- best
(0.10, 0.00)      0.635            0.854
(0.10, 0.10)      0.647   <- best  0.839
(0.20, 0.20)      0.619            0.817
(0.50, 0.00)      0.511            0.697
```

**The optimum did not transfer.** On docTR's parseq a 10% symmetric pad was
worth +2.2 points, which is what made this entry look promising. On baudm's
parseq every pad is worse than none, and 0.10/0.10 costs 3.1 points. The
stronger reader already handles a tight crop; padding only adds distractors it
has to ignore. Measured on the same 323 readable crops, 0.870 accuracy / 0.969
selective at pad 0 -- marginally above the 0.858/0.958 in the handoff because
the sweep re-extracts crops from the video rather than reading the stored crop
files.

The digit-loss hypothesis that motivated horizontal-only pads is **not settled
by this**, and cannot be from these numbers: padding changes a crop's aspect
ratio, so crops move between the aspect bands and the per-band populations are
not the same crops at different pads (the narrowest band goes n=27 at pad 0 to
n=10 at 0.10 horizontal). The bands cannot be compared across pads. Testing it
properly would mean holding the band assignment fixed at pad 0.

`scripts/sweep_number_crop_padding.py` gained `--reader {parseq,doctr}`; it
previously hardcoded docTR's `recognition_predictor`, which is why the first
sweep could only measure the wrong one. Results in
`runs/number_eval_1080p/crop_padding_sweep_baudm.json`.

---

## Retroactive number backfill (2026-09-11, `dfd616a`)

Shipped. A number is now displayed from the start of the segment that earned
it, not from the frame its third read lands.

**Measured on a fresh run of the 60s Melsungen clip
(`runs/full_pipeline/Melsungen_geo.json`):**

```
player-frames live                   18867
  labelled by the render              7810   41.4%
  backfill, stable segments only      3667  +19.4pp
  withheld, verdict changed later      284
  -> coverage after backfill                 60.8%
```

**The old entry's baseline was wrong and is corrected here.** It claimed 7045
labelled (37%) and +3273 (+17%). That 7045 came from a replay that started a
fresh `NumberVoter` at each identity break, which is *not* what the render
does: `NumberVoter.suspend` keeps the accumulated votes across a re-ID and
re-asserts on the first agreeing read, so a returning player is relabelled
immediately rather than after another `min_votes` reads. The render actually
labels 7810 (41.4%). Building the first version against the replay would have
*removed* 485 frames of labels the render was already showing -- p1 and p12 for
475 frames each, p16 for 10 -- a regression dressed as caution. The end state
(54.6%) barely moved; the gain is smaller because the baseline was understated.

**Correctness constraint, unchanged and still the whole design.** Backfill must
never cross an identity break, because re-ID moves an identity onto a different
person -- p6 is 15 until frame 560 and somebody else after. The unit is the
segment: the span between a `reid` or `suspected_id_switch`. Segments bound how
far back a label may reach; they no longer partition the evidence.

**Guard, measured:** 1 of 17 resolved segments changed its verdict mid-way (p20
goes `6` at frame 885 to `15` at 1245). Those frames keep what they were shown.
Costs 284 frames.

**Two bugs found by watching the render, after the first version passed every
test.** p1 backfilled nothing across its second segment. Both are fixed and
both have regression tests; together they cost 1178 player-frames.

- The carry-in that stops a `suspected_id_switch` losing a label the render was
  still drawing was anchored on the *break* frame rather than the segment's
  first *live* frame. p1 breaks at 540 and is off screen from 401, returning at
  541 -- so the pre-break verdict was planted at 540, back-dating `resolved_at`
  to the segment start and erasing the backfill. It also carried a verdict
  across a re-ID, which is the one thing segmentation exists to prevent. p16
  escaped only by being on screen at its own break frame.
- `causal_labels` walked only frames where somebody was live, so a revival at
  an unoccupied frame never applied its `suspend`. Invisible on a full clip,
  where somebody always is.

**What was built.**

- `causal_labels` / `segment_verdicts` / `Segment` in
  `src/handball_cv/jersey/identity.py`. `causal_labels` replays the run's own
  loop -- suspend, observe, arbitrate every `--ocr-every` -- so the labelled
  region is the render's by construction.
- `src/handball_cv/tracking/geometry_cache.py`: per-frame boxes, ids, palette
  index, team, and per-player masks cropped to their bounding box and
  bit-packed. **7.1 MB** for this clip (~395 B/player-frame), against 2.07 MB
  per player-frame unpacked. Not `MaskCache`, whose label map loses overlapping
  players -- see that module's docstring.
- `--redraw` on `scripts/render_full_pipeline.py`, plus `--redraw-from` (redraw
  one run into a different output) and `--redraw-causal` (label as the original
  pass did -- the reproduction check).

**Verification.**

- The tracking pass reproduces `Melsungen_sam2_vouched.json` exactly on every
  summary field, `numbers_resolved`, `identity_events`, `number_reads` and
  `player_teams`. `frame_players` differs by 230 frames of p10 only, because
  that older run predates the retirement rule in `c79db06`.
- `--redraw-causal` is **pixel-identical to the render on 1498 of 1499 frames**,
  and agrees with a live-voter replay on 18867/18867 player-frames.
- The single differing frame is 780, where p16 is revived. The redraw suspends
  the number on the revival frame; the render draws it once more and suspends
  on the next. `drive_sam2` records a checkpoint's re-ID event after the frame
  is yielded, so `NumberIdentityResolver.begin_frame` sees it one frame late --
  contradicting its own docstring ("stale from that instant"). **Left as-is: the
  redraw is the more correct of the two.** Worth fixing in the render itself.
- Redraw costs **82 seconds** against ~22 minutes for the tracking pass.

**Now scored.** The clip's numbers were unverified when this shipped; all ten
were read off the footage on 2026-09-11 and all ten are right (item 6). Since
every verdict is correct, the 3667 backfilled player-frames carry correct
numbers, and the coverage gain is a real gain rather than more of an unknown.
Recall remains unmeasured, so this scores what the run says, not what it missed.

---

## The migration manifest is out of the repository (2026-09-09, `ab26419`)

The regenerated audit reclassified 3,294 files as `directory_level_reference`
(up from 72) and put a reference list of up to 713 entries on nearly every row,
taking the CSV from 2.16 MB to 135 MB across the same 3,314 rows. Committing it
would have added that to history permanently -- git history cannot be trimmed
afterwards without a rewrite that invalidates every clone.

`docs/outputs-migration-manifest.csv` is now gitignored (`.gitignore:168`) and
untracked; `outputs-migration-reference-report.md` is what stays in git. The
142 MB file still sits in the working tree, which is the intended outcome, not a
leftover.

## Team separation on Kiel-Lemgo was a measurement error (2026-09-09)

The entry claimed Kiel-Lemgo was "the worst of the three matches" at 81/36 with a
correctly fitted model. It is the **best**. Measured with the model the pipeline
actually holds, sampled across the whole 10-minute window rather than the fit's
front-loaded one:

```
                     split       mean conf   below gate   visual/colour agreement
Kiel-Lemgo         209 / 160       0.676       1%              0.990
Melsungen-Berlin   213 / 159       0.587       3%              0.957
Eisenach-Hamburg   181 / 192       0.582       9%              0.894
```

The crops confirm it (`runs/team_grid/*/team_{0,1}.png`): Kiel white against
Lemgo dark blue at 0.86-0.97, Melsungen red against Berlin green, Eisenach maroon
against Hamburg navy. Kiel's five sub-gate crops are dark, occluded or a
close-up of the ball -- not kit confusion.

**The 81/36 figure was never a measure of separation.** It counted cluster
assignments over ~117 crops from a handful of sampled frames, so it measures who
happened to be on screen. Holding the same model fixed and changing only the
sample window moves it from 60/59 to 85/28 -- reproducing the "failing" number --
while mean confidence stays 0.61-0.70 and sub-gate stays under 14%. A balanced
split is not evidence of a good fit and an unbalanced one is not evidence of a
bad one; **confidence and the sub-gate fraction are the metrics, and the crops
are the check.**

What the user actually saw on Kiel ("many contested/unsure players") came from
the misaligned-crop fit below: every overnight render, Kiel included, sits in
`runs/overnight/_broken_team_fit/`. Re-rendered on 2026-09-10 with the corrected
model, `Eisenach_Hamburg_2min_1` goes from **41 identities all on team 1** to
25/15, confirmed on the video. Its two duplicate numbers (#11, #21) sit on
*opposite* teams, so part of the same-team duplicate complaint was the team
model and not the voter.

The retirement rule that shipped in the same commit is **not** verified by that
clip: it never approached the track cap (mean 12.7 live, max 16 of 20, identical
before and after), because its bench is out of frame. `Melsungen_Berlin_2min_1`
is where 17.8-live-against-14-detections and 34%-at-cap were measured, so that
is the clip that settles it.

`scripts.team_grid_examples` now takes `--video/--detections/--team-model`, so
any fitted model can be inspected against the boxes the pipeline used, with
crops ordered and annotated by confidence.

## Number-anchored linking had never fired on real data (2026-09-09)

`NumberIdentityResolver` links identities that are the same player -- same team,
same number, never simultaneously live. On the 60-second Melsungen clip it fired
**0 times**: every same-team duplicate there was simultaneously live, so none was
fragmentation, and the mechanism had unit tests and nothing else.

The overnight batch exercised it. Across seven 2-minute renders it fired **1 to 7
times per clip**, most on Eisenach_Hamburg_2min_1. Two minutes is long enough for
a player to leave and return, which is the case it was built for and a 60-second
clip cannot contain.

That run also showed zero same-team duplicate numbers by canonical identity on
six of seven clips -- and exposed the folded-identity vote bug below, found while
checking why a duplicate count looked wrong.

---

## Reader accuracy (2026-09-08, `9ee582b`)

Three docTR recognisers converged near 0.62 with a shared ~6% floor of
confidently-wrong reads, which read as a domain limit. It was not. The original
`baudm/parseq` checkpoint, same architecture trained on scene text rather than
docTR's document-text corpus, scores **0.858 / 0.958 selective** on the identical
crops -- paired McNemar b=4 c=80, **p=2e-19** -- at the same speed and abstention.
Wired in as `--reader parseq`. See the handoff for the full table and the
end-to-end check on the `#6` failure.

Fine-tuning on other sports was measured and rejected: hockey weights match on
accuracy but collapse abstention (0.49), SoccerNet weights fall to 0.266.

**Consequences for the crop-padding and fold-rule entries: both were measured
against the weak reader and should be re-run before being trusted.**

Remaining reader work, if 0.858 is not enough: no public handball video matches
our regime (TeamTrack is 6K fisheye, "Play by play" is JSON coordinates not
video, the GTS action set is practice footage), so training data means labelling
new broadcast matches -- starting with finishing the truncated
`data/raw/2026-06-07_1284786_VfL_GM_Loewen.mp4` transfer (65 MB of ~3.2 GB).
**The 660-crop evaluation set must stay out of any training set**; every
measurement in the handoff is anchored to it. Where the two readers agree on a
number they are right 92% of the time and agree on human-unreadable crops only
1% of the time, so agreement is a good *pre-fill* for a labelling pass -- but not
an auto-label, because agreement selects exactly the easy crops.

---

## `notebooks/*.py` have diverged from the modules that replaced them (2026-09-09, `5689a9b`)

The 29 legacy `notebooks/*.py` copies were deleted. Verified before removal: no
`.ipynb` imported any of them, they formed a closed import cluster referencing
only each other, and every one had a current counterpart (15 under a new name).
All were last touched between 2026-07-29 and 2026-08-25; their counterparts
carry changes through 2026-09-08.

The concrete drift that motivated this — the legacy
`render_raw_team_classification.py` copy sitting 32 lines behind `scripts/`,
missing `PERSON_CLASS_IDS`, `person_detections()` and `number_detections()` from
commit `6dc4ff1`, and so silently reproducing the bug where the tracker followed
jersey *number boxes* as people — is gone with the copy. `git log` is the
fallback now.

`notebooks/` itself stays: it holds the four `.ipynb` files plus `fonts/`
(`DEFAULT_FONT` in `scripts/render_full_pipeline.py`), `.env`, `models/` and the
two dataset directories.

---

## Number votes were filed against folded identities (2026-09-09, `fc7c259`)

An alias is an interpretation layer, not a renumbering: after two identities are
folded the tracker keeps emitting the folded id. Both render paths called
`voter.observe()` with that raw id, so every read taken after a link was filed
under an identity `best()` no longer answers for -- 500 of 7131 votes across the
seven overnight renders, 7% overall and 39% on the clip with the most links.

It rarely changed a verdict, but the hidden failure is worse than the waste:
contradicting evidence on a folded id could never correct the surviving verdict.
Both paths now observe against `registry.canonical(...)`.

---

## Bench players were tracked forever (2026-09-09, `c79db06`)

`TrackManager` rule 2 retires a track only when its mask *collapses*. A
substitute who sits on the bench stays fully visible, keeps a perfect mask, and
was tracked to the end of the clip. Measured on Melsungen_Berlin_2min_1: 17.8
live tracks against 14.0 detections per frame, live > detections on 97% of
frames, and **34% of frames pinned at `MAX_LIVE_OBJECTS=20`** -- so a genuinely
new player on court could not be admitted at all, and some fragmentation blamed
on re-ID was the tracker having no room.

`UNMATCHED_CHECKPOINTS_MAX = 20` now retires a track the detector has stopped
confirming, whatever its mask looks like. The threshold comes from linking raw
detections over three 10-minute windows (n=14232 dropouts of a person who
returns): p90=65 frames, p95=124, so 8 seconds of footage retires 1.6% of genuine
dropouts early -- and those recover through re-ID, while a bench-sitter never
leaves on its own.

---

## The overnight team models were fitted on misaligned crops (2026-09-09, `c79db06`)

`TeamModel.fit_from_video` calls `detect_fn` only on every `stride`-th frame but
hands it the frame, not the index. `run_overnight_batch` counted calls, so boxes
from frame *n* landed on video frame *10n*: the torso crops were background and
2-means had only noise to cluster. Five of seven overnight clips came back with
every player on one team.

Refitting correctly turns Eisenach from `{0:1, 1:102}` to `{0:46, 1:57}` on the
same sampled frames. The caller now steps its counter by `FIT_STRIDE` and passes
the same value to `fit_from_video`.
