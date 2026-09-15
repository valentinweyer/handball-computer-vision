# What body keypoints can and cannot buy us

Measured on the ViTPose demo (`scripts/render_pose_demo.py`, branch
`pose-keypoint-demo`) over FelixClaar: 249 frames, 2808 poses, one pose per
cached RF-DETR person box.

The question this answers: can we infer passes, throws and positioning from
keypoints. Short version -- positioning and whole-body events yes, throws with
a trained model and a ceiling around 0.7 F1, passes not without either the ball
or a labelled event set.

## 1. What the pose data actually is here

Players are bigger than the usual "pose on broadcast is hopeless" assumption:
median box height 146 px (p10 115, p90 211). Per-joint share scoring above 0.5,
over all 2808 poses:

| joint group | share > 0.5 |
|---|---|
| shoulders | 99.5 / 99.6 % |
| hips | 98.7 / 98.1 % |
| elbows | 92.1 / 92.9 % |
| knees | 92.8 / 91.6 % |
| ankles | 88.9 / 88.0 % |
| wrists | **85.3 / 86.3 %** |

The ordering is the whole story. Torso is effectively always available; the
extremities are not. Wrists are the weakest joint and wrists are exactly what a
throw is made of, so every throw-shaped question inherits a ~14% per-frame
dropout that has to be absorbed temporally.

Derived signal availability, same clip:

- arm elevation (both shoulders + one wrist confident): **91.4 %** of poses
- ankle ground point (both ankles confident): **81.1 %**
- shoulder line: **96.1 %**; shoulder and hip line together: **93.2 %**

## 2. Throws and shots

A naive geometric probe was run to avoid guessing: per pose, the higher wrist's
elevation above the shoulder line, normalised by torso length. It produces a
clean, temporally consistent signal -- the top responses persist across three
to five consecutive frames on the same player rather than flickering, so it is
measuring something real.

It is measuring the wrong thing. Every one of the clip's strongest responses,
inspected frame by frame, is a raised arm that is not a throw: gestures, a
celebration behind the play, a player jogging with an arm up. Arm elevation is
a *candidate generator*, not a throw detector, and this clip may contain no
shot at all.

That matches the published ceiling. On UNIRI-HBD -- 751 clips, 7-9 handball
action classes -- an I3D model over RGB plus optical flow reaches 0.78 average
F1, but the per-class spread is the point:

| class | F1 |
|---|---|
| Jump shot | 0.89 |
| Running | 0.83 |
| Passing | 0.79 |
| Catch | 0.73 |
| Dribbling | 0.71 |
| Shot | 0.70 |
| **Throw** | **0.66** |

Shot and Throw are the two worst classes and are confused with each other;
Dribbling leaks into Running, Catch and Throw. Two caveats make these numbers
optimistic for us: the clips are *pre-segmented single actions* from *static
GoPro cameras at training sessions*, so the model never has to find the action
in continuous play, and the camera never pans.

Jump shot being the best class is the useful signal. It has a whole-body
signature -- airborne, knees driving, trunk extended, arm cocked -- carried by
hips, shoulders and knees, which are our reliable joints, not by wrists.

**Verdict.** Jump-shot and airborne detection are reachable from these
keypoints. Separating "throw" from "shot" from "pass release" is a trained
temporal classifier (PoseC3D over ST-GCN: more robust to pose noise, better
cross-dataset, 94.1% vs 81.5% on NTU60-XSub) and lands near 0.7 F1 even in the
literature's easier setting. It needs labels we do not have.

## 3. Passes -- the hard one, and why

**There is no ball.** The detector emits goalkeeper and field player only
(`HOST_CLASSES = (GOALKEEPER, PLAYER)`); the current checkpoint is named
`large-handball-v4-w-numbers-no-ball`, and no cache in `outputs/` contains a
ball class. This was a deliberate choice, and the literature supports it: the
same handball group reports ball detection at **AP 35.44** -- the ball is small,
motion-blurred in flight, and occluded in a 40x20 m court with constant contact.

A pass is *defined* by the ball leaving one player and arriving at another. With
no ball, it has to be inferred from synchronised arm gestures plus trajectory
changes in the receiver. That is a real research direction, not a heuristic:
PathCRF (KDD 2026) does ball-free soccer event detection from player
trajectories alone, reaching **75.69 % event F1** by modelling possession as an
edge in a dynamic player graph with a CRF forbidding impossible transitions.

Its requirements are the obstacle. It consumes *metric* coordinates (see §4),
runs at 5 fps, and trains on matches with ground-truth event labels. Handball is
also harder than the soccer case: shorter distances and faster releases mean the
possession-transfer window is a few frames rather than a second.

**Verdict.** Not reachable now. The honest routes are (a) add a ball class back
to the detector and accept poor AP, using pose to disambiguate who released it,
or (b) label events on a few clips and train a trajectory model. Both are
projects, not features.

## 4. Positioning -- the most valuable, and least blocked by pose

This is where keypoints pay off, but the blocker is not the pose.

**What pose adds over the boxes we already have.** Boxes already give position
and speed. Keypoints add two things a box cannot:

- **Orientation.** The shoulder line gives facing direction, available on
  96.1 % of poses (93.2 % with the hip line too). Front/back is resolved by the
  face keypoints, which score 97-99 %. Which way a player faces is genuinely new
  tactical information -- it separates a defender tracking the ball from one
  turned away, and it is invisible to a bounding box. Caveat: on **12.4 %** of
  poses the shoulders project to under 0.08 of player height (player square to
  the camera axis), where orientation is ill-conditioned and should abstain.
- **A ground point that knows when it is wrong.** The ankle midpoint sits a
  median **21.7 px above** the box bottom, 12.4 % of player height. Neither is
  automatically better -- the ankle is the joint, a constant ~5-8 cm above the
  sole, while the box bottom is whatever the detector drew. The gain is that the
  ankle is *semantically defined*: it degrades visibly under occlusion (18.9 %
  of poses have no confident ankle pair, which is a usable abstain signal) and
  it tells you when a player is airborne, where the box bottom silently means
  nothing.

**The actual blocker is the court homography**, not the keypoints. Per
`45fdce0` and `76f3a0d` on `court-keypoint-mapping`: the landmark-to-vertex
mapping is fixed (fit residual 922.1 -> 32.9 cm, leave-one-out 1701.5 -> 47.8
cm), and inter-frame regularisation cut jitter from p50 23.1 px to 6.8 px. But
p90 is still 25.4 px, 4.1 % of frames move the projected court over 50 px, 4.4 %
of players land off court, and -- the commit says so directly -- none of it is
checked against ground truth. No frame has inliers spanning more than 55 % of
the court, so every homography is extrapolated roughly fourfold.

**Verdict.** Metric positioning, heat maps, formation width and inter-player
distances are the highest-value output and need no new models -- but they are
gated on validating the homography against ground truth. Orientation is
available today in image space and is the one tactical signal pose uniquely
unlocks.

## 5. The near-term win that is not an event at all

`teams/model.py::torso_boxes` currently cuts the torso as fixed fractions of box
height (`TORSO_TOP_FRAC`, `TORSO_BOTTOM_FRAC`). Shoulders and hips are our two
most reliable joints (99.5 % / 98 %). A pose-defined torso would be exact rather
than assumed, and would stay correct when a player is bent over, airborne or
leaning -- precisely the poses where a fixed fraction slices into shorts or
background and where crop contamination currently costs a team vote.

This is the cheapest real use of the keypoints, and the only one that improves a
component already in the runtime. It is **not** a free change: crop geometry is
one of the four invariant areas, so "Per-video team-model calibration" in the
handoff has to be read first, and the calibration re-measured -- changing crop
geometry moves the feature distribution the per-video clustering is fitted to.

## 6. Recommended order

1. **Pose-defined torso crops** for team classification -- reuses existing
   architecture, measurable against the current calibration.
2. **Orientation and airborne/jump-shot signals** -- available now, no labels,
   no homography.
3. **Validate the court homography against ground truth**, then metric
   positioning and heat maps. Highest value, and the work is court work.
4. **Throw/shot classification** only with a labelled set and PoseC3D; expect
   ~0.7 F1 and plan for Shot/Throw confusion.
5. **Passes** last, and only after deciding between a ball class and a labelled
   trajectory model.

## Sources

- UNIRI-HBD and I3D per-class results: [Analysis of Movement and Activities of Handball Players Using Deep Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC10144022/), [dataset](https://ieee-dataport.org/open-access/handball-action-dataset-uniri-hbd)
- Pose quality and failure modes in handball: [3D Pose Estimation and Tracking in Handball Actions Using a Monocular Camera](https://pmc.ncbi.nlm.nih.gov/articles/PMC9693185/)
- Ball-free event detection: [PathCRF](https://arxiv.org/abs/2602.12080v1)
- Skeleton backbone choice: [Revisiting Skeleton-based Action Recognition (PoseC3D)](https://arxiv.org/abs/2104.13586)
- Ball detection difficulty in broadcast handball: [Spatio-Temporal Analysis of Handball Players' Actions from Broadcast Videos](https://doi.org/10.3390/bdcc10040118)
