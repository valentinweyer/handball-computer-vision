# Handball CV handoff: tracking and team classification

This document records the decisions, experiments, measurements, and current implementation state from the team-classification/tracking work on branch `feat/team-classification`. Read it before changing the pipeline. Do not assume every experimental module is wired into the main runtime.

## Goal and constraints

The immediate goal is reliable per-player team classification in handball video, including brief partial overlaps. The two teams and their jerseys change from video to video, so the system must discover two anonymous teams independently for each video. It must not require a permanently labeled global catalogue of teams.

The user accepts a small amount of labeling when it is made easy, but does not want a large supervised labeling project. Existing labels are primarily evaluation and cluster-orientation data, not the basis of a universal supervised classifier.

The key architectural constraints are:

- Use the user's local RF-DETR/Roboflow detector.
- Use MCByte for multi-object tracking. Do not switch back to ByteTrack.
- SAM2 is not the primary multi-object tracker. Segmentation masks may be used as supporting evidence.
- Raw team classification must remain frame-local. A tracker must not supply or freeze the raw team prediction.
- Team labels attached to identities must be reversible.
- Abstaining is preferred to injecting a confident wrong observation.
- Avoid circular coupling: a team error must not force a tracking error, and a tracking error must not automatically become a team error.

## Current high-level design

There are three conceptually separate layers:

```text
RF-DETR detections
        |
        +--------------------------+
        |                          |
        v                          v
MCByte tracking              frame-local TeamModel
(track IDs + masks)          (team, confidence, quality)
        |                          |
        +-------------+------------+
                      v
              IdentityManager
       (stable player IDs + reversible
              temporal team evidence)
```

MCByte does not classify teams. TeamModel does not use previous team labels. IdentityManager aggregates independent observations after tracking.

## Per-video team-model calibration

`notebooks/team_model.py` fits two anonymous clusters per video.

Calibration flow:

1. Run the object detector on sampled frames.
2. Exclude detector-provided goalkeepers from the two-field-team fit.
3. Reject clearly unusable and heavily contaminated crops.
4. Extract the centered torso/waist crop.
5. Fit two anonymous jersey-color clusters.
6. Fit the SigLIP visual branch for re-identification and guarded fallback.

The retained crop geometry is `waist_v1`: a center-anchored crop using 40% of the player box width and 40% of its height.

The color descriptor has 62 values:

- Lab `a` histogram: 12 bins.
- Lab `b` histogram: 12 bins.
- HSV hue histogram: 12 bins.
- HSV saturation histogram: 8 bins.
- HSV value histogram: 8 bins.
- Mean and standard deviation for five Lab/HSV channels: 10 values.

The current color model is:

```text
62-D Lab/HSV features
    -> StandardScaler
    -> PCA(2)
    -> KMeans(2)
```

Cluster IDs are anonymous and are aligned to display labels A/B separately for each video.

Important thresholds in `team_model.py`:

- `CENTERED_CROP_SCALE_W = 0.4`
- `CENTERED_CROP_SCALE_H = 0.4`
- `MAX_TORSO_CONTAMINATION = 0.25`
- `MIN_TEAM_VOTE_CONFIDENCE = 0.30` for KMeans
- `MIN_COLOR_CONFIDENCE = 0.08`
- `MIN_VISUAL_FALLBACK_CONFIDENCE = 0.25`
- `MIN_VISUAL_COLOR_AGREEMENT = 0.70`

KMeans confidence is a normalized distance-margin proxy, not a calibrated posterior probability.

### Color versus visual embeddings

Color is the primary team signal. SigLIP is still useful for player re-identification and as a guarded fallback, but unconditional fusion was not retained.

Measured clean-crop results recorded in the implementation:

- Felix color: 98.59%.
- Felix visual: 88.73%.
- Felix fixed fusion: 97.18%.
- With the current KMeans confidence threshold of 0.30, the measured retained sets were 70/71 Felix crops and 58/64 Han-Ber crops, both at 100% accuracy.
- The current all-label cluster-orientation checks in the mask experiment were 69/71 (97.18%) for Felix and 61/64 (95.31%) for Han-Ber.

These numbers answer different questions: confidence-gated accuracy is the relevant metric for observations allowed to affect tracked state; all-label accuracy includes reads that should abstain.

## Runtime team observations

For a normal, non-masked observation, `TeamModel.observe()` currently does this:

1. Extract the 40% x 40% torso crop.
2. Run one SigLIP embedding pass for re-ID/visual fallback.
3. Predict the color team and confidence.
4. Use visual fallback only under the guarded low-color-confidence conditions.
5. Compute crop quality from size, contrast, and sharpness.
6. Compute how much another person box covers the target torso.
7. Reduce observation quality continuously as contamination approaches 25%; quality becomes zero at or above 25%.

`predict_masked()` in `team_model.py` still intentionally ignores masks. The new masked fallback is implemented and tested separately but is not yet wired into `IdentityManager`.

## Tracking and identity

### Tracker choice

Use plain `McByteTracker` from the `trackers` package. The user explicitly rejected ByteTrack because MCByte performed materially better on these videos.

The current comparison/correction renderer uses plain MCByte, not the experimental `TeamGatedMcByteTracker`. Team-aware association exists in `notebooks/team_aware_tracker.py`, but should not be enabled for this work: using the same uncertain team classification to gate tracking creates circular failure modes.

MCByte uses detector boxes for association and, when enabled, SAM + Cutie masks for mask-conditioned association. The frame supplied to MCByte must be RGB.

### IdentityManager

`notebooks/identity_manager.py` maps short-lived MCByte `tracker_id` values to longer-lived `player_id` values. It can reconnect retired fragments using SigLIP cosine similarity.

Current re-ID constants:

- `REID_COS_SIM_MIN = 0.7`
- `REID_MAX_GAP_FRAMES = 300`

Goalkeeper status always comes from the detector class. Goalkeepers do not contribute to the two-field-team fit.

### Reversible temporal team state

The original failure was that one early wrong team read could remain attached to a tracked player indefinitely. This was replaced with continuous, reversible monitoring:

- Sample team evidence approximately every 5 frames.
- A qualified observation needs team confidence at least 0.30 and quality at least 0.40.
- A same-team qualified observation clears any pending switch.
- Three consecutive qualified observations for the opposite team replace the current label.
- A switch clears old votes and starts evidence for the replacement team.
- Switches are logged as `team_switch` events.

Constants:

- `TEAM_OBSERVATION_INTERVAL = 5`
- `TEAM_SWITCH_OBSERVATIONS = 3`
- `TEAM_SWITCH_MIN_QUALITY = 0.40`

This mechanism fixed the persistent-wrong-label behavior in the Felix and Han-Ber comparison videos.

## Agreed overlap-mask fallback

The user and assistant agreed on this exact policy:

1. Use normal crop classification whenever the normal crop passes quality and overlap gates.
2. Only when an otherwise usable crop is rejected because of inter-player overlap, try guarded mask-color classification.
3. Accept the masked observation only if its mask geometry, mask confidence, team confidence, and final observation quality all pass their gates.
4. Otherwise abstain and wait for a better frame.
5. A masked observation must never override a good clean-frame observation merely because a mask exists.

Masks are therefore an evidence-recovery fallback, not the team-label authority.

### Guarded mask construction

Implemented in `notebooks/mask_team_features.py`.

MCByte's `_last_mask_output` is spatially aligned to the current frame, although it is produced before current-frame association using prior track state. It contains:

- `masks`: boolean `(K, H, W)` masks.
- `tracklet_mask_dict`: stable tracker ID to mask-row mapping.
- `mask_avg_prob_dict`: tracker ID to average winning-mask probability.

Do not select a target mask using tracker ID alone. A tracker association error would then become a confident team-color error.

The implementation:

1. Assigns current detector boxes to current masks one-to-one using Hungarian matching on mask/box geometry.
2. Separately checks that the spatial assignment agrees with MCByte's tracker-ID-to-mask mapping.
3. Abstains on disagreement or ambiguous spatial matching.
4. Intersects the target mask with the torso crop.
5. Erodes the target mask boundary.
6. Dilates nearby masks and removes that uncertain neighbor boundary.

The reliable pixels are conceptually:

```text
safe jersey pixels = torso ROI
                   AND eroded target mask
                   AND NOT dilated neighboring masks
```

Cutie masks are mutually exclusive already; neighbor dilation creates the useful uncertainty margin.

Current mask gates:

- Mask average confidence: at least 0.60.
- Fraction of target mask inside current box: at least 0.80.
- Mask fill of current box: at least 0.08.
- Spatial assignment score: at least 0.25.
- Assignment margin over alternatives: at least 0.02.
- Safe torso coverage: at least 0.15.
- Retained fraction of target torso mask: at least 0.50.
- Safe pixels before resize: at least 80.
- Safe pixels after 32x32 resize: at least 24.
- Erosion radius: 2.5% of the smaller torso dimension, clamped to 1-5 pixels.
- Neighbor dilation radius: erosion radius + 1, capped at 5 pixels.

The diagnostic also used a stricter MCByte mask-creation overlap threshold of 0.20 instead of the package default 0.60, so a heavily overlapped box is less likely to initialize a contaminated SAM mask.

### Masked color features

`masked_jersey_color_features()` in `notebooks/team_model.py` computes the same 62-D descriptor as the normal path, but histograms and moments use only selected safe pixels. Empty masks return zero features and invalid mask dimensions raise an error.

The mask experiment is deliberately color-only, so it measures what the mask changed rather than hiding the result behind an unmasked SigLIP fallback. It loads a lightweight placeholder classifier to avoid loading SigLIP alongside SAM/Cutie.

## Mask experiment results

Diagnostic: `notebooks/render_mask_team_comparison.py`.

It performs frame-local raw box-color classification and frame-local guarded mask-color classification on identical detections. Tracking supplies masks and diagnostic IDs only. There is no temporal team vote in this experiment.

Overlap begins at torso contamination 0.05. A mask is considered usable by the real team-state policy only when:

- Geometry is accepted.
- Team confidence is at least 0.30.
- Effective mask observation quality is at least 0.40.

### FelixClaar

- 249 frames.
- 2,598 field-player detections.
- 562 overlap detections.
- 259 geometrically accepted masks: 46.1% acceptance.
- 159 observations were unusable through the box-overlap gate but usable through the mask path.
- Four geometric mask classifications changed team.
- Two changes passed the real confidence/quality gate and look visually plausible in extreme overlap frames.
- There is no substantial manually labeled Felix overlap set, so do not claim a numerical accuracy improvement from Felix.

Primary rejection counts:

- Tracker/mask disagreement: 133.
- Unassigned: 71.
- Low box coverage: 57.
- Ambiguous spatial assignment: 31.

### Han-Ber4

- 199 frames.
- 2,206 field-player detections.
- 464 overlap detections.
- 245 geometrically accepted masks: 52.8% acceptance.
- 112 observations were unusable through the box-overlap gate but usable through the mask path.
- Six geometric mask classifications changed team.

Manual overlap ground truth is very small:

- Five substantial manually labeled overlap crops, all from team B.
- Three had geometrically accepted masks.
- Raw, mask, and chosen classification were all correct on those five.

The independent Han-Ber reference-mask evaluation is more useful:

- 339 overlap detections matched to existing SAM2 reference identities at IoU >= 0.65.
- Raw box-color accuracy: 95.28%.
- If every geometrically accepted mask is used, accuracy appears to rise to 96.46%: five corrections and one regression.
- Those six changes all had low team confidence.
- Under the actual confidence/quality gate, 138 mask observations were selected, none changed the team label, and accuracy remained 95.28%.

Interpretation: the demonstrated value is recovering additional qualified same-team evidence, not yet improving qualified per-frame accuracy on Han-Ber. Do not quote the 96.46% geometric result without the confidence-gating caveat.

### Runtime cost

On the available NVIDIA GB10 machine, the complete diagnostics ran at roughly 3.8-4.8 frames per second. Each 199/249-frame clip took about 52 seconds. SAM/Cutie mask propagation dominates this experiment; the color classifier itself is cheap.

## What was tried and rejected

### Tracker-first team labeling with sticky track labels

Rejected as originally implemented. A short tracker mistake or contaminated initial crop could lock the wrong team until the track ended. Temporal aggregation is still useful, but only after independent raw observations and with reversible hysteresis.

### Classify once and freeze

Rejected. Current labels remain mutable and switch after sustained qualified opposition.

### ByteTrack

Rejected by user based on observed tracking quality. Use MCByte.

### SAM2 as the main tracker

Not the current direction. The older SAM2-era track manager required prompt/add/remove/reprompt logic and produced unhelpful tracklets in these tests. MCByte re-anchors on detector boxes every frame. Existing SAM2 masks remain useful as independent Han-Ber evaluation references.

### Team-gated tracking association

Implemented experimentally in `team_aware_tracker.py` but not retained for the current tests. Gating tracking with uncertain team predictions creates circular errors. Keep tracking team-agnostic until team evidence is independently validated; even then, any later use should be soft and separately evaluated.

### Run the tracker twice, once per predicted team

Not implemented as the current solution. It may become a later downstream experiment after reliable team classification, but doing it before classification is stable risks fragmentation and circular dependency.

### Large supervised universal team classifier

Not selected. Team identities and jerseys change per video, so fixed semantic team labels do not generalize. Small labels are retained for evaluation and A/B orientation.

### GMM instead of KMeans

Rejected for now. It provides genuine posterior probabilities but cost about 9.6 percentage points in measured team purity. KMeans with abstention performed better.

### Unconditional color + visual fusion

Rejected. On Felix, color alone outperformed visual embeddings and fixed fusion. Keep color first and visual only as a guarded fallback/re-ID feature.

### Larger torso crops

Rejected based on measurements and crop inspection.

- 0.4 width x 0.4 height remains the best tested geometry.
- Increasing only height to 0.55 diluted the jersey with head/legs and measured worse.
- Increasing both axes to 0.7 pulled in neighboring players and measured much worse.

### Use tracker masks during base per-video fitting

Rejected for the base model. The base two-team discovery remains tracker-independent and sees the same ordinary crop distribution used by the normal prediction path. Masks are an overlap-only fallback.

### Trust mask geometry without classifier confidence

Rejected. This produced an apparent Han-Ber gain from 95.28% to 96.46%, but every changed label had a low KMeans margin. The real confidence/quality gates remove both the five apparent corrections and the one regression.

### Trust the tracker-ID mask mapping alone

Rejected. The guarded path requires spatial assignment plus tracker-ID agreement and abstains when they disagree.

## What is retained

- Local RF-DETR/Roboflow detections.
- Per-video anonymous two-team discovery.
- Small, easy manual label sets for evaluation and cluster orientation.
- Centered 0.4 x 0.4 torso crops.
- Lab/HSV color-first classification.
- KMeans distance-margin confidence and abstention.
- SigLIP for re-ID and guarded visual fallback.
- Plain MCByte tracking.
- Reversible team evidence with a three-observation switch.
- Mask interiors as a strictly guarded overlap fallback.
- Explicit unknown/abstain behavior.

## Current implementation status

Implemented and tested:

- Per-video TeamModel and color features.
- Raw per-detection classification renderer.
- MCByte IdentityManager with reversible team switching.
- Guarded mask-to-box matching and safe torso masks.
- Masked Lab/HSV features.
- Box-versus-mask diagnostic, overlays, JSONL rows, and metrics.

Not yet wired into the runtime IdentityManager:

- The agreed mask fallback. `IdentityManager._embed()` still calls ordinary `TeamModel.observe()` and therefore overlap-rejected normal observations still abstain in the actual tracked pipeline.

## Agreed next implementation step

Wire the mask fallback into the tracked observation path without changing clean behavior.

Recommended logic:

```text
normal crop quality accepted?
    no  -> abstain; masks must not rescue blur, tiny crops, or absent players
    yes -> normal overlap quality accepted?
              yes -> use normal TeamModel observation
              no, rejected specifically by overlap -> try guarded mask observation
                    mask geometry valid?
                    mask team confidence >= 0.30?
                    effective mask quality >= 0.40?
                        yes -> submit masked observation
                        no  -> abstain
```

Implementation cautions:

- Keep normal and mask crop quality separate so a mask cannot rescue an intrinsically bad crop.
- Pass current MCByte mask output explicitly; do not hide tracker state inside TeamModel.
- Build assignments across all current person detections, including goalkeepers, before selecting field-player evidence.
- Preserve the spatial-assignment/tracker-ID agreement guard.
- Do not feed team predictions back into MCByte association as part of this change.
- Do not change the three-opposite-observation switching rule.
- Add tests proving clean observations win, overlap can fall back, invalid masks abstain, and low-confidence mask labels cannot switch a team.

After integration, rerun Felix and Han-Ber tracked overlays and compare:

- Number of qualified observations per player.
- Time to initial stable team assignment.
- Number and direction of team switches.
- Wrong-label duration around overlaps.
- Any increase in tracker fragmentation or re-ID mismatch.

## Data and evaluation assets

Videos and caches:

- `FelixClaar.mp4`: 249-frame short clip.
- `outputs/team_comparison/.FelixClaar_detections_v1.npz`
- `outputs/team_dataset/Han-Ber4_cached.mp4`: 199-frame short clip.
- `outputs/team_dataset/.Han-Ber4_detections_v1.npz`
- `Hannover.mp4` and its existing detection/model assets are also available, but the recent overlap-mask evaluation focused on Felix and Han-Ber.

Manual manifests:

- `outputs/team_dataset/FelixClaar/manifest.json`: 71 labeled field crops, 42 A and 29 B. They are essentially clean; there is no useful substantial-overlap subset.
- `outputs/team_dataset/Han-Ber4/manifest.json`: 64 labeled field crops, 29 A and 35 B, plus six goalkeeper labels. Only five substantial overlap labels exist, all team B.

Independent Han-Ber reference evaluation:

- `source/.Han-Ber4_sam2_masks/00001.npz` through `00198.npz`
- `outputs/.Han-Ber4_track_team_labels.json`
- `outputs/.Han-Ber4_team_labels.json`
- `outputs/.Han-Ber4_track_label_sheet_order.json`

## Important source files

- `notebooks/team_model.py`: crop geometry, quality, color/visual model, prediction, persistence.
- `notebooks/identity_manager.py`: stable player IDs, re-ID, temporal observations, reversible switching.
- `notebooks/render_raw_team_classification.py`: raw frame-local baseline.
- `notebooks/render_mcbyte_team_correction.py`: plain MCByte plus reversible tracked team overlay.
- `notebooks/mask_team_features.py`: guarded spatial mask assignment and safe-pixel construction.
- `notebooks/render_mask_team_comparison.py`: isolated box-versus-mask experiment and metrics.
- `notebooks/team_aware_tracker.py`: experimental team-gated association; do not enable by default.
- `tests/test_team_model.py`: feature, mask-guard, and switching tests.

## Result artifacts

- `outputs/team_raw/FelixClaar_raw_team_h264.mp4`
- `outputs/team_raw/Hannover_raw_team_h264.mp4`
- `outputs/team_correction_mcbyte/FelixClaar_mcbyte_team_correction_h264.mp4`
- `outputs/team_correction_mcbyte/Han-Ber4_mcbyte_team_correction_h264.mp4`
- `outputs/mask_team_comparison/FelixClaar_box_vs_mask_h264.mp4`
- `outputs/mask_team_comparison/Han-Ber4_box_vs_mask_h264.mp4`
- `outputs/mask_team_comparison/FelixClaar_box_vs_mask_overlaps.jpg`
- `outputs/mask_team_comparison/Han-Ber4_box_vs_mask_overlaps.jpg`
- `outputs/mask_team_comparison/FelixClaar_box_vs_mask.json`
- `outputs/mask_team_comparison/Han-Ber4_box_vs_mask.json`
- Per-detection diagnostic rows are in the matching `*_rows.jsonl` files.

## Useful commands

Use the existing conda environment:

```bash
conda run -n NewEnv pytest -q tests
```

The scoped project suite currently passes: 28 tests. Do not run bare `pytest` from repository root because vendored `onnxruntime` and other upstream trees contain unrelated test entry points that break global collection.

Render Felix mask comparison:

```bash
conda run -n NewEnv python notebooks/render_mask_team_comparison.py \
  FelixClaar.mp4 \
  --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \
  --team-model outputs/team_comparison/.FelixClaar_team.pkl \
  --manifest outputs/team_dataset/FelixClaar/manifest.json \
  --output outputs/mask_team_comparison/FelixClaar_box_vs_mask.mp4 \
  --device cuda
```

Render Han-Ber with independent reference scoring:

```bash
conda run -n NewEnv python notebooks/render_mask_team_comparison.py \
  outputs/team_dataset/Han-Ber4_cached.mp4 \
  --detections outputs/team_dataset/.Han-Ber4_detections_v1.npz \
  --team-model outputs/team_correction_mcbyte/.Han-Ber4_team.pkl \
  --manifest outputs/team_dataset/Han-Ber4/manifest.json \
  --output outputs/mask_team_comparison/Han-Ber4_box_vs_mask.mp4 \
  --device cuda \
  --reference-mask-dir source/.Han-Ber4_sam2_masks \
  --reference-track-labels outputs/.Han-Ber4_track_team_labels.json \
  --reference-min-iou 0.65
```

## Repository-state warning

The working tree contains many pre-existing modified and untracked experiment files. They belong to the ongoing project. Do not run destructive cleanup, reset, or checkout commands. Work only on the requested files and preserve unrelated changes.
