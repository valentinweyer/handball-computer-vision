# Handball CV handoff: tracking and team classification

Detail companion to `CLAUDE.md`, which carries only the working rules.
This file is **not** auto-loaded into agent context — read it before
changing the detection, tracking, team-classification, or identity code.
It records the decisions, experiments, measurements, and implementation
state of the team-classification/tracking work on branch
`feat/team-classification`. Do not assume every experimental module is
wired into the main runtime.

## Repository migration note

The canonical reusable code now lives under `src/handball_cv/`; command-line
diagnostics live under `scripts/`; rejected or non-default approaches live under
`experiments/`; and protected label copies live under `data/annotations/`.

The older `notebooks/*.py`, `outputs/`, and `source/` paths referenced later in
this document still exist and still work as migration fallbacks. They were not
deleted or moved. New code should import from `handball_cv`, and new commands
should be run as modules, for example:

```bash
python -m scripts.render_mcbyte_team_correction --help
```

See `docs/architecture.md` for dependency rules and `docs/legacy-layout.md` for
the exact material intentionally retained pending deletion approval.

**`docs/tracking-evaluation.md` supersedes several tracking claims made in this
document.** It records per-frame identity measurements against human-verified
references, and in particular: MCByte-with-masks is the *worst* of the seven
box-tracker configurations tested on both clips, plain SORT is roughly twice
as accurate as MCByte on FelixClaar, and masks make identity worse while
dominating runtime *when they only nudge box association*. §8 then measured a
different SAM2 configuration — mask-memory propagation with periodic
detector-checkpoint reprompting — as the best tracker of all eight measured,
by a wide margin, which is why "SAM2 is not the primary tracker" below is no
longer a settled constraint. The document also lists the claims made during
that work that were later refuted, and why.

**The "Agreed next implementation step" below (wiring the mask fallback) is
superseded.** Identity, team evidence, goalkeeper role, and re-ID are now a
single `PlayerRegistry` (`src/handball_cv/tracking/identity.py`) shared by
both `IdentityManager` (McByte) and `TrackManager` (SAM2, via the new
`src/handball_cv/tracking/sam2_driver.py`), and jersey numbers are now read
against SAM2 tracklets too (`scripts.evaluate_number_pipeline --tracker
sam2`), which they never were before. See "Number-anchored identity across
both trackers" below for the measurements and what is still open.

This document records the decisions, experiments, measurements, and current implementation state from the team-classification/tracking work on branch `feat/team-classification`. Read it before changing the pipeline. Do not assume every experimental module is wired into the main runtime.

## Goal and constraints

The immediate goal is reliable per-player team classification in handball video, including brief partial overlaps. The two teams and their jerseys change from video to video, so the system must discover two anonymous teams independently for each video. It must not require a permanently labeled global catalogue of teams.

The user accepts a small amount of labeling when it is made easy, but does not want a large supervised labeling project. Existing labels are primarily evaluation and cluster-orientation data, not the basis of a universal supervised classifier.

The key architectural constraints are:

- Use the user's local RF-DETR/Roboflow detector.
- Use MCByte for multi-object tracking as the current production default. Do not switch back to ByteTrack — it measured worst on FelixClaar in `docs/tracking-evaluation.md`.
- **Tracker choice is under active reconsideration, not settled.** SAM2 with periodic detector-checkpoint reprompting (not the raw video predictor seeded once, and not masks merely nudging MCByte's box association — see `docs/tracking-evaluation.md` §8 for why those are different things) measured as the best tracker by a wide margin against per-frame ground truth on all three clips tested: 93.8% correct on FelixClaar (vs 81.5% for the prior-best box tracker, SORT), 89.2% with zero mixed tracklets/switches on Han-Ber4, and 98.9% on a third derived clip (BHC-FAG) with a hand-verified, architecturally independent reference (vs 95.9% for the best box tracker there). Two of the three references (FelixClaar, BHC-FAG) are built with SAM+Cutie, a different engine than the SAM2 tracker being scored — only Han-Ber4's is circular (its reference was itself built by SAM2). This overturns the reasoning that originally excluded SAM2 as primary. Before changing the production default, weigh the open caveats in §8: roughly 10x the runtime cost and not a drop-in swap into the existing per-frame tracker interface — the remaining open question is cost/integration, not validity.
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

**Unresolved architectural fork (2026-08-26):** there are now two parallel implementations of track lifecycle, team voting, and SigLIP re-ID, one per tracker family — `notebooks/identity_manager.py` (below) for MCByte, and `src/handball_cv/tracking/sam2_manager.TrackManager` for SAM2. They share some primitives from `handball_cv.teams.model` but are not the same code. If the SAM2 tracker direction above (see "SAM2 as the main tracker") is pursued further, this fork needs a decision — one identity layer, not two — before it compounds.

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
- Switches are logged as `team_switch` or `suspected_id_switch` events (see below).

Constants:

- `TEAM_OBSERVATION_INTERVAL = 5`
- `TEAM_SWITCH_OBSERVATIONS = 3`
- `TEAM_SWITCH_MIN_QUALITY = 0.40`

This mechanism fixed the persistent-wrong-label behavior in the Felix and Han-Ber comparison videos.

### Decoupling tracker error from team error (2026-08-24)

A tracker swap used to become a team error and then a re-ID error — the circular
failure this architecture forbids — because the reversible-switch mechanism could
not distinguish a classifier correction from a tracker swap. Two fixes landed in
`notebooks/identity_manager.py`. The invariants they establish, which later edits
must preserve:

- Team evidence **decays** (`TEAM_EVIDENCE_DECAY = 0.85`) before each new vote, so
  `team_confidence` caps near 0.93 and a *contested* label can fall below the
  stable gate and stop vetoing re-ID. Do not restore an unbounded accumulator.
- Prior evidence is **kept, not reset**, on a switch.
- A flip from a settled label (`>= MIN_STABLE_TEAM_CONFIDENCE` and
  `>= TEAM_SETTLED_OBSERVATIONS` = 5) logs `suspected_id_switch` rather than
  `team_switch`. Diagnostic only — no behavioral branch.
- A provisional label (`team_is_provisional`, zero qualified observations) is
  adopted by the first qualified read; it does not get the three-observation
  inertia of a real switch.
- `is_qualified(confidence, quality)` is the **single shared predicate** gating
  both the vote and the re-ID veto, and it gates *entry* to `record_team_vote` —
  an unqualified read must be a true no-op, never a decay-only call.
- Goalkeeper status is a running majority (`GOALKEEPER_EVIDENCE_MIN = 5`,
  `GOALKEEPER_EVIDENCE_MAJORITY = 0.70`), not frozen at creation.

`0.85` is a reasoned estimate (≈10 observations ≈ 50 frames to re-stabilize), not
a value tuned against labeled data — no labeled ID-switch dataset exists. Revisit
if `suspected_id_switch` rates look wrong on new videos.

Renderers show two distinct non-stable states: **new/uncertain** `(0, 220, 255)`
for `team_is_provisional`, and **contested** `(0, 0, 220)` for established-but-
below-`MIN_STABLE_TEAM_CONFIDENCE`. Contested is deliberately visible, not muted —
it surfaces the exact failure mode this work exists to expose.

Discovery narrative, per-clip verification tables, spot-checked frames, the
buggy-vs-fixed decay measurements, and the `summary()` / result-JSON key changes:
`docs/identity-decoupling.md`.

## Agreed overlap-mask fallback

The user and assistant agreed on this exact policy:

1. Use normal crop classification whenever the normal crop passes quality and overlap gates.
2. Only when an otherwise usable crop is rejected because of inter-player overlap, try guarded mask-color classification.
3. Accept the masked observation only if its mask geometry, mask confidence, team confidence, and final observation quality all pass their gates.
4. Otherwise abstain and wait for a better frame.
5. A masked observation must never override a good clean-frame observation merely because a mask exists.

Masks are therefore an evidence-recovery fallback, not the team-label authority.

### Guarded mask construction

Implemented in `notebooks/mask_team_features.py`. MCByte's `_last_mask_output`
(`masks`, `tracklet_mask_dict`, `mask_avg_prob_dict`) is spatially aligned to the
current frame but produced from prior track state.

**Never select a target mask by tracker ID alone** — a tracker association error
would become a confident team-color error. The path assigns detector boxes to
masks by Hungarian matching on geometry, separately checks that this agrees with
MCByte's tracker-ID-to-mask mapping, and **abstains on disagreement or ambiguous
matching**.

```text
safe jersey pixels = torso ROI
                   AND eroded target mask
                   AND NOT dilated neighboring masks
```

Cutie masks are already mutually exclusive; neighbor dilation supplies the
uncertainty margin. The eleven numeric gate thresholds live in
`mask_team_features.py` and are tabulated with rationale in
`docs/overlap-mask-experiment.md`.

### Masked color features

`masked_jersey_color_features()` in `notebooks/team_model.py` computes the same 62-D descriptor as the normal path, but histograms and moments use only selected safe pixels. Empty masks return zero features and invalid mask dimensions raise an error.

The mask experiment is deliberately color-only, so it measures what the mask changed rather than hiding the result behind an unmasked SigLIP fallback. It loads a lightweight placeholder classifier to avoid loading SigLIP alongside SAM/Cutie.

## Mask experiment results

Diagnostic: `notebooks/render_mask_team_comparison.py` — frame-local raw box-color
vs guarded mask-color on identical detections, no temporal vote. Overlap begins at
torso contamination 0.05; a mask is usable only with geometry accepted, team
confidence >= 0.30, and effective observation quality >= 0.40.

Headline across both clips: the mask path **recovers additional qualified
same-team evidence** (159 observations on Felix, 112 on Han-Ber that the
box-overlap gate rejected) but does **not** yet improve qualified per-frame
accuracy. On Han-Ber's independent reference evaluation, raw box-color accuracy is
95.28% and remains 95.28% under the real confidence/quality gate.

**Do not quote the 96.46% "every geometrically accepted mask" figure without the
confidence-gating caveat** — all six label changes behind it had low team
confidence, and the real gate removes both the five corrections and the one
regression. There is no substantial labeled Felix overlap set, so claim no
numerical Felix improvement at all.

Runtime: ~3.8-4.8 fps on the NVIDIA GB10, ~52 s per clip; SAM/Cutie propagation
dominates, the color classifier is cheap.

Per-clip detection counts, acceptance rates, rejection-reason breakdowns, and the
manual-label subsets: `docs/overlap-mask-experiment.md`.

## What was tried and rejected

### Tracker-first team labeling with sticky track labels

Rejected as originally implemented. A short tracker mistake or contaminated initial crop could lock the wrong team until the track ended. Temporal aggregation is still useful, but only after independent raw observations and with reversible hysteresis.

### Classify once and freeze

Rejected. Current labels remain mutable and switch after sustained qualified opposition.

### ByteTrack

Rejected by user based on observed tracking quality. Use MCByte.

### SAM2 as the main tracker

**Superseded (2026-08-26) — see `docs/tracking-evaluation.md` §8.** This rejection was based on an older SAM2-era track manager (prompt/add/remove/reprompt logic on a seed-once propagator) that produced unhelpful tracklets in informal review, predating this project's rigorous per-frame scorer. When the same reprompting design (`src/handball_cv/tracking/sam2_manager.TrackManager`, periodic detector checkpoints rather than a one-time prompt) was actually run through that scorer, it measured best on both clips by a wide margin. The original reasoning — "MCByte re-anchors on detector boxes every frame, SAM2 doesn't need to" — turned out to miss that SAM2's mask memory also survives frames where the detector itself misses the player, which was the source of its recall advantage, not just an identity-stability difference. Existing SAM2 masks are still also used as independent Han-Ber evaluation references (with the important exception that this makes `sam2_reprompt`'s own Han-Ber score partially circular — see §8.4). Kept as the original rejection reasoning, not deleted, since §8's caveats (runtime cost, integration shape, two-clip sample) mean this is not yet a settled reversal.

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

## Number-anchored identity across both trackers (2026-09-05)

Numbers are the only signal that discriminates teammates (§5 of
`docs/tracking-evaluation.md`), and the number pipeline had never been run
against SAM2's tracklets — `scripts/run_sam2_reprompt_tracker.py` had no
`NumberVoter` at all. This closes that gap, ahead of the mask-fallback work
below, because it directly targets teammate-level ID errors the mask fallback
does not.

**Shared identity layer.** `IdentityManager` and `TrackManager` were two
parallel implementations of team evidence, goalkeeper role, and re-ID (the
"unresolved architectural fork" noted above). Both now delegate to one
`PlayerRegistry` (`src/handball_cv/tracking/identity.py`). This also carried
two real fixes from the McByte side to the SAM2 side that had never applied
there: team-evidence decay/hysteresis (previously a plain accumulator with no
switch logic) and running-majority goalkeeper status (previously frozen at
track creation). Verified on all three §8 clips (FelixClaar, Han-Ber4,
BHC-FAG): tracker lifecycle events and final `evaluate_tracker_identity`
scores are byte-identical to the pre-change baseline on every clip — the two
fixes exist for cases these particular clips don't happen to exercise (no
goalkeeper-class flicker or contested team label landed near a re-ID
decision), so this is a clean, unexercised result, not evidence the fixes
never matter.

**Shared SAM2 driver.** The predictor/`TrackManager` checkpoint loop, previously
inlined in `run_sam2_reprompt_tracker.py`, is now
`src/handball_cv/tracking/sam2_driver.drive_sam2`, reused by both the
tracking-only dump and the number pipeline. Re-verified byte-identical
(events and full per-frame box dump) against the pre-extraction version on
all three clips.

**Numbers now run on SAM2.** `scripts/evaluate_number_pipeline.py` takes
`--tracker {mcbyte,sam2}`; the per-frame match→read→vote logic is shared
(`process_numbers_for_frame`) so only the outer loop differs. On FelixClaar,
`--tracker sam2 --reader easyocr` resolved 2 of 3 labeled players correctly
(players 3 and 4) versus `--tracker mcbyte --reader qwen`'s 1 of 6 (only
player 4, with 5 no-verdicts) from the same clip's earlier run
(`runs/number_pipeline/felix_qwen/report.json`) — SAM2 resolving more players
with the *weaker* reader shows reads-per-tracklet is a real bottleneck,
because SAM2 has a mask every propagated frame where McByte's mask manager
does not. One wrong read on this run (player 12 voted "3", truth "33")
matches the already-documented single-vs-double-digit truncation mode.

**Do not read that as "reader accuracy is not the bottleneck" — BHC-FAG
refuted the strong form of that claim.** Traced frame by frame on that clip
(`--tracker sam2 --reader easyocr`), player 2 wears 22 and EasyOCR produced
`92, 92, 22, 22, …` — at frame 45 the voter committed to **`92`, a wrong
number, and displayed it until ~frame 85** before retracting and settling on
`22` only at frame 175. Reader error is a first-order problem, not a
second-order one; see the vote-gate defect below.

**Vote-gate defect (open).** That wrong commit cleared on counts `92:3, 22:2`
→ `margin = (3-2)/5 = 0.200` against `min_margin = 0.200`, and the test is
`margin < self.min_margin`, so it passed by exactly zero. A 3-vs-2 plurality
is enough to display a number confidently, which violates this project's
"abstaining beats injecting a confident wrong observation" rule. Raising
`min_votes` from 3 to 4-5 blocks it without penalising a genuinely dominant
value; raising `min_margin` alone would also suppress correct late verdicts
(`22` resolved at margin 0.235). Not yet changed — `NumberVoter`'s defaults
are still `min_votes=3, min_margin=0.2`.

**Clip caveat: do not use `data/raw/Hannover.mp4` for read-rate or tracker
comparisons.** It is 120fps (8.3s of play over 999 frames, 3456x2168) while
every other evaluation clip is ~25fps. At `ocr_every=5` that is ~4.8x more
reads per second of real play, which inflates reads-per-player (measured 40.6
vs FelixClaar's 5.3) and makes association artificially easy — the same
distortion §8.7 of `docs/tracking-evaluation.md` avoided by subsampling
BHC-FAG 50->25fps. Hannover was never given that treatment. Its
`mcbyte+qwen` 5/5 result is correspondingly softer than it looks: one player
accumulated 81 near-identical reads of "10".

**Not yet done:** number-anchored merge -- resolving the same number on two
different `player_id`s (McByte fragment or SAM2 re-ID miss) does not yet fold
their evidence together, though `NumberVoter.merge()` already exists for
exactly this. Needs its own guard against two different physical players who
share a jersey number across a team boundary before it can be trusted as an
identity-correcting signal rather than just a reporting convenience.

## Agreed next implementation step (superseded in priority, not correctness)

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

- `data/raw/FelixClaar.mp4`: 249-frame short clip.
- `outputs/team_comparison/.FelixClaar_detections_v1.npz`
- `outputs/team_dataset/Han-Ber4_cached.mp4`: 199-frame short clip.
- `outputs/team_dataset/.Han-Ber4_detections_v1.npz`
- `data/raw/Hannover.mp4` and its existing detection/model assets are also available, but the recent overlap-mask evaluation focused on Felix and Han-Ber.

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
- `notebooks/render_mcbyte_team_correction.py`: plain MCByte plus reversible tracked team overlay, dense per-player diagnostic text (team/confidence/obs/switches).
- `notebooks/render_team_overlay.py`: same MCByte/IdentityManager pipeline, clean broadcast-style overlay (translucent team-colored mask fill + box border, small legend, no per-player text) matching the original notebook's "Full video team clustering" cell style. Masks are pulled directly from MCByte's `tracklet_mask_dict` for visualization only, not through the guarded spatial-assignment checks in `mask_team_features.py` (an occasional wrong mask here is cosmetic, not a label error).
- `notebooks/mask_team_features.py`: guarded spatial mask assignment and safe-pixel construction.
- `notebooks/render_mask_team_comparison.py`: isolated box-versus-mask experiment and metrics.
- `notebooks/team_aware_tracker.py`: experimental team-gated association; do not enable by default.
- `tests/test_team_model.py`: feature, mask-guard, and switching tests.

## Result artifacts

- `outputs/team_raw/FelixClaar_raw_team_h264.mp4`
- `outputs/team_raw/Hannover_raw_team_h264.mp4`
- `outputs/team_correction_mcbyte/FelixClaar_mcbyte_team_correction_h264.mp4`
- `outputs/team_correction_mcbyte/Han-Ber4_mcbyte_team_correction_h264.mp4`
- `outputs/team_correction_mcbyte/FelixClaar_idswitch_h264.mp4`, `Han-Ber4_idswitch_h264.mp4`: same renderer, post tracker/team-error decoupling (see above).
- `outputs/team_correction_mcbyte/FelixClaar_clean_overlay_v3_h264.mp4`, `Han-Ber4_clean_overlay_v3_h264.mp4`: `render_team_overlay.py`, current two-tone uncertain/contested scheme (see above). `_v3` because two earlier iterations (`_idswitch`-equivalent, then noise-gating-fixed) were superseded and deleted during this session; the suffix has no meaning beyond "latest."
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
  data/raw/FelixClaar.mp4 \
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
