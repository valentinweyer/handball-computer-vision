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

`src/handball_cv/teams/model.py` fits two anonymous clusters per video.

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

Important thresholds in `teams/model.py`:

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

`predict_masked()` in `teams/model.py` still intentionally ignores masks. The new masked fallback is implemented and tested separately but is not yet wired into `IdentityManager`.

## Tracking and identity

### Tracker choice

Use plain `McByteTracker` from the `trackers` package. The user explicitly rejected ByteTrack because MCByte performed materially better on these videos.

The current comparison/correction renderer uses plain MCByte, not the experimental `TeamGatedMcByteTracker`. Team-aware association exists in `experiments/team_gated_tracking/tracker.py`, but should not be enabled for this work: using the same uncertain team classification to gate tracking creates circular failure modes.

MCByte uses detector boxes for association and, when enabled, SAM + Cutie masks for mask-conditioned association. The frame supplied to MCByte must be RGB.

**Architectural fork resolved (2026-09-08):** track lifecycle, team voting and re-ID were once implemented twice, one per tracker family. They are now a single shared layer: `PlayerRegistry` in `src/handball_cv/tracking/identity.py` owns team evidence, goalkeeper role and re-ID, and both tracker-facing managers delegate to it — `IdentityManager` (McByte, same module) and `sam2_manager.TrackManager` (SAM2). Keep it that way: identity invariants belong in `PlayerRegistry`, not in a tracker-specific manager, or the decay, veto and reversibility rules below have to be re-proved per tracker.

### IdentityManager

`src/handball_cv/tracking/identity.py` maps short-lived MCByte `tracker_id` values to longer-lived `player_id` values. It can reconnect retired fragments using SigLIP cosine similarity.

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
`src/handball_cv/tracking/identity.py`. The invariants they establish, which later edits
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

Implemented in `src/handball_cv/teams/masks.py`. MCByte's `_last_mask_output`
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
`teams/masks.py` and are tabulated with rationale in
`docs/overlap-mask-experiment.md`.

### Masked color features

`masked_jersey_color_features()` in `src/handball_cv/teams/model.py` computes the same 62-D descriptor as the normal path, but histograms and moments use only selected safe pixels. Empty masks return zero features and invalid mask dimensions raise an error.

The mask experiment is deliberately color-only, so it measures what the mask changed rather than hiding the result behind an unmasked SigLIP fallback. It loads a lightweight placeholder classifier to avoid loading SigLIP alongside SAM/Cutie.

## Mask experiment results

Diagnostic: `scripts/render_mask_team_comparison.py` — frame-local raw box-color
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

Implemented experimentally in `team_gated_tracking/tracker.py` but not retained for the current tests. Gating tracking with uncertain team predictions creates circular errors. Keep tracking team-agnostic until team evidence is independently validated; even then, any later use should be soft and separately evaluated.

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

**Vote-gate defect (resolved 2026-09-06; evidence base is thin -- read the
caveats).** That wrong commit cleared on counts `92:3, 22:2` -> `margin =
(3-2)/5 = 0.200` against `min_margin = 0.200`, and the test is `margin <
self.min_margin`, so it passed by exactly zero. A 3-vs-2 plurality is enough to
display a number confidently, which violates this project's "abstaining beats
injecting a confident wrong observation" rule.

The fix that was tried first -- raising `min_votes` 3 -> 5 -- is wrong and was
reverted. It blocks the bad commit, but it also blocked a correct one: BHC-FAG
player 3 (visually confirmed jersey 53) read `53, 53, 53` and then nothing but
shorts-logo noise, so at `min_votes=5` that player never resolves. **Count does
not separate a safe early commit from a premature one; margin does.** Both
commits happened on three votes, at margin 1.000 and 0.200 respectively.
Defaults are now `min_votes=3, min_margin=0.35`, paired with verdict hysteresis
(a resolved value is held until a rival clears both gates).

Three caveats on how well-evidenced `0.35` actually is:

- **It is a lower bound, not a tuned value.** Sweeping `min_margin` from 0.15 to
  1.00 on BHC-FAG's three contested read sequences, behaviour changes at 0.25
  and at 0.35 and is then *flat from 0.35 through 1.00*. All BHC-FAG establishes
  is "must exceed 0.30 to block the 3-vs-2 split"; 0.35 is the conservative edge
  of a wide plateau, not a fitted point. No clip in the corpus contains a
  genuinely close-run correct answer, which is what would pin the upper end.
- **The held-out clips do not validate it.** Felix and Hannover produce
  identical output at every swept value, because their reports store only final
  tallies and cannot exercise the temporal early-commit path this gate governs.
  That is absence of harm, not confirmation. A real test needs per-frame read
  traces from a second clip -- now cheap, since `render_full_pipeline.py` writes
  a `<stem>_reads.json` cache.
- **This clip already had disproportionate influence on the vote layer.**
  `min_promote_ratio` exists because of BHC-FAG p27 and its docstring cites
  BHC-FAG p23. The folding constants and the margin gate were all set from one
  clip.

Known cost of the change: an EasyOCR-style verdict that only becomes dominant
late (`22` resolved at margin 0.235 on the trace above) is now permanently
suppressed rather than merely delayed, because hysteresis means a value that
never qualifies never gets held. That is acceptable while Qwen is the default
reader -- it resolves the same player at margin 0.831 -- but it would be a
regression if EasyOCR were ever restored as the default.

**Order-dependence is now a load-bearing property, by design.** Because
verdicts are held once set, the outcome depends on read *order*, not just final
tallies. Two of the nine verdicts on the BHC-FAG re-render could not qualify on
their final counts and survive only because they qualified early: p3 (`53`,
final margin 0.111, buried by 13 shorts-logo `0` reads) and p8 (`49`, final
margin 0.333). This is the intended behaviour -- early reads come from frames
where the player is well-resolved -- but it means a clip that happens to deliver
its noisy reads first will behave differently from one that does not, and that
sensitivity has not been measured.

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

## A 1080p number-reading evaluation set, and what it measured (2026-09-06)

**Why it exists.** Every labelled jersey set before this one was unusable for
benchmarking a production reader. `runs/jersey_audit` is sampled from the 640x640
COCO export RF-DETR trains on, where number boxes have median width **11 px**;
production runs at 1920x1080 where the median is **26-30 px**. That is a 2.7x
different regime, and a reader scored on the first says nothing about the second.
The two on-regime sets (`runs/ocr_labels/FelixClaar`, `runs/jersey_native_eval`)
are both the same clip, so at 1080p there was exactly one venue -- the same
single-clip weakness that made validating `min_margin` impossible.

`runs/number_eval_1080p` replaces them for reader work: **660 crops, 110 from each
of six 1920x1080 clips** (four full Bundesliga matches plus BHC-FAG, FelixClaar,
Han-Ber4), sampled from the production detector's own cached output, stratified by
box height and round-robined across clips. Built by
`scripts/build_number_eval_set.py`; ground truth in
`data/annotations/jersey/number_eval_1080p_labels.json`.

Composition: **323 readable, 298 unreadable, 39 unsure, 190 marked not-a-number.**

### Detector precision at 1080p (new -- never previously measured)

A `not_number` box status means the human judged the detection not to be a jersey
number at all. That makes this set a detector benchmark as well as a reader one:

| conf bucket | n | not_number | precision |     | band | n | precision |
|---|---|---|---|---|---|---|---|
| 0.3-0.4 | 206 | 103 | **50%** | | `<18` | 120 | **45%** |
| 0.4-0.5 | 121 | 51 | 58% | | `18-21` | 120 | 72% |
| 0.5-0.6 | 89 | 20 | 78% | | `22-25` | 120 | 73% |
| 0.6-0.7 | 104 | 13 | 88% | | `26-30` | 120 | **84%** |
| 0.7-0.8 | 118 | 3 | **97%** | | `31-40` | 120 | 78% |
| 0.8-1.0 | 22 | 0 | 100% | | `>=41` | 60 | 78% |

Per clip, precision ranges 62% (FelixClaar) to 81% (Eisenach-Hamburg).

**Confidence is a strong, monotonic filter, and the production threshold of 0.3
looks too low:**

| `--threshold` | boxes kept | junk kept | readable lost |
|---|---|---|---|
| 0.3 (current) | 660 | 190 | 0 |
| 0.5 | 333 | 36 | 73 |
| 0.6 | 244 | 16 | 117 |

Raising 0.3 -> 0.5 removes **81% of false positives** for **23% of readable
numbers**. For a voting pipeline that is likely a good trade -- reads accumulate
over frames, while false positives inject noise into every vote, which is the
mechanism behind both the `0`-on-the-shorts misread and p3's margin collapsing to
0.111. **Not yet changed:** confirm end-to-end first by re-running BHC-FAG at 0.5
and checking resolved-player count does not fall.

Separately, the player-overlap filter (`max_player_containment`, >=0.9) drops
**26180 of 65039** number detections across the seven 1080p clips -- **40% are on
no player at all** (hoardings, scoreboards, backdrop lettering). Per-clip keep
rate 39% (Melsungen) to 89% (Eisenach).

### Reader configuration: reasoning is actively harmful here

`Qwen3.8-Flash-Next` emits `reasoning_content` before `content`. Two consequences,
both of which produced wrong measurements before being caught:

1. **A token budget that only fits the answer yields an empty string.** At
   `--max-tokens 16` every request returned `''`; at 320, 10% did. Empty content
   parses as no-answer and *scores as an abstention*, so a misconfiguration
   masquerades as the model correctly declining -- corrupting the one metric the
   benchmark exists to measure. `benchmark_qwen_jersey_ocr.py` now marks these
   `truncated`, excludes them from scoring, and retries them (reasoning length is
   not deterministic: a crop that overran 1024 tokens answered in 96 on retry).
2. **Reasoning roughly halves abstention discipline for no accuracy gain.**
   Measured on 195 samples of this set, same prompt and images, only
   `enable_thinking` differing:

   | | coverage | accuracy | selective | abstention | wrong | tokens |
   |---|---|---|---|---|---|---|
   | thinking on | 0.99 | 0.69 | 0.70 | **0.41** | 24 | 133 |
   | thinking off | 0.89 | 0.68 | **0.76** | **0.81** | 17 | **2** |

   Paired McNemar on accuracy: p = 1.00 -- indistinguishable. A reasoning model
   reasons its way to *an* answer, which on a crop with no legible number is
   exactly the wrong instinct: a milder form of the SmolVLM failure. This also
   explains why whole-number reading scored 0.44 abstention here against the
   0.93 recorded on FelixClaar -- **that earlier benchmark ran without reasoning,
   so its 0.70/0.90/0.93 is not a like-for-like comparator for anything measured
   under this server config.**

### Whole-number reading on the full set (thinking on, 660 crops)

coverage 0.96, accuracy 0.64, selective 0.67, abstention 0.44. **37% of all errors
are silent truncations** -- a two-digit number answered with one of its digits:
`17->7` (x11), `11->1` (x4), `22->2` (x3), `23->2`, `21->2`, `54->5`. That is 38 of
103 errors, and 15% of all two-digit readable crops. It is the failure that forced
suffix-folding into `NumberVoter`, and it is what the digit-wise read mode
(`context_digits`) is being measured against.

### Caveats on this set

- **Half of it is temporally correlated.** FelixClaar and Han-Ber4 are short clips
  detected at stride 1, so 110 samples cover 44% and 55% of *all their frames*;
  109 near-duplicate pairs each. The four Bundesliga matches (stride 100, 440
  samples) are the temporally independent subset and should be checked separately.
  Use a minimum frame spacing rather than a fixed stride next time.
- **`>=41` was under-sampled on a bad call.** Its quota was halved on the basis of
  eyeballing six crops in a contact sheet; the labels show it is the *cleanest*
  band (78% precision, 52% readable). The `BAND_QUOTA_SCALE` entry should be
  dropped if the set is ever rebuilt.
- **`<18` is the weak band**: 45% detector precision, 23% readable. It is real
  production data, but it measures the detector far more than the reader.
- 39 `unsure` crops are excluded from all scoring.

## Digit-wise vs whole-number reading: not demonstrated (2026-09-07)

The plan was to test, before building a digit-level model, whether decomposing a
jersey number into digits helps *at all* -- using the best reader already
available rather than training something first. Four arms on
`runs/number_eval_1080p`, same model, same images, read mode x reasoning:

| read mode | reasoning | coverage | accuracy | selective | abstention | wrong |
|---|---|---|---|---|---|---|
| whole | on | 0.96 | 0.64 | 0.67 | 0.44 | 102 |
| whole | **off** | 0.87 | 0.64 | **0.74** | **0.81** | **73** |
| digit | on | 0.96 | 0.67 | 0.70 | 0.47 | 91 |
| digit | off | 0.90 | 0.67 | 0.74 | 0.74 | 74 |

641 crops all four arms answered; `unsure` and `truncated` excluded throughout.

**Reasoning off, unambiguously.** Paired McNemar on accuracy: p = 0.824 (whole)
and p = 1.000 (digit) -- reasoning changes accuracy not at all. It changes
abstention from 0.44 to 0.81, drops wrong answers 102 -> 73, and costs 2
completion tokens instead of 133. A reasoning model reasons its way to *an*
answer, which on a crop with no legible number is the wrong instinct.

**Digit decomposition: not demonstrated.** Digit mode leads on both settings
(p = 0.064 with reasoning, p = 0.053 without) but never clears the
pre-registered 0.05, and two things argue against reading the near-misses as a
real effect:

- **It does not survive the independence check.** Full set 16-6 for digit mode;
  restricted to the four Bundesliga matches (325 crops, stride 100) it is 5-3,
  p = 0.727. FelixClaar and Han-Ber4 sample 44% and 55% of *all their frames*, so
  their near-duplicate crops were inflating the discordant counts.
- **Selective accuracy is identical at 0.74.** Digit mode is not more reliable per
  answer; it answers more often (coverage 0.90 vs 0.87) and abstains less (0.74 vs
  0.81). Trading 7 points of abstention for 3 of accuracy is not obviously a gain
  under this project's "abstaining beats a confident wrong observation" rule.

**H2 is a clean negative, and it was the stronger half of the original argument.**
The premise was that per-position reading lets a model report *partial*
information -- answering `? 7` for a half-visible 17 instead of a confident `7`
the voter counts as a wrong vote. Measured on the 38 crops where whole mode did
exactly that: digit mode produced **0** explicit partials with reasoning on and 3
of 31 with it off. On the same crops it reproduced the truncation as a confident
single digit with status `exact`. Offering a `?` token does not make the model
report uncertainty it would otherwise hide.

### What this means for the small-reader plan

The digit-head architecture was motivated by three things, and the first two no
longer hold:

1. ~~Per-position uncertainty comes free~~ -- H2 says it does not.
2. ~~Decomposition improves reading~~ -- not demonstrated; the effect vanishes on
   temporally independent data.
3. **Class balance is still real**: 100-way is untrainable here (13 of 32 numbers
   had a single sample), and digits pool to a usable distribution. This argument
   is unaffected by the above and remains the reason a digit *output layer* may
   still be right -- but as a data-efficiency measure, not because decomposition
   makes reading more accurate or more honest.

The measured gap now worth transferring is **whole-number reading with reasoning
off**: 0.64 accuracy / 0.74 selective / 0.81 abstention, against EasyOCR's
0.37 / 0.64 / 0.84 on the same 660 crops. That is +27 points of accuracy and +28
of coverage over the production reader, and it is what distillation should target.

Artifacts: `runs/number_eval_1080p/all_arms_benchmark.json` (four arms),
`compare_thinking.json`, `compare_nothink.json`, `easyocr_benchmark.json`.

## An off-the-shelf recogniser already matches the VLM (2026-09-07)

Tested because the small-reader plan assumed a model had to be trained. It does
not. All of these are pretrained, already installed, and needed no domain data.

Scored on the same 660 crops and labels as every other reader
(`scripts/benchmark_doctr_readers.py`, 619 crops common to all arms):

| reader | coverage | accuracy | selective | abstention | ms/crop |
|---|---|---|---|---|---|
| `parseq` @conf 0.5 | 0.76 | 0.62 | **0.82** | **0.88** | **2** |
| `vitstr_small` @0.5 | 0.76 | 0.61 | 0.80 | 0.85 | 1 |
| `crnn_vgg16_bn` @0.5 | 0.73 | 0.56 | 0.77 | 0.89 | 1 |
| Qwen whole, no reasoning | 0.86 | 0.63 | 0.74 | 0.81 | ~700 |
| Qwen whole, reasoning | 0.96 | 0.64 | 0.67 | 0.44 | ~7400 |
| **EasyOCR (production)** | 0.57 | 0.37 | 0.64 | 0.84 | ~40 |

**PARSeq is statistically indistinguishable from Qwen**: paired McNemar on
readable crops, 40 / 37 discordant, **p = 0.82**. It has *better* selective
accuracy (0.82 vs 0.74) and *better* abstention (0.88 vs 0.81), at roughly 350x
the speed and no GPU-resident 89 GB model. Against the production reader it is
+25 accuracy points and ~20x faster.

Two things this settles:

**"CTC over digits" was never an untested architecture.** EasyOCR's recogniser is
already a CRNN -- feature extractor, BiLSTM, CTC prediction, 1.4M params over 96
characters. It is the production floor at 0.37. What separates it from PARSeq is
training data and decoder, not the digit-level shape. Any plan justified by "use a
CTC digit model" needs to explain what it adds over swapping the checkpoint.

**Each reader needs its own input, and they disagree about which.** `parseq`
scores 0.62 on the tight crop and **0.04** on the context crop -- a recogniser
trained on cropped text lines is out of distribution on a padded scene with a red
rectangle drawn on it. The VLM is the reverse (0.30 tight, 0.70 context). Comparing
them on a single shared input would have badly misrepresented one of them.

### What this does to the plan

Distilling Qwen into a small model was motivated by a 27-point gap over EasyOCR
that only an 89 GB VLM could reach. A 20M-parameter pretrained model closes most of
that gap for free, which makes distillation-from-scratch the expensive way to get
somewhere we can already stand.

The remaining headroom is different in kind: **PARSeq has had no handball data at
all.** Fine-tuning it on jersey crops -- with Qwen or human labels as the target --
is now the cheap experiment, and it starts from 0.62 rather than from nothing. The
digit-level output-layer question folds into that as a decoder choice, where it
can be measured against the same baseline instead of argued about.

Also worth noting: PARSeq abstains *more* than Qwen (0.88 vs 0.81) while being no
less accurate, which is the direction this project's invariant prefers.

Artifacts: `runs/number_eval_1080p/doctr_benchmark.json` (all three archs, all
confidence thresholds, raw reads).

## First minute-long clip, and the class-filter bug it exposed (2026-09-08)

Every tracker and identity claim in this repo rested on clips of 8-20 seconds.
`data/raw/Melsungen_window_cached.mp4` is the first longer test: 60s
at 25fps (1500 frames), carved from a real Bundesliga broadcast at 88:00 by
`scripts/extract_clip_window.py`, chosen for continuous 7v7 play (12.3 players
+-1.1 per frame, 11.2 numbers per frame). Team model fitted unsupervised from 1412
torso crops.

**It immediately exposed a bug the short clips could not.** The detection caches
were changed to keep every class (goalkeeper 1, player 2, referee 3, number 4) so
a later question would not force a re-detection. The number-reading consumer was
filtered in `799676d`; the *tracking* consumers were not. Both trackers were
handed the unfiltered cache and followed referees and jersey-number boxes as
people -- **16689 of 36686 cached detections on this clip are number boxes, so 45%
of what the tracker was asked to follow were not people.**

| | all classes | + pending fix | people only |
|---|---|---|---|
| players | 30 | 29 | **18** |
| tracker ids consumed | 87 | 79 | **35** |
| re-ID hits | 57 | 50 | **17** |
| fragmented players | 22 | 22 | **10** |
| max fragments / player | 7 | 6 | **4** |
| ocr reads | 1235 | 1333 | **392** |

The read count is the clearest tell: a tracked number box *contains itself*, so it
scored mask-IoS 1.0 against its own detection and produced a read every OCR frame.
Roughly 3.4x of the reads on this clip were numbers reading themselves.

Only this clip used a multi-class cache for `--detections`; FelixClaar, BHC-FAG and
Han-Ber4 all used two-class Roboflow caches, so the earlier tracker comparisons are
unaffected. Fixed in `6dc4ff1`: `person_detections` / `number_detections` now live
beside `frame_detections` and both scripts share them.

### New-player confirmation was unreachable for moving players

`_pending_new` binned unmatched detections by `round(x/50)_round(y/50)` and
required two hits in the same bin. Measured on this clip at `CHECK_EVERY=10`,
people move a median **30px** between checkpoints (p75 55, p90 93): 57% cross a bin
edge and 28% move more than a whole bin. The rule therefore admitted stationary
people and rejected running ones. The counter also never expired despite the
constant being named `MIN_CONFIRM_CHECKPOINTS`, so detection / gap / detection
confirmed a player who was never continuously present.

Both fixed in `a989e9f` by following candidates on centre distance scaled by box
height. IoU was tried first and is wrong here: a player box is ~40px wide, so a
30px sideways step -- the median -- drops IoU to 0.14.

### What the clip still shows, after both fixes

- **Fragmentation is the dominant remaining failure**: 10 of 18 players fragmented,
  worst into 4 pieces, 35 tracker ids for 18 identities.
- **Duplicate numbers persist**: `25` resolves on three player_ids, `15` and `18` on
  two each -- 13 resolved ids carrying only 9 distinct numbers. This is the
  evidence the number-anchored merge (`NumberVoter.merge()`, already written) was
  waiting for. It needs per-player team in the render output as a guard, since two
  players on opposite teams may legitimately share a number.
- **Bench players are still tracked**, correctly -- they are people. Excluding them
  needs a real court test; `court_test_fn` in `drive_sam2` is currently
  `lambda box: True`. A colour-based court mask was tried and **abandoned**: the
  bench sits at the court edge with the crowd directly behind it, so a floor-colour
  mask cannot separate them (it excluded 8% of person detections and none of the
  bench). The workable route is the homography in `scripts/run_court_mapping.py` --
  map the foot point to court coordinates and test the 40x20m rectangle.

  **That homography was fitted to a scrambled landmark correspondence until
  2026-09-11.** The keypoint model's slot order and the `sports` court
  template's vertex order are different orderings of the same 37 landmarks, and
  nothing translated between them, so every court coordinate this repository has
  produced is void. `handball_cv.court.keypoints.KEYPOINT_TO_VERTEX` is the
  missing translation; it takes the homography fit residual over the 892
  labelled images from 922 cm to 33 cm. See the TODO entry for the measurement
  and for the two blockers that remain (the keypoint model does not load under
  the pinned `inference`, and `ViewTransformer` has no RANSAC, no residual check
  and no way to abstain).

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
- `data/raw/Han-Ber4_cached.mp4`: 199-frame short clip.
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

- `src/handball_cv/teams/model.py`: crop geometry, quality, color/visual model, prediction, persistence.
- `src/handball_cv/tracking/identity.py`: stable player IDs, re-ID, temporal observations, reversible switching.
- `scripts/render_raw_team_classification.py`: raw frame-local baseline.
- `scripts/render_mcbyte_team_correction.py`: plain MCByte plus reversible tracked team overlay, dense per-player diagnostic text (team/confidence/obs/switches).
- `scripts/render_team_overlay.py`: same MCByte/IdentityManager pipeline, clean broadcast-style overlay (translucent team-colored mask fill + box border, small legend, no per-player text) matching the original notebook's "Full video team clustering" cell style. Masks are pulled directly from MCByte's `tracklet_mask_dict` for visualization only, not through the guarded spatial-assignment checks in `teams/masks.py` (an occasional wrong mask here is cosmetic, not a label error).
- `src/handball_cv/teams/masks.py`: guarded spatial mask assignment and safe-pixel construction.
- `scripts/render_mask_team_comparison.py`: isolated box-versus-mask experiment and metrics.
- `experiments/team_gated_tracking/tracker.py`: experimental team-gated association; do not enable by default.
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

The scoped project suite currently passes: 226 tests (measured 2026-09-09). Do not run bare `pytest` from repository root because vendored `onnxruntime` and other upstream trees contain unrelated test entry points that break global collection.

Render Felix mask comparison:

```bash
conda run -n NewEnv python -m scripts.render_mask_team_comparison \
  data/raw/FelixClaar.mp4 \
  --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \
  --team-model outputs/team_comparison/.FelixClaar_team.pkl \
  --manifest outputs/team_dataset/FelixClaar/manifest.json \
  --output outputs/mask_team_comparison/FelixClaar_box_vs_mask.mp4 \
  --device cuda
```

Render Han-Ber with independent reference scoring:

```bash
conda run -n NewEnv python -m scripts.render_mask_team_comparison \
  data/raw/Han-Ber4_cached.mp4 \
  --detections outputs/team_dataset/.Han-Ber4_detections_v1.npz \
  --team-model outputs/team_correction_mcbyte/.Han-Ber4_team.pkl \
  --manifest outputs/team_dataset/Han-Ber4/manifest.json \
  --output outputs/mask_team_comparison/Han-Ber4_box_vs_mask.mp4 \
  --device cuda \
  --reference-mask-dir source/.Han-Ber4_sam2_masks \
  --reference-track-labels outputs/.Han-Ber4_track_team_labels.json \
  --reference-min-iou 0.65
```

## Re-ID cannot tell teammates apart, and what was built because of it

Watching the 60s Melsungen overlay surfaced two complaints: jersey `15` moved to
a different player after its wearer left frame, and later two players on the same
team both wore `15`. Both trace to one mechanism.

`PlayerRegistry.reid_match` picks the retired player with the highest cosine
similarity above `REID_COS_SIM_MIN`. Team and goalkeeper role can veto, but
within a team both are silent -- every outfield player wears the same kit -- so
the appearance embedding is the entire discriminator.

### Measurement

`scripts/measure_reid_discriminability.py` scores that embedding against the
labelled 1080p set: two readable crops in the same clip carrying the same number
are the same person, different numbers are different people. Each labelled number
box is traced to the player box containing it (the containment rule the set was
built with) and embedded exactly as the runtime embeds it. Rank-1 retrieval is
the operation `reid_match` performs.

```
clip                     crops  players   same   other   >0.7   rank1  chance
Melsungen_Berlin            55       17  0.802   0.822   0.98    0.19    0.10
Kiel_Lemgo                  54       18  0.829   0.825   0.96    0.26    0.11
Eisenach_Hamburg            56       17  0.829   0.829   0.91    0.06    0.11
BHC-FAG                     61        9  0.801   0.822   0.95    0.55    0.21
FelixClaar                  41        8  0.862   0.845   1.00    0.89    0.50

overall rank-1 within team: 90/243 = 0.37   (chance 0.19)
```

**Different-person same-team pairs are as similar as same-person pairs** -- 0.822
vs 0.802 on Melsungen -- and 91-100% of them clear the 0.7 floor, so the
threshold rejects nothing within a team, and no threshold could: the two
distributions sit on top of each other. The two clips that look competent are the
two with 8-9 labelled players, where chance is 0.21 and 0.50.

**PRTReID, already in `models/` and consumed by nothing, is substantially
better.** Identical 256 queries and galleries:

```
clip                      team-model      prtreid       chance
Melsungen_Berlin        0.17 ( 9/52)   0.27 (14/52)      0.06
Kiel_Lemgo              0.26 (13/50)   0.42 (21/50)      0.07
Eisenach_Hamburg        0.04 ( 2/54)   0.33 (18/54)      0.06
BHC-FAG                 0.53 (32/60)   0.87 (52/60)      0.18
FelixClaar              0.85 (34/40)   0.90 (36/40)      0.27
OVERALL                 0.35 (90/256)  0.55 (141/256)
```

Two caveats on these figures. The measurement embeds a **single crop** on both
sides, while `TrackManager` refreshes a live track's embedding with an EMA
(`EMBEDDING_EMA_ALPHA = 0.3`), so the runtime *gallery* is smoother than measured
-- the query side is still a single crop. And the measured gallery is ~25
same-team crops against a runtime gallery of retired players within 300 frames,
usually far fewer. Both make the absolute numbers pessimistic; neither changes
the ordering, which holds on every clip.

### The margin sweep, and why the floor was replaced

Pooled over five clips, for a rule that requires the winner to beat the runner-up
by delta (PRTReID):

```
 delta   match rate   precision   wrong match when the player is NEW
  0.00         1.00        0.59                                 1.00
  0.01         0.46        0.66                                 0.41
  0.02         0.23        0.78                                 0.14
  0.03         0.11        0.86                                 0.04
  0.05         0.02        1.00                                 0.01
```

The `delta = 0` row is what shipped. **Re-ID always claims a match, including for
players it has never seen** -- there is no path through `reid_match` that says
"this is somebody new". That is the renaming mechanism, stated exactly.

`REID_MARGIN_MIN = 0.02` is the chosen point. The two errors it trades between
are not equal: a fragment is repairable from number evidence afterwards, while a
wrong revival silently contaminates a vote tally for the rest of the clip.

### Number evidence, applied where it can act

At the instant of re-ID the returning track is a brand-new tracker id with **zero
reads**, so the number cannot gate that decision -- it takes a few reads for
`NumberVoter` to qualify a value. `NumberIdentityResolver`
(`scripts/render_full_pipeline.py`) is where the later evidence acts, in two
steps that need the same two facts:

1. **link** -- same team, same resolved number, never simultaneously live. That
   is one player the tracker split; `PlayerRegistry.link` aliases the ids and
   `NumberVoter.merge` folds the tallies. An alias, not a rewrite: SAM2's
   `obj_id` is a live handle into its predictor session and cannot be renumbered,
   and an alias stays reversible.
2. **arbitrate** -- same team, same resolved number, but they *were* on screen
   together. That cannot be one player, so `NumberVoter.arbitrate` withholds the
   weaker claim. It reports no number rather than a wrong one, per the standing
   rule that abstaining beats injecting a confident wrong observation.

Order matters: linking first means arbitration only ever sees duplicates that
genuinely coexisted.

**Arbitration is not a repair.** The losing identity is still whoever the tracker
mistakenly grabbed; it is now unlabelled instead of mislabelled.

The 60s clip is the worked example. Three identities resolved to `25`, and two
pairs of them were read in the *same frame* -- so they are different people, and
a number-anchored merge without the simultaneity guard would have welded them
together. `15` and `18` sat on two identities each with no shared read frame.

### A verdict cannot outlive the fragment that earned it

The first number-aware run removed every same-team duplicate, but the overlay
still showed a *different* player wearing 15 after the real one left frame. The
mechanism is one step upstream of the duplicate: a number is evidence about a
person, filed against an identity, and re-ID moves an identity onto whoever it
believes reappeared. The verdict travels with it, and nothing told the voter.

```
p6   re-ID revivals at frames [560, 1390]
   f   65..110   read 15  x9        -> settles on 15
   f  120..340   read 5/1/11/6      -> held; no rival qualifies
   f  865        read 20            <- after revival @560
   f  870        read 29            <- after revival @560
   f 1060        read 2             <- after revival @560
```

Every read after frame 560 comes off a different shirt. The label stayed `15`
because the hysteresis in `_verdict` holds a value until a *rival* qualifies --
three votes past a 0.35 margin -- and `29` got one. That is why the wrong label
was stable rather than flickering, and why it only came off when the real 15 was
read again and arbitration stripped the loser. Arbitration was cleaning up after
a defect one layer down.

`NumberVoter.suspend`, called on every revival by `NumberIdentityResolver`,
withholds the inherited verdict until a read from the new fragment backs it --
including a single-digit partial, since the fold rule already treats "5" as
evidence for "15". Two contradicting reads instead disown the tally outright.

Keeping the votes was the first attempt, and a test caught why it fails: nine
stale votes stay in the denominator, so the new player needs ~25 reads to clear
`min_margin` against evidence about somebody else. One contradiction is a misread
at these crop sizes (the readers run ~0.6 accurate); two is a different shirt.

`claim` is split from `best` for the same reason: on this clip the inherited 15
carried more folded votes (13) than the real one (17 including folds, but only 6
raw), so an unvouched verdict left in the contest would have suppressed the
correct player.

Replaying p6's actual read sequence through the fixed voter: `15` through frame
340, then no label at all from the first post-revival read onward.

### Output added for the guards

`identity_report` (shared by both tracker paths) now emits `player_teams`,
`frame_players`, `player_aliases`, and a `number_row`/`box` on every read. Before
this, simultaneity could only be approximated by read co-occurrence, which is far
too sparse to conclude from, and reads could not be traced back to the crop that
produced them.

### The reader was never the domain's fault -- it was docTR's weights

Three docTR recognisers converged near 0.62 on the labelled 1080p crops, shared a
~6% floor of confidently-wrong reads, and reached an oracle of only 0.71 between
them. That pattern says "the domain beats this model family", and the conclusion
drawn from it -- collect handball data and fine-tune -- was wrong.

docTR trains its own recognisers, largely on document text. The **original
`baudm/parseq` checkpoint**, the same architecture trained on scene-text
benchmarks, scores on the identical crops, labels and scoring function:

```
reader                     accuracy   selective   abstention   ms/crop
docTR parseq (shipped)        0.622       0.817         0.88       2.0
baudm parseq (original)       0.858       0.958         0.89       1.6
baudm parseq (hockey FT)      0.851       0.893         0.49       0.9
baudm parseq (SoccerNet FT)   0.266       0.336         0.39       1.5
```

Paired McNemar, docTR vs baudm-original: **b=4, c=80, p=2e-19**. Eighty crops
docTR misses that this reads, four the other way, at the same speed and with
better abstention.

The errors it removes are exactly the ones that were driving pipeline failures:
`22` read as `2`, `17`/`10`/`11`/`21` abstained on, `7` read as `1`, `54` as
`51`. Those were the "digit loss" failures the crop-padding sweep chased and
could not fix -- because the crops were fine and the reader was not.

**Fine-tuning on another sport does not help.** The same repository's hockey
weights match the original on accuracy but collapse its abstention (0.49 vs
0.89), and its SoccerNet weights fall to 0.266 -- adapting to SoccerNet's tiny
blurred tracklet thumbnails costs more than the jersey-domain match buys.

Verified end to end on the failure that prompted this. Player 20 wears 15;
docTR read `6` three times in a row, which cleared `min_votes=3` at margin 1.0
and put `#6` on screen. Re-reading the same 24 crops:

```
    frame   docTR   baudm
      875       6       6      genuinely ambiguous -- both say 6
      880       6      15      conf 0.99
      885       6      15      conf 0.85
      890       5      15      conf 0.99
      ...
    correct   6/24   15/24
```

The first four reads become `6, 15, 15, 15`, so `6` never reaches three votes
and the wrong label never appears.

Two consequences worth stating plainly:

- The crop-padding sweep and the ensemble analysis both measured a weak reader,
  not a hard domain. Their negative results stand for docTR parseq and say
  nothing about this one; they should be re-run if they matter.
- **No new footage is needed to fix reading.** The data question stays open only
  for whatever comes after 0.858.

LICENCE: the checkpoints come from `mkoshkina/jersey-number-pipeline` (CC BY-NC
3.0). The *original* parseq checkpoint it redistributes is Apache-2.0 upstream
(`baudm/parseq`) and is the one wired in; the hockey and SoccerNet fine-tunes
are the NC-licensed ones and are measured here but not used.

## Repository-state warning

The working tree contains many pre-existing modified and untracked experiment files. They belong to the ongoing project. Do not run destructive cleanup, reset, or checkout commands. Work only on the requested files and preserve unrelated changes.
