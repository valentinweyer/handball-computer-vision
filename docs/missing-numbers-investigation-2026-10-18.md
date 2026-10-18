# Missing jersey numbers: investigation, 2026-10-18

The offline number-label replay fix is now implemented on
`investigate/missing-jersey-numbers`. The OCR size gate remains unchanged;
removing it outright would admit confident errors.

## Implementation and validation

Offline table assembly now prefers a stable source-track verdict only when
that track's own recorded OCR reads independently earn the same number under
the existing vote rules. Support is scoped to the track's identity-break
segment. A partial trailing digit cannot replace the full family number.
Unresolved/revised local segments retain the previous family backfill, and
legacy runs without explicit `track_id` read provenance retain prior behavior.
The duplicate-number guard still runs after selecting these labels.

On the complete saved v4 run:

- **1,284 additional labelled player-frame observations**, with **zero
  previously displayed labels removed** on this run.
- **1,574 existing numbers changed**: track 71's earlier incorrect `4` becomes
  `3`, and track 161's incorrect `3` becomes `4`. Representative crops confirm
  the printed jersey numbers; this is not a new whole-video accuracy benchmark.
- Both #3 and #4 are labelled throughout the 520-frame collision episode.
- All other track columns, including `number_live`, IDs, teams, boxes, court
  positions, and mask references, compare identically to the original.
- No same-team duplicate field-player numbers are introduced. All original
  bundle files retain their SHA-256 hashes.
- **549 tests pass**. Seven new regressions cover later links, absent legacy
  provenance, genuine duplicates, trailing digits, revised early reads,
  revival boundaries, and insufficient source evidence. The main regression
  fails against the pre-fix implementation.

The corrected bundle is saved separately at
`runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4_backfill_fix`.
Validation metrics and a 22-second preview of the #3/#4 case are under
`runs/number-backfill-fix-2026-10-18/`. The original ten-minute video is
preserved. Detection, tracking, OCR inference, and their caches were reused.

The initial investigation below describes the unmodified v4 result.

## Evidence and limits

Branch: `investigate/missing-jersey-numbers`, based on `main` at `20741a0`.
Examined the complete 15,000-frame Kiel–Lemgo v4 bundle, extracted at
`39eb1b3`. The relevant detection/OCR/voting/table code is unchanged between
that extraction and this investigation's base.

- Reconstructed all **6,658 OCR attempts**, including empty reader outputs,
  and matched **all 4,384 accepted reads** to their original frame, track and
  number-box row. Both checks require exact set equality.
- Inspected **30 deliberately selected missing-number episodes**, three frames
  each, plus six labelled controls. This is an assistant visual review, not a
  random sample or an independently human-labelled accuracy benchmark. Two
  selected episodes are duplicate tracks on the same person.
- Reproduced **all 2,136 pre-OCR crop exclusions** from the original pixels.
  Probed those crops with the existing FP32 PARSeq model and unchanged 0.5
  confidence/valid-number gate, in separate audit outputs.
- Original bundle tables and cached reads were not overwritten. The audit
  records SHA-256 fingerprints for the input tables.

The bundle has 139,478 player-frame rows, 46,387 without a displayed number.
Some rows have empty tracking boxes: among 136,864 nonempty boxes, 45,202 lack
a number. These are observation counts, not counts of distinct real players.

At the five-frame OCR cadence, 9,271 observations lack a final displayed
number. Their immediate evidence stages are:

| Recorded stage | Observations |
| --- | ---: |
| No number detection centred inside the player's box | 7,752 |
| Matched number rejected by crop gate | 756 |
| OCR attempted but abstained | 346 |
| Nearby detection below mask-overlap threshold | 204 |
| Read accepted, final number still absent | 172 |
| Ambiguous mask owner | 36 |
| One-to-one match conflict | 5 |

These stages do not establish digit visibility or detection recall. A person
facing away from the camera, an obscured number, a small front number, an empty
mask, and an actual detector miss can all contribute to the first category.

## 1. Confirmed offline replay error: real #3 disappears

At **5:44.04–6:04.80** (frames 8601–9120), blue #3 is track/player 113.
It receives **86 accepted reads of `3`** during the missing-label episode.
Visual review confirms the printed number. Its causal label is already `3`
at frame 8670, yet its final, backfilled `number` is missing.

At frame 8850, a second blue player, **track 114, visibly #4**, is also assigned
prospective `3` by the final folded replay. The duplicate-number guard then
correctly withholds both proposed `3` labels. The wrong proposal is upstream.

`_segments_by_player(..., fold_links=True)` maps every historical frame and
read through the run's **final** alias map before replaying the votes.
Track 114 links to player 23 at frame 8935; player 23 only links into player
71 at frame **12915**. Nevertheless, that later fold is applied to track
114's earlier frames. The resulting player-71 segment spans frames
2571–12660 and holds `3` from frame 5635, including when the actual current
member is #4. Reads of `4` on track 114 do not undo that inherited result.

The genuine #3 therefore loses **520 frames / 20.8 seconds**. Across the whole
bundle, 1,280 rows lose otherwise proposed numbers at the final duplicate
guard; that total is **not** a proven recoverable gain. The guard also handles
real conflicts and should remain enabled.

Relevant code at investigation commit `8d9ee56`:

- `src/handball_cv/pipeline/tables.py:44` — final-alias folded replay.
- `src/handball_cv/pipeline/tables.py:96` and `:156` — duplicate suppression.
- `src/handball_cv/jersey/identity.py:694` — segment backfill behavior.

Identified patch target: preserve the timing and source-fragment ownership of number
evidence when assembling offline labels. Backfill within a supported physical
track/revival segment; a later link must not transplant an older fragment's
verdict onto an earlier, independently readable different shirt. Preserve
identity-break suspension and per-team duplicate abstention. Do not replace
all backfilled labels with causal labels, since that discards useful existing
backfill.

Regression evidence must cover this #3/#4 case, a valid later link that should
retain backfill, real duplicate-number conflicts, and re-ID breaks.

## 2. Confirmed useful evidence lost at the crop-size gate

`readable_number_crop()` reuses the team torso `crop_quality()` predicate.
That predicate requires an area of **400 pixels**, despite digit crops often
being narrower than useful team-colour crops.

Of the 2,136 matched crops rejected before OCR:

- 2,134 fail **only** the area rule.
- Two also fail the 10-pixel minimum side.
- Every rejected crop passes the contrast and sharpness thresholds.
- Areas range from 168 to 399 pixels; the median is 339.

The unchanged reader produces **818 valid, confidence-qualified outputs** on
these rejected crops; **235** occur where the video lacks a number. These are
candidate reads, not 818 verified correct reads or a measured coverage gain.

Useful examples include:

- Track 162, white **#91**, currently unlabelled for 95.6 seconds: the probe
  produces 28 additional `91` reads. At frame 11830 / 7:53.2, the rejected
  crop is **19×17 pixels** and reads `91` at confidence 0.997.
- Track 175, blue **#4**: 17 additional `4` reads. A **17×18** crop at frame
  12925 reads `4` at 0.996.
- Track 36, white **#91**: 11 additional `91` reads.

There are also counterexamples. Blue **#28** on track 182 repeatedly produces
`20` on the small crops: six extra qualified outputs, including confidence
**0.932** at frame 12935. White #91 on track 94 also produces repeated `97`.
High confidence alone does not make these new reads safe.

Relevant code: `scripts/render_full_pipeline.py:341` and
`src/handball_cv/teams/model.py:86`.

Next experiment: evaluate a separate, number-specific crop gate in shadow
mode on a labelled set containing both the useful small crops and these
counterexamples. Keep the current confidence, ownership, voting, and identity
safeguards. Do not simply lower the shared team-crop threshold or insert all
probe outputs into live voting.

## 3. Smaller findings

Most accepted-but-unresolved examples outside the #3 collision have only one
or two agreeing reads, or conflicting values. For example #21 has `21,21,11`.
That does not justify lowering the three-vote minimum or the margin threshold.

At **9:41.0–9:41.6**, tracks 190 and 196 cover the **same white #61**.
Their mask IoUs are 0.980, 0.968 and 0.961 at sampled frames 14525, 14535 and
14540. Number ownership is ambiguous because tracking has duplicated the
person; the new ambiguity guard is correctly abstaining. This is a separate
tracking-duplicate investigation, not a reason to restore row-order matching.

Goalkeeper #1 often faces the camera with a small, low-contrast front number
and very few accepted reads. The sample confirms both poor visibility and
some apparent detector opportunities. Measuring that detector loss needs
manually marked visible-number boxes; the current cache cannot measure recall
against detections it never emitted.

## Order of work

1. Correct offline replay ownership/timing, keeping the existing safeguards
   (implemented and validated above).
2. Test a number-specific crop gate in shadow mode with correct and incorrect
   small-crop controls before changing runtime defaults.
3. Investigate duplicated tracks and measure visible-number detector misses.

A larger appearance-ReID project or more training data is not yet the most
direct response to these demonstrated mechanisms.

## Reproduction and artifacts

On Spark, repository `/home/valentinweyer/projects/handball-computer-vision`:

```bash
.venv/bin/python -m scripts.audit_missing_numbers \
  --bundle runs/bundles/Kiel_Lemgo_66m00_10min_expanded_validation_v4 \
  --work data/interim/Kiel_Lemgo_66m00_10min_expanded_validation_v4 \
  --output runs/missing-numbers-audit-recheck
```

The output directory must be new. The original audit is at
`runs/missing-numbers-audit-2026-10-18/`: opportunity and number-box Parquet
tables, input hashes, missing episodes, final duplicate-guard losses, visual
sample selection, and crop-probe outputs. The exact pixel/OCR probe script is
preserved with the investigation under `experiments/missing_numbers/probe.py`.

Local review sheets `review-1.jpg` through `review-5.jpg` contain the 30
episodes; `controls.jpg` contains six controls. `rejected-crop-examples.jpg`
shows 24 exact digit crops. `small-crops-benefit-and-risk.jpg` places useful
and misleading probe reads side by side.
