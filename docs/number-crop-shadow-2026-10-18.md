# Small-number crop gates: shadow results, 2026-10-18

**Keep the current 400-pixel gate.** Smaller crops recover useful reads, but
the tested alternatives also admit mistakes that can turn into stable wrong
numbers. No production defaults, canonical annotations, or saved results were
changed. The source-track backfill fix remains in place.

## What was compared

Six candidates were fixed before scoring:

- Current area minimum: 400 pixels.
- Smaller area minima: 350, 300, 250 and 200 pixels.
- Guarded 250: area at least 250, with OCR confidence at least 0.95 for crops
  below 400 pixels. Crops already accepted at 400 retain their 0.5 threshold.

All candidates keep the 10-pixel minimum side, contrast and sharpness gates,
reader weights, legal-number check, matching, and vote thresholds. The
baseline predicate was checked against the production predicate on every
benchmark crop. No candidate removes a currently admitted crop; any lost
number labels result from the later voting/arbitration rules.

The base code is `ecaeee9` on `investigate/missing-jersey-numbers`.

## Human-labelled crop benchmark

Reused all **660 native-resolution crops across six clips**. The stored labels
include 323 readable crops, 298 unreadable and 39 unsure; 190 are additionally
marked `not_number`. PARSeq was rerun locally in FP32 on the tight JPEG crops.
This measures the candidate gates on stored native crops, not a new detector's
recall. The Kiel probe below uses the original decoded production pixels.

Agreement with the original annotations, relative to the current gate:

| Gate | Additional correct reads | Additional wrong readable-crop reads | Additional outputs on crops marked not-number |
| --- | ---: | ---: | ---: |
| 400 current | 0 | 0 | 0 |
| 350 | 14 | 0 | 0 |
| 300 | 21 | 2 | 0 |
| 250 | 23 | 5 | 1 |
| 200 | 24 | 5 | 1 |
| Guarded 250 / 0.95 | 17 | 2 | 1 |

The current gate has 253 correct and seven wrong readable-crop outputs, with
63 abstentions. These are annotation-agreement counts on a stratified crop
set; they are not estimates of production error frequency.

### Annotation review and sensitivity

Reviewing the new disagreements against the original source frames exposed
two questionable stored annotations:

- **Index 168:** stored as `10`; the green jersey appears to be **11** in the
  source frame. OCR also says `11`.
- **Index 415:** marked `unsure` and `not_number`; the wider frame appears to
  show a red goalkeeper wearing **10**. OCR says `10`.

Neither canonical label was edited. Proposed corrections and source-frame
references are preserved in `annotation_review.json`. A separate sensitivity
analysis excludes these two cases instead of silently changing their truth.

The guarded candidate then adds **17 correct reads and one wrong read**, with
no additional output on the remaining 189 not-number crops. The remaining
error is unambiguous: **#51 → `15`, confidence 0.9877**, on a 16×18 crop.
Thus a 0.95 confidence requirement does not remove every genuine extra error.
The 350-pixel candidate remains +14 correct / zero extra crop errors in this
sensitivity check.

## Effect on final displayed numbers

Replayed every candidate against the saved Kiel v4 extraction, using the
corrected backfill code. Every original read is retained. New reads are drawn
only from the 2,136 matched crops previously rejected by the quality gate.

**Tracks, team observations, re-ID events and number-link events are frozen.**
New read owners are resolved through links at their recorded time, before
same-frame links are applied. This owner mapping reproduces **all 4,384
original read owners**. This is a controlled label replay, not a fresh
end-to-end extraction; new reads could change linking and later tracking
decisions in a real rerun.

Twelve deliberately selected sequences cover useful cases (#91 and #4), a
known #28 challenge, and labelled controls. Their numbers were checked against
source-only contact sheets, five samples per sequence, with generated labels
hidden. These are assistant-reviewed challenge labels, not a random sample
or exhaustive per-frame human ground truth. Identity is inferred between the
sampled frames within the recorded track. Very small trailing tracking boxes
are excluded using a fixed width >=20 / height >=70 visibility filter.

The twelve sequences contain 11,537 eligible player-frame observations.

| Gate | Additional correct observations | New wrong observations | Lost previously correct observations |
| --- | ---: | ---: | ---: |
| 400 current | 0 | 0 | 0 |
| 350 | 2,487 | 0 | 0 |
| 300 | 6,592 | 975 | 0 |
| 250 | 6,592 | 975 | 0 |
| 200 | 6,592 | 975 | 0 |
| Guarded 250 / 0.95 | 6,592 | 0 | 0 |

The 975 wrong observations are **track 182, actual #28, displayed as `20`**.
Repeated small-crop misreads qualify the wrong value; offline backfill then
extends it over the segment. More labelled frames alone would misleadingly
make these candidates look better.

### Additional regression found outside the initial challenge set

Inspection of *all changed existing labels*, not only the twelve selected
sequences, found another failure in the 350-pixel candidate:

- **Track 109, actual blue #28**, changes from correct `28` to `20` over
  frames 7940–8080, around **5:17.6–5:23.2**.
- Source-frame review confirms #28. There are 141 affected tracked rows,
  including **127** that pass the visibility filter.
- A recorded suspected identity switch at frame 7940 opens the segment.
  New small reads of `20` at frames 7990, 7995 and 8050 combine with the
  original accepted `20` at frame 8040; source-track voting can now earn the
  wrong value instead of preserving the correct family label.
- All four plain smaller-area candidates make this mistake. The guarded
  candidate retains correct `28` on all 127 eligible observations.

This case is saved separately as **post-hoc** evidence. It was found after
the full replay and is not presented as part of the original challenge set.

Across the full unlabelled run, the guarded candidate admits 354 extra reads,
adds 8,970 number-labelled rows and removes 105 previously labelled rows,
with no changes from one existing numeric label to another. The removed
labels are withheld during new duplicate-number conflicts. These totals do
not establish correctness outside the reviewed sequences.

## Decision and next experiment

The simple 350-pixel change passes the crop comparison but fails the temporal
regression check. Smaller plain gates also fail the original #28 challenge.
The guarded candidate is promising on the selected temporal cases, but still
admits a verified high-confidence read error on a different clip and has
unreviewed full-run changes. It is not established as a safe default.

Keep the production gate at 400. The next focused experiment should use small
reads as **corroboration for independently supported numbers**, rather than
letting several correlated small-crop reads establish or replace a number
alone. Evaluate it on fresh sequences as well as the saved regressions; the
cases above are now development evidence, not a fresh holdout.

## Verification and reproduction

- Every candidate's saved crop and challenge score was independently
  recomputed from its saved predictions and label tables.
- The baseline matches the production crop predicate. All candidates preserve
  currently accepted reads.
- No same-team duplicate field-player numbers were introduced by assembly;
  abstention still handles conflicts.
- All protected input SHA-256 hashes remain unchanged. Production code under
  `src/` and `scripts/` is unchanged from `ecaeee9`. No detector, tracking or
  OCR caches were overwritten, and no paid API was used.

From the repository on Spark, reproduce into a **new** output directory:

```bash
.venv/bin/python -m experiments.missing_numbers.shadow_gate \
  --output runs/number-crop-shadow-recheck
```

The saved experiment is `runs/number-crop-shadow-2026-10-18/`. It contains the
protocol and input hashes, all 660 crop predictions/features, per-clip human
scores, annotation sensitivity results, six complete frozen-replay label
tables, the twelve-sequence results, separate post-hoc results, and source
review sheets. Versioned scripts and review labels are under
`experiments/missing_numbers/`.

Useful review images: `challenge-review-1.jpg` through `challenge-review-3.jpg`,
`area350-adversarial-review.jpg`, and `human-label-context-review.jpg`.
