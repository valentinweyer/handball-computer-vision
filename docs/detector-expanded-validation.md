# Expanded detector validation branch

`feat/detector-expanded-validation` starts at pipeline commit `37a37a3`.
It includes the 15 pipeline commits beyond `origin/main`, plus the changes below.
The branch is a validation candidate; it has not been merged into `main`.

## Default detector

Fresh `scripts.process_video` runs use the upgraded export in the sibling
fine-tuning repository:

- Export: `exports/detector-expanded-v1/rfdetr-large-handball.ts`
- Matching checkpoint: `runs/detector-retrain-2026-10-17/expanded/checkpoint_best_ema.pth`

All 510 state tensors in the export match that EMA checkpoint exactly. The
expanded training set added 295 approved frames. The weights, datasets and
rendered videos remain ignored artifacts in the fine-tuning/pipeline checkouts;
Git contains their consuming code and configuration, not the large artifacts.

Existing stage caches are reused. To validate the new detector and tracking
changes together, use a fresh work directory and output bundle, for example:

```bash
.venv/bin/python -m scripts.process_video data/raw/Kiel_Lemgo_66m00_10min.mp4 \
  --work data/interim/Kiel_Lemgo_expanded_validation \
  --out runs/bundles/Kiel_Lemgo_expanded_validation
.venv/bin/python -m scripts.render_result runs/bundles/Kiel_Lemgo_expanded_validation
```

For an explicit old-detector comparison, pass
`--detector ../rfdetr-handball-finetune/exports/detector-local/rfdetr-large-handball.ts`
and use separate work/output directories. Neither command needs to overwrite
existing run artifacts.

## Number ownership follows visibility

Only currently visible identities compete over a same-team jersey number.
Arbitration refreshes every frame, including between OCR frames. Historical
co-visibility still prevents unsafe identity linking, and two visible teammates
claiming the same number still trigger suppression. Live processing and saved-run
label replay apply the same visibility rule.

The Kiel checkpoint replay now labels P13 as 41 at 2:12. Its rendered bundle is
`runs/bundles/Kiel_Lemgo_66m00_10min_expanded_number_visibility_v2`. The 15,000-frame
render was checked at 2:12; geometry and OCR reads are unchanged. Number-label
coverage changed from 58.0% to 65.8%; this is coverage, not measured accuracy.

## Conflicting role detections

Before SAM2 seeding and detector checkpoints, a Player/Goalkeeper pair with box
IoU at least 0.95 keeps the higher-confidence detection. Surviving rows retain
their order. Same-role detections, ordinary player overlaps and raw detection
caches are unchanged.

The exact cached boxes and masks at Kiel frames 3530 and 3540 were replayed through
TrackManager with controlled appearance observations. Unfiltered inputs created
one extra track; filtered inputs created none. The shared-driver tests also cover
filtering at both seeding and subsequent checkpoints. These checks do not replace
a full SAM2 rerun.

The watchable number-visibility-v2 video still uses the old tracking pass and does
not contain this role-duplicate fix. A new tracking pass is needed to validate
its full-video effect.

## Validation

The focused lifecycle, team, re-ID and number suite passed 89 tests before
branch preparation. The full scoped repository suite (`.venv/bin/python -m pytest -q tests`)
passed 505 tests. The pipeline CLI and both default model paths were checked.
