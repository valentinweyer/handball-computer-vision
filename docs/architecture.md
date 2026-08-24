# Architecture

## Dependency direction

```text
frame + detections ────────> team observation
        │
        └──────────────────> tracker

team observation + tracker output ──> temporal identity manager ──> overlay
```

Team classification is tracker-independent by default. It consumes image crops,
detection geometry, and optionally a supplied mask. The team package does not
import MCByte or any tracker. This keeps raw classification measurable without
letting tracking errors define team labels.

MCByte owns short-lived track IDs. `IdentityManager` combines those IDs with
qualified, reversible team observations and optional appearance evidence. A
tracked team label may switch after sustained contradictory evidence; it is not
permanently frozen from an early crop.

Team-aware association is deliberately isolated under
`experiments/team_gated_tracking/`. It is not enabled in the default
configuration because uncertain classification and association can amplify one
another.

## Package boundaries

- `handball_cv.detection`: detector adapters and cached inference.
- `handball_cv.teams`: crop geometry, features, per-video discovery, masks,
  calibration, dataset contracts, and evaluation.
- `handball_cv.tracking`: tracker-independent identity management, mask caching,
  and the retained SAM2 lifecycle manager.
- `handball_cv.embeddings`: optional appearance-model adapters.
- `handball_cv.jersey`: jersey-number evidence.
- `scripts`: thin labeling, evaluation, rendering, and comparison tools.
- `experiments`: non-default baselines and rejected/uncertain approaches.

## Artifact boundaries

- Human labels: `data/annotations/` and tracked.
- Raw video: `data/raw/` and ignored.
- Reproducible caches: `data/cache/` and ignored.
- Checkpoints: `models/` and ignored.
- Generated results: `runs/` and ignored.

The old `notebooks/*.py`, `outputs/`, and `source/` layouts remain intact until
the new package and commands have been verified. They are compatibility copies,
not the canonical location for new work.
