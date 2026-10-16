# Cutie comparison — 2026-10-16

Standalone Cutie-base-mega completed the same 141-frame FelixClaar prefix on
NVIDIA GB10. Its 480-pixel setting is a promising faster candidate on this clip.

| Tracker | Total tracking FPS | Correct identity | Coverage | ID switches |
|---|---:|---:|---:|---:|
| SAM2.1-L | 2.51 | 96.84% | 97.45% | 7 |
| Cutie, 1024 short edge | 1.60 | 97.51% | 97.69% | 4 |
| Cutie, 480 short edge | 8.25 | 98.91% | 99.03% | 4 |

Cutie-480 was **3.28x faster**, with **2.07 percentage points higher**
correct-identity score on this reference. It had 2 wrong-identity observations
and 16 missing observations, versus SAM2's 10 and
42. The 480 setting also outscored 1024 on this clip;
higher mask resolution did not produce a higher score here.

CAMEL's earlier replay, rescored on the same frames, scored
83.70% correct identity, 90.09% coverage
and 26 ID switches. Its earlier 5.47 FPS timing used a different
warmup/lifecycle and FP32 appearance/pose extraction; it is not the paired timing
protocol used for the Cutie/SAM2 table above.

## What was held fixed

- Exact RF-DETR detection cache and source-video hashes, frame prefix 0–140,
  team-model cache, production TrackManager/PlayerRegistry, checkpoint interval
  10 and existing CPU mask post-processing.
- BF16 autocast, 12-frame warmup in a fresh state, eager uncompiled models.
- GPU models loaded strictly with no missing or unexpected tensors.
- No training or local threshold tuning. The two Cutie resolutions were
  exploratory settings, not a held-out model-selection evaluation.

SAM2.1-L's image head generated Cutie's initial/correction masks; all temporal
propagation used Cutie. Native generic-video long-term memory was retained.
Small, recorded compatibility fixes support partial corrections and safe object
removal without clearing surviving memory; see README and upstream-delete.patch.
This is the adapted Cutie pipeline rather than untouched upstream inference.

## Timing scope

Total tracking FPS includes fresh-state initialization, seeds, all propagation,
corrections, checkpoint appearance work, CPU mask processing and replay collection.
Model loading, warmup and artifact compression/writing are excluded. All runs use
cached detections; live RF-DETR, OCR and downstream rendering are excluded.
Cutie's prompt batches are deferred into the first propagation step, so total
tracking FPS is the comparison authority rather than seed time alone.

| Tracker | Seed seconds | Loop seconds | Loop FPS | Peak allocated GPU GiB |
|---|---:|---:|---:|---:|
| SAM2.1-L | 0.37 | 55.70 | 2.51 | 3.63 |
| Cutie, 1024 short edge | 0.08 | 87.84 | 1.59 | 10.04 |
| Cutie, 480 short edge | 0.08 | 17.01 | 8.23 | 3.11 |

## Quality scope and next validation

All quality figures use the unchanged repository IoU>=0.5 greedy box matcher,
trusted human verification spans, and all 1,644 reference player observations as
the denominator. Missing players therefore reduce correct-identity score.
This metric is not HOTA/IDF1 and does not establish correct trajectories through
complete occlusion.

**Felix's reference was generated with SAM+Cutie.** Human verification reduces
but does not remove the risk of favouring the same propagation family, including
brief swaps missed by sampling. This result is not proof of general SAM2-level
quality or a production replacement. The next useful independent-engine check
is Han-Ber4 (SAM2-generated, verified player-only reference), followed by the full
Felix/BHC clips and dense visual review of overlap and reappearance events.
No tracker default was changed.

## Artifacts

- `runs/cutie/felix141_480/`: completed fast candidate, manifest, replay, scores.
- `runs/cutie/felix141_1024/`: completed higher-resolution candidate and scores.
- `runs/cutie/sam2_felix141/`: fresh paired SAM2 baseline and events.
- Corresponding `.log` files and `_score.log` files are beside the run folders.
- `runs/cutie/felix141/` is the earlier **failed** partial-correction diagnostic,
  not a valid benchmark. The scorer rejects incomplete runs.

Worktree: `/home/valentinweyer/projects/handball-computer-vision-cutie`;
branch `experiment/cutie`. Main, CAMEL and SAM3.1 worktrees were not modified.
