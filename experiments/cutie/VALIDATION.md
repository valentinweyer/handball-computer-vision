# Full Cutie-480 validation — 2026-10-16

## Verdict — 2026-10-16: closed, SAM2 stays the tracker

The user reviewed Cutie-480 by hand on two clips that no tracker had been run or
tuned on — Hannover (999 frames, Roboflow detections) and
VfL_GM_Loewen_30m00_60s (1500 frames, local v4 detections, team model fitted
like the overnight batch) — and judged its identity keeping worse than SAM2's
and not good enough to adopt. The 3x speed advantage does not compensate.

The scored results below stay as recorded; they did not show this, partly
because Felix and BHC references were built with SAM+Cutie and Han's with SAM2
(see "Evaluation boundaries"). Of Cutie's 30 sub-0.5 Han observations, 27 had a
box on the player at IoU 0.1–0.5, i.e. outline rather than identity.

Review videos: `runs/cutie_review_20261016/{Hannover,VfL_GM_Loewen_30m00_60s}`
(+ `_sam2` for the same overlay from production SAM2), rendered with
`render_review.py --backend cutie|sam2`. VfL frames ~610–640 are a crowd pan
with no players; the empty-frame crash this exposed in `masks_from_logits` is
fixed on main (ce8da56) and included here.


All six GPU runs completed: Cutie-480 and fresh SAM2.1-L on full FelixClaar,
Han-Ber4 and the BHC-FAG validation window. There were no model, threshold,
resolution or lifecycle changes between clips. The candidate is identical to
that selected on the earlier 141-frame Felix screen.

Cutie was 3.09–3.25x faster across these clips. It scored higher on Felix and BHC,
but lower on Han, so **a no-loss replacement across every clip is not established**.

| Clip | Frames | Cutie correct identity | SAM2 correct identity | Cutie FPS | SAM2 FPS | Quality difference |
|---|---:|---:|---:|---:|---:|---:|
| FelixClaar | 249 | 98.09% | 95.01% | 8.31 | 2.62 | +3.08 pp |
| Han-Ber4 | 199 | 98.31% | 99.91% | 7.69 | 2.36 | -1.60 pp |
| BHC-FAG window | 500 | 99.45% | 98.88% | 7.62 | 2.46 | +0.57 pp |

FPS is total tracking FPS, including fresh-state setup, masks, propagation,
corrections, checkpoint appearance work and CPU post-processing. Model loading,
12-frame warmup and artifact writing are excluded. Detection is cached RF-DETR;
live detection, OCR and rendering are excluded.

## Coverage, identity errors and fragmentation

Each pair below is **Cutie / SAM2**. All percentages are per-clip; they are not
averaged to hide a regression on an individual clip.

| Clip | Coverage | Wrong identity observations | Missing/unmatched observations | ID switches | Mean track IDs per reference player |
|---|---:|---:|---:|---:|---:|
| FelixClaar | 98.25% / 95.76% | 4 / 18 | 42 / 102 | 5 / 8 | 1.23 / 1.38 |
| Han-Ber4 | 98.31% / 99.91% | 0 / 0 | 39 / 2 | 0 / 0 | 1.00 / 1.00 |
| BHC-FAG window | 99.57% / 98.99% | 6 / 6 | 22 / 51 | 5 / 6 | 1.33 / 1.42 |

Han's 1.60-point deficit is entirely coverage in this matcher: both methods have
zero measured identity switches, zero wrong-identity observations and one
predicted ID per reference player. Cutie leaves 39 reference observations
unmatched versus SAM2's 2. An unmatched observation can mean a missing mask or
insufficient box overlap at IoU 0.5; these metrics alone do not distinguish the
two or prove that an ID was physically lost. This is a coverage regression under
the existing metric, and it must not be dismissed to claim equal quality.

Felix and BHC show higher coverage and fewer measured switches for Cutie. The
full Felix result also improves on SAM2 beyond the initial 141-frame screen.

## Evaluation boundaries

The unchanged repository matcher uses greedy one-to-one IoU>=0.5 box matching,
then a dominant reference identity per predicted track. Correct identity divides
matched-and-correct observations by **all trusted player observations**, counting
misses against the score. Fragmentation is reported separately because a clean
new ID can escape the dominant-identity error metric.

Han excludes verified reference IDs 12 (standing bench player) and 14 (referee)
from player scoring and the denominator. All other trusted player spans remain.

The references are model-derived and human-verified, not dense independent
manual ground truth. Han was generated with SAM2 and may favour SAM2; Felix and
BHC were generated with SAM+Cutie and may favour Cutie. Brief swaps missed by
verification, mask geometry and complete-occlusion identity remain limitations.
These scores are not HOTA/IDF1 or proof of general equivalence.

## Reproduction and artifacts

The runner now accepts `--case felix|han|bhc`. Use the matching full prefix:
249 Felix, 199 Han, 500 BHC. Keep `--cutie-size 480` and `--warmup-frames 12`.
Run Cutie and SAM2 in separate fresh processes, serially on the GPU, with new
output directories. The CPU scorer selects the matching reference and excluded
IDs from `cases.py`.

Example pair (use new output paths when repeating):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.run_comparison \
  --case han --backend cutie --cutie-size 480 --max-frames 199 \
  --warmup-frames 12 --output runs/cutie_validation_20261016/han_cutie_480
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.run_comparison \
  --case han --backend sam2 --max-frames 199 \
  --warmup-frames 12 --output runs/cutie_validation_20261016/han_sam2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.cutie.score \
  --cutie runs/cutie_validation_20261016/han_cutie_480 \
  --sam2 runs/cutie_validation_20261016/han_sam2
```

Artifacts are under `runs/cutie_validation_20261016/`: per-run manifests,
`replay.npz`, events, frame timings, per-pair `identity_score.json`, logs,
`plan.json`, `summary.json` and `verification.json`.

Verification confirmed matching video/detection/team/source hashes in each pair,
exactly frozen Cutie settings and weights, and every frame processed without
skipping. After extending case selection, the earlier 141-frame Felix metrics
reproduced exactly; Han's bench/referee exclusion was checked explicitly. Syntax
and focused whitespace checks passed. Model loading and native lifecycle checks
from setup remain applicable because the model/adapter were not changed.

Work remains isolated on branch `experiment/cutie` in
`/home/valentinweyer/projects/handball-computer-vision-cutie`.
