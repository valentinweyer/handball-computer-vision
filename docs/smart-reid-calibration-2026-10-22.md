# Smart re-ID: label-free per-video score calibration (exploratory, 2026-10-22)

Branch `prototype/smart-reid-shadow`, never merged. Development only: split_v2's
8 tuning matches. Code: `experiments/teammate_reid/calibrate.py`. Output:
`runs/smart-reid-shadow/calibrate/lomo.json`.

## Why

split_v2 uses one gate (τ, margin) for every video. The finetune_v1 B diagnosis
showed that similarity levels differ between videos and kits. For example,
Pforzheim's median unknown score is above Altenholz's median correct score. A
shared gate then has to sit above the highest-scoring video's impostors.

## Method (variants fixed before any result)

**Impostor level.** Teammates on court at the same time are different people:
same team, different tracks, overlapping by at least 25 frames. Their pairwise
scores, with split_v2's aggregation, give each (video, team) an impostor level.
**No labels and no numbers are used.**

| Variant | Score | Margin |
| --- | --- | --- |
| raw | split_v2 as locked | unchanged |
| shift | score − median impostor score | unchanged |
| z | (score − median) / (1.4826·MAD) | margin / (1.4826·MAD) |

**Evaluation.** split_v2's leave-one-match-out: the gate is chosen on 7 matches
and applied unchanged to the 8th. This is the shared-gate case. Halves guard on;
targets `tight` (precision 0.97, FMR 0.02) and `at_bounds` (0.95, 0.03).

## Result

556 known and 582 unknown queries.

| Variant | Folds passing | Precision | Correct | Wrong known | Unknown false accepts | Unknown FMR | Coverage (universe) | Wrong s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw, tight | 5/8 | 0.987 | 303 | 4 | 21 | 0.036 | 0.521 | 614 |
| **shift, tight** | **7/8** | 0.993 | 300 | **2** | **11** | **0.019** | 0.515 | **388** |
| raw, at bounds | 4/8 | 0.990 | 311 | 3 | 20 | 0.034 | 0.534 | 637 |
| shift, at bounds | 6/8 | 0.988 | **330** | 4 | 20 | 0.034 | **0.567** | 584 |
| z, tight | 1/8 | 1.000 | 76 | 0 | n/a | 0.029 | 0.131 | 625 |
| z, at bounds | 4/8 | 1.000 | 134 | 0 | n/a | 0.033 | 0.230 | 686 |

**Readings:**

- **The raw shared gate misses its own target on held-out matches.** At `tight`
  it aims for FMR 0.02 and reaches 0.036. Pforzheim alone is at 0.172.
- **shift at `tight` hits the target, at the same coverage.** FMR falls to 0.019
  (11 instead of 21 false accepts), wrong-known accepts fall from 4 to 2, and
  wrong seconds from 614 to 388. Correct names stay about the same (300 vs 303).
- **shift at `at_bounds`** gives +19 correct names (330 vs 311) at the same FMR.
- **z is clearly worse.** The MAD scale is noisy, with as few as 19 impostor pairs
  in a team.

**shift helps most folds, but not all:**

| Match | Raw FMR (tight) | Shift FMR (tight) |
| --- | ---: | ---: |
| Pforzheim | 0.172 | 0.000 |
| Göppingen–Leipzig | 0.086 | 0.000 |
| **Balingen–Kiel** | 0.000 | **0.086** (coverage 0.50 → 0.80) |

Balingen–Kiel breaks because its team-0 impostor level is low and widely spread
(median 0.523, scale 0.107, 35 pairs). The shift over-corrects there.

## Status

- **Exploratory and on development rows.**
- **It is not independent of split_v2.** split_v2 itself was chosen on these
  rows, so raw had a home advantage.
- **Small numbers.** The differences are 10 false accepts and 2 wrong names.
- **Not lockable from this alone.** The next step would be to lock `shift` and
  test it once on development matches that played no part here.
