# Smart re-ID: fixed tuning/test split, split v1 (2026-10-19)

Branch `prototype/smart-reid-shadow`. Offline shadow experiment: nothing in the
runtime changes. The full history is in `docs/smart-reid-shadow-2026-10-18.md`.

## Why this exists

The earlier locked tests each spent their matches, and those matches were then
folded into the development pool without asking. The 11-match encoder
comparison then saw every processed match, which left no isolated validation
footage. This split restores the separation using matches already processed:

- **Tuning:** the detector project's `train/` matches.
- **Test:** its `eval/` matches.

The split was committed (`1c5a45e`) before any selection ran on it.

| role | matches |
| --- | --- |
| tuning (8) | Ludwigshafen–Wetzlar, RNL–SGF, Balingen–Kiel, Berlin–BHC, Lemgo–Magdeburg, Göppingen–Leipzig, Altenholz–Lübeck, Pforzheim–Bietigheim |
| test (3) | Kiel–Lemgo, Eisenach–Hamburg, Gummersbach–Löwen |

## Procedure

1. **Select on the tuning matches only** (`split_test select`). Test rows are
   dropped first, and a test asserts it. Every candidate is scored by
   leave-one-match-out over the 8 tuning matches. There are 28 candidates:
   - 7 feature sets: PRTReID stripes, PersonViT-B, SportsReID ViT-B, the two
     fused, each full-crop or `strict`
   - × rule {absolute, 8-crop minimum}
   - × training target {at bounds, tight}
2. **Lock the winner** in `locked/split_v1.json`, commit `fb36758`:
   - SportsReID ViT-B, full crop
   - absolute gate: score ≥ 0.766, margin ≥ 0.071
   - 6/8 tuning folds passed
3. **Score the three test matches once** (`split_test test`) against criterion
   v2: known precision ≥ 0.95, unknown false-match rate ≤ 3%,
   coverage ≥ 0.20.

## Result: pass

| pooled over the 3 test matches | bound | result | 95% interval |
| --- | ---: | ---: | --- |
| known-player precision | ≥ 0.95 | **1.000** (76 / 76) | exact one-sided lower bound 0.961 |
| unknown false-match rate | ≤ 3% | **0.0%** (0 / 213; natural 0 / 34) | exact one-sided upper bound 1.4% |
| coverage | ≥ 0.20 | **0.42** (76 / 181) | [0.30, 0.53] (identity-clustered) |

| test match | correct / known queries | coverage | correctly linked footage |
| --- | ---: | ---: | ---: |
| Kiel–Lemgo | 28 / 74 | 0.38 | 1,012 s |
| Eisenach–Hamburg | 25 / 63 | 0.40 | 484 s |
| Gummersbach–Löwen | 23 / 44 | 0.52 | 980 s |

The identity-clustered bootstrap interval for precision is degenerate at
[1, 1], because there are no errors to resample. The exact bounds above are
the honest ones.

## Caveats (they limit the claim)

- **The test matches are not perfectly virgin.** They were part of the earlier
  11-match encoder comparison, which shaped the candidate list (PersonViT and
  SportsReID). The selection itself ran without them.
- **Clubs overlap:** Kiel, Lemgo and Löwen also play in tuning matches. The
  encoders are frozen and each match builds its own galleries, so player
  identity is not learned across matches.
- **Queries are number-vouched fragments**, the visible-back case. Baseline
  unresolved segments are harder. Measuring the real gain there is Stage 2,
  which needs human-verified labels.
- **All three test matches are 1080p HBF broadcasts.** The lower-league
  failure mode (sealed test, look-alike teammates) is not tested here.
- **The final word is new footage**, processed and scored once against this
  lock without any change.

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.split_test select   # refuses to overwrite the lock
PYTHONPATH=src:. $PY -m experiments.teammate_reid.split_test test
```

Results are in `runs/smart-reid-shadow/split_v1/test.json`.
