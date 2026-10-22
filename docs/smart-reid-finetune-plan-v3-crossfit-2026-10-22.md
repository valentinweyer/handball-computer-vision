# Smart re-ID fine-tune: paired parity test (plan v3, draft for approval, 2026-10-22)

> **Not run. Withdrawn after an external review (2026-10-22).**
> - The fold-strength correction is unjustified: M is not additive.
> - P3 is vacuous.
> - Half-data folds cannot certify the full-data adapter.
> - Löwen is counted twice.
>
> See the Corrections section in `smart-reid-finetune-result-2026-10-22.md`.

Branch `prototype/smart-reid-shadow`, never merged. **Nothing runs until this is
approved.** Once approved, the bars are frozen in `locked/finetune_v3.json`
before any training.

## Why a new plan

finetune_v1 (`smart-reid-finetune-result-2026-10-22.md`) gained on unseen teams:
M rose by +0.184, with a 95% CI of [+0.011, +0.243]. Seen teams in a new match
gained less, by +0.149. **P2 still failed:** the CI of the seen advantage G was
[−0.21, +0.16]. G compared *different* teams, and per-team Δ has a standard
deviation of about 0.12–0.13, which buried any difference.

v3 removes that noise by comparing **each team with itself**: the same team and
the same queries, scored by one adapter that saw the team and by one that did
not. The recipe stays exactly v1's; only the training subsets change. No
hyperparameter is touched.

## 1. Two-fold cross-fit over A (training matches only, `train/`)

Six eval teams also appear in exactly one training match. The split places each
of them in only one fold. Folds are balanced by crop count (v1 training-pool
sizes).

| Fold | Training matches | Crops | Eval teams it has seen |
| --- | --- | ---: | --- |
| X | Balingen–Kiel, Ludwigshafen–Wetzlar, RNL–SGF (Löwen), Berlin–BHC | 5350 | Kiel, Wetzlar, Löwen |
| Y | Lemgo–Magdeburg, Dresden–Melsungen, Göppingen–Leipzig, Vinnhorst–Potsdam | 4901 | Lemgo, Magdeburg, Melsungen |

The split follows from the matches. Lemgo and Magdeburg share one training
match, so they must go together. With Kiel and Lemgo in different folds, each
eval match gets one team seen by X and one seen by Y.

**Paired teams.** In each pair below, a team is seen by one fold and unseen by
the other:

| Eval match | Seen by X | Seen by Y |
| --- | --- | --- |
| Kiel–Lemgo | Kiel | Lemgo |
| Magdeburg–Wetzlar | Wetzlar | Magdeburg |
| Melsungen–Löwen | Löwen | Melsungen |
| Gummersbach–Löwen | Löwen | none |

That gives 7 paired team instances: 4 seen by X, 3 by Y.

**Control.** D (Eisenach, Hamburg, Stuttgart, Erlangen, Saarlouis, Münster,
Bielefeld, Aurich, Würzburg, Regensburg and Gummersbach) is unseen by both folds.

## 2. Statistics

The benchmark protocol is unchanged from v1: split_v2 rows, M = coverage at
precision ≥ 0.95 and unknown FMR ≤ 0.03, gates recomputed per condition.

- **Seen and unseen conditions.** The same C rows are scored twice, giving two
  row sets:
  - *seen*: each team's queries scored by the fold that saw it;
  - *unseen*: the same queries scored by the other fold.

  Unknown rows travel with their team.
- **G_raw** = M(seen) − M(unseen). The queries are identical, so team-to-team
  variation cancels.
- **Fold-strength correction.** X and Y may differ in overall strength:
  - F = M_X(D) − M_Y(D), measured on D, which neither fold saw;
  - w = (rows of C seen by X − rows seen by Y) / all C rows;
  - **G = G_raw − w·F.**
- **CI.** Bootstrap clustered by team, 2000 replicates, seed 0:
  - C teams are resampled with their paired rows;
  - D teams are resampled independently for F;
  - both gates are recomputed in every replicate.
- **Unseen gain.** Δ_unseen = the mean over both folds of M_fold(D) − M_base(D),
  with the same bootstrap.

## 3. Pass bars (v1's bars, applied to the paired G)

| # | Condition |
| --- | --- |
| P1 | Δ_unseen > 0 and lower 95% bound > 0 |
| P2 | G ≤ max(0.01, ¼·Δ_unseen) **and** upper 95% bound of G ≤ 0.04 |
| P3 | Unknown FMR on D, each fold adapter ≤ base + 0.01 |

Each fold trains on about half of the training data, so its gain will be
smaller than v1's. P1 can fail for that reason alone.

Reported, not bars: per-pair Δ, rank-1, and the B and A-block controls per fold.

**Power, stated openly.** There are only 7 paired instances. If the
within-team difference is small, the CI should be far narrower than v1's.
**This cannot be checked in advance without looking at C**, so v3 is the last
attempt on this footage.

## 4. The B anomaly (step 0, before locking)

v1 lowered M on B (−0.059) while rank-1 rose. Step 0 diagnoses this on B only.
B is the inner-validation group, and C and D are not looked at.

1. Which constraint binds: precision 0.95 or FMR 0.03?
2. Whether synthetic unknowns now score higher against the gallery.
3. Score and margin distributions, base against adapter.

**Outcome:**

- **A genuine pipeline bug** is fixed and recorded before the lock. It is the
  only allowed recipe change.
- **A real property of the adapter**, for example unknowns pulled closer, is
  written down. v3 runs unchanged, and that finding counts against the system
  check later.

## 5. If it passes

1. **The final adapter is v1's recipe on all of A.** It is deterministic and
   reproduces adapter sha `6754abc6…`. It is re-admitted only through this pass.
2. **Then the v1 plan §7 system check, unchanged:**
   - variants a and b;
   - gate recalibrated on B;
   - D Stage 2 populations;
   - blind assistant queue with controls;
   - lock bar: newly correct player-time ≥ +25%, added-link precision ≥ 0.95,
     synthetic FMR ≤ 3%.

## 6. If it fails

Fine-tuning is a dead end on this footage. Write it up and stop. A later attempt
needs new matches with new teams in both C and D.

## 7. Data protection

- **C and D get a second look.** It is legitimate only because the recipe is v1's,
  frozen, and the design and bars are fixed here before anything runs.
- **No seeds, folds or hyperparameters are tried.** The fold split above is the
  only one, and it follows from the matches.
- **B is used only for the step-0 diagnosis.** It never decides a bar.

## 8. Effort

| Step | Work | Time |
| --- | --- | --- |
| 0 | B diagnosis | about 30 min |
| 1 | Lock, then code the fold training and paired benchmark, with tests | about 1 h |
| 2 | Train the two folds, run parity once | CPU, minutes |

No GPU is needed: the embeddings are cached.

## Questions for approval

1. Is the fold split in §1 right?
2. Do the bars in §3 stay as in v1, including the 0.04 upper bound on G?
3. Is it acceptable for C and D to get a second look under these conditions (§7)?
