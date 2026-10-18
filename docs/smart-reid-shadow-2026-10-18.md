# Smart global ID, shadow pass: Stage 1 held-out transfer, 2026-10-18

Branch `prototype/smart-reid-shadow`. It builds on `prototype/smart-player-reid`
(`f0d212c`, report `docs/teammate-reid-prototype-2026-10-18.md`).

This is offline and shadow only:

- No runtime default, tracker, bundle or label changes.
- Production code ran unmodified from the main checkout at `d0fea4d`.
- Results are in `runs/smart-reid-shadow/` (gitignored).

## Verdict: Stage 1 fails. Stage 2 is not run.

The question was whether the prototype's matcher keeps its Kiel precision on
matches it has never seen, at a threshold locked beforehand. The setup:

- **Matcher:** the frozen PRTReID global embedding, a multi-view gallery and an
  8-crop query, with digits masked.
- **Locked threshold:** score ≥ 0.943 and margin ≥ 0.012.
- **Data:** two unseen matches, pooled.

| criterion (fixed before processing) | bound | held-out result | 95% CI (clustered) | |
| --- | ---: | ---: | ---: | --- |
| precision | ≥ 0.90 | **1.00** (11 / 11) | [1.00, 1.00] | pass |
| unknown-player false-match rate | ≤ 5% | **0.0%** (0 / 129) | [0, 0] | pass |
| coverage of known players | ≥ 0.20 | **0.10** (11 / 107) | [0.02, 0.19] | **fail** |

On Kiel, where the threshold was chosen, coverage was 0.31.

- **Every accepted link was correct.** But the matcher accepted only 11 of 107
  known-player queries.
- **A precision of 11/11 does not show near-perfect precision.** The interval
  is degenerate because there were no errors to resample. With 11 accepted
  links, the exact binomial 95% lower bound on precision is 0.72.
- **Per match:**
  - Eisenach–Hamburg: coverage 0.13, 8 accepted.
  - Gummersbach–Löwen: coverage 0.07, 3 accepted.
  - Neither match reaches the bound on its own.

Under the agreed staging, a fail means stop. The overnight SAM2 baseline run
and the labelling effort for Stage 2 are not spent. The bundles from this stage
already exist, so they are not wasted if a later test passes.

## Why it failed: ranking transferred, the score floor did not

These diagnostics are descriptive. None of them chose anything.

- **The matcher still ranks teammates well on unseen matches.**
  - Rank-1 is 0.75, and 0.72 on re-entries. Kiel gave 0.87.
  - The oracle two-gate coverage at P90 is 0.31, the same ceiling as Kiel
    (0.31).
- **The absolute cosine level shifted down.**
  - The median top score of a correct known match is 0.920 held-out. Only 18
    of 107 known queries reach 0.943 at all.
  - The margin gate is not the bottleneck: 55 of 107 pass it.
  - The floor was tuned to Kiel's score scale, and that scale is a property
    of the broadcast (lighting, kit, camera), not of identity.

So the failure is calibration, not discrimination. A fixed absolute cosine
floor does not transfer between broadcasts.

## Secondary configurations (locked at the same time, reported, not promoted)

| config (each at its own locked Kiel point) | coverage | precision | unknown FMR (natural) | wrong-link s / correct s |
| --- | ---: | ---: | ---: | ---: |
| `digits` frozen global (primary) | 0.10 | 1.00 | 0.0% (0.0%) | 0 / 213 |
| `digits` frozen stripes | 0.46 | 0.86 | 5.4% (9.1%) | 308 / 1232 |
| `strict` frozen global | 0.08 | 0.90 | 0.8% (4.5%) | 9 / 190 |
| `strict` frozen stripes | 0.31 | 0.94 | 1.6% (0.0%) | 52 / 1039 |

- **`strict` frozen stripes would clear all three bounds.** That is a
  selection made on these test rows, so it is **not** a result.
  - Its interval is [0.86, 1.00] for precision and [0.18, 0.42] for coverage.
  - Promoting it now would be exactly the test-set threshold choice the plan
    rules out.
- **It is a hypothesis worth a fresh test.** The pattern matches the
  prototype's own finding that frozen stripes are better calibrated for
  abstention.
- **`digits` stripes misses the pass mark.** Its precision is below 0.90 and
  its natural-unknown rate above 5%.

## What a legitimate next test looks like

Eisenach–Hamburg and Gummersbach–Löwen are now development data, alongside
Kiel. A credible follow-up has four steps:

1. **Pre-register the method and rule before any new window is processed.**
   Either:
   - (a) `strict` frozen stripes at its locked Kiel point, or
   - (b) a per-broadcast score floor calibrated without labels (for example
     from each video's own impostor-score distribution), designed on the
     three development matches.
2. **Test it on fresh windows.** Ludwigshafen–Wetzlar and RNL–SGF are already
   listed in `heldout.py`. They are detector-training matches, so report them
   separately. RNL–SGF shares the Löwen players with Gummersbach–Löwen.
3. **Cost:** about 1.5 h per window with SAM2 (`process_video`), and minutes
   to evaluate.
4. **Stage 2** only if that test passes.

## Follow-up: leave-one-match-out check on development data (2026-10-19)

Kiel, Eisenach and Löwen are all development data now. Before spending GPU
time on fresh windows, `experiments/teammate_reid/calibration.py` checks
whether a rule set on two matches holds on the third. Each match takes a turn
as the test match. The setup:

- **Configurations:** five feature/calibration combinations. The full-crop
  configuration is unmasked: jersey text is legitimate evidence for telling
  teammates apart.
- **Threshold choice:** the highest-coverage gate on the two training matches
  that keeps precision ≥ target and false matches ≤ 5%, at targets 0.90 and
  0.95.
- **Pass mark:** the same three bounds as Stage 1, checked on each test match.

Folds that pass all three bounds, out of 3:

| features | threshold | train precision ≥ 0.90 | train precision ≥ 0.95 |
| --- | --- | ---: | ---: |
| number box hidden, global (Stage 1 primary) | absolute | 0 | 0 |
| full crop, stripes + global | absolute | 0 | **3** |
| full crop, stripes + global | per-broadcast calibrated | 1 | 1 |
| shirt+shorts hidden, stripes + global | absolute | 0 | 1 |
| shirt+shorts hidden, stripes + global | per-broadcast calibrated | 1 | 1 |

**Full crop, absolute, train ≥ 0.95, per test match:**

| test match | gate chosen on the other two | coverage | precision | false matches (natural) | wrong s / correct s |
| --- | --- | ---: | ---: | ---: | ---: |
| Eisenach–Hamburg | 0.856 / 0.023 | 0.41 | 0.93 | 2.7% (8.3%) | 16 / 546 |
| Gummersbach–Löwen | 0.855 / 0.020 | 0.34 | 0.94 | 1.9% (10.0%) | 55 / 514 |
| Kiel–Lemgo | 0.858 / 0.020 | 0.42 | 0.97 | 1.2% (0.0%) | 22 / 1085 |

**What this shows:**

- **The gate is stable across rotations:** score 0.855–0.858, margin
  0.020–0.023. A floor set on two broadcasts transferred to the third, where
  a floor set on Kiel alone did not.
- **Every error at this rule is an unknown player matched to a teammate.** No
  known player was linked to the wrong teammate. The natural-unknown rate is
  above 5% in two folds, but that is 1 of 12 and 1 of 10 players, so it is
  the risk to watch.

**Calibration did not help.** Label-free per-broadcast normalisation (median
and MAD of non-top candidate scores) made transfer worse. The impostor
baseline does move between broadcasts (median 0.797 on Eisenach against
0.84 on Kiel and Löwen), but correct-match scores do not move with it, so
normalising by it misplaces the gate. It is dropped.

**This is still selection on development data.** One rule out of ten passed.
That rule now has to be locked and tested on fresh matches before it means
anything. Set on all three matches pooled (training precision ≥ 0.95), it
is score ≥ 0.860 and margin ≥ 0.020, with in-sample coverage 0.38 and
precision 0.96. Results are in `runs/smart-reid-shadow/calibration/lomo.json`.

```bash
PYTHONPATH=src:. $PY -m experiments.teammate_reid.calibration
```

## Data and review

| window | match | detector exposure | fragments mined / accepted | identities | known queries | natural unknowns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Eisenach_Hamburg_66m00_10min` | Eisenach–Hamburg | unseen (`eval/`) | 477 / 82 | 19 | 63 | 12 |
| `VfL_GM_Loewen_66m00_10min` | Gummersbach–Löwen | unseen (`eval/`) | 376 / 64 | 15 | 44 | 10 |

The window rule was fixed before processing: 10 minutes from 66:00 of the
broadcast at 25 fps, the same as Kiel. Both matches were checked for live play
at 66–76 min, and nothing else.

Neither match shares a team with the development data (Kiel, Lemgo, BHC, FAG,
Melsungen, Berlin).

**Mining.** Mining used the prototype's rules unchanged:

- source-track own reads
- `NumberVoter` gates
- ≥ 5 agreeing reads and ≥ 80% agreement
- a rival-number veto
- stable team evidence
- no goalkeepers

**Review.** The labels are each fragment's own number reads, not human labels.
- I visually reviewed all 34 identity contact sheets (`stage1/review.json`,
  copied to `experiments/teammate_reid/annotations/stage1_review_v1.json`)
  before scoring. This is an assistant review, not an independent human one.
- One fragment was excluded: Gummersbach #8, track 46, frames 2201–2370. Two of
  its five crops are dominated by another person.
- Three tracker-id handovers sit at re-ID breaks, and mining cut each of them
  correctly:
  - track 182: #6 → #74
  - track 205: #4 → #33
  - tracks 102 and 109 swap #77 and #78

**Protocol.** The protocol is identical to the one the thresholds came from:

- Galleries are same-team, same-video vouched fragments that end ≥ 2 s before
  the query.
- Unknowns are natural (no gallery yet) and synthetic (true identity removed).
  The criterion pools them, as the locked Kiel point did.
- Intervals come from 2,000 bootstrap resamples of (match, identity) clusters
  within each match.

## Checks

- **Evaluation code.** Replaying Kiel through `transfer.evaluate`, with the
  match renamed only to pass the development-match guard, reproduces the
  fold-A operating points exactly for all four configs:
  - coverage 0.3108 and precision 0.92 for the primary
  - accepted counts and unknown rates identical
- **Encoder.** Re-encoding prototype crops gives cosine 0.998–0.9998 against
  the stored prototype features, so the weights loaded.
- **Ordering.** Thresholds and criterion were committed in `e3a3bb3` before any
  held-out window existed. The evaluation code was committed in `46c94ae`
  before mining.
- **Inputs untouched.** All 12 held-out input files keep the SHA-256 they had
  before mining. The threshold file still hashes `09c437d8…`.
- **Tests.**
  - `tests/unit/test_teammate_reid.py`: 19 passed, 6 of them new. The new
    ones cover:
    - the locked file is never rewritten
    - natural and synthetic unknowns are reported apart
    - every bound must hold
    - the clustered CI resamples whole identities
    - development matches are refused
    - held-out teams are disjoint from development data
  - Full `tests/`: 558 passed, 10 skipped. The skips are gitignored assets
    absent from the worktree.

## Side measurement: what Stage 2 would be worth (Kiel, development data)

`runs/smart-reid-shadow/kiel_unresolved_sizing.json`:

- **How much is unresolved.** The production baseline shows no number for
  1,506 of 5,129 field-player seconds (29%).
- **How much the matcher could act on.** 81 of the 100 unresolved fragments
  have at least 3 clean crops, holding 97% of that time. 70 of the 81 are fresh
  tracks, not re-IDs.
- **Caveat.** The court test is a no-op, so the pool can include bench or
  crowd tracks.

The opportunity is real. The matcher just cannot claim it at a transferable
threshold yet.

## Reproduce

```bash
PY=/home/valentinweyer/projects/handball-computer-vision/.venv/bin/python
# worktree: .claude/worktrees/smart-reid-shadow
PYTHONPATH=src:. $PY -m experiments.teammate_reid.transfer lock          # once; refuses to overwrite
PYTHONPATH=src:. $PY -m experiments.teammate_reid.heldout cut --count 2
# from the main checkout, production code, defaults:
$PY -m scripts.process_video data/raw/Eisenach_Hamburg_66m00_10min.mp4
$PY -m scripts.process_video data/raw/VfL_GM_Loewen_66m00_10min.mp4
OUT=/home/valentinweyer/projects/handball-computer-vision/runs/smart-reid-shadow/stage1
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run --set heldout --out $OUT mine
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run --set heldout --out $OUT extract
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run --out $OUT sheets
# review.json in $OUT: assistant contact-sheet review (copy in experiments/teammate_reid/annotations/)
PYTHONPATH=src:. $PY -m experiments.teammate_reid.transfer evaluate
```

## Result files (`runs/smart-reid-shadow/`)

| file | contents |
| --- | --- |
| `thresholds.json` | locked points, primary, criterion, source hash |
| `heldout_windows.json` | window rule, sources, frame counts |
| `input_hashes_before.txt` | SHA-256 of every held-out input before mining |
| `stage1/fragments.json`, `crops.json`, `mining_summary.json` | mined fragments with exclusion reasons, crop provenance, input hashes |
| `stage1/features.npz`, `crops/` | frozen features (`raw`, `digits`, `strict`) and crop pixels |
| `stage1/review.json`, `sheets/` | assistant review and contact sheets |
| `stage1/transfer.json` | per-config pooled and per-match points, CIs, accepted errors, verdict |
| `stage1/rows_heldout.json` | per-query outcomes |
| `kiel_unresolved_sizing.json` | Stage 2 opportunity on Kiel |
| `logs/` | processing logs; bundles are in `runs/bundles/<window>/` |
