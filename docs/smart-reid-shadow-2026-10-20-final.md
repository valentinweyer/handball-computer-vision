# Smart re-ID: split v2 lock and final test on new footage (2026-10-20)

Branch `prototype/smart-reid-shadow`, never merged. Production defaults are unchanged.

The split v1 test passed on three HBF broadcasts
(`smart-reid-shadow-2026-10-19-split.md`). After that, the rule was widened to
name more players, and the wider version was locked as split v2. It was then
scored once on four lower-league matches that nothing had touched before.

## What split v2 changes (locked before the new footage existed)

The rule is chosen on the eight tuning matches only, by leave-one-match-out.
The code is `experiments/teammate_reid/symmetric.py`; the lock is
`locked/split_v2.json` (sha256 `84aae4f1bf942257…`, commit 06aabdb).

- **Galleries from both sides.** Galleries draw on vouched fragments that end at
  least 2 s before the query, and also on fragments that start at least 2 s
  after it. The query's own track is excluded in both directions.
- **Spread query crops.** Eight crops are taken evenly over the fragment instead
  of the first eight.
- **Half-split guard.** The matcher abstains when the first and second half of
  the fragment (each with at least 2 crops) pick different players. This
  guards against a box that switches person.
- **Everything else as in v1.** Encoder SportsReID ViT-B/16 on raw crops. The
  absolute two-gate rule at the `tight` target: τ = 0.784, margin ≥ 0.0226.

On the tuning LOMO this went from 190 correct names with 1 wrong to 296 correct
with 1 wrong. That is a pooled precision of 0.997 and an unknown false-match
rate of 2.2%, passing 5 of 8 folds. Berlin fails narrowly, and Göppingen and
Pforzheim fail in every variant because of look-alike unknowns. At a fixed gate,
the guard costs 1 of the 296.

## Final test bank

The final bank is four 10-minute windows from `eval/smart-id-eval/`:

- Saarlouis–Münster
- Bielefeld–Aurich
- Balingen II–Erlangen
- Würzburg–Regensburg

They are lower-league halls with TV floors (handball lines only) and come from
`heldout.FINAL_WINDOWS`. None of these clubs appears in tuning or in the split
v1 test.

- **Processing.** Production `process_video` (SAM2) ran overnight from the main
  checkout at commit d0fea4d (`investigate/missing-jersey-numbers`).
- **Court check.** The court model saw a court on 87%, 97%, 99.5% and 87% of
  frames. All four are above the 70% cut-off, so the no-court-pause fallback
  was not used.
- **Mining.** The rules are unchanged, giving 241 accepted fragments and 46
  identities.
- **Contact sheets.** I reviewed all of them myself before scoring. This is an
  assistant review, not an independent human one. Two Bielefeld fragments were
  excluded:
  - Aurich #20, track 161, switches to #6 and then a third player.
  - Aurich #8, track 160, is a substitute in a warm-up shirt with no visible
    number.

  The review is in `annotations/final_review_v1.json`.
- **Scoring.** The scorer is `experiments.teammate_reid.final_test`, committed
  (35cdd15) before it ever ran. It reads only the lock and refuses to run twice.

## Result: pass

| pooled over the 4 final matches | bound | result | 95% interval |
| --- | ---: | ---: | --- |
| known-player precision | ≥ 0.95 | **1.000** (113 / 113) | exact one-sided lower bound 0.974 (rows); 0.861 counting only the 20 named identities |
| unknown false-match rate | ≤ 3% | **0.0%** (0 / 229; natural 0 / 10) | exact one-sided upper bound 1.3% |
| coverage of known queries | ≥ 0.20 | **0.52** (113 / 219) | [0.39, 0.62] (identity-clustered) |

| final match | correct / known queries | coverage | unknown false matches | correctly linked footage |
| --- | ---: | ---: | ---: | ---: |
| Saarlouis–Münster | 9 / 21 | 0.43 | 0 / 21 | 131 s |
| Bielefeld–Aurich | 45 / 84 | 0.54 | 0 / 88 | 1,313 s |
| Balingen II–Erlangen | 14 / 33 | 0.42 | 0 / 36 | 640 s |
| Würzburg–Regensburg | 45 / 81 | 0.56 | 0 / 84 | 1,023 s |

Every match passes on its own. Over the universe of vouched fragments with at
least 3 crops (229, 6,141 s), the rule names 49% of fragments and 51% of the
seconds, all correctly. The half-split guard vetoed one fragment the gate would
have accepted, and the vetoed name was correct.

The identity-clustered bootstrap interval for precision degenerates to [1, 1]
because there are no errors to resample. The exact bounds are the honest ones.
The row bound, 0.974, treats the 113 names as independent. The identity bound,
0.861, is the conservative reading, because names of the same player are
correlated.

## Caveats (they limit the claim)

- **Queries are number-vouched fragments**, the case where the player's back is
  visible. The labels come from each fragment's own OCR reads. Mixed fragments
  were removed by assistant review, not by a human.
- **This does not measure gain on unresolved segments.** Baseline-unresolved
  segments are harder. Stage 2 measured them on Kiel, Eisenach and Gummersbach:
  34/34 with the user's adjudication (v3), at about 11% segment coverage.
  Stage 2 has not been run on these matches.
- **Unknowns are mostly synthetic.** 219 of the 229 unknowns are synthetic,
  made by removing the player's gallery; only 10 are natural unknowns. A real
  unseen player, for example a substitute coming on, is represented by just
  those 10.
- **Saarlouis is thin:** 5 identities and 21 known queries. The low yield comes
  from few readable number reads in that footage.
- **The bundles came from the main checkout at d0fea4d.** The scoring code is on
  this branch.

## Reproduce

```bash
# main checkout, production defaults (overnight; runs/smart-reid-shadow/run_final.sh)
python -m scripts.process_video data/raw/<window>.mp4
# worktree: mining, PRTReID + SportsReID features, sheets (runs/smart-reid-shadow/mine_final.sh)
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run --out runs/smart-reid-shadow/final/<window> \
    --set final --only <window lowercase> mine   # then extract, sheets
# review.json per window (assistant review), then once:
PYTHONPATH=src:. $PY -m experiments.teammate_reid.final_test
```

`--only` used to take several space-separated names and swallowed the subcommand after it, so the first overnight mining attempt failed. It now takes one comma-separated value and rejects names not in the set.

Results are in `runs/smart-reid-shadow/final/test.json`, with all rows,
per-match figures, the lock hash and input hashes.
