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

## Stage 2 on the final bank: names on baseline-unresolved segments

The test above asks whether the names are right on segments where the player's
back, and so the number, was visible. Stage 2 asks the question that matters in
production: when the matcher names a segment that the production pipeline left
unnamed, is the name right? It uses the same protocol as the first Stage 2, run
as `stage2 --bank final` with the split-v2 rule.

- **Population.** These are segments where the production bundle shows a number
  on fewer than half the rows, with at least 3 clean crops: 663 segments,
  10,803 s of player-time. Another 368 segments are too short or crop-poor to
  judge and are listed separately.
- **Proposals.** The matcher's proposals were sealed before any labelling
  (sha256 `26f66da8…`). It named 52 segments; 487 fell below the gate, 75 had a
  single candidate, 47 had none, and the half-split guard stopped 2.
- **Labelling.** The user labelled a blind queue of 92 segments: the 52 named
  segments plus 40 random others, shuffled, with only the target box drawn.
  Each review clip carries a follow panel cut from the source frame. The labels
  were frozen (sha256 `c7206e19…`) before scoring.

| | correct | wrong | unscored | precision | exact 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| **every name reviewed (v3 + agreement audit)** | **47** | **1** | 4 | **0.979** | 0.889–0.999 |
| blind labels only (v1) | 45 | 3 | 4 | 0.938 | 0.828–0.987 |

All of these reviews were made after unsealing, so none of them is blind. They
went in both directions.

**Disagreements.** The 3 cases where the blind label and the matcher disagreed
were checked on full-resolution close-ups
(`annotations_human_v1_adjudication.json`, sha256 `78f2dda6…`):

- **Saarlouis S449:** the matcher said #29 and the blind label was #23. It is
  #29, misread at 720p.
- **Bielefeld S278:** the matcher said #97. The blind label was "not a player",
  noted as "timeout", meaning not in play. It is #97 walking off during a
  timeout.
- **Balingen II S158:** the matcher said #10. It is **#7**, a genuine
  teammate confusion, and it stays wrong.

**Agreements.** All 45 names where the blind label and the matcher agreed were
checked on full-resolution strips (`agreement_audit.json`, sha256
`8ca50716…`). In 40 of them the claimed number is readable in the strip; this
was checked by the assistant visually, not by a human. The other 5 went to the
user:

- Bielefeld S280 #7 and S170 #18: the number is visible in the video.
- Würzburg S625 #13, S612 #13 and S611 #9: the user knows the players.

None of the 45 was contradicted.

So the matcher made one error in 48 scorable names. On these hard,
mostly-720p segments, the user's single blind pass made two label errors.
Blind labels are an incomplete reference on footage like this, which is why
every name was reviewed.

Per match on the blind labels:

| match | correct | wrong | after review |
| --- | ---: | ---: | --- |
| Würzburg | 17 | 0 | 17 / 0 |
| Bielefeld | 16 | 1 | 17 / 0 |
| Balingen II | 8 | 1 | 8 / 1 |
| Saarlouis | 4 | 1 | 5 / 0 |

- **Coverage.** The matcher names 7% of the unresolved segments correctly (45 of
  663), which is 5% of their seconds (586 s). This is close to the first Stage 2
  (11% / 9%).
- **Headroom.** In the random sample, the user could name 28 of 40 segments. For
  19 of those the right player had a gallery, and in 14 the matcher's top guess
  was that player but fell below the gate. The co-presence and number vetoes
  removed the true player twice.
- **Criterion.** With every name reviewed, precision is 0.979, above the 0.95
  bound. With n = 48 the exact interval (0.889–0.999) still reaches below
  0.95, so this bank supports the bound but cannot prove it on its own. The
  blind-only figure, 0.938, is the lower reading.

## Caveats (they limit the claim)

- **Queries are number-vouched fragments**, the case where the player's back is
  visible. The labels come from each fragment's own OCR reads. Mixed fragments
  were removed by assistant review, not by a human.
- **The test above does not measure gain on unresolved segments.** That is the
  Stage 2 section on this bank.
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
