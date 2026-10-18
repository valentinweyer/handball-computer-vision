# Teammate re-ID prototype: gallery memory vs learned appearance, 2026-10-18

Branch `prototype/smart-player-reid`, based on `d0fea4d`. This is a shadow,
offline prototype. It does not change the tracker, team classification, the
number voter or resolver, any runtime default, or any saved bundle or video.
Code is in `experiments/teammate_reid/`, tests are in
`tests/unit/test_teammate_reid.py`, and results are in
`runs/smart-reid-prototype-2026-10-18/results/` (gitignored).

## Verdict

1. **Gallery memory is the upgrade worth having, and it needs no training.**
   The test is cross-match: Kiel–Lemgo, with 16 identities, 74 later
   fragments of known players and 84 unknown-player queries. Digits are
   masked. Several trusted views per player plus an 8-crop query raise
   within-team rank-1 from **0.53 to 0.87**. This uses the same frozen
   PRTReID embedding the runtime uses, compared with the runtime's EMA vector
   against a single crop. The paired, identity-clustered bootstrap gives
   +0.34, with a 95% CI of [0.22, 0.46]. Chance is 0.16.
2. **Precision at useful coverage is still not good enough to act on
   automatically.**
   - Runtime-shaped matching at the runtime gates (cos ≥ 0.7, margin ≥ 0.02):
     coverage is 0.22 at precision 0.49. 15.5% of unknown players get matched
     to someone, and wrong links cover 498 s of footage against 546 s
     correctly linked.
   - Gallery + aggregation at its best 90%-precision operating point: coverage
     is 0.31, with 23 correct links, 0 wrong links among known players and 2
     unknown-player false matches. That threshold was picked on the test set
     itself (oracle).
   - Margin-only gating does much worse. It reaches 95% precision at only 4%
     coverage, because unknown players produce confident margins between two
     teammates.
3. **The learned pilot is inconclusive to negative, and I am not claiming
   specialised learning.** Trained on other matches (fold A), neither head
   beats frozen gallery + aggregation on any metric.
   - The heads help where the frozen baseline is weakest: single-crop / EMA
     matching with the torso hidden. The stripe head gains +0.12 rank-1 (CI
     [0.05, 0.21]) in fold A and +0.28 in fold B.
   - Fold B trains on the same broadcast with disjoint people. There the
     strict stripe head lifts closed-set coverage at 95% precision from 0.31
     to 0.76. Open-set coverage barely moves, and fold B is not a held-out
     match.
   - That is a lead, not a result. The training set had 22 identities and
     904 crops.
4. **Next recommendation.** Prototype an offline post-pass with a trusted
   multi-view gallery, two-gate abstention and number veto, still in shadow.
   Spend data effort on more matches, which means more identities, before any
   backbone fine-tuning. See "Next" below.

## Data actually available

Mining used every existing run summary that has tracking plus OCR reads.
Nothing was reprocessed.

| video | match (split unit) | fragments | accepted | identities | crops |
| --- | --- | ---: | ---: | ---: | ---: |
| kiel_lemgo_10min (v4 + backfill fix bundle) | Kiel–Lemgo | 263 | 93 | 17 | 2805 |
| bhc_fag | BHC–FAG | 45 | 10 | 7 | 291 |
| bhc_fag_window | BHC–FAG | 18 | 3 | 3 | 99 |
| melsungen_window | Melsungen–Berlin | 36 | 13 | 9 | 478 |
| felixclaar | FelixClaar | 18 | 1 | 1 | 10 |
| han_ber4 | Han-Ber4 | 12 | 2 | 2 | 26 |

That is 39 identities in 122 accepted fragments (121 after review) and 3,709
crops.

- **Only Kiel–Lemgo has enough later, independent fragments for a
  within-match re-ID test.** BHC–FAG and Melsungen together yield 6 known
  queries (fold C), which supports no conclusion. They are reported only for
  completeness.
- **This is the binding limitation.** A credible held-out learned-model
  comparison needs several more matches with 10+ minutes each.
  `bhc_fag_window` shows the same people as `bhc_fag` (visually confirmed:
  #49 appears in both), so the two are grouped as one match.

## Method

**Mining (`mining.py`).** A fragment is one tracker id's uninterrupted
presence. It is cut at `reid` and `suspected_id_switch` events, the same
breaks `segment_verdicts` uses, and at gaps longer than 1 s.

It is labelled only by OCR reads recorded on that physical track within its
own span. This is the source-track rule of the backfill fix
(`tables._supported_track_segments`). A displayed, folded, linked or
backfilled family number never labels a fragment.

The fragment must also pass these checks:

- The existing `NumberVoter` gates.
- At least 5 exact own reads.
- At least 80% agreement among full-length reads.
- No rival number read at least twice and at least 15% as often. A trailing
  digit counts as support, using the voter's fold rule.

It is excluded, with the reason recorded in `fragments.json`, for any of:

- a goalkeeper
- a mixed team
- `team_state` less than 90% stable
- a team switch
- touching a suspected switch
- the final bundle withholding the number (an arbitration or duplicate-guard
  loss)
- the final bundle contradicting the number
- a co-live duplicate of the same (team, number)

Legacy runs (BHC–FAG, Melsungen, FelixClaar, Han-Ber4) have no read
`track_id`. There the track is reconstructed from the single player box
containing the number box, and kept only if it resolves to the same final
identity. Ambiguous reads are dropped: 28 + 17 + 11 + 2 + 6.

**Crops.** Crops are the runtime's raw player box (`sam2_manager._appearance`),
with no segmentation masks, so mask artifacts cannot become cues. They are
taken only at detector frames, every 10 frames, up to 60 per fragment. A crop
is dropped when any other tracked player or detected referee covers more than
10% of its box (1,630 drops), when it touches the border, or when it is under
64 px tall. Inputs are hashed in `mining_summary.json`.

**Review (`review.json`).** I reviewed all 39 identity contact sheets
(`sheets/*.jpg`) by eye: an assistant visual review, not an independent human
annotator. Each sheet has an on-image legend, one row per fragment and the
detector boxes marked. Anchors and every query fragment were checked against
the printed number and the person.

- All 39 identities are consistent.
- One fragment is excluded: Kiel #8, track 74, where close-up crops contain a
  second player's #28.
- The review prompted one pipeline fix: referees now join the overlap filter.

**Digit hiding (`masking.py`).** The same masking is applied to gallery,
training and query crops.

- `digits` mode:
  - Each detector number box is covered by a fixed-size grey square, 0.45 of
    the crop width, so one-digit and two-digit numbers look the same.
  - Crops without a box get a seeded decoy square, so patch presence is not a
    cue.
  - 505 of 3,709 crops needed a grown patch because the box was larger than
    the standard square. There, patch size still scales with the number box.
- `strict` mode adds a fixed torso-and-shorts band (12–88% of the width,
  12–62% of the height).
- Zoomed audit of 96 seeded crops (`sheets/masking_audit_*`):
  - `digits` leaves a legible or partial digit in about 8. These are
    undetected small front numbers below the patch, shorts numbers, and edge
    digits beside a patch.
  - `strict` leaves 1, and that is another player's number at the box bottom.
- Coverage limit: digit boxes exist only where the number detector fired
  (1,873 of 3,709 crops). Shorts and sleeve numbers are not targeted by
  `digits` mode. A conclusion is only trusted if it survives `strict`.

**Queries (`evaluation.py`).**

- A query is an accepted fragment of a test video. It starts at least 50
  frames (2 s) after the end of every gallery fragment it is compared with,
  and its own crops are never in a gallery.
- Galleries hold only mined, reviewed, number-vouched fragments of the same
  team. Nothing appearance-derived is ever added, so there is no
  self-confirmation.
- 68 of the 74 known queries are re-entries: none of their own gallery
  fragments is on the same tracker id.
- Unknown players are represented two ways:
  - Natural: a fragment whose identity has no gallery yet (12).
  - Synthetic: every known query rescored with its true identity removed
    (72). Any accepted proposal there is a false match.
- Candidates are all same-team gallery identities. The runtime's 300-frame
  revival window and its co-live exclusion are not applied, which makes the
  benchmark harder than the runtime.

**Matching.**

| method | gallery | query |
| --- | --- | --- |
| `ema_single` (runtime shape) | EMA (α 0.3) over gallery crops | first crop |
| `ema_agg` | EMA | mean of first 8 crops |
| `gallery_single` | every gallery crop; identity score = mean of its top-3 cosines | first crop |
| `gallery_agg` | every gallery crop, top-3 mean | mean over first 8 query crops |
| `…|anchor` | only the identity's first fragment | |

Decisions accept when the best score and the margin over the runner-up both
clear their gates. With a single candidate the margin gate abstains, except at
`runtime_gates`, which reproduces `reid_match`.

**Metrics.**

- rank-1 on known queries, overall and for re-entries
- closed-set coverage at 95% precision (known queries only, margin sweep)
- open-set coverage at 95% precision, for a margin-only gate and for the
  two-gate (score floor, margin) Pareto front. The front is an oracle chosen
  on the test rows, shared equally by every method.
- wrong links, the seconds of footage they would mislabel, and the false-match
  rate for unknown players

**Learned prototype (`train.py`, `features.py`).** The PRTReID backbone and
its pooling heads stay frozen; features are precomputed. Exactly these layers
train:

- `proj_global`: Linear 256→128 on the runtime global embedding (32.9k
  parameters).
- `stripe_head`: LayerNorm + Linear 1920→64 for each of four fixed horizontal
  stripes of the HRNet feature map, pooled with PRTReID's own learned
  foreground attention (no keypoints); LayerNorm + Linear 256→64 on the global
  embedding; ReLU; Linear 320→128 (565k parameters). The stripes are
  head/shoulders, torso, thighs and lower legs/feet.

Training details:

- Loss: batch-hard triplet whose negatives are restricted to the same video
  and same team, i.e. teammates, plus a cosine-softmax identity loss over the
  crop's own video's identities only.
- Batches are built from (video, team) groups.
- 400 steps of AdamW at lr 1e-3. Feature and stripe dropout.
- 3 seeds, about 2 s per head on the GB10.

Folds (the split unit is the match):

- **A**: train on all non-Kiel matches (22 identities, 904 crops), test
  Kiel–Lemgo.
- **B**: test one Kiel team; train on the non-Kiel matches plus the other Kiel
  team. Same broadcast, disjoint people. Weaker separation; never pooled
  with A.
- **C**: train Kiel, test BHC–FAG + Melsungen. Only 6 known queries.

The benchmark asserts that no test-fold crop trains a head.

## Results, fold A (cross-match, Kiel–Lemgo test)

- 74 known queries from 16 identities, plus 84 unknown queries.
- Chance rank-1 is 0.163.
- Learned rows are the mean of 3 seeds; seed spread is at most ±0.03 on
  rank-1.
- "Two-gate cov@P90/95" are oracle operating points.

| config (`digits` masking unless noted) | rank-1 | closed-set cov@P95 | margin-only cov@P95 | two-gate cov@P90 | two-gate cov@P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen global, EMA + single crop (runtime shape) | 0.527 | 0.10 | 0.00 | 0.04 | 0.04 |
| + query aggregation only (`ema_agg`) | 0.568 | — | 0.04 | — | — |
| + multi-view gallery only (`gallery_single`) | 0.716 | — | 0.01 | — | — |
| multi-view gallery + aggregation | **0.865** | 0.58 | 0.04 | 0.31 | 0.30 |
| same, anchor fragment only | 0.716 | 0.32 | 0.01 | 0.16 | 0.15 |
| frozen stripes, gallery + aggregation | 0.838 | **0.72** | 0.04 | **0.54** | **0.46** |
| learned `proj_global`, gallery + aggregation | 0.833 | 0.54 | 0.03 | 0.16 | 0.10 |
| learned `stripe_head`, gallery + aggregation | 0.824 | 0.63 | 0.08 | 0.21 | 0.14 |
| learned `stripe_head`, EMA + single crop | 0.572 | 0.17 | 0.07 | 0.14 | 0.11 |
| raw crops (digits visible), gallery + aggregation | 0.865 | 0.55 | 0.01 | 0.37 | 0.28 |
| `strict`: frozen global, EMA + single crop | 0.365 | 0.05 | 0.00 | 0.07 | 0.07 |
| `strict`: frozen global, gallery + aggregation | 0.851 | 0.31 | 0.01 | 0.28 | 0.27 |
| `strict`: learned `stripe_head`, EMA + single crop | 0.464 | 0.10 | 0.06 | 0.06 | 0.06 |
| `strict`: learned `stripe_head`, gallery + aggregation | 0.820 | 0.52 | 0.10 | 0.23 | 0.10 |

Runtime-gate operating point, frozen global EMA single (`digits`):

| | value |
| --- | ---: |
| coverage | 0.216 |
| precision | 0.485 |
| correct / wrong-known / unknown false matches | 16 / 4 / 13 |
| unknown-player false-match rate | 15.5% |
| wrong-link footage | 498 s, against 546 s correct |

Best two-gate point at precision ≥ 0.90, gallery + aggregation (`digits`):

| | value |
| --- | ---: |
| coverage | 0.311 |
| precision | 0.92 |
| correct / wrong-known / unknown false matches | 23 / 0 / 2 |
| unknown-player false-match rate | 2.4% |
| wrong-link footage | 49 s, against 887 s correct |

The thresholds (score 0.943, margin 0.012) were chosen on these test rows, so
this is an upper bound, not a deployable setting.

### What gallery memory alone buys, and what learning does not (yet)

- **Most of the gain is memory.** Keeping several trusted views per identity
  instead of one EMA vector adds +0.19 rank-1 on its own. Accumulating views
  across *several* number-vouched fragments matters: anchor-only scores 0.716
  against 0.865 for all trusted fragments. Averaging 8 query crops adds the
  rest, and adds little on top of an EMA gallery (+0.04).
- **The gains are not digits.**
  - Raw crops with digits visible score the same rank-1 as masked crops
    (0.865 against 0.865).
  - With the whole torso and shorts hidden (`strict`), frozen gallery +
    aggregation still reaches 0.851. Head, arms, lower legs and shoes carry
    enough to rank teammates when there are many views.
  - Hiding the torso does cost confidence: closed-set cov@P95 drops from 0.58
    to 0.31.
- **Learned heads.** Across masking modes, the learned heads do not beat
  frozen gallery + aggregation in fold A. Their only fold-A gain is on the
  weak single-crop path under `strict` masking: +0.12 rank-1 [0.05, 0.21].
  - In fold B (same broadcast, other team, 31 identities), the strict stripe
    head reaches rank-1 0.883 and closed-set cov@P95 0.76, against 0.851 /
    0.31 frozen. Open-set two-gate cov@P90 is 0.37 against 0.28.
  - This is the only hint that teammate-discriminative training helps, and
    it comes from in-domain data that is not held out by match.
- **Which regions the learned head uses.** Dropping one input region of the
  fold-A `stripe_head` at evaluation time:
  - lower legs/feet hurt most: rank-1 0.824 → 0.775, precision at 50%
    coverage 0.565 → 0.478
  - then torso: 0.806 / 0.498
  - head/shoulders barely matter: 0.820 / 0.559

  This matches what the contact sheets show varies between teammates: shoes,
  shin sleeves, tights, knee pads and arm sleeves. It is evidence about which
  regions the head weights, not proof that it learned hair, skin or build.
  The head never beat the frozen features, so no new cue can be credited to
  learning.
- **Frozen stripes are the surprise.** The untrained concatenation of
  foreground-weighted stripes gives the best open-set behaviour (two-gate
  cov@P90 0.54). It also has the lowest single-crop rank-1. The richer
  regional features look better *calibrated* for abstention, not better at
  ranking. This is worth confirming on more data before relying on it.

Representative cases: `sheets/examples_A_cross_match_digits_frozen_global_gallery_agg.jpg`
shows the five most confident correct and wrong proposals. The confident
errors are mostly unknown-player queries assigned to a visually similar
teammate:

- #93 → #22 and #22 → #53 (white, dark hair, similar build)
- #61 → #93 (black arm sleeves, neon shoes)
- #23 → #41 (blue)

The successes lean on the same accessories: #21's black sleeves and knee pads,
#61's sleeves and shoes.

### Leakage checks

- **Splits.** The match is the split unit. Fold A trains on no Kiel–Lemgo crop,
  and the benchmark asserts this.
- **Gallery separation.** Galleries end at least 2 s before a query; queries
  never contain their own gallery crops. Both properties are tested.
- **Labels.** Labels are never derived from appearance. Proposals are scored
  read-only and never feed a gallery or a training set (tested).
- **Possible replays.** Broadcast replays are not detected. As a proxy
  (`near_duplicate_check.json`), I compared raw-feature similarity between
  each query and its true identity's gallery.
  - The median best match is 0.980, against 0.966 for impostor identities.
    For reference, adjacent crops from one fragment have median 0.948 and a
    90th percentile of 0.977.
  - 37/74 queries have a true-identity crop at ≥ 0.98. Impostors also exceed
    the adjacent-p90 level in 48/473 cases.
  - The four highest pairs (`sheets/near_duplicate_top_pairs.jpg`) are
    similar standing poses at different moments, not pixel replays.
  - Replays cannot be ruled out; they would inflate every method equally.

## Runtime

Measured on a GB10 shared with four other GPU jobs.

| step | cost |
| --- | --- |
| PRTReID global embedding (the runtime encoder) | 4.7 ms/crop at batch 64, 5.2 ms at batch 8 |
| stripe features (same forward pass) | +0.2 ms/crop |
| learned head | ~0.003 ms/crop |
| head training | ~2 s |
| mining + extraction for all six videos | 378 s, dominated by decoding the 15,000-frame Kiel video |
| full benchmark (3 folds × 2 masks × 2 heads × 3 seeds) | ~2 min |

Gallery + aggregation needs about 8 sampled crops (~3 s) of a new fragment
before it proposes.

## Limitations

- **Thin data.** One usable test match (Kiel–Lemgo, 10 minutes, 16 query
  identities) and 22 cross-match training identities. Learned-model
  conclusions are restricted accordingly. Fold C is not interpretable.
- **Review.** Query and anchor labels were checked by assistant visual review,
  not by an independent human.
- **Oracle operating points.** The two-gate points are picked on the test set.
  Only the runtime-gate point is a fixed, pre-existing threshold.
- **Masking coverage.**
  - `digits` mode misses undetected numbers (about 8/96 audited crops show a
    digit).
  - 14% of crops have a grown patch whose size follows the number box.
  - Shorts and sleeve numbers are not targeted.
  - Results quoted as non-digit evidence rely on `strict`.
- **Easier than runtime in one way, harder in another.** Fragments come from
  the existing tracker, and crops with any overlap are dropped. Crowded
  revivals, the hard runtime case, are under-represented. In the other
  direction, candidates ignore the runtime's revival time window and co-live
  exclusion.
- **Old numbers do not transfer.** The historical 55% single-crop rank-1 was
  not used. On this benchmark the matched runtime-shaped baseline is 0.527
  (masked) and 0.459 (raw).

## Next

1. **Shadow post-pass on Kiel–Lemgo.** Link unnumbered fragments to a trusted
   multi-view gallery: frozen global + stripes, gallery + aggregation,
   two-gate abstention. A number contradiction or co-live presence vetoes a
   link, and links are emitted as reversible `appearance_link` proposals.
   Measure how many unlabelled player-frames it would cover at a threshold
   *fixed in advance* from this benchmark, and spot-check the wrong-link
   seconds.
2. **More matches before more model.** Mine 2–3 more full-length matches with
   the same rules. That needs `process_video` GPU time, so it needs its own
   go-ahead. Then repeat fold A with ≥ 50 training identities. Only if the
   strict stripe head's fold-B advantage holds there, try fine-tuning the last
   HRNet stage with the same teammate-negative loss.
3. **Calibrate abstention on held-out matches.** Fix the score/margin
   thresholds on training matches, then report test precision at that point.
   This was impossible here: the non-Kiel matches have too few queries.

## Reproduce

From the worktree, with the project venv. All inputs are read-only and
outputs go under `--out`.

```bash
PY=/home/valentinweyer/projects/handball-computer-vision/.venv/bin/python
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run mine       # fragments.json, crops.json, mining_summary.json
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run extract    # crops/, features.npz, mask_info.json
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run sheets     # sheets/ (review + masking audit)
# review.json is the hand-written review record; benchmark reads it
PYTHONPATH=src:. $PY -m experiments.teammate_reid.run evaluate   # benchmark.json, rows_<fold>.json
$PY -m pytest -q tests/unit/test_teammate_reid.py
```

## Result files

All under `runs/smart-reid-prototype-2026-10-18/results/`:

| file | contents |
| --- | --- |
| `fragments.json` | every fragment, its own reads, status and exclusion reasons |
| `crops.json` | crop provenance: video, frame, track, box, digit boxes |
| `mining_summary.json` | per-video counts and SHA-256 of every input |
| `features.npz` | 256-d global + 4×1920 stripes for `raw`, `digits` and `strict` |
| `review.json` | the review record and masking audit |
| `benchmark.json` | per fold: summaries, seed mean/sd, bootstrap CIs, training info, timing |
| `rows_<fold>.json` | per-query outcomes |
| `near_duplicate_check.json`, `encoder_timing.json` | leakage proxy and encoder cost |
| `sheets/` | contact sheets, masking audits, examples |
