# Open tasks

Deferred work, each with the evidence that motivates it and a concrete first
step. Detail and measurements live in `docs/team-classification-handoff.md`;
this file is the index of what is *not* done.

---

## 1. Retroactive number backfill (deferred 2026-09-08, measured, not started)

A number is only displayed from the frame its third read lands. Everything
before that is `P<id>`, even though the evidence explains those frames too.

**Measured on `runs/full_pipeline/Melsungen_sam2_vouched.json`:**

```
player-frames live                   19097
  labelled today                      7045   37%
  backfill, stable segments only      3273  +17%
  withheld, verdict changed later      284
  -> coverage after backfill                 54%
```

Half again as many labelled frames, no new reads, no new model.

**The correctness constraint is the whole design.** Backfill must never cross an
identity, because re-ID moves an identity onto a different person -- p6 is 15
until frame 560 and somebody else after. The unit is the **segment**: the span
between identity breaks (a `reid` revival or a `suspected_id_switch`), which is
the same boundary `NumberVoter.suspend` already uses. Within a segment it is one
person, so a verdict earned late legitimately describes the start.

**Guard, measured:** 1 of 14 segments changed its verdict mid-way (p20 went
`6 -> 15`). Backfilling a changed verdict paints 15 over frames whose own
evidence said 6. Backfill only segments whose verdict never changed; costs 284
frames, leaves +17%.

**Why it needs two passes.** The render draws while it tracks, so labels can only
ever be causal. Pass 1 caches per-frame geometry `(frame, player_id, box, packed
mask)` -- a mask cropped to its box and `np.packbits`'d is ~500 bytes, so ~10 MB
for this clip. Pass 2 re-reads the video and draws from that cache with final
per-segment verdicts: no SAM2, no OCR, seconds instead of 28 minutes. That also
makes every future label change free to re-render, the same reason the reads
cache exists.

**First step:** `segment_verdicts(reads, breaks, live_frames)` in
`src/handball_cv/jersey/identity.py` (pure, testable), then the geometry cache,
then `--redraw` on `scripts/render_full_pipeline.py`. Nothing in the tracking or
voting path changes.

---

## 2. Reader accuracy -- RESOLVED by swapping weights (2026-09-08)

Three docTR recognisers converged near 0.62 with a shared ~6% floor of
confidently-wrong reads, which read as a domain limit. It was not. The original
`baudm/parseq` checkpoint, same architecture trained on scene text rather than
docTR's document-text corpus, scores **0.858 / 0.958 selective** on the identical
crops -- paired McNemar b=4 c=80, **p=2e-19** -- at the same speed and abstention.
Wired in as `--reader parseq`. See the handoff for the full table and the
end-to-end check on the `#6` failure.

Fine-tuning on other sports was measured and rejected: hockey weights match on
accuracy but collapse abstention (0.49), SoccerNet weights fall to 0.266.

**Consequences for items 3 and 4 below: both were measured against the weak
reader and should be re-run before being trusted.**

Remaining reader work, if 0.858 is not enough: no public handball video matches
our regime (TeamTrack is 6K fisheye, "Play by play" is JSON coordinates not
video, the GTS action set is practice footage), so training data means labelling
new broadcast matches -- starting with finishing the truncated
`data/raw/2026-06-07_1284786_VfL_GM_Loewen.mp4` transfer (65 MB of ~3.2 GB).
**The 660-crop evaluation set must stay out of any training set**; every
measurement in the handoff is anchored to it. Where the two readers agree on a
number they are right 92% of the time and agree on human-unreadable crops only
1% of the time, so agreement is a good *pre-fill* for a labelling pass -- but not
an auto-label, because agreement selects exactly the easy crops.

## 3. Crop padding is untested between its two extremes

`NUMBER_CROP_PAD = 0`. The eval set only carries tight (`crop_path`) and 0.8x
padded (`context_path`) crops; PARSeq scores **0.62 tight vs 0.04 padded**.
Nobody has tried 10-20%. Would only address the partial-digit band above, not the
6% floor. Cheap: re-crop the 323 labelled boxes at several pads and rescore.

---

## 4. The fold rule is one-sided

`_Votes.resolved_counts` folds a single digit into a 2-digit value that *ends*
with it, arguing a partial view catches the trailing digit and "the reverse
cannot happen". The labelled set contains **5 leading-digit-only reads**, and
player 20 produced a `1` for jersey 15. Re-check against labels before changing
anything -- the rule was tuned on real failures, and folding both ways would let
`1` capture `15`, `13`, `18`... which is exactly the ambiguity the current rule
avoids.

Related risk, unresolved: player 20's claim to `15` rests on 11 bare `5` reads
folded in against only 6 raw `15`. If that player is actually #5, folding
manufactured the number.

---

## 5. Number-anchored linking has never fired on real data

`NumberIdentityResolver` links identities that are the same player (same team,
same number, never simultaneously live). On the 60s Melsungen clip it fired
**0 times** -- every same-team duplicate was simultaneously live, so none were
fragmentation. It has unit tests and nothing else. Needs a clip where a player
genuinely fragments across a gap.

---

## 6. The court test is a no-op

`sam2_driver.drive_sam2` passes `court_test_fn=lambda box: True`. A colour-mask
approach was tried and abandoned -- the bench sits inside the mask, so it
excluded 8% of detections and none of the bench. The identified route is
homography from the keypoints in `scripts/run_court_mapping.py`.

Lower value than it first appeared: most of what looked like crowd-tracking was
number boxes being tracked as people (fixed in `6dc4ff1`), and only ~8% of
person detections have feet off the floor.

---

## 7. Bench occupancy as an identity constraint (blocked on 6)

A player on the bench cannot be on the pitch at the same instant. Once
court/bench is distinguishable this is a *hard* mutual-exclusion constraint, and
a stronger version of the simultaneity guard the linker already uses. Deliberately
deferred until the court test exists.

---

## 8. Detector threshold 0.3 -> 0.5

Detector precision measured monotonic with confidence (50% -> 100%) on the 1080p
set, recommending the raise. Documented, **never verified end to end** -- it
would change what the tracker and the reader see on every clip.

---

## 9. Missing ground truth

- **The 60s Melsungen clip has none.** Every number claim on it is unverified;
  today's before/after comparisons are self-consistent but not scored.
- **Real Bundesliga rosters.** Two roster simulations were run on invented squad
  lists and both were retracted as worthless. The roster question -- constrain
  reads to numbers that exist in the squad -- cannot be answered honestly without
  them.

---

## 10. Housekeeping

- `pytest -q tests` fails collection: `test_team_model.py`,
  `test_team_dataset.py`, `test_team_calibration.py` and `test_team_evaluation.py`
  exist in **both** `tests/` and `tests/unit/`, and without `__init__.py` pytest
  cannot import both. Run the two directories separately (216 + 41 pass) until
  the basenames are disambiguated.
- `runs/reid_analysis/` is gitignored, so the re-ID discriminability reports live
  on disk only. Regenerate with `scripts/measure_reid_discriminability.py`.
