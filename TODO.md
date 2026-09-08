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

## 10. Do not commit `docs/outputs-migration-manifest.csv` as it stands

```
committed in git    2.16 MB
working tree      135.70 MB    same 3,314 rows
```

The regenerated audit reclassified 3,294 files as `directory_level_reference`
(up from 72) and puts a reference list of up to 713 entries on nearly every row,
so rows went from short to ~41 KB each. Committing it adds **135 MB to history
permanently** -- git history cannot be trimmed afterwards without a rewrite that
invalidates every clone.

Either regenerate the manifest without per-row reference lists (a count plus the
report's summary table carries the same information), or keep the CSV out of git
and track only `outputs-migration-reference-report.md`.

The tracked 2.16 MB version is fine and unaffected.

---

## 11. `notebooks/*.py` have diverged from the modules that replaced them — RESOLVED 2026-09-09

The 29 legacy `notebooks/*.py` copies were deleted. Verified before removal: no
`.ipynb` imported any of them, they formed a closed import cluster referencing
only each other, and every one had a current counterpart (15 under a new name).
All were last touched between 2026-07-29 and 2026-08-25; their counterparts
carry changes through 2026-09-08.

The concrete drift that motivated this — the legacy
`render_raw_team_classification.py` copy sitting 32 lines behind `scripts/`,
missing `PERSON_CLASS_IDS`, `person_detections()` and `number_detections()` from
commit `6dc4ff1`, and so silently reproducing the bug where the tracker followed
jersey *number boxes* as people — is gone with the copy. `git log` is the
fallback now.

`notebooks/` itself stays: it holds the four `.ipynb` files plus `fonts/`
(`DEFAULT_FONT` in `scripts/render_full_pipeline.py`), `.env`, `models/` and the
two dataset directories.

---

## 12. Hardcoded absolute paths in tracked files

`/home/valentinweyer/...` remains in three scripts: `build_jersey_audit_set.py`,
`cache_number_detections.py`, `evaluate_checkpoint_against_labels.py`. Those
cannot run on another machine. `.env.example` already defines the override
pattern (`HANDBALL_CV_VIDEO`, `SAM2_UPSTREAM_DIR`); these should use it.

The five notebook copies that also carried absolute paths went with the deletion
in item 11, and `data/annotations/team/Han-Ber4.json` was made repo-relative in
`c616040`. The remaining tracked annotations under `data/annotations/` still
carry absolute paths in their provenance fields.

---

## 13. Chunked SAM2 propagation, without resetting identity at the seam

**This is the hard cap on clip length, and it is a memory cap, not a compute
one.** `SAM2VideoPredictor.init_state` allocates the whole video up front:

```python
images = torch.zeros(num_frames, 3, 1024, 1024, dtype=torch.float32)
```

12.58 MB per frame at model resolution, before a single frame is tracked:

```
 1,500 frames   60s @25fps    17.6 GiB   proven (the Melsungen clip)
 3,000 frames    2min         35.2 GiB   comfortable
 9,000 frames    6min        105.5 GiB   at the limit
15,000 frames   10min        175.8 GiB   exceeds the machine
```

The machine has 121 GiB. `offload_video_to_cpu=True` is the usual escape and
**does not help on GB10**, whose memory is unified -- CPU and GPU draw on the
same pool. A second term grows on top: `output_dict_per_obj[obj]
["non_cond_frame_outputs"][frame_idx]` accumulates for every propagated frame
per object, and `clear_non_cond_mem_around_input` defaults to False, so nothing
is pruned as propagation advances.

Compute, by contrast, is flat: ~1.07 s/frame at any length, because the
attention memory bank is bounded (`num_maskmem=7`). Frame 15,000 costs what
frame 100 costs.

### What "seamless" has to mean

Naively restarting per chunk would retire every track at each boundary and let
re-ID re-acquire them -- and re-ID is 0.55 rank-1 within a team, so a boundary
every 3,000 frames would inject a burst of exactly the identity errors the rest
of this work removes. The seam must not go through re-ID at all.

Carry across the boundary:

- **`PlayerRegistry` whole.** It is already tracker-agnostic: identities, team
  votes, embeddings and the retired gallery survive a predictor swap untouched.
- **`NumberVoter` whole.** Keyed by `player_id`, so it needs nothing.
- **The `obj_id -> player_id` mapping.** This is the actual seam. Re-seed the
  new chunk with the *final masks/boxes of the live tracks*, not with fresh
  detections, and assign each new `obj_id` the `player_id` it already had. No
  retire, no revive, no re-ID call, no `suspend()`.
- **Overlap a few frames** so the new state has real content to prompt from and
  SAM2's memory bank refills before the first output frame is kept.

What is genuinely lost is SAM2's own memory bank (7 frames) -- unavoidable, and
small against a 3,000-frame chunk. Expect a brief quality dip in the overlap,
which is why the overlap frames should be discarded from the output rather than
written.

### Verification that would actually prove it

Render one clip that fits in memory both ways -- unchunked, and chunked with a
boundary deliberately placed mid-possession -- and require the run summaries to
match: same `players`, same `numbers_resolved`, and no `reid`/`link`/
`suspected_id_switch` event within the overlap window. Anything else means the
seam is leaking.

---

## 14. Housekeeping


- ~~`pytest -q tests` fails collection on duplicated basenames~~ -- fixed in
  `a0a5a9e` by deleting the superseded root copies. The documented command now
  collects and passes 226 tests in one run.
- `runs/reid_analysis/` is gitignored, so the re-ID discriminability reports live
  on disk only. Regenerate with `scripts/measure_reid_discriminability.py`.
