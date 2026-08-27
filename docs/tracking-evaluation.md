# Tracking and identity: measured findings

Session of 2026-08-25/26. Records what was **measured**, what was **refuted**,
and what infrastructure now exists to re-measure it. Several claims made during
this work turned out to be wrong; they are kept below with the reason, because
the way they failed is the most reusable part.

Scope: identity quality of the tracking layer. Team classification and jersey
numbers appear only where they bear on identity.

---

## 1. The headline numbers

Two clips, seven tracker configurations, scored per frame against
human-verified identity references (§4). `correct` is the share of trusted
reference detections that a tracker both found *and* attached to the right
identity.

**FelixClaar** (249 frames, harder)

| tracker | recall | wrong-identity | **correct** |
|---|---|---|---|
| **sort** | 88.3% | 7.7% | **81.5%** |
| mcbyte_masks_off | 89.5% | 15.6% | 75.5% |
| botsort | 89.3% | 15.6% | 75.4% |
| mcbyte_masks_on | 89.6% | 16.9% | 74.5% |
| ocsort | 86.0% | 13.5% | 74.4% |
| cbiou | 89.5% | 19.1% | 72.4% |
| bytetrack | 89.3% | 19.4% | 72.0% |

**Han-Ber4** (198 frames, easier): botsort 86.7%, sort 86.5%, ocsort 85.5%,
bytetrack 85.0%, mcbyte_masks_off 84.9%, cbiou 84.9%, **mcbyte_masks_on 82.6%**.

Conclusions that hold on **both** clips:

- **SORT is top-2 on both**, and roughly twice as accurate as MCByte on Felix.
  It is also the simplest tracker available — no appearance model, no masks.
- **MCByte with masks is last on both.** That is the configuration the pipeline
  currently runs.
- **Masks make identity worse**, consistently (16.9 vs 15.6 on Felix, 8.0 vs
  5.4 on Han-Ber), while dominating runtime.
- **Recall is flat at 86-90% across every tracker.** No tracker wins by
  tracking less. That ceiling is the *detector*: RF-DETR matches ~86% of
  reference players at IoU ≥ 0.5. Roughly 12% of the remaining error is
  detection, not association.

### Why SORT wins

SORT has the *most* mixed tracklets on Felix (8, vs MCByte's 7) yet by far the
lowest wrong-identity time. It **mixes more often but briefly** — it loses the
track and breaks. MCByte mixes less often but **stays wrong far longer**,
because mask conditioning keeps it confidently locked onto the wrong player.

This matters architecturally: a fragment is recoverable by re-ID, a wrong
identity is not. Prefer a tracker that breaks over one that persists wrongly.

---

## 2. How bad is it, in absolute terms

On Felix, with hand-labelled tracklets (§4.1): **7 of 17 MCByte tracklets hold
more than one player**, and ~25% of tracked frames sit under a wrong identity.
The automated scorer independently reports 7 mixed tracklets for the same
configuration — two different methods, same answer.

Handball is the reason. Play is 1v1 with body contact, so crossings are
constant, and the crossing partner is usually an opponent (user's domain
knowledge, consistent with the data).

---

## 3. Claims that were made and then refuted

### 3.1 "Cutie masks are actively hurting: 6 → 2 switches"

**Refuted, then partly re-established on better evidence.**

The original evidence was `suspected_id_switch` counts. That metric fires only
when a *settled* team label flips, so it is:

- blind to same-team swaps entirely,
- **suppressed by fragmentation** — a track that breaks instead of sliding
  starts a fresh provisional identity and never registers a switch.

Hand-labelling both configurations showed **7 mixed tracklets either way**
(masks on and off) while the heuristic showed 6 vs 2. The metric moved; reality
did not. Later per-frame scoring against a reference did show masks-on worse on
both clips — so the conclusion survives, but nothing about the original
evidence justified it.

**Lesson:** a metric that can be gamed by a change in failure *mode* cannot
compare two configurations that differ in failure mode.

### 3.2 "PRTReID has no usable signal"

**Refuted.** The statistic pooled all tracklets, including the 7 later labelled
mixed. A mixed tracklet's representative embedding is a blend of two people,
which compresses every distance. Restricted to verified-clean tracklets,
prtreid separates two **teammates** at 4.3-5.0× their own variation — clearly
usable.

**Lesson:** the contamination being measured had corrupted the measurement.

### 3.3 "So appearance-based split will work"

**Refuted.** Between-tracklet separability does not transfer to within-tracklet
detection. On masked crops, with the human changepoint and a ±15-frame
exclusion zone, the unsupervised statistic on mixed tracklets (min 0.0245,
median 0.0433) sits entirely inside the clean range (0.0157-0.1152). No
threshold separates them.

The mechanism: **the visual information needed to split a tracklet is destroyed
by the same occlusion that causes the swap.** Crops around the handover are
blends of two bodies, and masks degrade or vanish there — the frames that would
prove the switch are the ones that do not survive. Masking also discards ~60%
of frames (tid 3: 238 → 91 crops), leaving too few near the changepoint.

GTA-style split/connect (`handball_cv/tracking/tracklets.py`) was implemented
and tested against these labels: **split caught 0 of 7**, and **connect got 0 of
3 correct while making ~4 wrong merges**. The code and its unit tests are kept;
the approach is not viable on this footage with these features.

### 3.4 "SAM2/mask propagation is consistently the best"

**Refuted.** Seed-once propagation looked excellent — 14/14 identities clean on
Han-Ber, 10/13 on Felix — but that verdict came from sheets sampling **12
frames per identity**, while trackers were scored **per frame**.

Seeding two propagation runs at different frames and comparing them over the
overlap (`scripts/evaluate_propagation_drift.py`, no labels needed) gives
**14.7% disagreement** on Felix, with 4 of 12 identities breaking. That places
propagation in the same band as the trackers, and worse than SORT's 7.7%.

Two cross-checks on that test:

- it independently found id 10 breaking at **f112** against a hand label of
  **f113**;
- it flagged id 6 (hand-labelled clean), i.e. the 12-frame sheets **did** miss
  swaps;
- it missed id 1 (hand-labelled mixed at f210) because **both runs failed
  identically** — propagation errors are correlated, so 14.7% is a lower bound
  on per-run drift.

What survives: propagation's Felix failures include a **lifecycle** failure (a
bench player walked in and took a mask) rather than pure drift. Lifecycle is
fixable with detector checkpoints — `handball_cv/tracking/sam2_manager.py`
exists for this — in a way mask drift is not.

---

## 4. Ground truth and evaluation infrastructure

### 4.1 Hand-labelled tracklets

`runs/tracklet_labels/FelixClaar_masks_{on,off}/answers.txt` — one line per
MCByte tracklet: clean, or mixed with the frame it changed hands. Produced from
sheets rendered by `scripts/label_tracklet_identity.py` (12 crops across each
tracklet's life, frame-numbered, upscaled).

Costly to produce: every tracker configuration yields its own `tracker_id`
partitioning, so comparing N configurations needs N label passes. Superseded
for tracker comparison by §4.2, still the authority on MCByte specifically.

### 4.2 Per-frame identity references

A reference is one `label` map per frame whose pixel values are persistent
identity ids. Identities are assigned once, on a seed frame, and carried purely
by mask propagation — nothing re-associates them to detections, so the result
is **independent of the box-association logic every tracker under test uses**.

| clip | path | ids | verified |
|---|---|---|---|
| Han-Ber4 | `source/.Han-Ber4_sam2_masks` (pre-existing, SAM2) | 14 | 14/14 clean |
| FelixClaar | `source/.FelixClaar_ref_masks` (SAM+Cutie) | 13 | 9 clean, 3 mixed, 1 leaves frame |
| FelixClaar | `source/.FelixClaar_ref_masks_seed60` | 12 | drift test only |

Han-Ber id 12 is a bench player and id 14 is a referee — exclude via
`--exclude-ids` for player-only metrics.

Label once, score any number of trackers. Identities are scored only over their
**verified span**: "clean until f90" means frames after 90 are dropped, so a
tracker is never punished for the reference's own errors.

Build with `scripts/build_identity_reference.py`, verify with
`scripts/render_reference_sheets.py` (non-mask pixels dimmed to 35% — a bare
bounding box in a scrum routinely contains two players and is genuinely
ambiguous).

**Known limits:** seeding once means players entering later are never
represented; mask *shape* degrades late (Felix id 4 collapses to 55×31 at
f176) even where identity holds; and verification sampled 12 frames, so brief
swaps can be missed (§3.4).

### 4.3 Scripts

| script | purpose |
|---|---|
| `evaluate_tracker_identity.py` | score any tracker against a verified reference |
| `build_identity_reference.py` | seed-once propagation → per-frame label maps |
| `render_reference_sheets.py` | verification sheets for a reference |
| `label_tracklet_identity.py` | verification sheets for a tracker's own tracklets |
| `evaluate_propagation_drift.py` | cross-seed drift, no labels required |
| `evaluate_reid_separability.py` | can an embedding separate teammates |
| `evaluate_tracklet_split.py` | can split detect labelled mixed tracklets |

### 4.4 What the metric does and does not punish

`wrong_identity_pct` counts, among detections that **matched** a reference
identity, those attached to a non-dominant one.

- A player leaving frame is **not** punished — the reference has no mask there
  either, so nothing is counted. Correct by design.
- Losing a player who is still visible is **also not** punished by that figure
  alone. Always read it alongside `recall`; the `correct` column in §1
  multiplies the two. Recall turned out flat (86-90%), so the ranking is about
  identity, not coverage — but that had to be checked, not assumed.

---

## 5. Jersey numbers (deferred, measured)

Numbers are the only signal that discriminates **teammates** — team colour
cannot, and appearance embeddings fail inside mixed tracklets (§3.3). Current
state on Felix:

- ~33% correct on legible crops; **0 of 8** on the one player checked closely
  (a clearly visible **#13** read as `8`,`18`,`16`,`18`,`8`,`18`,`17`,`8`).
- Failures are systematic, not random: `3`↔`8` confusion at 25-40px, where the
  two digits differ by a few pixels. Distinctive numbers (`24`, `54`) read
  correctly at the same crop size.
- The OCR VLM never abstains — handed an unreadable crop it emits a confident
  digit (`11` overwhelmingly), which votes as hard as a real read. Junk must be
  rejected *before* the model.
- Fixes applied in `scripts/render_full_pipeline.py`: detector confidence
  0.3 → 0.5 (matching the notebook; at 0.3 the class-4 head fired on sponsor
  lettering), a minimum box area, a `crop_quality` gate, and one-to-one
  number↔player matching (previously one player's mask could absorb a
  neighbour's number — 15% of matches).
- `NumberVoter` correctly **abstained** on the contaminated player rather than
  committing to the wrong number. The abstention discipline works.

Also note: the earlier per-player vote histograms were contaminated by identity
errors — a tracklet holding two players accumulates both their numbers. Number
accuracy is not well defined until identity is trustworthy.

---

## 6. Open questions

1. **Adopt SORT?** Top-2 on both clips, ~2× better than MCByte on Felix.
   `CLAUDE.md` records "Use MCByte, do not switch back to ByteTrack" as a hard
   constraint — ByteTrack is indeed poor here (worst on Felix), but SORT and
   OC-SORT were never in that comparison. Needs a full-pipeline run before
   changing the default. **Superseded — see §8: SAM2 with periodic reprompting
   beats every tracker measured here, including SORT, by a wide margin.**
2. **Turn masks off regardless.** Worse identity on both clips and ~3-4×
   slower. The only cost is that the team mask-fallback and overlay mask fill
   depend on them.
3. **Detection is ~12% of the loss** and is the shared ceiling for every
   tracker. `notebooks/Player-and-Handball-detection-1/` has train/valid/test.
4. **Motion/trajectory is unexplored** — the one cue not destroyed by
   occlusion. Two players in contact still have distinguishable velocities
   entering and leaving it.
5. **Two clips, ~450 frames total, one detector, library-default parameters.**
   No tracker was tuned. This ranks trackers on this footage, not in general.

---

## 7. Environment note

The conda env (`NewEnv`) is gone — the project moved to a new device using
`uv`. Current setup: `.venv`, Python 3.11.16, `torch==2.9.1+cu130` from the
cu130 index (aarch64/GB10). `requirements.txt` documents versions verified
under conda and warns it "is not a from-scratch install recipe" — accurate:

- `inference==0.62.0` is **unsatisfiable** against `transformers==5.9.0`
  (`huggingface-hub>=1.5` vs `<1.0`). pip installed it anyway; uv resolves
  strictly. It is currently **left out** — it is only needed for the deferred
  jersey-number stage and the court/keypoint scripts.
- torch must be installed first from the cu130 index, then the rest from PyPI;
  uv will not cross indexes by default.
- `models/prtreid/prtreid-soccernet-baseline.pth.tar` needs a source checkout
  for the model definition. It used to live in `/tmp` and was lost to a reboot,
  leaving a 378 MB unloadable checkpoint. Now `prtreid-upstream/` (gitignored),
  documented in `models/README.md`.
- Restoring the SAM2 baseline (needed for §8) after `bbc9493` archived
  `sam2-upstream/`: `git clone --depth 1 https://github.com/facebookresearch/sam2.git
  sam2-upstream`, then `uv pip install iopath` (the only package `sam2`'s own
  `setup.py` lists that wasn't already in `requirements.txt`) and
  `uv pip install -e .` (this project's own `handball_cv` package was not
  installed into `.venv` at all before this). The checkpoint at
  `segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt` loads fine
  against the freshly cloned upstream config. This host also has no system
  `ffmpeg` and no passwordless `sudo`; `uv pip install imageio-ffmpeg` provides
  a working static binary without root.

---

## 8. Update (2026-08-26): SAM2 with periodic reprompting beats everything in §1

A user review of this document, on being told "SAM2 is not the primary
tracker" per `CLAUDE.md`, pushed back: their own manual review found SAM2
tracks players through occlusion much better than what's documented above.
That pushback was right, and tracing it down found a real gap in what §1 and
§3.4 had actually measured — not a visual-impression artifact.

### 8.1 What was measured before, versus what this adds

§1's `mcbyte_masks_on` (worst tracker measured) is masks merely nudging
MCByte's box-based association — not mask-memory doing the association
itself. §3.4's naive seed-once propagation was checked only by a cross-seed
drift proxy, not the per-frame scorer used for every tracker in §1. Neither
is what "SAM2 as the tracker" usually means, and neither had been scored
against ground truth the same way as SORT, MCByte, etc.

This entry adds that measurement: the real SAM2 video predictor, seeded once
and then given periodic detector-checkpoint reprompting via
`src/handball_cv/tracking/sam2_manager.TrackManager` (`CHECK_EVERY=10`
frames) — the design `experiments/sam2_baseline/run_pipeline.py` already
implements, driven here by the new `scripts/run_sam2_reprompt_tracker.py` and
scored as `sam2_reprompt` through `evaluate_tracker_identity.py` via a small
replay adapter (`Sam2ReplayTracker`) — SAM2 propagates in chunks against a
prebuilt video state rather than accepting one frame at a time, so it can't
implement the `trackers` package's per-frame `.update()` shape directly; the
adapter replays a precomputed run through that interface so the same scorer
applies unmodified.

Both the frame-0 seed and every checkpoint's reprompt boxes come from the
exact same cached detections every other tracker in §1 consumes
(`.FelixClaar_detections_v1.npz`, `.Han-Ber4_detections_v1.npz`) — SAM2 is not
given a stronger detection input than the box trackers got. `court_test_fn` is
left permissive (accepts any new detection as a candidate track), matching
the box trackers, which also add every unmatched detection without a
court-membership check.

### 8.2 Results

**FelixClaar** (249 frames) — `sam2_reprompt` added to the §1 table:

| tracker | recall | wrong-identity | **correct** |
|---|---|---|---|
| **sam2_reprompt** | 94.5% | 0.8% | **93.8%** |
| sort | 88.3% | 7.7% | 81.5% |
| mcbyte_masks_off | 89.5% | 15.6% | 75.5% |
| botsort | 89.3% | 15.6% | 75.4% |
| mcbyte_masks_on | 89.6% | 16.9% | 74.5% |
| ocsort | 86.0% | 13.5% | 74.4% |
| cbiou | 89.5% | 19.1% | 72.4% |
| bytetrack | 89.3% | 19.4% | 72.0% |

13 tracklets, 5 mixed (vs. 7-8 for every box tracker), 8 id switches (vs.
19-25), matched 2271/2403 trusted detections.

**Han-Ber4** (198 frames): `sam2_reprompt` scored recall 89.2%, wrong-identity
**0.0%**, correct 89.2% — 12 tracklets, **0 mixed, 0 id switches**, matched
2305/2585. Previous best was botsort at 86.7% correct.

`sam2_reprompt` beats every previously-measured tracker on both clips: by a
wide margin on Felix (+12.3 points over SORT, the prior best), and by a
smaller margin but a qualitatively different result on Han-Ber (zero errors
of any kind, where every other tracker has nonzero mixed tracklets and
switches).

### 8.3 Why recall moved too, not just identity accuracy

§1 read recall as flat at 86-90% across all seven trackers and attributed the
ceiling to the detector, since every box tracker's `.update()` only returns
objects present in that frame's detection input. That ceiling doesn't apply
to mask-memory propagation: between checkpoints, SAM2 does not need a
detection on every frame to keep tracking a player — it propagates on its own
visual memory and only re-syncs with the detector every `CHECK_EVERY=10`
frames. A player the detector drops for a few frames (partial occlusion,
motion blur) isn't immediately lost the way it would be for a box tracker.
That is the likely mechanism behind `sam2_reprompt`'s higher recall (94.5% on
Felix, above every box tracker and above the ~86% ceiling §1 attributed to
the detector), not an artifact of this evaluation.

### 8.4 Caveat: the two references aren't equally independent of what was tested

Both ground-truth references are themselves mask-propagation-based, but not
equally close to the configuration just tested:

- **FelixClaar's reference** (`source/.FelixClaar_ref_masks`) was built by
  `scripts/build_identity_reference.py` using SAM (ViT-B, single-image) +
  Cutie propagation — a different engine from the real SAM2 video predictor
  tested here. The 93.8% Felix result is architecturally independent
  evidence, not two runs of the same system agreeing with itself.
- **Han-Ber4's reference** (`source/.Han-Ber4_sam2_masks`) is, per §4.2,
  "pre-existing, SAM2" — also built from real SAM2 mask propagation, seeded
  once, on the same footage. A perfect 0.0%-wrong match between two different
  SAM2 runs (different seeding/reprompting policy, same underlying mechanism)
  is weaker evidence than the Felix result: correlated failure modes between
  two SAM2 runs on the same ambiguous frames could inflate the apparent
  agreement. Read the Han-Ber result as *consistent with* the Felix result,
  not as an independent second confirmation on its own.

Net: the Felix result alone — independent reference, +12.3 points over the
previous best — is enough to reopen open question #1. Han-Ber corroborates it
but leans on a same-family reference, so don't quote it as independent
confirmation without this caveat attached.

### 8.5 Cost and integration, not yet resolved

- **Runtime**: ~1.0-1.2 it/s on both clips (Felix: 248 frames / 3m26s;
  Han-Ber: 198 frames / 3m17s) on the available GB10 machine — an order of
  magnitude slower than the box trackers in §1, and slower than the SAM/Cutie
  mask-fallback path (3.8-4.8 fps, per `CLAUDE.md`'s mask-experiment runtime
  note). Feasible for this project's offline batch pipeline; not for anything
  with a latency budget.
- **Integration shape**: not a drop-in swap. Adopting it means running the
  pipeline the way `experiments/sam2_baseline/run_pipeline.py` (and now
  `scripts/run_sam2_reprompt_tracker.py`) already do — SAM2 predictor +
  `TrackManager` checkpoint loop — rather than through the generic `trackers`
  interface MCByte/SORT/etc. share.
- **Sample size unchanged from open question #5**: still two clips, ~450
  frames total, one detector, library-default parameters for the box trackers
  being compared against. `sam2_reprompt`'s own parameters (`CHECK_EVERY=10`)
  were not tuned either.

### 8.6 Bottom line (two clips)

Open question #1 is superseded. **SAM2 with periodic detector-checkpoint
reprompting is the best-measured tracker on both clips by a substantial
margin** — pending the cost/integration tradeoffs above and validation past
these two clips. §8.7 adds that third validation.

### 8.7 Third clip, with a fully independent reference (2026-08-27)

§8.4 flagged that Han-Ber4's reference was itself SAM2-built, making that
result corroborating rather than independent. This closes that gap with a
third clip whose reference was verified by hand against the specific failure
mode §4.2/§3.4 warned about (sparse verification missing a real swap).

**Clip**: `source/BHC-FAG.mp4` (4554 frames, 50fps, 1080p), a match not used
anywhere else in this evaluation. A 500-frame derived clip was cut at
`scripts.extract_clip_window` from original frames 3030-4029 (the densest
continuous span found, 11-14 detections/frame, no dropouts), subsampled
every 2nd frame to land at ~25fps and match the motion difficulty of
FelixClaar/Han-Ber4 rather than testing at native 50fps, which would make
association easier for every tracker and compress the gaps between them.
Detections and team model were sliced/reused from the existing full-match
cache (`outputs/team_confidence_v2/.BHC-FAG_detections_v1.npz` /
`.BHC-FAG_team.pkl`), re-indexed to the new frame numbering.

**Reference**: built with `scripts.build_identity_reference.py` — SAM ViT-B +
Cutie, the same non-SAM2 engine used for FelixClaar's reference, seeded at
frame 0 (14 identities). Verified at `--samples 40` (vs. the original 12) by
viewing all 14 sheets directly, specifically hunting for the failure mode
§3.4 documents: a real swap surviving sparse sampling. It found one.

- **Identity 13 was genuinely mixed**: a dark #14-like jersey for roughly the
  first 50 frames, then a full handoff onto a different white #25 jersey for
  the rest of the clip. This is a real tracker error caught by verification
  working as intended — not a hypothetical.
- The `answers.txt` truncation format only expresses a valid *prefix*
  (`clean until fN`, i.e. frames `0..N`); it cannot express "bad prefix, clean
  suffix." Since identity 13's good segment is the suffix, not the prefix,
  the honest choice was to exclude it entirely (`X`) rather than force it into
  a format that would mislabel one half.
- Identities 3 and 4 showed a different jersey number briefly surfacing
  during the same frame window (~281-345) that coincides with an apparent
  multi-player pileup — also visible disrupting id6 and id12 momentarily,
  though those two recovered by the next sample. 3 and 4 were truncated
  (`clean until f280`) rather than trusted through the recovery, on the same
  "a smaller trustworthy reference beats a larger uncertain one" principle
  the FelixClaar reference already applies. Identity 14 was excluded outright
  — user-confirmed on review: no real player mask anywhere in this identity,
  pure noise fragments throughout rather than a real body degrading over
  time (unlike FelixClaar id4's late-life mask collapse, which does start
  from a genuine mask).
- Net reference: **12 of 14 seeded identities usable**, two truncated at
  frame 280, matching `evaluate_tracker_identity.parse_labels` exactly
  (verified programmatically before scoring — see the trap noted in §4.4:
  an unannotated line does not silently default to excluded here, since the
  auto-generated frame-count comment already makes the line non-empty).

**Result**:

| tracker | recall | wrong-identity | **correct** |
|---|---|---|---|
| **sam2_reprompt** | 99.0% | 0.1% | **98.9%** |
| mcbyte_masks_on | 98.0% | 2.2% | 95.9% |
| bytetrack | 97.8% | 2.3% | 95.6% |
| sort | 97.6% | 2.3% | 95.3% |
| mcbyte_masks_off | 98.0% | 4.3% | 93.8% |
| botsort | 98.0% | 4.3% | 93.8% |
| cbiou | 98.0% | 4.3% | 93.8% |
| ocsort | 96.9% | 3.9% | 93.1% |

14 tracklets, 3 mixed (vs. 5-7 for every box tracker), 6 id switches (vs.
9-13), matched 5022/5073 trusted detections — again the highest recall of
any tracker, consistent with §8.3's mechanism (mask propagation survives
detector misses between checkpoints).

**This replicates the finding on independent evidence.** The margin here
(+3.0 points over the best box tracker) is smaller than FelixClaar's +12.3,
which tracks: this clip has less-severe occlusion than Felix (per-identity
wrong-identity rates are lower across every tracker here than on Felix), so
there is less room for any tracker to lose ground. `sam2_reprompt` still wins
outright, with the lowest mixed-tracklet and switch counts of any config, on
a reference built with a different engine than the tracker being tested and
personally verified against the specific blind spot that made Han-Ber4's
result suspect.

### 8.8 Bottom line (three clips)

The advantage is not an artifact of Han-Ber4's circular reference. Three
clips, three wins, two of them (FelixClaar, BHC-FAG) on architecturally
independent references. Open question #1 is superseded with higher
confidence than §8.6 alone supported. The unresolved parts are now cost and
integration (§8.5), not validity.
