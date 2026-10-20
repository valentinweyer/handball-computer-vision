# Can a box tracker replace SAM2? Oracle study, 2026-10-20

Overnight investigation following `docs/tracking-evaluation.md` §8.13, CPU only
while the GPU ran a pipeline; the every-frame detector runs and the SAM2
scaling benchmark were added the next morning on the GPU. No production code
or default changed.
Code: `experiments/box_tracker_oracle/`. Artifacts:
`runs/tracker_eval/oracle_2026-10-20/` (per-run JSON, event renders, feature
caches, logs).

## Question

SAM2 is slow; a box tracker would be ~5x+ faster. Is the gap between box
trackers and SAM2 a detector problem (fixable by a better detector) or an
association problem (needing something else)?

## Short answer

**The information is in the boxes, but our detector does not provide it
where it matters.**

1. With perfect boxes, a box tracker is about as identity-safe as SAM2 on all
   three clips -- *if* it is set to break tracks instead of handing them over
   (`sort_strict`: 0.0-0.5% wrong-identity against SAM2's 0.0-0.8%; correct
   95.2 / 97.5 / 98.2% against 95.0 / 99.9 / 98.9%). The price is 2-3.5 fragments per
   player instead of SAM2's 1.0-1.4, which the identity layer would have to
   stitch. A plain motion linker stitches them for free on Han-Ber4 and
   BHC-FAG, but re-creates the swap at FelixClaar's hard crossing.
2. With the real detector, box trackers stay well short on the hard clip
   (best FelixClaar 83.8% vs 95.0%). The swaps are **not** caused by missed
   players: filling every missed player with its reference box does not
   reduce wrong-identity. They happen when two players are in contact and
   the detector emits one box that matches both.
3. Appearance cues do not rescue those contacts: in contact the occluded
   player's box shows mostly the occluder. A cheap kit-colour gate does
   remove the remaining swaps on the two easier clips (wrong-identity
   3.2 -> 0.0% Han-Ber4, 2.3 -> 0.1% BHC-FAG). PRTReID was no better than the
   colour histogram: its gate removed fewer swaps (Han-Ber4 3.2 -> 2.0%,
   BHC-FAG unchanged), and its fused variant helped BHC-FAG (0.2%) but
   wrecked FelixClaar (7.7 -> 21.7% wrong).
4. 8 of the 9 persistent error events are **cross-team** (attacker vs
   defender); the one same-team event may be a reference error.

5. With the production detector run on every frame (14-20 ms/frame), a box
   tracker plus kit-colour gate is within 0.9 / 2.9 points of SAM2 on BHC-FAG
   / Han-Ber4 but 11+ points short on FelixClaar.
6. A SAM2-only-near-contact hybrid would save only ~1.8x: contact is within
   10 frames on ~87% of match frames, and SAM2 costs 93 ms fixed + 17.8 ms
   per player.

So "box tracker + better detector" is viable only if "better" means
**separate, tight boxes for players in contact**. That is now a measurable
detector target (below): the production detector gives both players of a
touching pair their own box 75% / 87% / 69% of the time (FelixClaar /
Han-Ber4 / BHC-FAG).

## Setup

Three clips with verified per-frame identity references (FelixClaar 249
frames, hard; Han-Ber4 198; BHC-FAG 500, fully independent reference).
Scoring is unchanged from §1/§8 (`scripts/evaluate_tracker_identity.py`).
Detection arms:

- **detector**: the cached RF-DETR `_v1` detections, every frame (the §1 input).
- **oracle**: the reference's own boxes, every identity, confidence 1.0.
- **filled**: the detector's boxes plus the reference box of every player the
  detector missed (no detection at IoU >= 0.5) -- perfect recall with the
  detector's own geometry.

Tracker variants (all `trackers` 2.x, CPU):

- `sort`, `bytetrack`: §1 settings.
- `sort_strict`: SORT with `minimum_iou_threshold=0.5`, `lost_track_buffer=10`
  -- "break, don't swap". Fixed before running, not tuned.
- `colour_*`, `prtreid_*`: SORT plus an appearance term in association
  (`appearance_sort.py`). `gate` vetoes a match whose appearance distance
  exceeds a threshold; `fuse` is BoT-SORT's max(IoU, 1 - distance). Kit-colour
  = HSV histogram of the box's torso region; PRTReID = the production re-ID
  model, run on CPU. Thresholds were calibrated **on Han-Ber4 only** (95th
  percentile of same-player distance 5 frames apart: colour 0.28, PRTReID
  0.23; colour also at the 99th, 0.57) and applied unchanged to the others,
  so Han-Ber4 rows for these variants are in-sample.
- None of the trackers in `trackers` uses appearance by default (its BoT-SORT
  is motion plus camera-motion compensation). §1's comparison therefore never
  tested appearance-based association; this is the first test of it.

The appearance cue is computed inside the tracker experiment and never reads
the team model, so it does not couple team classification into tracking.

## Results

Cells are correct % / wrong-identity % / fragments per player (player ids
only). SAM2 (stored replay, real detector): FelixClaar 93.8 / 0.8 / 1.38
(older dump; current code 95.0), Han-Ber4 99.9 / 0.0 / 1.00, BHC-FAG
98.9 / 0.1 / 1.42.

| arm | variant | FelixClaar | Han-Ber4 | BHC-FAG |
|---|---|---|---|---|
| oracle | bytetrack | 90.5 / 8.9 / 1.54 | 99.4 / 0.0 / 1.08 | 97.6 / 2.1 / 1.25 |
| oracle | colour_fuse | 90.8 / 6.0 / 3.08 | 97.1 / 0.3 / 2.00 | 98.6 / 0.2 / 2.33 |
| oracle | colour_fuse (q99) | 91.3 / 6.6 / 2.38 | 98.2 / 0.1 / 1.50 | 99.3 / 0.0 / 1.33 |
| oracle | colour_gate | 83.7 / 13.1 / 2.69 | 97.4 / 0.0 / 1.92 | 97.0 / 1.9 / 1.92 |
| oracle | colour_gate (q99) | 89.4 / 8.9 / 1.92 | 98.2 / 0.1 / 1.58 | 99.3 / 0.0 / 1.25 |
| oracle | colour_gate_strict | 94.1 / 0.5 / 3.92 | 96.6 / 0.0 / 2.25 | 97.7 / 0.1 / 3.25 |
| oracle | prtreid_fuse | 92.6 / 5.3 / 2.54 | 96.5 / 0.7 / 2.25 | 98.5 / 0.2 / 2.67 |
| oracle | prtreid_gate | 89.4 / 8.9 / 1.92 | 97.2 / 0.0 / 2.17 | 98.7 / 0.0 / 2.33 |
| oracle | sort | 89.4 / 8.9 / 1.92 | 98.4 / 0.0 / 1.33 | 99.3 / 0.0 / 1.33 |
| oracle | sort_strict | 95.2 / 0.5 / 3.54 | 97.5 / 0.0 / 1.75 | 98.2 / 0.0 / 2.83 |
| filled | bytetrack | 83.8 / 14.7 / 3.23 | 96.0 / 3.2 / 2.17 | 95.8 / 3.4 / 2.75 |
| filled | sort | 86.6 / 10.0 / 3.77 | 94.9 / 3.2 / 2.25 | 95.4 / 3.1 / 2.67 |
| filled | sort_strict | 90.3 / 3.9 / 4.00 | 93.9 / 3.5 / 2.58 | 97.9 / 0.1 / 2.75 |
| detector | bytetrack | 72.0 / 19.4 / 2.15 | 89.3 / 5.5 / 1.42 | 95.6 / 2.3 / 1.75 |
| detector | colour_fuse | 80.7 / 8.1 / 2.77 | 92.9 / 0.1 / 2.08 | 97.3 / 0.1 / 1.75 |
| detector | colour_fuse (q99) | 81.1 / 8.2 / 2.54 | 89.3 / 4.9 / 1.75 | 95.4 / 2.2 / 1.67 |
| detector | colour_gate | 81.1 / 7.5 / 2.62 | 92.9 / 0.0 / 2.08 | 97.3 / 0.1 / 1.83 |
| detector | colour_gate (q99) | 81.5 / 7.7 / 2.46 | 90.9 / 3.2 / 1.58 | 95.3 / 2.3 / 1.83 |
| detector | colour_gate_strict | 82.2 / 5.4 / 2.92 | 92.5 / 0.0 / 2.33 | 97.1 / 0.1 / 2.25 |
| detector | prtreid_fuse | 69.1 / 21.7 / 2.54 | 91.3 / 1.8 / 2.08 | 97.0 / 0.2 / 2.25 |
| detector | prtreid_gate | 81.5 / 7.7 / 2.46 | 91.0 / 2.0 / 2.25 | 95.0 / 2.3 / 2.42 |
| detector | sort | 81.5 / 7.7 / 2.46 | 90.9 / 3.2 / 1.58 | 95.3 / 2.3 / 1.83 |
| detector | sort_strict | 83.8 / 4.2 / 2.77 | 90.1 / 3.3 / 2.17 | 97.2 / 0.1 / 2.00 |

### Error anatomy

Every persistent error event (>= 3 frames) of SORT/ByteTrack across the arms,
inspected against pixels (`events/*.jpg`; red/cyan = reference masks):

| clip, ids, frame | teams | detector at the contact |
|---|---|---|
| Felix 1/6 f121 (oracle arm only) | cross (white jump-shooter vs blue #15 blocking) | boxes coincide; reference correct |
| Felix 4/5 f88 | cross | one detection best for both, f82-89 |
| Felix 3/4 f90 | cross | one detection best for both, f92-93 |
| Felix 6/7 f64 | cross | both detected; blue player hidden behind white |
| Felix 10/1 f111 | cross | id 10 thin sliver (reference truncated f113) |
| Han 8/3 f132 | cross | one detection best for both, f128-131 |
| Han 3/5 f100 | cross | one detection best for both, f100-101 |
| BHC 5/8 f438 | cross | one detection best for both, f435-439 |
| BHC 10/5 f455 | same (light kit) | id 10 poorly covered f452-458; reference mask fragmented at f450, possibly a reference error |

Handball contact is mostly attacker against defender, so the cross-team cue
is the common case on these clips. Felix 6/7 is the instructive failure: both
players are detected, but around f64 the blue player is almost entirely
behind the white one, so his box's pixels are largely the occluder's and
neither colour nor PRTReID separates them (inferred from the render, not
measured per pixel). Only a mask (or a
tracker that keeps per-pixel memory) resolves it -- which is what SAM2 does.

### Missed players versus bad boxes

`filled` (perfect recall, detector geometry) raises recall to 96-99% but
leaves wrong-identity where it was (SORT: Felix 7.7 -> 10.0, Han 3.2 -> 3.2,
BHC 2.3 -> 3.1). Recall is a coverage problem, not the swap problem. The swap
problem is box geometry in contact.

### Production detector, every frame (added after the GPU was free)

The detector arm above uses the `_v1` caches from the first comparison. The
production default is `detector-expanded-v1` (retrained 2026-10-17); the
older `detector-local` export made the stride-5 `data/interim/score_*`
caches. Both were run on every frame of the three clips
(`detect_every_frame.py`, threshold 0.3 as in `process_video`): **14-20 ms
per frame including decode**, against ~460 ms for the SAM2 loop. Running the
detector every frame is cheap.

Detector quality on every reference frame (confidence 0.5;
`detector_versions.py --every-frame`). "Separated" = of reference player
pairs in contact (box IoU > 0.2), the share where each player gets its own
detection at IoU >= 0.5 under one-to-one matching:

| clip | recall v1 / local / expanded | separated v1 / local / expanded | contact pairs |
|---|---|---|---|
| FelixClaar | 90.7 / 91.6 / 92.1 | 66.2 / 66.2 / **75.2** | 133 |
| Han-Ber4 | 95.6 / 98.6 / 98.5 | 71.1 / 86.3 / **87.3** | 197 |
| BHC-FAG | 98.4 / 99.0 / 98.7 | 76.4 / 69.7 / **68.5** | 89 |

The production detector separates contacts better on FelixClaar and Han-Ber4
and slightly worse on BHC-FAG. Roughly a quarter of contact pairs on the hard
clips still get one shared box.

Trackers on the production detector. The activation threshold was chosen on
Han-Ber4 only (0.3 and 0.5 tie, 0.7 is worse: SORT 95.5 / 95.6 / 93.8%) and
fixed at 0.5 for every clip and variant; colour gate 0.28 as before:

| variant | FelixClaar | Han-Ber4 | BHC-FAG |
|---|---|---|---|
| sort | 80.9 / 11.0 / 2.77 | 95.5 / 2.4 / 1.58 | 97.9 / 0.2 / 2.25 |
| sort_strict | 83.6 / 7.6 / 3.08 | 95.3 / 2.5 / 1.67 | 97.9 / 0.2 / 2.25 |
| bytetrack | 80.3 / 12.2 / 2.00 | 88.2 / 10.5 / 1.42 | 95.3 / 3.1 / 1.58 |
| colour_gate | 83.4 / 8.2 / 2.69 | 97.0 / 0.2 / 2.00 | 98.0 / 0.1 / 2.25 |
| colour_gate_strict | 83.1 / 8.1 / 3.00 | 96.8 / 0.0 / 2.00 | 97.9 / 0.1 / 2.42 |
| *SAM2 (reference)* | *95.0 / ~0.8 / 1.38* | *99.9 / 0.0 / 1.00* | *98.9 / 0.1 / 1.42* |

(`detector-local` rows are in `current_detectors.json`; it is no better.)
With the production detector a box tracker plus kit-colour gate is within
0.9 points of SAM2 on BHC-FAG and 2.9 on Han-Ber4 (in-sample for both the
gate and the activation threshold), but FelixClaar stays 11+ points short:
its contacts (6/7 at f64, 3/4 at f63) still swap.

### Is a contact-triggered SAM2 hybrid worth it?

All swaps sit at contacts, so the natural hybrid runs a box tracker
everywhere and SAM2 only near contacts. Two measurements size it.

**How much of a match is contact** (`contact_coverage.py`, 15 full 10-minute
games, production detector at stride 5, court-in-view frames, person boxes
at confidence >= 0.5, contact = box IoU > 0.1, widened by +/-10 frames):

| | median over games | range |
|---|---|---|
| frames with any contact, instantaneous | ~65% | 49-79% |
| frames with a contact within +/-10 frames | ~87% | 78-95% |
| player boxes near a contact (+/-10 frames) | ~39% | 35-47% |
| visible players per frame | ~10 | 8-12 |

Contact is the normal state of handball, not an event: SAM2 would be needed
on ~87% of frames. Skipping whole frames (Selective Mask Propagation's
dispatch) saves almost nothing here.

**How SAM2's cost scales with players** (`sam2_object_scaling.py`, GPU,
production predictor and accelerations, bf16, Han-Ber4, 60 steady frames):

| tracked players | 1 | 2 | 4 | 8 | 14 |
|---|---|---|---|---|---|
| ms / frame | 110.8 | 128.6 | 165.3 | 234.7 | 343.3 |

Linear fit: **93 ms fixed + 17.8 ms per player.** A hybrid that runs SAM2 on
87% of frames for ~40% of ~10 players costs about
0.87 x 93 + 0.39 x 10 x 17.8 = ~150 ms/frame against ~271 ms for SAM2 on
those 10 players: **~1.8x, not an order of magnitude**, before any cost of
handing identities between the two trackers. A narrower trigger -- only
where the detector actually merges two players, or where a box track is lost
in contact -- would fire less often, but needs its own measurement.

### Stitching the fragments

`sort_strict` buys its identity safety with fragments, so `stitch.py` links
its tracklets offline with the simplest plausible rule (an ending tracklet
continues as one starting within 20 frames and 0.75 box heights of its
extrapolated position, one-to-one by Hungarian; optionally also within the
kit-colour gate). A proxy for the identity layer, not `PlayerRegistry`.

| clip | arm | unstitched | motion | motion + colour |
|---|---|---|---|---|
| FelixClaar | oracle | 95.2 / 0.5 / 3.54 | 89.7 / 6.2 / 1.54 | 92.1 / 3.8 / 1.92 |
| Han-Ber4 | oracle | 97.5 / 0.0 / 1.75 | 97.5 / 0.0 / 1.00 | 97.5 / 0.0 / 1.17 |
| BHC-FAG | oracle | 98.2 / 0.0 / 2.83 | 98.2 / 0.0 / 1.33 | 98.2 / 0.0 / 1.33 |
| FelixClaar | detector | 83.8 / 4.2 / 2.77 | 77.7 / 11.2 / 2.00 | 80.9 / 7.6 / 2.15 |
| Han-Ber4 | detector | 90.1 / 3.3 / 2.17 | 90.1 / 3.3 / 1.50 | 90.1 / 3.3 / 1.50 |
| BHC-FAG | detector | 97.2 / 0.1 / 2.00 | 97.1 / 0.2 / 1.50 | 97.1 / 0.2 / 1.50 |

On the easier clips stitching is free: fragments collapse with no new
errors. On FelixClaar it re-creates the very swap the strict tracker avoided
(ids 1/6 at f121, even with the colour gate): breaking a track at a crossing
only moves the decision to the stitcher, and the stitcher has no better
evidence than the tracker had. So fragmentation is a real cost on hard
contacts, not a free pass -- the identity layer would need evidence that
survives the contact (jersey number, a mask-based appearance, or abstaining
and leaving the fragments unlinked).

## What this means

- The plan "fast box tracker + better detector" is not ruled out. With
  perfect boxes a deliberately fragmenting SORT is as identity-safe as SAM2.
- "Better detector" has a specific meaning: separate boxes for players in
  contact. Recall is already high and is not what causes swaps. The contact
  separation rate above is a direct target for further fine-tuning (more
  crowded/contact frames in training data), and a cheap metric to track.
- Fragmentation becomes the identity layer's problem. On easy contacts a
  trivial motion linker removes it for free. On the hard FelixClaar crossing
  it re-creates the swap, so the identity layer must either have evidence
  that survives contact (jersey numbers) or leave such fragments unlinked --
  abstaining, which this project already prefers to a confident wrong link.
- A SAM2-near-contacts hybrid buys ~1.8x at best, because contact is the
  normal state of handball. It is not the route to a fast tracker.
- Appearance in association is worth keeping only as a cheap kit-colour gate
  and only as a soft, separately evaluated option (CLAUDE.md: no team-gated
  association by default). PRTReID adds CPU/GPU cost with no gain here.

## Caveats

- Three short clips, nine error events; FelixClaar carries most of the signal.
- The references are partly mask-propagated (§8.4); the oracle boxes come from
  them and are cleaner than any detector.
- The production-detector runs use an activation threshold and colour gate
  chosen on Han-Ber4; its rows are in-sample.
- Thresholds were calibrated on Han-Ber4 and are in-sample there.

## Next steps

1. Fast tier: a box tracker (SORT, optionally strict, optional kit-colour
   gate) on the production detector run every frame is ~20x cheaper than
   SAM2, and on the same `_v1` detections `sort_strict` beats MCByte on all
   three clips (83.8 / 90.1 / 97.2% against 75.5 / 89.2 / 93.8%). Worth adopting as the fast
   option in place of MCByte -- a decision for the user, since CLAUDE.md
   names MCByte. The colour gate stays off by default (no team-gated
   association) unless evaluated separately.
2. Detector: fine-tune on contact-heavy frames, with the every-frame contact
   separation rate (`detector_versions.py --every-frame`) as the acceptance
   metric. It is the only lever that moves a box tracker on FelixClaar-type
   play.
3. Stitch with the real identity layer: feed `sort_strict` tracklets through
   `PlayerRegistry` (re-ID plus jersey numbers) and score it. The motion proxy
   shows the easy cases are free and the hard crossing is not; the question
   is whether number reads resolve it, or whether the registry correctly
   declines to link.
4. Hybrid: not worth building on an any-overlap trigger (~1.8x). Only revisit
   with a narrow trigger (detector merge or box track lost in contact), after
   measuring how often that fires.
5. Build the contact-event evaluation set discussed on 2026-10-19 before any
   production decision; three clips cannot carry it.

## Reproduce

All CPU (`CUDA_VISIBLE_DEVICES=`), from the repository root with
`PYTHONPATH=src`; outputs land in `runs/tracker_eval/oracle_2026-10-20/`.

```bash
python -m experiments.box_tracker_oracle.run --variants sort bytetrack --tag baseline
python -m experiments.box_tracker_oracle.run --variants sort_strict --tag sort_strict
python -m experiments.box_tracker_oracle.run --arms filled \
    --variants sort sort_strict bytetrack --tag filled
python -m experiments.box_tracker_oracle.run --calibrate han --feature colour   # -> 0.28 / 0.57
python -m experiments.box_tracker_oracle.run --variants colour_gate colour_fuse \
    --gate colour=0.28 --tag colour_q95                                          # also 0.57 -> colour_q99
python -m experiments.box_tracker_oracle.run --variants colour_gate_strict \
    --gate colour=0.28 --tag colour_q95_strict
python -m experiments.box_tracker_oracle.run --calibrate han --feature prtreid  # -> 0.23
python -m experiments.box_tracker_oracle.run --variants prtreid_gate prtreid_fuse \
    --gate prtreid=0.23 --tag prtreid_q95                                        # ~40 min CPU
python -m experiments.box_tracker_oracle.stitch --arm oracle    # and --arm detector
python -m experiments.box_tracker_oracle.detector_versions
python -m experiments.box_tracker_oracle.table
python -m experiments.box_tracker_oracle.render_event felix 1 6 121 out.jpg
# GPU
python -m experiments.box_tracker_oracle.detect_every_frame detector-expanded-v1  # and detector-local
python -m experiments.box_tracker_oracle.run --arms expanded local --activation 0.5 \
    --variants sort sort_strict bytetrack colour_gate colour_gate_strict \
    --gate colour=0.28 --tag current_detectors
python -m experiments.box_tracker_oracle.detector_versions --every-frame
python -m experiments.box_tracker_oracle.contact_coverage
python -m experiments.box_tracker_oracle.sam2_object_scaling
```
