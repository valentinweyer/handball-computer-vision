# EfficientTAM-S paired results — 2026-09-10

**The trained candidate runs through the existing lifecycle, but does not yet
justify replacing SAM2.** Eager EfficientTAM-S at 1024 improved complete-loop
throughput by 17–26% in the first paired runs. Aggregate correctness fell on two
clips and was nearly unchanged on the third; continuity/mixing results vary.
The hoped-for 2× gain did not appear. Actual paired masks confirm a same-team
tracking collapse on FelixClaar that SAM2 avoids through the inspected window.
Plain EfficientTAM-S therefore fails the continuity gate for default replacement.

## What was executed

Six full clip runs: SAM2.1 Hiera-L and official EfficientTAM-S, one warmed run per
backend per clip, sequentially on GB10. Each measured state/manager was fresh
after a separate 21-frame warmup. Both used 1024 input, BF16 prompts/propagation,
eager execution, cached RF-DETR, checkpoint interval 10, the existing fitted team
model, identical mask post-processing and all live objects. Backend order was
SAM2/ET on Han-Ber4, ET/SAM2 on FelixClaar, SAM2/ET on BHC-FAG.

This is the historical **tracking-only** configuration with team-model appearance
features. It does not include live RF-DETR inference, PRTReID, PARSeq, rendering
or video writing; those deployed components and their defaults were not changed.
The full loop includes propagation, CPU masks/filtering/boxes, manager updates,
checkpoint reads/decisions/prompts and collection of output boxes. Model build,
fresh-state initialization/seeding and compressed artifact writing are separate.
CUDA was synchronized at full-interval boundaries, not per kernel.

Checkpoint: [official efficienttam_s.pt](https://huggingface.co/yunyangx/efficient-track-anything/blob/9bdd8ab/efficienttam_s.pt),
SHA-256 `2b572be30d9e96ee29c8d785fe157c6b079ede7d56fbc8a3671d4120e63c89cd`.
Source: EfficientTAM `abcd061ebd3cc6e7527d152d75b890126aaa53f6`;
SAM2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`; project `9952d0c` plus the
recorded factory change. Plain `_s` was tested, **not** `_s_1`, `_s_2`, or 512.
Encoder compilation was explicitly disabled despite EfficientTAM's YAML default.
Both source checkouts lack the optional hole-filling CUDA extension; both
reported skipping that step. The project's OpenCV fragment filter stayed active.

## Measured pair

Correct percentages below restrict **both numerator and denominator** to the
same trusted player reference IDs. See the Han-Ber4 reporting correction below.

| Clip | Frames propagated | SAM2 FPS | ET FPS | Speed ratio | Correct SAM2 → ET |
|---|---:|---:|---:|---:|---:|
| felix | 248 | 2.38 | 3.00 | 1.26× | 95.01% → 94.26% |
| han | 198 | 2.13 | 2.49 | 1.17× | 99.91% → 99.26% |
| bhc | 499 | 2.22 | 2.80 | 1.26× | 98.88% → 98.94% |

| Clip/arm | Wrong-ID frames | Unmatched reference frames | Mixed tracklets | Within-tracklet switches | Fragments/reference ID |
|---|---:|---:|---:|---:|---:|
| felix / sam2 | 18 | 102 | 5 | 8 | 1.38 |
| felix / efficienttam | 13 | 125 | 5 | 11 | 1.38 |
| han / sam2 | 0 | 2 | 0 | 0 | 1.00 |
| han / efficienttam | 2 | 15 | 2 | 4 | 1.17 |
| bhc / sam2 | 6 | 51 | 3 | 6 | 1.42 |
| bhc / efficienttam | 9 | 45 | 4 | 8 | 1.58 |

Unmatched means failure to match a trusted reference **box** at IoU ≥ 0.5. It
must not automatically be interpreted as a player disappearing or a true ID
loss. Within-tracklet switches count changes in matched reference identity;
predicted-ID changes along one reference player's timeline are also exported.

| Clip | Mean live objects SAM2 → ET | Peak live objects SAM2 → ET | Peak CUDA allocation GB SAM2 → ET |
|---|---:|---:|---:|
| felix | 11.60 → 11.48 | 13 → 13 | 7.04 → 5.80 |
| han | 13.38 → 13.69 | 14 → 14 | 6.64 → 5.45 |
| bhc | 12.42 → 12.34 | 14 → 14 | 13.15 → 11.91 |

The candidate did not achieve speed by imposing a lower object cap. Small natural
occupancy differences result from the shared manager acting on different masks.
Its model build was roughly 0.45–0.49 seconds versus SAM2's roughly 1.4 seconds
on Han/Felix; fresh state plus seeding still took several seconds. Full per-run
startup, resume-latency and memory numbers are in the manifests. CUDA allocation
is not a measurement of all physical memory used on the unified-memory GB10.

These are **single warmed measurements**, not a three-repeat latency study.
During some runs, CPU-only scoring/tests ran concurrently; no competing GPU job
was used. Small timing differences therefore deserve a more isolated repeated
benchmark before a deployment decision. The data supports a modest initial gain,
not a precise universal speed factor or a claim of real-time full-application FPS.

## Continuity and visual review

- Han-Ber4: candidate reference-box matching exchanges IDs 3/8 at frame 131 and
  returns at 132. Existing verified labels make this an opposite-team pair.
  SAM2 has no such scored exchange. Actual candidate masks at frames 130–132
  retain the correct white-shirt and red-shirt players under IDs 3 and 8. The
  diagnostic replay/events match the measured run exactly. This scored exchange
  is a box-matching artifact, **not a verified body swap**.
- FelixClaar: wrong-ID frames improve 18 → 13, but unmatched frames rise
  102 → 125. Most of the extra misses are reference 11: 27 → 85, with its longest
  constant-ID match run shrinking 116 → 50 frames. Other players improve.
  Pixel/box review at frames 1, 15, 29, 45, 60 and 90 shows the candidate often
  retains the partly obscured white-shirt player in its box but extends the box
  down across a foreground blue-shirt occluder. This is evidence of a geometry
  problem, not proof of 58 extra frames of lost identity.
- Felix's reference IDs 8/10 are both blue-striped teammates in the inspected
  reference crops. Candidate track 8 later matches reference 10, but the original
  reference-8 trusted span ends at frame 90 and the next match is at 109. This
  cannot by itself locate or prove an acute same-team handoff. Dense actual-mask
  review below supplies the missing evidence without changing the trusted spans.
- BHC-FAG: correct percentage rises slightly because six extra matched frames
  outweigh three extra wrong-ID frames. Mixing increases 3 → 4 tracklets and
  fragments/reference ID 1.42 → 1.58. Candidate track 14 is retired/revived twice
  and matches references 4, 3 and 8 at widely separated times. Inspected reference
  crops show all three in dark jerseys. These events concern reused identity
  across lifecycle gaps, not a verified uninterrupted same-team crossing.
  Candidate re-ID revivals are 2 versus SAM2's 0; reprompts are 3 versus 2.

Separate mask-capture runs reproduce the measured candidate boxes/events exactly
on Han-Ber4 and FelixClaar. Actual-mask review confirms that Felix reference 11
keeps its white-shirt torso in mask 11, while wrongly including a blue-striped
foreground sleeve in the inspected early frames. By frame 60 that extra sleeve
is absent. This explains much of the box-IoU loss without inventing an identity
loss. The overlap masks remain relevant to downstream crop/jersey usability.

For the same-team Felix case, dense candidate masks locate the collapse at
**frame 107**: IDs 8 and 10, previously following separate blue-shirt players,
now cover the same teammate. Their mask IoU rises from 0 at frame 106 to 0.765
at 107 and 0.966 at 110. The duplication persists through frame 120; the manager
then retires ID 8 as a duplicate and revives it at 150. Thus the visual evidence
is stronger than the isolated scored transition at 109.

The completed **SAM2 actual-mask comparison confirms a candidate regression**:
SAM2 keeps ID 8 on the partly occluded original player and ID 10 on the other
blue-shirt teammate at frames 106, 107, 110 and 120. Its mask IoU between those
IDs stays at zero through 117 and reaches only 0.006 at 120, versus 0.904 for
EfficientTAM. The SAM2 diagnostic replay and events also reproduce its measured
arm exactly. This is a verified same-team collapse beginning at frame 107,
persisting for 14 returned frames through 120, followed by manager retirement;
a later re-ID revival does not restore uninterrupted tracking. The original
reference-8 trusted span ends at 90, explaining why the aggregate score misses
much of this failure. No reference span or identity label was extended. SAM2
also has later scored failures; this finding concerns the inspected overlap
window, not a claim that its entire clip is error-free.

Review artifacts: `review/han_masks_131.jpg`, `review/felix_masks_ref11.jpg`,
`review/felix_same_team_8_10.jpg`, `review/felix_same_team_paired_masks.jpg`,
`review/felix_same_team_overlap.json` and `review/felix_same_team_event.json`.

The scorer exports missing runs, constant-ID runs and raw per-frame matches;
these do not silently reconnect fragments through jersey aliases. Human labels
and trusted spans were kept unchanged. Han-Ber4 still uses a SAM2-built reference;
Felix/BHC use SAM+Cutie references with known sparse-verification limits. New
entrants, complete occlusions and BHC's truncated pileup spans still need a
small explicit event ledger for a definitive trajectory-quality comparison.

## Han-Ber4 denominator discrepancy

The historical 89.2% is reproduced as **2305 / 2585 = 89.17%**. Its numerator
excludes bench/reference IDs 12 and 14, while its denominator includes them.
Within the same player scope the baseline is **2305 / 2307 = 99.91%**. The
candidate is **(2292 − 2) / 2307 = 99.26%**, or **88.59%** using the historical
all-reference denominator. The normalized paired difference is −0.65 percentage
points. This does not change any matches, wrong-ID counts, or ranking. Both
percentages and both denominators are saved; previous result artifacts were not
rewritten. Felix and BHC do not have this player/all-reference denominator mismatch.

## Implementation, proof and next decision

The production diff only adds an optional predictor factory to `drive_sam2` and
makes its default SAM2 builder import lazy. RF-DETR inputs, `TrackManager`,
`PlayerRegistry`, object memory/lifecycle, checkpoint policy and `Sam2FrameResult`
remain shared. SAM2 is still the default. Experiment scripts select and score
explicit, separate replays rather than overwriting old baselines.

The same six lifecycle smoke scenarios also passed with the **trained** Small
checkpoint. The full project suite passes **266 tests**, including new tests for
default/injected construction, middle removal, same-ID reset, mask/ID ordering,
clean fragmentation and gaps under a reused ID. Fresh SAM2 arrays (frame indices,
IDs and boxes) and lifecycle events match the existing `*_policy_reprompt`
artifacts **exactly on all three clips**.

**Decision:** retain SAM2. Plain EfficientTAM-S is runnable and somewhat cheaper,
but the verified same-team collapse fails the required continuity gate despite
its modest speed gain. Do not use downstream identity fixes to declare parity.
This does not reject the entire EfficientTAM family: `_s_1`/`_s_2` compressed-memory
variants are distinct future experiments, each needing matching weights and the
same paired gates. Compilation and EdgeTAM reset/reseed remain rejected; DAM has
not been integrated or evaluated.

Artifacts are under `runs/efficienttam_pair/`: `summary.json`, per-arm
`manifest.json`, `replay.npz`, `events.json`, `frames.json`, `matches.json` and
`score.json`. `review/` contains diagnostic contact sheets. Mask-capture runs
have separate directories and must not be used as speed measurements.
Reproduction commands: [experiment README](../experiments/efficienttam/README.md).
