# SAM3.1 multiplex local evaluation — 2026-10-15

**Full-clip quality is mixed, and a temporal-memory crash on the third clip
blocks adoption of this adapter.** SAM3.1 scores above SAM2 on full Felix and
slightly below it on Han. This does not establish general quality equivalence or
regression. Production SAM2 remains the default. The earlier short comparison
and reassessment are retained below.

## Worktree follow-up

Further work is isolated on `experiment/sam31-multiplex` in the sibling worktree
`handball-computer-vision-sam31`. The user accepts a 2-to-3.3 FPS improvement as
worthwhile; 30 FPS is not an adoption minimum.

A real-GPU lifecycle probe now verifies late addition, reprompting, middle-object
removal and remove/re-add of an individual ID, while checking survivor history.
An edge case with missing conditioning anchors for late-added objects was found
and fixed in the isolated adapter. This fix does not change initial seeding or
the existing-object reprompt policy. It does not edit upstream source.

## Full-clip follow-up

Results live under `runs/sam31_full_20261015/`. SAM2 baselines are copies of
historical full runs, rescored with the same labels and matching
tracking-driver/detection/team-model hashes. SAM3 uses the isolated adapter with
the late-object conditioning fix. Complete frame counts and contiguous replay
frames were checked for both successful runs.

| Clip | Frames | Trusted observations | SAM2 correct | SAM3 correct | Difference |
|---|---:|---:|---:|---:|---:|
| FelixClaar | 249 | 2,403 | 95.01% (2,283) | 95.76% (2,301) | +0.75 pp |
| Han-Ber4 | 199 | 2,307 | 99.91% (2,305) | 99.22% (2,289) | −0.69 pp |
| BHC-FAG | 500 requested | — | 98.88% historical | Incomplete: runtime error | — |

Felix has 7 wrong-identity matched observations versus SAM2's 18, and 95 misses
versus 102. Its gain is concentrated in reference player 1 (+35 correct), while
the original reference-13 deficit remains (−21). Han has 3 wrong-identity
matched observations versus zero, and 15 misses versus 2. Reference player 8
accounts for 12 of Han's 16 fewer correct observations. These are scorer counts,
not visual proof of body swaps; the metric combines box coverage and identity.
No target is excluded to improve either result.

The three-clip suite did not complete. BHC reached frame 370 of 499 propagated
frames before `torch.cat` in `_prepare_memory_conditioned_features` encountered
object pointers with conflicting bucket dimensions (1 and 2). Its directory
contains the initial manifest and captured diagnostic masks but no complete
replay or score. The traceback is preserved in `bhc_sam3.log`; no partial quality
score is presented as a completed clip.

A separate real-GPU reproduction seeded 16 IDs, propagated two frames, then
added ID 17. It reproduced the same upstream failure immediately: the original
conditioning frame retained pointer shape `[1,16,256]`, while the insertion frame
had `[2,16,256]`. Report:
`runs/sam31_bucket_growth_20261015/report.json`. Repeated real boxes were used as
API stress inputs, not quality evidence. Upstream insertion expands the current
output without expanding older pointers; removed slots remain reserved, so
bucket growth can occur through player churn even with fewer than 17 active IDs.
The BHC exception has the same incompatible bucket dimensions; its full state
was not serialized, so its exact insertion sequence is not claimed here.

The successful lifecycle probe covers a single bucket. Multi-bucket history
handling remains an adoption blocker; passing add/remove/reprompt in the short
probe is not a general lifecycle guarantee. No global reset or memory-discarding
workaround was substituted, because that would change the tracking behavior
whose quality is under evaluation.

These full runs are **not a fresh paired speed comparison**. SAM3 loop throughput
was 1.68 FPS on Felix and 3.20 FPS on Han under a shared GPU workload. The historic
SAM2 timings were collected separately. Use the matched short-run 1.97 versus
3.31 FPS measurement below for the existing speed evidence, with its stated
limits. Two completed short clips and one failed clip cannot establish
whole-match accuracy or deployment reliability.

## Matched test

FelixClaar frames 0–140 inclusive; 13 initial targets, one SAM3 multiplex bucket,
the same cached RF-DETR boxes, shared production manager and detector checks
every ten frames. Each backend ran in a fresh process with a separate 12-frame
warmup and fresh measured state, BF16 on GB10, eager/uncompiled. Native model
resolutions are SAM3 1008 and SAM2 1024. Historical team-model re-ID features
are shared, not the separate PRTReID pipeline. Neither test includes live
detection, PARSeq or rendering. No production code/defaults were changed.

| Prefix metric (frames 0–140) | SAM2.1-L | SAM3.1 multiplex |
|---|---:|---:|
| Trusted reference observations | 1,644 | 1,644 |
| Correct matched observations | 1,592 | 1,573 |
| Correct percentage | **96.84%** | 95.68% |
| Matched observations | 1,602 | 1,577 |
| Missing reference observations | **42** | 67 |
| Wrong-identity matched observations | 10 | **4** |
| Scorer identity switches | 7 | **6** |
| Measured loop seconds | 71.23 | **42.33** |
| Diagnostic loop FPS | 1.97 | **3.31** |
| Median / p95 resume latency | 385 / 890 ms | **261 / 533 ms** |
| Peak CUDA allocation | **5.06 GiB** | 5.72 GiB |

The quality regression is **19 fewer correct observations, or 1.16 percentage
points**. SAM3 has fewer scorer identity errors but more missed matches, so
describing this as a wholesale identity collapse would be incorrect. Speed is
1.68x in this single diagnostic measurement, far below a demonstrated 30 FPS.
Mask capture and frame preparation are included; initial SAM3 prompt work is
deferred into its loop. This is not an optimized or statistically repeated
speed comparison. Shared GPU services remained running, and memory pressure
increased after the runs; no unrelated services were stopped.

## Reassessment after the user's uncertainty question

The 1.16-point difference is an exact score on this fixed prefix, not a measured
confidence interval for performance on other videos. The 1,644 observations are
temporally correlated and cannot be treated as independent statistical trials.
No defensible margin of error or cross-video non-inferiority conclusion has been
established by this run. Repeating the same deterministic footage is not a
substitute for more independent clips.

Paired analysis using each run's prefix-wide dominant-ID mapping shows:

- Target 13 contributes **21 fewer correct observations** for SAM3.
- All other reference targets together contribute **two more correct
  observations** for SAM3. This is diagnostic, not permission to exclude target
  13 from acceptance scoring.
- Frames 0–20 account for 20 fewer correct observations; frames 21–140 yield
  **one more** correct observation for SAM3.
- SAM3 has more correct matches on 19 frames, fewer on 38, and ties on 84.

Thus the evidence points to a concentrated initialization/partial-mask problem,
not a demonstrated broad identity-quality loss. Check that initialization against
the upstream prompt path and evaluate additional clips before deciding whether
the configuration can meet the no-regression requirement. No accuracy-loss
allowance is inferred from the user's observation about uncertainty.

## What caused the measured regression

Reference target 13 accounts for most additional misses: SAM3 misses 22 trusted
matches versus SAM2's one. Its early mask-derived boxes are too short. At frame
1, for example, SAM3's box is approximately `(57,469,112,540)` versus the
reference `(56,466,131,615)`; SAM2 gives `(56,469,117,572)`. SAM3 misses the
reference IoU gate during frames 1–20, and its detector reprompt at frame 20
restores more complete geometry from frame 21. These are matching/coverage
failures, not proof of a wrong physical identity for that target.

In the overlap window, frames 95–130, SAM2 scores 94.72% (359/379) versus SAM3's
94.20% (357/379); both have zero wrong-identity matches within that window's
local dominant-ID mapping. Both differ in mask geometry and partial-occlusion
coverage. Reference 8 is trusted only through frame 90 and reference 10 only
through frame 113, so automatic scoring cannot certify their entire crossing.

All 36 frames 95–130 were inspected in paired mask contact sheets for predicted
IDs 8 and 10. Both models keep distinct masks on the two bodies through the
reviewed crossing; no equivalent of EfficientTAM's two-target mask collapse was
seen. SAM3 sometimes retains more visible fragments during partial occlusion,
but that does not eliminate the prefix coverage regression. This visual review
covers that pair and interval, not every player or an entire match.

Review artifacts:
[frames 95–106](../runs/sam31_quality_20261015/review/paired_overlap_095_106.jpg),
[107–118](../runs/sam31_quality_20261015/review/paired_overlap_107_118.jpg),
[119–130](../runs/sam31_quality_20261015/review/paired_overlap_119_130.jpg).

## Setup and limits

The existing released checkpoint was loaded with strict completeness: 931
tensors / 486,715,574 parameters, including the shared vision encoder and both
trained mask heads. Model and temporal state run on GPU. Batched mask seeding
kept the measured run in one bucket. Periodic correction and target removal
were exercised. A first three-frame smoke attempt found an upstream standalone
encoder-output error; the isolated adapter requests only tracker necks and the
repeated smoke passed. No upstream checkout or installed dependency was edited.

RF-DETR boxes are converted using SAM3's trained interactive box head and passed
as masks to the multiplex tracker. This conversion and reconditioning policy
are part of the tested configuration; the result is not proof that every possible
SAM3.1 integration must be worse. Initialization, late-entry behavior, compilation
and a possible SAM2-seeded hybrid would require separate validation. None is
silently substituted into this result or counted as an accuracy pass.

Reproducible commands and caveats:
[experiment README](../experiments/sam3_multiplex/README.md).
Raw manifests, replays, masks, events and scores:
`runs/sam31_quality_20261015/{sam2_felix141,sam3_felix141}/`.
