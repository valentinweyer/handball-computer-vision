# Player-tracker research — 2026-10-19

This is a source review, not a new tracking benchmark. Papers, author repositories,
inference code and public checkpoint listings were checked on 19 October 2026.
No model was installed or run, and production code and defaults were not changed.

## Decision

Keep production SAM2.1-L with periodic RF-DETR corrections. Under the requirement
to preserve its tracking quality, the primary new experiment should be
**SAMIDARE's coordinated lifecycle, reprompt and memory policy on SAM2**.
Use the verified exact SAM2 optimization stack as the speed foundation.

ARC-SORT and SRITrack belong in a separate, optional fast-tier investigation,
compared first with SORT and MCByte. A cheap adapter alone is insufficient reason
to prioritize a box tracker as a SAM2 replacement: the local benchmark already
shows substantial coverage and identity advantages for mask propagation.
Selective Mask Propagation (SMP) first needs a cheap dispatch-feasibility check.

This revises the original ARC-first experiment order after review against our
local failure evidence. No candidate has established equivalent handball quality,
and SAMIDARE itself has no established speed advantage over our manager.

| Candidate | Reason to investigate | Main uncertainty | Suggested experiment |
|---|---|---|---|
| SAMIDARE | Targets crowded mask contamination and duplicate track births | Policy transfer to our manager; no demonstrated speed benefit | Primary quality experiment on SAM2 |
| Selective Mask Propagation (SMP) | Retains SAM on ambiguous intervals | Dispatch may cover most frames or miss actual swaps; GTA overlaps existing identity repair | Dispatch-feasibility check before an adapter |
| SORT versus MCByte | Already measured fast options; SORT reduced wrong-ID time on Felix | MCByte with masks scored better on BHC; full mask contract and runtime | Establish the fast-tier baseline before new trackers |
| ARC-SORT | Explicit RF-DETR support; revisits ambiguous crossing decisions | Same-jersey appearance; offline gains dominate its headline | Optional fast-tier screen against existing baselines |
| SRITrack | Online sports tracking designed for re-entry and identity continuity | Re-entry differs from visible teammate contact; ReID and GB10 cost | Lower-priority fast-tier experiment |

All preserve RF-DETR in principle. Published box-tracker protocols use detections
every frame; retaining our ten-frame detector cadence would be a separate,
unvalidated configuration. They also need a separately timed mask-generation
policy for downstream overlap/jersey consumers. Sparse boxes alone do not satisfy
that mask contract. Image-only or selective masks could have different costs;
their total runtime and downstream quality remain unmeasured.

## Local baseline and lessons

- [Current tracking evaluation](tracking-evaluation.md): SAM2's corrected
  player-only JPEG-input benchmark scores are 95.01% FelixClaar, 99.91% Han-Ber4
  and 98.88% BHC-FAG.
  These are correctly matched trusted observations, including misses in the
  denominator; they are not HOTA/IDF1 or complete-trajectory accuracy. The
  October 17 direct-video experiment scores 96.21%, 99.91% and 98.96%, respectively;
  keep frame source fixed within each candidate/baseline pair.
- [CUTIE validation](../experiments/cutie/VALIDATION.md): the user's unseen-clip
  review rejected its identity keeping despite favorable aggregate scores and
  roughly 3x speed. Do not rank candidates using model-derived reference scores
  alone. [EfficientTAM](efficienttam-paired-results.md) also failed a visually
  confirmed same-team overlap that SAM2 survived.
- [SAM3.1 results](sam31-multiplex-results.md): the bucket-growth crash is fixed.
  The completed paired suite is faster, but regresses on Han and BHC. It remains
  an existing quality investigation, not a new untested replacement.
- [Exact SAM2 optimization stack](sam2-speed-research.md#further-exact-sam2-speedups-on-gb10-2026-10-16):
  fused RoPE, CPU/GPU overlap and small-module CUDA graphs already achieved
  approximately 1.20x complete tracking-loop throughput on all three clips,
  with identical IDs, boxes and binary masks in both repeats. These are isolated
  experiment patches; production adoption is not established by this report.
  Compare future runtime claims with this available optimized baseline as well
  as the production baseline, using a matched harness.

### What the earlier box-tracker evidence establishes

The [historical recall ceiling](tracking-evaluation.md#83-why-recall-moved-too-not-just-identity-accuracy)
applies to the tested adapters, which returned detection-supported observations.
Online motion prediction can bridge short detection gaps in other output policies;
it supplies no new visual localization evidence during the miss. That distinction
does not remove SAM2's measured advantage on our footage.

SORT's better local result does not isolate the causal effect of appearance.
The [evaluator](../scripts/evaluate_tracker_identity.py) constructs library-default
BoT-SORT without supplying an appearance model; the installed implementation
inspected during review has IoU/score association and camera-motion compensation,
with no ReID path. The historical package revision was not reconstructed. Those
scores are not a controlled appearance-on/off experiment. Separately, our failed
appearance-based split/connect tests are direct evidence against those particular
features and policies on our footage.

The [0.55 PRTReID rank-1 measurement](team-classification-handoff.md#measurement)
is single-crop retrieval with a different gallery from runtime. The handoff
explicitly notes that EMA and a smaller runtime gallery can improve the absolute
result. It warrants skepticism about same-team appearance, but is not a measured
limit on DINOv3 or deferred decisions using later clear views.

The same caution applies to metrics: a published gain of three HOTA points cannot
be compared numerically with our correct-observation margin. The often cited
Felix +12.3 points used the old 93.8% SAM2 result; current JPEG SAM2 is 95.01%
against SORT's historical 81.5%. The local comparison remains the stronger reason
to prioritize retaining mask propagation.

## ARC-SORT: optional fast-tier baseline

The [paper](https://arxiv.org/html/2610.22306) and
[released implementation](https://github.com/Athlitix/arcsort) combine multiple
motion models, camera-motion compensation and appearance matching. Ambiguous
crossings remain under review until later appearance evidence supports one
corrective identity swap. This is a limited pairwise correction, not general
reasoning through a multi-player pileup.

The SportsMOT test headline is **81.82 HOTA / 88.47 IDF1**, including interpolation
and offline tracklet merging. On Deep-EIoU's identical detections and embeddings,
it reports **80.31 / 86.92**, versus Deep-EIoU's **77.2 / 79.8**; that matched row
does not include the newer deferred-arbitration mechanism. These results cannot
be compared numerically with our local correct-observation percentages.

The authors' [RF-DETR compatibility check](https://github.com/Athlitix/arcsort#tested-detectors)
uses three SportsMOT validation clips, 1,616 frames: **67.6 HOTA online**, **71.6
with offline merging**, versus BoT-SORT's **65.9** on the same RF-DETR-L outputs.
The authors explicitly identify this as a small compatibility check, not a
benchmark. It supports feasibility, not handball accuracy.

The [API](https://github.com/Athlitix/arcsort#quick-start-track-with-any-detector)
accepts scored detection arrays and images. No tracker-specific training is
required; its appearance path uses existing pretrained weights. Its CPU core
and optional PyTorch ReID path make an isolated adapter plausible on our stack,
but neither aarch64 execution nor GB10 throughput has been tested here.

**Timing correction:** the paper labels 50.7 FPS on an RTX 3060 Laptop as
"full", but the [benchmark source](https://github.com/Athlitix/arcsort/blob/2a580c592c838bce99b89a6a2dbd7ad6f50c84be/tools/benchmark_fps.py)
reads cached detections, optionally uses cached camera-motion transforms, and
measures tracking, live OSNet and JPEG decoding. It excludes live detection and
offline post-processing. Do not call this
detector-to-final-result throughput.

**Inference:** inexpensive to screen within the fast tier, particularly for
two-player crossings and camera motion. That does not establish a strong case
for replacing SAM2. Later appearance evidence may still be ambiguous for teammates;
the offline appearance-only merge can also join different same-jersey players.
Score raw output and retrospective corrections separately.

## SRITrack: optional sports-trained fast-tier candidate

The authors report **85.2 HOTA / 91.3 IDF1** on SportsMOT, with online inference
and sports-trained assets. It explicitly addresses visibility at entry/exit and
re-association after frame-out periods.
[Author publication](https://scholar.lib.ntnu.edu.tw/en/publications/sritrack-online-multi-object-tracking-for-sports-broadcasting-wit/),
[code and published weight link](https://github.com/kaoyuyukao/SRITrack).

The released default uses DINOv3 ViT-B/16 appearance features on 384-pixel crops,
with FP32 configured in the runner. No matched 14-player GB10 latency was found.
The sports ReID checkpoint is linked on Google Drive; download and tensor
completeness were not verified. A public Hugging Face name search returned no
SRITrack repository, which does not invalidate the published Drive release.
[Backbone](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/reid/backbones.py),
[extractor](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/reid/reid_extractor.py),
[configuration](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/setting.yaml).

Source inspection found three reasons to adapt its core rather than run the
public-detection demo unchanged:

1. The [det.txt path](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/tracker/track_engine.py)
   converts coordinates to integers, replaces every confidence with 0.9, and
   rereads the full detection file each frame. Pass RF-DETR's original scored
   floating-point boxes directly to the tracker instead.
2. [track.py](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/track.py)
   builds YOLOX even when public detections are selected. Bypassing detection
   during frame processing does not eliminate that startup dependency.
3. The [weight loader](https://github.com/kaoyuyukao/SRITrack/blob/80ca110a3e0e4874a8700fa8bd57f84cdfa2d919/reid/tools.py)
   accepts partial key/shape matches and continues even if nothing matches.
   A local trial must audit every inference-relevant tensor before accepting
   any quality result. Changing FP32/TF32 behavior is a separate speed trial.

**Inference:** a stronger sports-specific appearance hypothesis than another
generic SORT variant, but lower priority under our quality requirement. Its
re-entry focus does not establish that it solves close same-team contact while
both players remain visible. The new features are untested locally.

## Selective Mask Propagation: establish useful dispatch sparsity first

The [latest paper, v3](https://arxiv.org/html/2606.13033v3), reports SportsMOT
test **87.2 HOTA / 93.6 IDF1** for SAM3+global track association (GTA), and
**85.5 / 91.2** for SAM2+GTA. The older 86.8/93.2 headline belongs to v1.
Crucially, the validation ablation is base **78.9 HOTA**, selective SAM3 alone
**80.4**, GTA alone **86.7**, both **88.5**. Most of the gain comes from
retrospective linking, not simply dispatching SAM selectively.

The [author implementation](https://github.com/holma91/selective-mask-propagation)
starts from a cheap tracker and uses assignment ambiguity to choose intervals
for SAM. Its 13 FPS figure on RTX PRO 6000 measures SAM+merge only, excluding
the detector, base tracker and GTA. The more relevant indoor basketball examples
show 8.7/9.5 versus 6.0/6.1 FPS for selective versus uniform SAM3 on RTX 5090.
These are neither GB10 measurements nor a guaranteed order-of-magnitude gain.

The [augmentation API](https://github.com/holma91/selective-mask-propagation/blob/6472b52e79c47a7793cd4c6cabd64e812a8e5852/selective_mask_propagation/augment.py)
accepts external box tracks and assignment margins, so RF-DETR can remain.
The SAM2 path uses our SAM2.1-Large checkpoint family. Public Hub metadata
confirms actual OSNet, ViTPose+ and PARSeq files in the
[author checkpoint collection](https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/tree/main/checkpoints),
checked with the installed `hf` CLI without downloading weights.

Its public output is corrected box tracks. The paper's default mode preserves
base box geometry; mask-authoritative mode is a different configuration.
Downstream masks need lower-level output and a measured generation policy.
GTA additionally uses pose, jersey OCR and Gemini-derived team labels. Our
existing number-anchored linking already repairs some clean fragmentation,
although it cannot repair a live body swap. Establish the additional failures
GTA would fix before duplicating it. Reusing our independent team/jersey evidence
changes that published system and must be evaluated as our own configuration.

**Inference:** conditional follow-up if later frames may revise earlier identities
and dispatch is sparse enough to save work. A confidently wrong base assignment
can avoid the ambiguity trigger, and dense handball may dispatch SAM so often that
expected savings disappear. Measure merged dispatch intervals, active object-frames
and coverage of known failure events before building a full adapter. Frame
percentage alone is insufficient because targets and overlapping windows affect
cost. A cached-box proxy is diagnostic, not proof of the native SAM schedule.
Only pursue integration if that check supports useful savings; then validate
Python 3.12/native-extension dependencies in an isolated environment.

## SAMIDARE: primary new quality experiment

[SAMIDARE](https://arxiv.org/html/2604.22162v1) combines density-aware detector
reprompts, selective per-object memory updates and state-aware reactivation/new
track creation. It uses existing SAM2.1-L weights and needs no new tracker
training. Test SportsMOT is **77.3 HOTA / 78.6 IDF1**, versus SAM2MOT's
**76.4 / 76.5**. Its validation HOTA rises **83.7 to 86.2**; memory gating alone
falls to **83.2**, whereas state-aware association alone reaches **85.6**.
Test the coordinated policy rather than promising that memory freezing solves
swaps by itself.

The [released inference script](https://github.com/ZabuZabuZabu/SAMIDARE/blob/4e663fa384dcbfd0868f39db0d78933dfe1170c1/samidare_for_vis.py)
reads detector results. RF-DETR can replace the published SportsMOT-trained
YOLOX, but the paper uses detections every frame and validation-selected
thresholds. Our detector and ten-frame correction cadence are a new protocol.
This is a plausible accuracy improvement, not evidence of faster propagation.
Porting its SAM2 memory policies into SAM3.1 multiplex would be a separate
adaptation with different shared-state constraints.

## Other candidates and reasons to defer

| Candidate | Verified evidence | Decision for this project |
|---|---|---|
| DAM4SAM / d4sm | Training-free distractor-aware SAM2 memory; released [multi-target code](https://github.com/alanlukezic/d4sm). [Paper](https://arxiv.org/html/2509.13864v1) reports better distractor tracking with a speed cost. | Good quality idea after SAMIDARE; initialize/track wrapper does not demonstrate independent late add/remove/correction. |
| SENTRY | Released [code](https://github.com/HamadYA/SENTRY); [paper](https://arxiv.org/html/2606.24449v1) reports SAM2-L 44.0 to 32.8 FPS on A100. | Reverse-checking rescue predictions may help wrong identities, but adds cost. Released defaults use severe-rescue memory policy, not unconditional memory rewriting. |
| SAM2++ | [Code](https://github.com/MCG-NJU/SAM2-Plus) and [801 MB checkpoint](https://huggingface.co/MCG-NJU/SAM2-Plus/tree/main) are released; Hub listing independently verified. | Separate trained model, not an exact SAM2 optimization; no established handball lifecycle or GB10 advantage. |
| NOOUGAT | [Paper v2](https://arxiv.org/html/2509.02111v2): online SportsMOT 81.0 HOTA/85.3 IDF1; offline 85.6/92.3, learned association across temporal horizons. | Strong research fit, but no official implementation or checkpoint was located; not an immediate experiment. |
| LiVOS | [Paper](https://arxiv.org/html/2411.02818v1) reports faster inference with worse segmentation than CUTIE; [core](https://github.com/uncbiag/LiVOS/blob/main/livos/eval.py) needs correction/removal adaptation. | Weak quality bet after CUTIE's rejection. |
| MAVOS | [Paper](https://arxiv.org/html/2403.17937v2): 63.3 LVOS J&F at 37.1 FPS on V100/480p, with acknowledged similar-object occlusion failures. [Code](https://github.com/Amshaker/MAVOS) has weights. | Default engine capacity is ten objects, so fourteen players need two propagation groups; no convincing same-team advantage. |
| StreamDAM | [Paper](https://arxiv.org/html/2608.03912v1) has encouraging presence-aware streaming results at 30 FPS on RTX6000D, single-object. | No official released code/weights located; 14-player runtime unestablished. |
| DeepStream MaskTracker | TensorRT SAM2 route. [DeepStream 9.1](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Installation.html) supports a Spark container, but the [MaskTracker-specific sample](https://github.com/NVIDIA/DeepStream/blob/main/src/apps/reference_apps/deepstream-masktracker/README.md) requires x86+dGPU. | Verify full tracker support before a port. Overall Spark support is insufficient evidence; no matched 14-player speed found. |

[TeamTrack](https://github.com/AtomScott/TeamTrack) supplies actual handball
annotations, unlike SportsMOT's basketball/football/volleyball footage. It is
useful for a later independent evaluation, but its fixed full-pitch cameras
differ from our broadcast close-ups. It is not evidence that its tracker beats
our baseline, and dataset collection is not a prerequisite for the first screen.

## Concrete next evaluation

1. Evaluate SAMIDARE's coordinated reprompt, per-object memory and lifecycle
   policy through an isolated SAM2 experiment, retaining RF-DETR and the existing
   identity/team contracts. Keep the ten-frame correction cadence fixed in the
   first comparison; adapting the paper's every-frame policy is a new experiment.
   Prove additions, exits, empty frames, per-ID correction and survivor history.
2. Score complete Felix/Han/BHC clips against unchanged trusted spans; separately
   render the unseen Hannover/VfL clips already useful in rejecting CUTIE.
   Review same-team contact and every persistent disagreement against pixels.
   Evaluate immediate output and retrospective output separately, recording
   exactly how long each correction takes to become available.
3. Record wrong-ID observations, coverage, fragmentation, swaps, duplicate births
   and masks needed downstream. Clean fragmentation must not count as success
   merely because a dominant-identity scorer hides it. Do not let one clip's
   improvement compensate for a severe new failure elsewhere.
4. Measure quality and runtime against matched production SAM2 and the available
   exact optimization stack. Include model setup, mask post-processing, corrections
   and detector cost in named intervals. Run GPU measurements serially. SAMIDARE
   may improve quality without improving speed; report those outcomes separately.
5. If a cheap cached-box replay can expose usable assignment margins, estimate
   SMP's merged dispatch demand and whether it catches known swaps. Stop before
   the full adapter if demand remains near dense propagation or failures escape.
6. For a separately chosen faster fallback, first compare SORT and MCByte across
   all three clips, mask needs and runtime. SORT's Felix advantage alone does not
   justify switching: MCByte with masks scored 95.9% on BHC versus SORT's 95.3%.
   Screen ARC-SORT against that fast-tier baseline only if further work is useful;
   consider SRITrack later after auditing its checkpoint and score-preserving
   adapter. These trials do not inherit SAM2-quality claims.

There is no evidence here that justifies switching production trackers or
claiming 30 FPS with preserved handball identity quality.

## Source revisions and checks

Public GitHub head revisions read during this review:

| Repository | Revision |
|---|---|
| Athlitix/arcsort | `2a580c592c838bce99b89a6a2dbd7ad6f50c84be` |
| kaoyuyukao/SRITrack | `80ca110a3e0e4874a8700fa8bd57f84cdfa2d919` |
| holma91/selective-mask-propagation | `6472b52e79c47a7793cd4c6cabd64e812a8e5852` |
| ZabuZabuZabu/SAMIDARE | `4e663fa384dcbfd0868f39db0d78933dfe1170c1` |

Only primary papers, author code/model listings and vendor documentation support
the recommendations. Public metadata checks used `hf datasets list` and
`hf models list`; weights were not downloaded or executed. Published FPS is
kept with its hardware, resolution and measurement scope. Existing failures
were not rerun and no acceptance threshold was relaxed.
