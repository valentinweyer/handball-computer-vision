# Quality-first tracking research — 2026-10-15

## Decision and acceptance rule

**Local test update:** the first matched GPU quality gate is complete. The tested
SAM3.1 adapter scores 95.68% versus SAM2’s 96.84% on FelixClaar frames 0–140,
with 3.31 versus 1.97 diagnostic FPS. Paired reassessment shows the deficit is
concentrated in one target’s initial partial mask; this short test is insufficient
to reject SAM3.1 generally. Full-clip follow-up and a successful GPU lifecycle
probe are recorded in the isolated `experiment/sam31-multiplex` worktree.
Full Felix scores higher than SAM2, Han slightly lower, and BHC stops with a
reproduced bucket-growth memory error. Adoption is blocked by that lifecycle
failure; broader accuracy equivalence remains unproven. See
[measured results](sam31-multiplex-results.md). The public research below predates
these local tests.

**SAM3.1 Object Multiplex is the strongest next architecture to investigate.**
It retains mask-conditioned spatial memory, changes the scaling of the costly
multi-object memory path, and has encouraging published VOS quality. This is a
research recommendation, not a claim that it matches our handball identity
accuracy or reaches 30 FPS on GB10. No replacement has passed those local gates.

The user's explicit requirement is **no reduction in tracking accuracy relative
to production SAM2.1-L**. Faster output with worse identities is a failed
candidate. The user subsequently clarified that the measured 2-to-3.3 FPS improvement is
worth pursuing. 30 FPS remains a longer-term target, not an adoption minimum.
Masks may be generated when needed,
but removing mask output is not permission to degrade internal tracking.

Retain RF-DETR and the established identity/team/jersey contracts. Production
SAM2 stays the default. SAM-MT was rejected locally and its suite was explicitly
stopped; this investigation does not restart it. The original research pass inspected public papers and released source only;
the linked local experiments followed later. Production defaults remain unchanged.

## 1. SAM3.1 Object Multiplex: best architecture lead

### Published quality

The following combines the SAM2.1-L comparison in Meta's SAM3 paper (Table 5)
with the multiplex VOS results in Appendix H, Table 43 and the SAM3.1 release.
These are published checkpoint/pipeline results, not an ablation isolating
multiplexing from architecture and training changes.

| Benchmark | SAM2.1-L | SAM3.1 multiplex | Metric |
|---|---:|---:|---|
| DAVIS17 validation | 90.7 | 92.7 | J&F |
| SA-V test | 78.4 | 85.1 | J&F |
| LVOSv2 validation | 79.6 | 89.2 | J&F |
| MOSEv1 validation | 77.9 | 79.6 | J&F |
| YouTube-VOS19 validation | 89.3 | 89.3 | G |

[Meta paper, Tables 5 and 43](https://arxiv.org/html/2511.16719),
[official SAM3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

These measure segmentation overlap/boundary quality, not our player identity
scorer. They justify a quality investigation, not an adoption decision. MOSEv2
also reports a large gain, but the SAM2 baseline is explicitly marked zero-shot;
do not present that difference as controlled evidence of the architecture alone.
The paper's interactive online comparison also has individual datasets where
SAM3 regresses against SAM2 despite a better average. General benchmark gains
cannot guarantee that handball overlaps preserve identity.

### Why the speed evidence is relevant

SAM2 and original SAM3 process the temporal memory separately for each target.
Multiplex groups up to 16 targets into one bucket, sharing spatial memory
encoding, attention and mask decoding, while keeping object-specific embeddings
and separate output masks. This directly addresses our 12–14-target workload.
[Architecture, Appendix H.1](https://arxiv.org/html/2511.16719).

Meta's released H100 graph shows **30.2 FPS at 16 targets**, compared with
9.8 FPS for November SAM3 and 12.2 FPS for SAM3 with newer implementation
optimizations. Thus the relevant gain is 3.08x against old SAM3 or 2.48x against
optimized SAM3 at this count. The headline approximately 7x is at 128 objects,
not fourteen players. None of these is a measured speedup over our SAM2 driver.
[Official graph](https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/sam3.1_efficiency.png).

Hardware, compilation and workload matter. H100 results do not establish GB10
throughput. The separate sports project previously inspected reports only
7.3 FPS aggregate for its complete SAM3.1 text-detection pipeline on RTX 5090;
that pipeline changes the detector and does not isolate the multiplex tracker.
[Sports project's native SAM3.1 measurements](https://github.com/holma91/selective-mask-propagation#postscript-sam-31).

### RF-DETR-preserving integration feasibility

The official source exposes `build_sam3_multiplex_video_model`, separate from
the combined text detector/tracker predictor. The released tracker demo provides
`add_new_points`, `add_new_masks`, object removal and propagation. Points are
converted to masks by the interaction head, then inserted into multiplex state.
Thus an external-detection-seeded tracking experiment is feasible in principle;
we need not adopt SAM3's text detector to investigate the tracker.
[Model builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py),
[tracker input and lifecycle methods](https://github.com/facebookresearch/sam3/blob/main/sam3/model/video_tracking_multiplex_demo.py).

This is source inspection, not a working adapter. Checkpoint key selection,
complete weight loading, correction behavior, add/remove isolation, streaming
state and compilation equivalence remain to be verified. In particular, a
`strict=False` load that leaves missing tracker weights cannot count as setup.
The combined builder replaces the tracker backbone with the detector's shared
encoder; a standalone builder must retain a correctly loaded vision backbone.

## 2. DeepStream MaskTracker: same-model engineering lead

This is distinct from NvDCF. NVIDIA accelerates **all four SAM2 networks** with
TensorRT FP16 and implements memory/lifecycle work in C++/CUDA. It supports
periodic detections and configurable batching for the individual networks.
Keeping the same SAM2 checkpoint is attractive when quality must be preserved.
[Official design and configuration](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#masktracker-developer-preview).

However, it changes memory-update, association and numerical behavior. The same
weights do not establish identical trajectories. No comparable 12–14-player
speed/accuracy table was found in the inspected NVIDIA documentation, and the
published DeepStream performance page does not report MaskTracker throughput.
The sample README specifies x86 dGPU; Spark/aarch64 full temporal-tracker support
still needs verification. Jetson segmentation-only support is not proof of
Spark support or lack of it. This is a secondary feasibility lead with unknown
speed benefit, not evidence of a 15x gain.
[Sample prerequisites](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker),
[DeepStream performance page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html).

## 3. Other new leads screened

| Method | What the evidence actually shows | Decision for this requirement |
|---|---|---|
| Efficient-SAM2 (ICLR 2026) | Authors headline 1.68x speedup on SAM2.1-L with a 1.0-point SA-V test quality drop | Do not prioritize: stated quality loss and insufficient speed gain |
| Lean-SAM2 (July 2026) | Large: LVOSv2 84.2 to 83.6 J&F at 1.412x; Base+: 83.6 to 82.4 at 1.417x | Do not prioritize: its headline quality gain is over Efficient-SAM2, not the original SAM2 baseline |
| SENTRY (ECCV 2026) | Improves SAM2 tracking robustness; SAM2-L 44.0 to SENTRY 32.8 FPS on A100 | Quality idea, not an acceleration: 25.5% slower and no established fourteen-target timing |
| SAM2++ | Released unified mask/box/point tracking model with task-specific memory; retains mask-form internal outputs | Quality lead without verified multi-player speed improvement; not a ready speed recommendation |
| SAMIDARE (CVPR sports workshop 2026) | SportsMOT test 77.3 HOTA / 78.6 IDF1 vs SAM2MOT 76.4 / 76.5; controls mask regeneration and memory contamination in dense scenes | Relevant quality design, but retains SAM2.1-L and offers no demonstrated 30 FPS pipeline |
| RS3-Prune (August 2026) | SAM2 DAVIS J&F 88.2 to 79.7 for 9.6 to 11.4 FPS in its own Table 1 | Reject as a no-loss speed claim; sizeable quality regression, weak speed gain |
| DiTTA (CVPR 2026) | Adapts image semantic segmentation to video semantic segmentation | Different task: semantic classes do not establish individual teammate identities |

Primary sources:
[Efficient-SAM2](https://arxiv.org/html/2602.08224),
[Lean-SAM2 Table I](https://arxiv.org/html/2607.19811),
[Lean-SAM2 authors' table](https://github.com/DeawhaleQwQ/Lean-SAM2#main-results),
[SENTRY results and timing](https://hamadya.github.io/SENTRY/),
[SAM2++](https://arxiv.org/html/2510.18822),
[SAMIDARE](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Hirano_SAMIDARE_Advanced_Tracking-by-Segmentation_for_Dense_Scenarios_CVPRW_2026_paper.pdf),
[RS3-Prune Table 1](https://arxiv.org/html/2608.22526),
[DiTTA authors' implementation](https://github.com/jihun1998/DiTTA).

SAMIDARE's explicit handling of contaminated masks and risky detector reprompts
is relevant to our overlaps. Its comparison baseline is the external SAM2MOT
system, not our production periodic-reprompt SAM2 manager. Its average per-video
times are reported in minutes (Table 7), without enough timing protocol to turn
those into a portable per-frame speed claim. Public source availability for
SAMIDARE itself was not established in this search.

The earlier CUTIE, LiVOS, MAVOS and StreamDAM evidence remains in
[the broader research note](realtime-player-tracking-search.md). None establishes
quality-preserving 30 FPS for our workload. MOTIP/PuTR learn association, but
there is no local evidence that they fix the failure modes already demonstrated
by box pipelines. They are not the current priority under the no-regression rule.

## Next comparison, if implementation work follows

1. Verify SAM3.1 multiplex tracker weights and standalone image encoder load
   completely, using an isolated adapter and the current CUDA environment.
2. Compare against **production SAM2 with its periodic RF-DETR corrections**.
   A native seed-once SAM2 control may diagnose the model, but cannot replace
   the production baseline. Preserve frame spacing, target set, detector inputs
   and usable prompt information; count initialization and corrections in cost.
3. Screen the FelixClaar teammate overlap densely first. Reject additional body
   swaps, ID switches or lost-player intervals even if an average score improves.
   Then check each full annotated clip separately with the corrected player-only
   scoring protocol; do not compensate one clip's regression with another's gain.
4. Only candidates that preserve identity quality proceed to performance
   acceptance. Measure warmed 12–14-target tracking latency and then complete
   RF-DETR + tracking throughput, including any requested mask workload. Include
   p50/p95 latency, deadline misses and memory growth. A single-object headline,
   stale-mask output or cached-detector timing does not establish 30 FPS live IDs.
5. Keep SAM2 as the default until the quality gate passes. No new tolerance for
   identity errors or numeric accuracy loss is authorized.
