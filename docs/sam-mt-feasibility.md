# SAM-MT feasibility for handball tracking

**Prioritize a bounded SAM-MT prototype when seeking a substantial speed improvement.** Its published target-count scaling is a better match for this project's bottleneck than image-encoder-only acceleration. Adoption still requires measured GB10 throughput and preserved player continuity.

## Evidence and scope

At fifteen objects the authors report SAM-MT at 35.7 FPS and SAM2.1-B+ at 4.9 FPS: about 7.3× throughput. Both use the paper's 1024p setting on A6000; the synthetic scaling suite contains twenty selected sequences, padded or truncated to 100 frames. These are propagation measurements, not the project's detector/checkpoint/CPU-mask loop. The comparator is B+, whereas the project uses Large. In the paper's quality comparison, SAM-MT versus B+ scores 76.6 versus 74.6 on LVOSv2 J&F and 68.2 versus 65.1 on MOSEv1. Both receive two positive clicks per target. This justifies a local experiment, not a promise of equal handball identity quality or sevenfold application throughput. [Paper, tables 1 and 3](https://arxiv.org/html/2607.08688v1).

## Current code inspection

Directly inspected public revision: `55952679e062feb05004b598bb835d4a7a8205a1`, retrieved September 10, 2026. The current source differs from the cached web rendering used in the earlier comparison.

- `_obj_id_to_idx` sets `allow_new_object=True`. Late ID registration is permitted; the remaining rejection branch is unreachable. Extracting and executing that method alone confirmed registration succeeds with an existing ID and `tracking_has_started=True`.
- That does not establish a working end-to-end dynamic API. `_consolidate_temp_output_across_obj` reads `temp_output_dict_per_obj[0]['cond_frame_outputs'][0]` and selects slot zero. These assumptions require investigation for later prompts and additional groups.
- The LVOS evaluator handles later arrivals by creating a new state for each birth group, renumbering its first frame to zero, propagating it separately and merging outputs. It does not demonstrate inserting a player into an already running shared group.
- Ordinary client IDs identify groups of internal targets in the demo. A per-player public-ID adapter must distinguish those internal targets from the predictor's client IDs before removal and same-ID reset can be trusted.

Sources: [pinned predictor](https://github.com/FudanCVL/SAM-MT/blob/55952679e062feb05004b598bb835d4a7a8205a1/sam2/sam2_video_predictor.py), [pinned late-arrival evaluator](https://github.com/FudanCVL/SAM-MT/blob/55952679e062feb05004b598bb835d4a7a8205a1/evaluation/evaluate_lvos.py), [demo](https://github.com/FudanCVL/SAM-MT/blob/55952679e062feb05004b598bb835d4a7a8205a1/inference.py).

The remaining issue is engineering feasibility, not a demonstrated model-quality failure. Earlier wording that late registration is rejected should not be used as the reason to dismiss this candidate.

## Smallest useful prototype

1. **Measure its native shared propagation first.** Use trained weights and the same handball frames on GB10, with 1, 8, 14 and 16 targets. Record initialization, propagation, original-resolution output conversion and post-processing separately. Keep BF16 and all intended targets. For the initial architecture comparison, seed SAM-MT and the SAM2 control with identical points derived from the existing detector/seed masks, recording their construction. This is an isolated model comparison; compare against the existing box-prompted production baseline in the later complete-loop test.
2. **Check the known difficult overlap immediately.** Inspect FelixClaar IDs 8 and 10 through frames 100–130. EfficientTAM collapsed them at 107; require SAM-MT to preserve both people. Review actual masks and continuity, not only the existing sparse-span aggregate score.
3. **Exercise one-player lifecycle operations.** Seed several targets, add a late target, correct one, remove one, and reset one under the same public ID. Record surviving state and following masks. Passing the registration helper alone is insufficient.
4. **Integrate only after those checks succeed.** Preserve RF-DETR, the existing manager and persistent identity/jersey contracts. Score all three clips; benchmark at least three fresh-state warmed repetitions, including checkpoint work and complete CPU masks. Add an entrant/exit-heavy sequence to detect growth in group count.

A proposed investment gate is **at least 2× complete tracking-loop throughput with no observed quality regression**. The speed factor is a project target for deciding whether a substantial adapter is worthwhile, not a measured expectation. The existing three clips and trusted spans do not prove universal equivalence, so same-team overlap and entrant events require explicit review.

If native individual insertion proves impractical, an experimental fallback could preserve the initial shared group and place entrants in additional groups. This resembles the evaluation structure and need not reset existing players. However, each group adds work, histories can become contaminated within a group, and hiding a retired output does not remove its influence from shared memory. Its speed and reset behavior would need separate proof. Do not silently treat this fallback as equivalent to the original fifteen-target timing.

## Decision

SAM-MT merits a focused proof-of-concept ahead of a broad CUTIE integration when the objective is a large throughput gain. The first decision should come from native GB10 speed and the known same-team overlap case. Keep the existing SAM2 implementation as the quality reference until the complete lifecycle and paired evaluation pass.

This assessment inspected source and ran only the isolated registration-helper check. It did not download model weights, run SAM-MT inference or alter production tracking behavior.
