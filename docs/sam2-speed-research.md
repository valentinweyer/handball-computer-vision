# Faster SAM2 multi-object tracking for handball

The trained eager EfficientTAM-S 1024 comparison is now complete: first paired runs measured 17–26% more complete-loop throughput, with mixed quality results and no default replacement. See [paired results](efficienttam-paired-results.md) for counts, continuity diagnostics, visual review and limitations. Same-weight SAM2.1 compilation and EdgeTAM reset/reseed remain rejected below. The [lifecycle investigation](efficienttam-lifecycle-investigation.md) explains the compatible API and the encoder-compilation default that must be disabled. EfficientTAM efficient-memory variants, SAM3.1 multiplex, DeepStream MaskTracker and SAM-MT remain separate unmeasured candidates.

Evidence is current to September 9, 2026. Local inspection covered the working tree based on commit `fc7c259`, the active Python environment, and relevant upstream checkouts. Published performance below is author-reported. **The initial research pass did not benchmark candidates. Subsequent measurements are recorded below and in the [trained EfficientTAM comparison](efficienttam-paired-results.md); none has established faster equivalent-quality tracking on this machine.**

The recommendation prioritizes preserving the measured identity advantage of the current tracker. It includes alternatives with quality tradeoffs, but distinguishes a faster implementation of the same computation from a new model or a change in how often segmentation runs.

The companion [comparison CSV](sam2-speed-candidates.csv) lists 18 configurations or implementation families, including timing scope, lifecycle fit and sources. All local speed-verification fields are false; this is a research inventory, not a results file.

## Current implementation and comparison baseline

The runtime is the official `sam2-upstream` video predictor wrapped by `drive_sam2` and `TrackManager`. Although the default checkpoint path points into `segment-anything-2-real-time/checkpoints`, the inference implementation is imported from `sam2-upstream`. The historical README's real-time-fork attribution therefore does not identify the current execution path. Local source references are collected in the evidence table below.

| Property | Observed baseline | Consequence for comparison |
|---|---|---|
| GPU and CPU architecture | NVIDIA GB10; aarch64 | A100, H100, A6000, and iPhone timings cannot predict local FPS |
| Active software | Python 3.11.16; PyTorch 2.9.1+cu130; CUDA build 13.0; driver 580.173.02 | CUDA is available; compatibility must be checked on this stack |
| Model | SAM2.1 Hiera-L; 1024 × 1024 model input | Smaller checkpoints and 512 inputs are separate quality experiments |
| Precision | CUDA BF16 autocast during prompts and propagation | Comparisons against FP32 baselines can exaggerate the gain |
| Compilation | `vos_optimized` omitted; image-encoder compilation false in config | Official full-model compilation is available locally but unused |
| Multi-object execution | One image-feature computation shared across objects; temporal inference loops with `batch_size=1` | Object-dependent work remains after accelerating the image encoder |
| Detector policy | Cached RF-DETR detections; checkpoints every 10 frames | A replacement must handle corrections and lifecycle changes |
| Input and state | JPEG cache; synchronous initialization; video and state offloading disabled | Startup and retained memory need separate measurement |
| Output | Original-resolution masks transferred to CPU, filtered individually, converted to boxes | Predictor-only throughput excludes substantial application work |

The environment values were read directly without executing a model. GPU memory capacity was not inferred from `nvidia-smi`: its capacity query returned `N/A`.

The repository's historical measurements report approximately **1.0–1.2 frames/s**: FelixClaar processed 248 frames in 206 seconds; Han-Ber4 processed 198 in 197 seconds. Those were documented run timings, not a synchronized benchmark. [Measured stage profile](#measured-stage-profile-han-ber4-2026-09-09) below now supplies one; it reproduces the Han-Ber4 timing exactly, attributes it by stage, and records the post-processing defect it found. With the two post-processing defects it found fixed, the same clip runs at **2.17 frames/s** and model inference is 87.9% of the frame. The identity results are the acceptance baseline, not a segmentation J&F score. “Correct” means the share of trusted reference detections that were matched and assigned to the reference identity dominant within their predicted tracklet. It is not the percentage of players with entirely correct trajectories; clean fragmentation can escape the wrong-identity measure. [Local evaluation](tracking-evaluation.md)

| Evaluation clip | Current SAM2 correctly identified reference detections | Best measured box-tracker configuration | Interpretation |
|---|---:|---:|---|
| FelixClaar, 249 frames | 95.0% | 81.5%, SORT | Largest measured benefit from the current design |
| Han-Ber4, 198 frames | 89.2% | 86.7%, BoT-SORT | Reference uses SAM2; correlated errors can inflate agreement |
| BHC-FAG, 500-frame derived window | 98.9% | 95.9%, MCByte with masks | Different match; SAM+Cutie reference with manual verification |

These references contain trusted identity spans derived from propagated masks. They are useful paired evaluation assets, but they do not constitute exhaustive independent pixel ground truth. Later entrants and brief errors missed by sampled verification remain limitations. Preserve the current exclusions and trusted spans when comparing candidates; inspect crossings and entrants separately. [Evaluation methodology and caveats](tracking-evaluation.md)

### Measured stage profile (Han-Ber4, 2026-09-09)

The first synchronized profile of the running loop: 198 frames at 13.4 objects/frame, GB10, BF16, `CHECK_EVERY=10`, cached RF-DETR detections, warm JPEG frame cache. CUDA is synchronized at every stage boundary, so asynchronous GPU work is charged to the stage that issued it rather than to the next `.cpu()` call. Startup (model build, `init_state`, frame-0 seed) is excluded at 8.2-8.6 s and reported separately.

| Stage | Original | After mask filter | After centroid fix |
|---|---:|---:|---:|
| Propagation (model) | 407.0 ms | 407.8 ms | 405.0 ms |
| `masks_from_logits` | 495.6 ms | 16.9 ms | 24.3 ms |
| `sv.mask_to_xyxy` | 1.8 ms | 2.0 ms | 1.8 ms |
| `TrackManager.update_from_propagation` | 57.1 ms | 57.6 ms | 1.9 ms |
| Checkpoint prompts | 1.4 ms | 1.4 ms | 1.4 ms |
| Unattributed | 26.5 ms | 27.3 ms | 26.6 ms |
| **Total** | **989.4 ms (1.01 fps)** | **512.9 ms (1.95 fps)** | **461.0 ms (2.17 fps)** |

The "Original" column reproduces the historical 198 frames / 197 seconds exactly, which is what validates the instrumentation. In each step only the changed stage moves; the rest are equal within run-to-run variance of a few ms.

**Bottleneck 1 below was the dominant term, at 50.1% of the frame — not the model.** Supervision's `filter_segments_by_distance(mode="edge")` computes its distance transform in `_chamfer_distances`, a pure-Python row loop of `2 x height` iterations over full-width `int64` arrays. Its fixed-point weights `62587/65536` and `89738/65536` are exactly OpenCV's `DIST_L2, maskSize=3` constants and the two agree bit-for-bit, so it reimplements `cv2.distanceTransform` in Python. At 1080p that cost about 31 ms per mask regardless of mask content, plus about 8 ms in a per-component loop making two full-image passes each.

`handball_cv.tracking.sam2_driver.filter_edge_fragments` replaces it with the OpenCV call, a single `np.unique` pass over the label image, and bounding-box-local work, with the threshold still taken from the full image diagonal. Output is byte-identical: 11 constructed cases in `tests/unit/test_filter_edge_fragments.py`, plus all 2650 real SAM2 masks on this clip (233 multi-component, 58 with fragments actually dropped, 0 differing pixels). No re-scoring or reference rebuild is required, and the identity results in [tracking-evaluation.md](tracking-evaluation.md) stand unchanged.

**Second defect: `np.nonzero` for a centroid.** `TrackManager.update_from_propagation` recorded each track's area and centroid with `m.sum()` then `np.nonzero(m)` and two `.mean()` calls. `np.nonzero` scans all two million pixels and allocates two int64 index arrays sized to the true-pixel count -- about 576 KB per player, 13.4 players a frame -- purely to take two means. It measured 56.6 ms of the stage's 57.6 ms. `mask_area_and_centroid` finds the bounding box with two `any()` reductions and calls `cv2.moments(binaryImage=True)` on the crop, which returns the pixel count and the summed coordinates directly: 44x faster on 826 real masks, and bit-identical on all 2650 masks of the clip.

One subtlety is worth keeping: the crop offset must be folded into the moment's numerator, not added to the quotient. Dividing on the crop and adding `x0` afterwards rounds twice and drifts by about an ulp, which an exact-equality test caught; `(m10 + area * x0) / area` divides once and reproduces `xs.mean()` exactly. Both outputs drive lifecycle decisions -- areas the mask-collapse test, centroids the duplicate-removal jump test -- so exactness is the requirement, not closeness.

**End-to-end verification.** Per-function equality implies pipeline equality only if the functions are pure, which they are, but it was checked directly anyway: today's code was run twice per clip, once with both previous implementations patched back in, on all three benchmark clips. Every frame index, tracker id, box, per-frame mask-pixel total and lifecycle event matched, and the compressed artifacts hash the same.

| Clip | Frames | Rows | Events | Result |
|---|---:|---:|---:|---|
| Han-Ber4 | 198 | 2664 | 5 | identical, sha256 `2c90f943b9277f42` |
| FelixClaar | 248 | 2889 | 13 | identical, sha256 `1f9fc7edeb011d58` |
| BHC-FAG window | 499 | 6211 | 7 | identical, sha256 `bed0fab091c2d22b` |

The pre-existing `runs/sam2_reprompt/*.npz` artifacts were deliberately not used as the baseline: `c79db06` changed `sam2_manager.py` after they were written, so a diff against them would show that lifecycle change rather than these two. The three identity results in [tracking-evaluation.md](tracking-evaluation.md) therefore stand exactly as measured; no re-scoring or reference rebuild is required.

**Consequence for the shortlist.** Model inference was 41.1% of the frame and is now 87.9%. Before these fixes a predictor twice as fast would have returned at most about 1.26x end to end; it now returns about 1.78x. That raises the value of priorities 1 and 2 (same-weight compilation, then smaller checkpoints) and still does not on its own justify a lifecycle port. The largest remaining non-model term is the 26.6 ms/frame of unattributed loop work (5.8%), which includes `TrackManager.checkpoint` -- unprofiled, and holding several more full-frame `mask.sum()` calls plus a pairwise `(ma & mb).sum()` overlap scan.

The duplicate `masks_from_logits` in `experiments/sam2_baseline/run_pipeline.py` was deliberately left unfixed: it is a frozen baseline behind published results.

### Stage A result: same-weight compilation, rejected (2026-09-09)

Priority 1 was run: SAM2.1-L, same checkpoint, `vos_optimized=True`, existing `TrackManager` lifecycle, on Han-Ber4's 198 frames. It is **slower than eager on this workload and it changes the output**. Not "rejected pending tuning" -- rejected on two independent grounds.

**Three toolchain blockers had to be cleared first, none of them SAM2's.** They are recorded because they are machine-wide, not experiment-specific:

1. `ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'`. Triton ships its own `ptxas` (V12.8.93) and prefers it over anything on the system; it predates GB10's `sm_121`. Every Inductor kernel failed codegen. `torch/bin/ptxas` (V13.0.48) and `/usr/local/cuda-13.0/bin/ptxas` (V13.0.88) both support the target, so `TRITON_PTXAS_PATH` selects a working one. **Until this is set, `torch.compile` is unavailable for everything on this machine** -- detector, re-ID and jersey reader included -- which is the practical form of the `cuda capability 12.1 ... maximum supported 12.0` warning.
2. `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`, from `memory_attention.py`. Upstream compiles at `mode="max-autotune"`, which enables CUDA graph trees; the plain VOS benchmark consumes outputs immediately, while this driver keeps masks alive across invocations through `TrackManager`.
3. Setting `torch._inductor.config.triton.cudagraphs = False` does not help: the `mode` string wins over the global config. The mode itself has to become `max-autotune-no-cudagraphs`, at all five compile sites -- four in `SAM2VideoPredictorVOS._compile_all_components` and one in `SAM2Base.__init__` for the image encoder.

**Timing** (warm Inductor cache, `TORCH_LOGS=recompiles` for the diagnostic run):

| | Eager | Compiled |
|---|---:|---:|
| Startup (build, compile, `init_state`, seed) | 8.6 s | 39.7 s |
| First frame | 0.3 s | 46.4 s |
| Median frame | 414.8 ms | **360.7 ms** |
| Mean frame, excluding frame 1 | **512.9 ms** | 563.4 ms |
| Total wall | **110.0 s** | 197.0 s |

The kernels really are faster: the median frame improves 1.15x. Three recompiles occur, all inside frame 1 -- `image_encoder.forward` at 16:03:53, `prompt_encoder.forward` at 16:04:26, `mask_decoder.forward` immediately after -- so the image-encoder recompile alone is roughly 33 s of that first frame. Those are one-time warmup, not a per-frame tax.

What kills it is the tail. Mean minus median is 98 ms eager and 202 ms compiled: compiled's tail is twice as heavy, enough to invert the average. The mechanism fits the architecture -- propagation restarts and reprompts every `CHECK_EVERY` frames, and that boundary work costs more compiled than eager. **The penalty is per checkpoint, so it does not amortize with clip length**; a full match makes it worse, not better. This was measured in aggregate; the specific operation carrying the heavier tail was not isolated.

**Output changes, on identical weights.** `tracker_id`, `frame_index` and all five lifecycle events matched, but 733 of 2664 boxes differ, by up to 99 px, and per-frame mask pixel totals differ on 198 of 198 frames (max 1566 px). The plausible chain is that compilation perturbs mask logits, pixels flip across the `> 0.0` threshold, and a marginal fragment crosses the edge filter's keep/drop boundary, moving a box edge sharply. Per this document's own rule, a changed same-weight result is to be investigated rather than accepted against a tolerance -- and here it would have to be scored against all three references to buy a configuration that is slower.

**Consequence.** The easy route to the 87.9% of frame time the model now occupies is closed: compiling the current predictor does not reach it. Priorities 2 and 3 (smaller checkpoints, EfficientTAM) are unaffected by this result, since they change the model rather than its compilation, but they carry accuracy gates that this experiment did not.

### Local evidence map

| File and inspected lines | Evidence |
|---|---|
| [sam2_driver.py](../src/handball_cv/tracking/sam2_driver.py), 34–35, 140–157 | Large configuration, checkpoint interval, builder call, initialization, BF16 prompting |
| [sam2_driver.py](../src/handball_cv/tracking/sam2_driver.py), 51–69, 173–188 | JPEG cache, CPU masks and filtering, propagation output |
| [sam2_driver.py](../src/handball_cv/tracking/sam2_driver.py), 195–249 | Detector checkpoints; remove, reprompt, add, reset actions |
| [run_sam2_reprompt_tracker.py](../scripts/run_sam2_reprompt_tracker.py), 44–64, 96–155 | Import path, checkpoint path, current CLI, replay output schema |
| [Upstream builder](../sam2-upstream/sam2/build_sam.py), 100–129 | `vos_optimized=False` default and optimized class selection |
| [Upstream predictor](../sam2-upstream/sam2/sam2_video_predictor.py), 42–98, 583–630, 976 onward | State defaults, per-object loop, full-resolution output, compiled components |
| [Large config](../sam2-upstream/sam2/configs/sam2.1/sam2.1_hiera_l.yaml), 88–89, 120 | Seven mask memories, 1024 input, encoder compilation disabled |
| [Evaluation](tracking-evaluation.md), 337–408, 435–503 | Accuracy, timings, reference limitations and third-clip result |
| [Requirements](../requirements.txt), 1–3, 25–27 | Documented aarch64/GB10 CUDA environment |

## Ranked shortlist

Priority reflects usefulness for this repository, including integration and evidence quality. It is not a measured speed ranking.

| Priority | Candidate | Why test it | Main limitation |
|---|---|---|---|
| ~~1~~ | ~~Official SAM2.1-L with `vos_optimized=True`~~ | **Measured and rejected 2026-09-09** | Slower in steady state (per-checkpoint penalty, does not amortize) and changes 27.5% of boxes on identical weights -- see [Stage A result](#stage-a-result-same-weight-compilation-rejected-2026-09-09) |
| 2 | Official compiled SAM2.1-B+ and Small | Cheap controlled model-size comparison | Accuracy tradeoff; object-dependent memory cost remains[^2] |
| 3 | EfficientTAM-S and efficient-memory variants | Lightweight encoder and memory; familiar video API | Published FPS largely single-object; new weights[^3][^4] |
| 4 | SAM3.1 multiplex | Shared processing across object buckets; assets already present | New predictor and prompt adapter; local speed unknown[^7][^8] |
| 5 | DeepStream MaskTracker | Full temporal SAM2 TensorRT path and target lifecycle | Larger integration; developer preview; no matched FPS proof[^10] |
| 6 | SAM-MT | Strongest directly relevant published dense-target scaling | Released target grouping conflicts with per-player lifecycle[^13][^15] |
| ~~7~~ | ~~EdgeTAM~~ | **Closed 2026-09-09** | Its mandatory reset-and-reseed pattern was priced on SAM2 first and costs up to 1.1 points with a 9x wrong-identity increase on one of three clips -- see [tracking-evaluation.md §8.10](tracking-evaluation.md) |
| 8 | Lean-SAM2 / Efficient-SAM2 | Research into memory and encoder sparsity | Headline timings use FP32, unlike this baseline[^18][^19] |

For a short implementation cycle, stop after the first three priorities and compare the speed/quality frontier. For a larger architecture experiment, favor SAM3.1 or SAM-MT according to whether integration readiness or published object-count scaling matters more. DeepStream is the strongest deployment-oriented branch when accepting a separate runtime and association implementation.

## Published speed evidence and its limits

The following rows belong to different experiments. Ratios are meaningful only within a stated comparison; the table is not a cross-hardware leaderboard.

| Source experiment | Reported result | Conditions | What it establishes |
|---|---|---|---|
| Official SAM2.1 table | Large 39.5; B+ 64.1; Small 84.8; Tiny 91.2 FPS | A100 benchmark context; official video benchmark uses one prompted object | Smaller encoders improve the author's benchmark, not necessarily 14-player GB10 throughput[^2][^20] |
| EfficientTAM paper, November 2024 version | S 85.0; S/2 109.4 FPS; its SAM2 comparator 43.8 FPS | A100, batch 1, 1024 input | Full temporal lightweight-model evidence, with a different historical SAM2 baseline[^3] |
| EdgeTAM paper | 150.9 vs SAM2.1-B+ 64.1 FPS | A100, compilation enabled | GPU acceleration exists; its 22× mobile headline is a different experiment[^5] |
| SAM3.1 release | About 7× at 128 objects versus original SAM3 | Single H100 | Strong high-object-count scaling, not 7× versus SAM2[^7] |
| SAM3.1 announcement | 16 → 32 FPS at a medium object count | Single H100, versus SAM3 | Relevant direction; exact local workload match absent[^8] |
| SAM-MT | 35.7 vs SAM2.1-B+ 4.9 FPS at 15 targets | A6000 48 GB; 1024 input; synthetic scaling suite | Directly relevant target-count evidence; GPU propagation timing excludes application work[^13][^14] |
| Efficient-SAM2 | 1.68× on SAM2.1-L; SA-V test about 1 point lower | RTX A6000, FP32, 20 SA-V clips | Sparse acceleration under those conditions; paper reports little/no BF16 gain[^18] |
| Lean-SAM2 | 1.412× Large; 1.417× B+ | RTX 3090, strict FP32; LVOSv2 timing subset | Another sparse implementation; no BF16 GB10 result[^19] |
| flybroken TensorRT C++ | About 33 FPS for one target | A10, FP16, 1080p source; encoder input 1024² | A functioning temporal runtime claim, not a multi-player result[^12] |

The official and EfficientTAM example benchmarks load state and prompts before timing, repeatedly propagate a small example, and discard outputs. EfficientTAM's current example additionally selects a 512 configuration and sets the checkpoint to `None`. It is useful for architecture timing, but cannot simultaneously validate tracking accuracy or reproduce a 1024 model's quality. Neither example should be copied unchanged as the handball benchmark. [Official benchmark][^20], [EfficientTAM benchmark][^21]

### Official SAM2.1 compilation and smaller checkpoints

Full compilation was introduced in December 2024. The optimized predictor compiles memory attention, memory encoding, and mask decoding in addition to the image backbone. It preserves independent object handling, including mid-video insertion; it does not turn the temporal loop into joint multi-object inference. Meta explicitly allows small prediction differences from compilation.[^1]

The exact first experiment is to construct the current predictor with `vos_optimized=True`, retain BF16 and the same checkpoint, and leave detector cadence and lifecycle rules fixed. This is a proposed experiment, not an implemented change. Record initial compilation separately, then measure fresh video sessions after representative warmup.

Warmup must exercise more than uninterrupted propagation: include a later player entry, a correction, removal, and reset. These paths may use different shapes or prompt signatures and trigger additional compilation. Record graph breaks and failures instead of silently falling back and labeling the result compiled.

For the model-size comparison, the official SA-V test scores are Large 79.5, B+ 78.2, Small 76.6 and Tiny 76.5 J&F.[^2] These are useful quality indicators, but the task here is same-team identity continuity through contact. B+ and Small are worthwhile controlled candidates; neither should become default from a generic VOS score alone. The driver hardcodes its model configuration, so replacing only the checkpoint CLI argument is insufficient.

### EfficientTAM

EfficientTAM changes both the encoder and, in efficient-memory variants, the memory computation. The November 2024 paper reports S/2 at 74.0 SA-V test J&F versus 74.5 for S and 74.7 for its SAM2 comparator. These are historical paper values, not a claim of parity with the current Hiera-L baseline.[^3]

The maintained predictor follows the newer SAM2 interaction model: adding objects after tracking starts is allowed, and per-object outputs are propagated independently. It exposes a similar prompt/state interface and a compiled builder. This makes it a more practical early model replacement than a fork with fixed initial targets.[^4]

Start with the 1024 Small family and record the exact config and matching checkpoint. The repository contains both `_1` and `_2` efficient-attention implementations, so the paper's “S/2” label alone does not uniquely identify a run. Treat 512 variants as an additional experiment. Code is Apache-2.0.[^22]

The adapter should preserve `Sam2FrameResult`'s IDs, mask ordering, original image coordinates and boxes. Keep the existing `TrackManager` and `PlayerRegistry`; the experiment should answer whether a different segmenter improves the tradeoff under the same lifecycle. Since the object loop remains independent, measure the slope from 1 to 16 objects as well as the single-object rate.

### SAM3.1 multiplex

SAM3.1, released March 27, 2026, groups targets into fixed-capacity buckets and performs shared tracking. Its release notes report improved VOS scores on six of seven listed benchmarks, while concept-segmentation results are mixed. It is a distinct checkpoint and tracking architecture from SAM3.[^7]

This repository already has `sam3/RELEASE_SAM3p1.md`, multiplex implementation files and `sam3-checkpoints/sam3.1_multiplex.pt`. Availability is established; successful execution and speed are not. The local builder defaults to `multiplex_count=16`, `max_num_objects=16`, and `compile=False`. A default cap must not silently exclude additional players, referees or sideline candidates. [Local builder](../sam3/sam3/model_builder.py), lines 1070–1105.

The high-level builder creates a tracker-plus-detector stack. A comparison that replaces RF-DETR with text-prompted person discovery changes both detection and tracking. Preserve the project's detector and test a geometric-prompt or tracker-only adaptation first. The lower-level multiplex builder exists, but its returned class takes dimensions and cached features in `init_state`, rather than the current video-path call. [Local model and demo](../sam3/sam3/model/video_tracking_multiplex_demo.py), lines 3218–3250.

The demo implementation has `add_new_points`, `add_new_masks` and object removal, rather than a matching `add_new_points_or_box` signature. Box-to-prompt semantics, output arity and object-local reset require explicit verification. A box's center is not an equivalent prompt when players overlap. Shared memory also means resetting one player's contaminated appearance must preserve the others' useful history. [Local interaction implementation](../sam3/sam3/model/video_tracking_multiplex_demo.py), lines 1249–1277 and 2955–2982.

Use 8-, 16-, and 17-object runs to expose bucket boundaries. Test the full wrapper and tracker-only path as separate configurations if both are built. The source README recommends Python 3.12+, while local metadata is looser and the active environment is 3.11; validate in an isolated compatible environment. SAM3 uses the SAM License, not SAM2's Apache license.[^9]

### NVIDIA DeepStream MaskTracker

MaskTracker provides a real temporal SAM2 implementation: image encoder, memory attention, mask decoder and memory encoder run through TensorRT FP16, with shared frame features and per-target memory. Its own association, Kalman estimation and target management handle detection updates, new targets and termination. The documentation labels it developer preview. Its segmentation-only mode omits temporal SAM2 and is not the equivalent tracking candidate.[^10]

DeepStream 9.0 explicitly supports DGX Spark through the `9.0-triton-sbsa-dgx-spark` container; native installation is unsupported. This resolves the blanket concern that an ARM desktop cannot use DeepStream. Platform support alone does not prove MaskTracker's speed or correctness on the local GB10.[^11]

The integration is larger because the application currently owns its association and identity lifecycle. For an initial evaluation, inject the same cached RF-DETR boxes and export boxes, masks and tracker IDs into the existing scoring format. Avoid substituting the reference application's PeopleNet detections: doing so would obscure the source of any improvement.[^23]

Benchmark default and project-matched prompt-update policies separately. Confidence-gated memory updates and fused boxes may produce beneficial changes, but they also mean the default runtime is not numerically equivalent to the current Python tracker. No trustworthy matched 12–16-player FPS figure was established from the inspected documentation. Record a local result before accepting a larger runtime migration.

### SAM-MT

SAM-MT uses target queries with shared context and sparse query memory. Its July 2026 paper starts from SAM2.1-B+ and reports 35.7 FPS for 15 targets against 4.9 FPS for SAM2.1-B+, with 3,627 versus 5,436 MB memory. The scaling suite uses 20 synthetic sequences spanning 1–20 targets and 100 frames each. This is the most directly relevant published object-count experiment in the shortlist.[^13]

The efficiency script uses BF16, disables `vos_optimized` and postprocessing, and places GPU timing around propagation after initialization. It discards masks. That supports a promising neural tracking result, but excludes reprompting, detection, CPU mask work and end-to-end output.[^14]

The public demo packs several internal targets into one client `obj_id=1`. LVOS later arrivals are handled by separate inference states grouped by birth time, then merged. **Inspection inference, not an executed failure:** inherited per-object removal would address the group, and the temporary frame-zero assumptions in consolidation make ordinary later reprompting questionable. Do not equate familiar SAM2 method names with a compatible lifecycle.[^15]

Inference code and a checkpoint are available. The checkpoint metadata is CC-BY-NC-SA-4.0; a root code license was not established in the inspected tree.[^16] Treat this as a research adapter project. First prove independent add, correction and reset for one target while others continue; only then compare the complete detector-reprompt loop. Splitting every entrant into a separate state may recover functionality while sacrificing the shared-computation benefit.

### EdgeTAM

EdgeTAM combines a small encoder with compressed memory. Its paper reports A100 throughput of 150.9 FPS versus SAM2.1-B+'s 64.1, and SA-V test J&F of 71.7 versus 77.0 in that experiment. The 22× claim is for iPhone deployment, not GB10.[^5]

The reference PyTorch predictor retains the old restriction against adding a new object once tracking begins; it also warns about later box refinements. That conflicts with this project's entry and reset behavior.[^6] Code and checkpoints are Apache-2.0.[^24]

EdgeTAM remains useful for a fixed-target speed/quality test. For a complete replacement, budget a lifecycle port or independently verify another maintained implementation.

**Closed 2026-09-09.** The concern above -- that reinitializing the video state every checkpoint discards history -- turned out to be measurable rather than hypothetical, and it was measured without writing any EdgeTAM integration. McByte++ does not patch the restriction: its vendored predictor still raises `"Cannot add new object id ... after tracking starts"`, and `mask_manager__edgetam.reseed_at_frame` works around it with `reset_state` plus a full re-seed. Reproducing that pattern on SAM2 through a `checkpoint_policy` seam in `drive_sam2`, with `TrackManager`'s decisions and `IdentityManager` held fixed, gives -0.15, +1.5 and -1.1 points across the three clips. BHC-FAG fails the per-clip gate: wrong-identity rises 9x, with an extra tracklet, an extra switch and worse fragmentation.

So a port would start from an identity regression on at least one clip, before EdgeTAM's own SA-V J&F of 71.7 against 77.0, and buy at most about 1.78x given inference is now 87.9% of the frame. That is a worse trade than the 2.15x the post-processing fixes returned byte-identically. Full numbers in [tracking-evaluation.md §8.10](tracking-evaluation.md).

## Other implementations and misleading comparisons

| Candidate or family | Evidence found | Decision for this project |
|---|---|---|
| Gy920 real-time SAM2 | Frame-by-frame camera API, compilation option, 512 Tiny config; README links to it historically[^25] | Useful streaming reference. “Real-time” does not establish a faster 14-player model loop |
| flybroken SAM2 TensorRT C++ | Temporal engines, multi-target API, FP16, memory gates and motion heuristics; quoted FPS is single-target[^12] | Secondary runtime experiment; substantial C++ and lifecycle verification |
| Microsoft ONNX Runtime SAM2 example | Image encoder/decoder export and profiling; image decoder batching limitation[^26] | Building blocks, not a complete temporal MOT replacement |
| PyTorch SAM2Fast/AOTInductor | Up to 13× serving-latency improvement for image prompting and mask generation[^27] | Techniques may transfer; headline does not describe video tracking |
| Efficient-SAM2 | Sparse windows and memory; 1.68× FP32 Large result; negligible BF16 gain in its appendix[^18] | Low priority on this BF16 machine |
| Lean-SAM2 | Target-preserving sparse memory and routing; FP32 published speedups[^19] | Research option after dense compiled baselines |
| TinySAM 2 | May 2026 paper studies extreme memory compression and a RepViT encoder[^28] | Public deployable release not verified; monitor, do not confuse with original TinySAM |
| Q-SAM2 | Quantization research with video quality evaluation[^29] | No verified GB10 temporal latency result; low-bit storage alone is insufficient |
| Fast SAM2 with Text-Driven Token Pruning | December 2025 paper reports up to 42.50% faster inference and lower memory[^30] | Additional text-routing dependency; executable lifecycle-compatible release not established |
| Selective Mask Propagation | Detector-led tracking with SAM only in ambiguity windows; public sports experiments[^17] | Alternative architecture; does not provide every player's mask on every frame |

Lean-SAM2's README reports LVOSv2 Large J&F **83.6 versus baseline 84.2**, and B+ **82.4 versus 83.6**. Its abstract's larger quality gains are comparisons against Efficient-SAM2, not gains over the original SAM2.1 model. Its README also distinguishes BF16 inference defaults from strict-FP32 speed reproduction. Those distinctions prevent overstating the benefit.[^19][^31]

The selective-propagation repository quotes 15.8 versus 5.5 FPS for selective versus uniform processing on four RTX5090 clips, but defines time as SAM plus merging, excluding detection and base tracking. Its larger SportsMOT run has another hardware/configuration context. These are useful workload-reduction results, not a faster SAM2 kernel.[^17]

This project's masks support downstream jersey-number matching and overlap handling. A sparse-mask architecture therefore changes more than tracker speed. It would need explicit rules for when masks exist, how detector misses are bridged and when identity corrections become available. Offline global trajectory association should also be reported separately from causal tracking. Existing MCByte-with-mask results already show that adding mask information to a box tracker does not automatically improve handball identity accuracy. [Local findings](tracking-evaluation.md)

## Bottlenecks to measure before choosing a replacement

For analysis, approximate one frame as:

`T(frame, N) = T(image encoder) + N × T(object memory and decoder) + T(mask output, N) + T(lifecycle)`

This is a diagnostic model, not a fitted timing result. It explains why an encoder-only export or a smaller encoder can disappoint at high object counts. For example, if the encoder accounted for 20% of total time, halving its time would yield only `1 / (0.8 + 0.2 / 2) = 1.11×` overall. Shared-memory architectures target a different term from encoder acceleration.

Four observations deserve measurement in the present wrapper:

1. **CPU mask work was on the measured path and dominated it (resolved 2026-09-09).** Every output transfers all original-resolution Boolean masks to CPU and calls a filter per object. Fourteen 1080p Boolean masks contain about 29 MB before extra copies and filtering. On GB10, shared physical memory does not make synchronization, tensor conversions or CPU processing free. The [measured stage profile](#measured-stage-profile-han-ber4-2026-09-09) puts this at 495.6 ms/frame, 50.1% of the loop, with almost all of it in the per-object filter rather than the transfer; `filter_edge_fragments` reduced it to 16.9 ms/frame at byte-identical output. Transfer and per-object filtering remain on the path at that reduced cost.
2. **Chunk boundaries repeat model work.** Propagation restarts at frame `t`, and the wrapper skips the duplicate only after the predictor yields it. The upstream predictor recomputes non-conditioning outputs instead of treating them as cached. With ten-frame chunks, this can approach one extra propagation step per ten new frames for ordinary objects. The exact excess depends on which objects receive conditioning prompts. This is a code-level inference, not a measured 10% wall-time saving.
3. **Initialization processes more than the requested output window.** `max_frames` limits the wrapper's iteration, but `init_state` still receives the full JPEG directory. A short benchmark can therefore preload the entire video. Use a physically bounded input cache for comparable startup and memory measurements.
4. **Prompt work can produce unused outputs.** The driver discards return values from initial and checkpoint prompts. Upstream prompting can still resize/consolidate masks for those returns. If this is costly, an adapter could expose a cheaper update operation while preserving its state effects.

The boundary case needs semantic care: a checkpoint correction is intended to update memory before subsequent frames. Avoid simply advancing `start_frame_idx` without testing that corrections and fresh-object state are consolidated correctly. Likewise, removing mask filtering or reducing output resolution alters the boxes consumed by the identity logic and must remain a separate quality experiment.

## Benchmark and integration plan

### Fixed workload and acceptance conditions

Use FelixClaar, Han-Ber4 and the existing 500-frame BHC-FAG window first, with identical cached detections, exclusions and trusted identity spans. Keep detector interval 10, BF16 where supported, and the current mask filtering. Use 1024 input for SAM2-family comparisons; retain and report native resolution for other architectures, including SAM3.1's 1008 × 1008. Changing a model's native resolution is a separate experiment. Use original input ordering and timestamp spacing. The BHC-FAG window already uses every second source frame; additional frame skipping changes the task.

Add a long unlabelled clip for resource growth and a short lifecycle case with an entrant, disappearance and body swap. The former measures runtime stability; it must not be advertised as independently scored accuracy. Inspect all identity-disagreement frames against pixels, especially when a reference shares the candidate's model family.

Before any speed test, each adapter must demonstrate: seed several players; add one later; correct only one; remove one; reset one under the same public identity; retain every other player's state; resume a chunk; and output masks and boxes with stable ID ordering. Preserve the project's reversible identity/team rules. Keep RF-DETR and `PlayerRegistry` fixed.

For same-weight compilation, investigate any changed identity result rather than accepting a percentage tolerance automatically. For new models, report a Pareto frontier of speed and quality. A useful provisional decision threshold is at least 2× complete tracking-loop throughput with no more than one percentage point loss in correctly identified reference detections on any trusted clip and no new severe same-team swap. This is a proposed engineering criterion, not a statistical equivalence guarantee or an agreed product requirement.

### Measurements

| Measurement | Include | Purpose |
|---|---|---|
| Cold startup | Model load, compile, video preload, initial prompts | Cost of a new process/video |
| Warm propagation | Shared image features, all active objects, memory read/write | Compare neural tracking at fixed object count |
| Complete tracking loop | Propagation, CPU masks, filtering, boxes, checkpoint decisions and prompts | Main replacement decision |
| Full application | Detection if uncached, team/re-ID/OCR, rendering and writing | Actual delivered throughput |
| Object-count sweep | 1, 4, 8, 12, 16; additionally 17 and 24 for multiplex runtimes | Scaling and bucket-capacity effects |
| Latency distribution | Median and p95; checkpoint frames separately | Avoid hiding correction stalls in average FPS |
| Memory over time | Device allocations/reservations and process/system memory | Distinguish retained history from working memory |
| Quality | Correct identities, recall, wrong identity, switches, fragmentation, mask usability | Preserve downstream usefulness |

Use synchronized wall-clock measurement around the complete measured interval; CUDA events can additionally isolate GPU work. Do not synchronize every kernel when measuring throughput. Perform warmup using a separate state, then create a fresh state for each measured repeat. Three independent warm runs per configuration are a reasonable first comparison; report variability rather than selecting the fastest run.

Record exact repository revisions, checkpoints and hashes, input dimensions, precision, compile settings, loaded frames, active target counts and prompt schedule. Model FPS must count completed video frames, not object-frames. If output masks are discarded or boxes remain on GPU, label that configuration explicitly.

### Experiment order and stop points

| Stage | Configurations | Stop condition |
|---|---|---|
| ~~A~~ | ~~Current baseline; same Large checkpoint with full compilation~~ | **Done 2026-09-09: rejected.** Speed and output differences explained in [Stage A result](#stage-a-result-same-weight-compilation-rejected-2026-09-09) |
| B | Compiled B+, Small; EfficientTAM-S and an efficient-memory Small variant | Select best quality/speed tradeoff under the current manager |
| C | SAM3.1 multiplex adapter; DeepStream full temporal mode | Lifecycle tests pass and comparison retains RF-DETR inputs |
| D | SAM-MT target/lifecycle adapter | Individual correction/reset proven before quoting benchmark FPS |
| ~~E~~ | ~~EdgeTAM lifecycle port~~; sparse or quantized experiments | **EdgeTAM closed 2026-09-09** on measured identity cost of its mandatory reseed pattern; the remaining entries still apply |

The smallest useful implementation would add a predictor factory/configuration seam to `drive_sam2`, explicit compile/model options, and timing around its existing operations. It should preserve `Sam2FrameResult` and delegate lifecycle to the existing manager. That makes official model variants and EfficientTAM comparable without rebuilding the application.

The replay schema already consists of `frame_index`, `tracker_id`, `boxes` and `source`. `Sam2ReplayTracker` can consume those outputs, but the existing factory discovers one fixed `runs/sam2_reprompt/<video>.npz` path. Add an explicit candidate replay path for future experiments rather than overwrite the baseline artifact. Evaluate tracking quality from the saved replay; **replay speed is not tracker speed**. [Replay implementation](../scripts/evaluate_tracker_identity.py), lines 93–154.

SAM3.1 and SAM-MT require their own prompt/state adapters. DeepStream additionally owns association and therefore deserves a separate experiment identity. New adapters should enter the same scorer and expose enough events to explain a score change. No implementation, checkpoint download, or model benchmark is required to use this report's shortlist.

## Sources

All live code and documentation links were inspected for this comparison on September 9, 2026. Publication/version dates are specified where material. Local evidence links above describe the inspected working tree and can change with subsequent development.

[^1]: Meta. [SAM2 release notes](https://github.com/facebookresearch/sam2/blob/main/RELEASE_NOTES.md), December 11, 2024. Full compilation, independent object semantics and numerical-variance caveat.
[^2]: Meta. [SAM2 model description](https://github.com/facebookresearch/sam2#model-description), SAM2.1 checkpoint table. Model size, published FPS and quality.
[^3]: Xiong et al. [Efficient Track Anything](https://arxiv.org/html/2411.18933v1), November 28, 2024, Table 1 and experimental setup. Values here are explicitly from this preprint version; the work also appeared at ICCV 2025.
[^4]: EfficientTAM authors. [Video predictor](https://github.com/yformer/EfficientTAM/blob/main/efficient_track_anything/efficienttam_video_predictor.py) and [builder](https://github.com/yformer/EfficientTAM/blob/main/efficient_track_anything/build_efficienttam.py). Mid-video insertion, per-object inference and compile API.
[^5]: Zhou et al. [EdgeTAM: On-Device Track Anything Model](https://arxiv.org/html/2501.07256v1), January 2025, Table 2 and GPU-efficiency discussion; CVPR 2025. Device-specific speed and quality comparisons.
[^6]: Meta. [EdgeTAM video predictor](https://github.com/facebookresearch/EdgeTAM/blob/main/sam2/sam2_video_predictor.py). `_obj_id_to_idx` and `add_new_points_or_box` restrictions.
[^7]: Meta. [SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), March 27, 2026. Multiplexing, 128-object comparison and VOS results.
[^8]: Meta. [SAM3.1 announcement](https://ai.meta.com/blog/segment-anything-model-3/), updated March 27, 2026. Medium-object-count H100 comparison.
[^9]: Meta. [SAM3 repository](https://github.com/facebookresearch/sam3), README installation guidance and license. Local API evidence is separately linked above.
[^10]: NVIDIA. [DeepStream 9.0 Gst-nvtracker: MaskTracker](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvtracker.html#masktracker-developer-preview). Temporal network components, lifecycle, modes and preview status.
[^11]: NVIDIA. [DeepStream 9.0 installation: DGX Spark](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Installation.html#dgx-spark-setup-for-ubuntu). Supported container route and platform constraints.
[^12]: flybroken. [SAM2-TensorRT](https://github.com/flybroken/sam2-tensorrt). Single-target A10 timing, model input shape, multi-target interface, motion/memory changes and MIT code license.
[^13]: Shen, Liu and Ding. [SAM-MT: Real-Time Interactive Multi-Target Video Segmentation](https://arxiv.org/html/2607.08688v1), July 9, 2026, experimental setup and Tables 3–4. Object-count scaling and memory.
[^14]: FudanCVL. [SAM-MT efficiency evaluation](https://github.com/FudanCVL/SAM-MT/blob/main/evaluation/evaluate_efficiency.py). Precision, timing scope and inference settings.
[^15]: FudanCVL. [SAM-MT inference demo](https://github.com/FudanCVL/SAM-MT/blob/main/inference.py), [LVOS evaluator](https://github.com/FudanCVL/SAM-MT/blob/main/evaluation/evaluate_lvos.py), and [video predictor](https://github.com/FudanCVL/SAM-MT/blob/main/sam2/sam2_video_predictor.py). Target grouping and lifecycle inspection.
[^16]: FudanCVL. [SAM-MT repository](https://github.com/FudanCVL/SAM-MT), [checkpoint files](https://huggingface.co/FudanCVL/SAM-MT/tree/main/checkpoints), and [model metadata](https://huggingface.co/FudanCVL/SAM-MT). Release availability and checkpoint license.
[^17]: Holmberg. [Selective Mask Propagation repository](https://github.com/holma91/selective-mask-propagation) and [benchmark implementation](https://github.com/holma91/selective-mask-propagation/blob/main/scripts/fps_benchmark.py), 2026. Throughput scope and sparse-mask output policy.
[^18]: Zhang et al. [Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](https://arxiv.org/html/2602.08224v1), February 9, 2026, especially Appendix B; ICLR 2026. FP32 benchmark and BF16 limitation. [Implementation](https://github.com/jingjing0419/Efficient-SAM2).
[^19]: Ouyang et al. [Lean-SAM2 repository](https://github.com/DeawhaleQwQ/Lean-SAM2), 2026. Strict-FP32 timing instructions, checkpoint paths and baseline quality table.
[^20]: Meta. [SAM2 benchmark source](https://github.com/facebookresearch/sam2/blob/main/sam2/benchmark.py). Single object, BF16, full compilation and timing scope.
[^21]: EfficientTAM authors. [Benchmark source](https://github.com/yformer/EfficientTAM/blob/main/efficient_track_anything/benchmark.py). Current 512 configuration, no checkpoint, single object and discarded outputs.
[^22]: EfficientTAM authors. [Repository](https://github.com/yformer/EfficientTAM), [Small efficient config 1](https://github.com/yformer/EfficientTAM/blob/main/efficient_track_anything/configs/efficienttam/efficienttam_s_1.yaml), and [config 2](https://github.com/yformer/EfficientTAM/blob/main/efficient_track_anything/configs/efficienttam/efficienttam_s_2.yaml). Release, license and configuration distinctions.
[^23]: NVIDIA-AI-IOT. [DeepStream MaskTracker reference application](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker). Detector example, export tooling and output integration.
[^24]: Meta. [EdgeTAM repository](https://github.com/facebookresearch/EdgeTAM). Apache-2.0 code/checkpoint licensing and public release.
[^25]: Gy920. [Segment Anything 2 real-time](https://github.com/Gy920/segment-anything-2-real-time). Camera API, compile option and 512 configuration.
[^26]: Microsoft. [ONNX Runtime SAM2 exporter](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers/models/sam2). Image export, profiling scope and batching limitation.
[^27]: PyTorch Foundation. [Accelerating Generative AI with PyTorch: Segment Anything 2](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/), February 26, 2025. Image-serving benchmark and AOTInductor techniques.
[^28]: Ding et al. [TinySAM 2: Extreme Memory Compression for Efficient Track Anything Model](https://arxiv.org/html/2605.18013v1), May 18, 2026. Architecture and memory-compression research; deployable code/checkpoints not verified here.
[^29]: Farronato et al. [Q-SAM2: Accurate Quantization for Segment Anything Model 2](https://arxiv.org/html/2506.09782v1), June 2025. Quantization and video-quality research.
[^30]: Mandal et al. [Fast SAM2 with Text-Driven Token Pruning](https://arxiv.org/abs/2512.21333), December 24, 2025. Reported pruning speed and memory benefits; local deployment not established.
[^31]: Ouyang et al. [Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2](https://arxiv.org/html/2607.19811v1), July 22, 2026. Timing protocol and comparator interpretation.
