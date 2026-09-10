# SAM2 performance and alternatives for handball tracking

**Keep SAM2.1-L as the quality reference. The best immediate opportunity is to remove redundant work around its existing lifecycle; the best larger opportunity is to accelerate the per-object temporal computation. No alternative examined has demonstrated equal tracking quality and higher throughput on this project's current hardware.** CUTIE deserves the first independent propagation experiment. DeepStream MaskTracker and SAM3.1 deserve narrowly scoped feasibility checks. SAM-MT has strong multi-object evidence but a difficult lifecycle interface.

This assessment uses the repository and public sources available on September 10, 2026. The objective is maximum speed **without reducing tracking quality on the existing GB10 machine**. Existing experiments are distinguished from the new diagnostic below. Production tracking code was not changed for this investigation.

## Project requirements

The relevant workload is handball video with roughly 12–14 live player/goalkeeper objects, occasional additional targets, frequent overlap, similar uniforms, entrants and departures. The implementation uses SAM2.1 Hiera-L at 1024 input resolution, BF16 autocast, and detector checkpoints every ten frames. It imports the official `sam2-upstream` predictor; the checkpoint directory's `segment-anything-2-real-time` name does not identify the running implementation. [Driver](../src/handball_cv/tracking/sam2_driver.py), lines 34–35 and 285–417.

A useful replacement must satisfy the following contract, not just produce plausible masks:

| Requirement | Why it matters | Acceptance implication |
|---|---|---|
| Stable individual identities through same-team overlap | Appearance alone is weak between teammates | Inspect uninterrupted identity continuity, not just aggregate box matches |
| One usable mask per live identity | Jersey matching uses mask intersection-over-smaller; team classification can use overlap masks | A box-only tracker or sparse mask output is a different product configuration |
| RF-DETR box initialization and correction | The detector is already trained for this domain | Include any box-to-mask conversion in candidate timing |
| Late insertion, removal and individual reset | The manager handles entrants, departures and contaminated memory | Retain every unaffected player's history and preserve mask/ID ordering |
| Tracker-independent team observations | Incorrect tracking must not freeze team labels | Preserve the existing team-classification dependency direction |
| Separate persistent identity and jersey evidence | Retirement/revival and number votes have established rules | Preserve `PlayerRegistry`/identity mappings and voting contracts |
| Long-video resource control | Full-frame preload and retained outputs grow with duration | Measure whole-process memory and continuity at any chunk seam |
| Current local hardware | GB10, aarch64, unified memory | Benchmark locally; CUDA support alone does not establish deployment compatibility |

These requirements follow [architecture](architecture.md), [tracking evaluation](tracking-evaluation.md), [jersey matching](../src/handball_cv/jersey/identity.py), and [open memory work](../TODO.md). No new numerical quality-loss allowance is assumed. The older research's provisional one-percentage-point tolerance is **not** the acceptance rule for this comparison.

## Existing speed and quality evidence

Two substantial CPU improvements already exist: an OpenCV-based equivalent of Supervision's edge-fragment filter and cropped image moments for mask centroids. The recorded Han-Ber4 stage profile fell from **989.4 to 461.0 ms/frame**, or **1.01 to 2.17 FPS**. Model propagation remained approximately 405 ms/frame. These are previous improvements, not changes made by this investigation. Existing per-mask and end-to-end equivalence evidence is documented in [the earlier speed report](sam2-speed-research.md).

More recent paired tracking-loop records provide the practical reference:

| Clip | SAM2 FPS | EfficientTAM-S FPS | SAM2 correct reference detections | EfficientTAM-S correct reference detections |
|---|---:|---:|---:|---:|
| FelixClaar | 2.38 | 3.00 | 95.01% | 94.26% |
| Han-Ber4 | 2.13 | 2.49 | 99.91% | 99.26% |
| BHC-FAG window | 2.22 | 2.80 | 98.88% | 98.94% |

These are existing single warmed measurements with cached detection and historical team-model appearance features. They exclude PRTReID, PARSeq, live detection, rendering and writing. Thus **2.1–2.4 FPS is tracking-loop throughput, not full-application throughput**. Their current driver and runner hashes match the inspected files. [Paired results](efficienttam-paired-results.md); [saved manifests](../runs/efficienttam_pair/summary.json).

The Han-Ber4 percentage differs from the README's historical 89.2% because an earlier denominator included excluded bench/reference identities. The comparable player-only denominator gives 2305/2307 = 99.91%. It is not a newly improved tracker. The reference itself is SAM2-derived, so agreement is not an independent guarantee of accuracy.

Most importantly, EfficientTAM's aggregate numbers conceal a verified same-team failure: IDs 8 and 10 collapse onto the same blue-shirt teammate at Felix frame 107, persisting through 120. SAM2 preserves their separation in that window. Plain EfficientTAM-S therefore fails the continuity requirement despite its 17–26% throughput gain. [Paired mask review and caveats](efficienttam-paired-results.md).

The earlier same-weight `vos_optimized=True` experiment was also rejected: its median frame improved, but complete wall time and mean processing time worsened, and output boxes changed. That result applies to the tested GB10/toolchain/driver configuration; it does not establish that every possible compiled or TensorRT implementation is slower. Re-enabling the same full-compilation recipe is not the next recommended action. [Compilation results and toolchain details](sam2-speed-research.md).

## New diagnostic: where the current work goes

A fresh diagnostic ran the unchanged SAM2 driver on the first 151 Han-Ber4 frames, yielding 150 new frames. It used the existing trained checkpoint, a physically bounded JPEG cache, a separate 21-frame warmup, BF16 and the existing detector/manager. CUDA events surrounded five non-nested modules; they did not synchronize every module call. The resident `llama-server` was left running.

| Module | Measured time per returned frame, including repeated boundaries | Calls in measured loop |
|---|---:|---:|
| Shared image encoder | 115.57 ms | 150 |
| Per-object memory attention | 293.12 ms | 2,264 |
| Per-object memory encoder | 49.70 ms | 2,278 |
| Mask decoder | 36.77 ms | 2,264 |
| Prompt encoder | 3.38 ms | 2,264 |

The measured leaf intervals sum to 74.78 seconds of a 106.85-second instrumented loop. The remaining time includes other tensor operations, CPU processing, checkpoint decisions, scheduling, synchronization and instrumentation overhead. The diagnostic's 1.40 FPS must not be treated as a regression against historical uninstrumented runs: it is a different measurement configuration, and service activity was not sampled continuously. CUDA-event intervals can also include GPU scheduling and host-launch gaps; these are not isolated kernel benchmarks.

The robust findings are the execution counts and the direction of the cost imbalance. The image encoder runs once per new frame and already shares its features across objects. Memory attention runs separately per object and accounts for about **59% of the measured leaf-module time**. Simply substituting a smaller encoder cannot remove that work. Even eliminating the entire measured encoder interval would remove only 16.2% of this diagnostic's wall time; this is an illustrative bound for this run, not a speed forecast.

Every returned box and tracker ID matched the corresponding prefix of the saved SAM2 baseline **exactly**, including the frame-120 removal and frame-140 reprompt events. This confirms that the instrumentation followed the existing behavior. Full mask equality against the saved baseline was not checked because that baseline replay stores boxes rather than complete masks.

Reproduction and raw evidence:

- [Diagnostic source](../runs/sam2_speed_audit_20260910/profile_components.py), run from the repository root with `.venv/bin/python`.
- [Profile JSON](../runs/sam2_speed_audit_20260910/profile.json), including software, source hash, counts and equality checks.
- [Component CSV](../runs/sam2_speed_audit_20260910/component_timings.csv).
- [Checkpoint latency CSV](../runs/sam2_speed_audit_20260910/saved_checkpoint_latencies.csv), derived from the previous paired runs.

## Concrete optimization opportunities

### 1. Avoid repeated boundary inference when no state update requires it

The driver calls `propagate_in_video(start_frame_idx=t, max_frame_num_to_track=chunk_len)` after each detector checkpoint, then discards the returned frame `t`. Upstream propagation includes both endpoints and recomputes every non-conditioning object on that frame. Discarding the output happens **after** the neural work. [Driver](../src/handball_cv/tracking/sam2_driver.py), lines 334–338; [predictor](../sam2-upstream/sam2/sam2_video_predictor.py), lines 575–630.

The diagnostic counted **193 repeated object inferences**, in addition to 2,070 object inferences for new frames. The repeated calls consumed **4.86 seconds in the measured leaf modules**, or 4.55% of instrumented wall time. Their associated Python work and output resizing are additional, unisolated costs. This establishes real avoidable work, not a promised 10% end-to-end improvement.

The existing uninstrumented records independently show checkpoint stalls:

| SAM2 clip | Ordinary-frame resume mean | Resume after checkpoint mean |
|---|---:|---:|
| Han-Ber4 | 417.94 ms | 956.82 ms |
| FelixClaar | 374.29 ms | 851.86 ms |
| BHC-FAG | 398.72 ms | 932.71 ms |

These resume intervals include manager/checkpoint work and repeated propagation together. The whole difference cannot be attributed to the repeated neural step.

A conservative prototype should initially skip a boundary **only when the checkpoint emitted no actions and there are no pending predictor outputs**. Starting at `t+1` also requires adjusting the inclusive propagation count to keep the same final frame. Compare masks, boxes, areas, centroids and lifecycle events against the current implementation on all three clips before adopting it. Expand to state-changing checkpoints only after proving their semantics separately.

### 2. Investigate correction overwrite before changing boundary semantics

The repeated work exposes a correctness issue. With the default `add_all_frames_to_correct_as_cond=False`, an existing player's box correction on an already tracked frame becomes a temporary **non-conditioning** output. Preflight consolidates its corrected mask and encodes memory. The following propagation of that same frame then executes an unprompted inference and replaces the corrected output. Non-conditioning temporal memory for frame `t` comes from earlier frames, so the corrected current-frame entry is not used to retain that correction. [Predictor](../sam2-upstream/sam2/sam2_video_predictor.py), lines 232–279, 480–542 and 591–614; [memory selection](../sam2-upstream/sam2/modeling/sam2_base.py), lines 539–570.

The diagnostic observed this on Han-Ber4 frame 140: the eighth live object, ID 8, is reprompted; its consolidated corrected mask and memory both change when the repeated unprompted inference replaces them. The other twelve repeated outputs at that boundary retain exactly equal mask and memory tensors. The saved baseline also has that ID-8 reprompt event.

This is evidence of a correction being overwritten, **not evidence that retaining it would improve the clip's final accuracy**. Late births and remove/re-add resets create different state and must not be conflated with this case. A correction-preserving fix is a behavior change: explicitly test its effect on occlusions, duplicate retirement and downstream identity. Do not assume that moving the start index preserves current results on corrected frames, and do not globally mark every corrected frame conditioning without measuring the resulting memory-policy change.

### 3. Stop materializing unused removal outputs

The driver ignores both calls to `remove_object`'s return value, but upstream defaults `need_output=True`. After deleting the target and remapping the surviving state, it reconstructs original-resolution masks for the removed object's prompted frames solely for the caller's return value. [Driver](../src/handball_cv/tracking/sam2_driver.py), lines 384 and 403; [predictor](../sam2-upstream/sam2/sam2_video_predictor.py), lines 867–953.

Using `need_output=False` is the narrowest apparent optimization. Inspection shows that the actual removal/state transitions precede that conditional output block. Its benefit is concentrated on removal/reset frames and may be small on clips with little turnover. Validate survivor state, same-ID reset, output order and following masks; the existing fake predictor's signature would need the optional argument. No speed gain for this change has been benchmarked.

### 4. Target per-object temporal computation for substantial gains

The current predictor runs `_run_single_frame_inference(..., batch_size=1)` inside the object loop. A larger engineering experiment should target memory attention, memory encoding and mask decoding, with compatible objects batched while keeping individual histories. The existing shared image-feature cache already solves the simpler encoder duplication problem.

Grouping objects by compatible memory shapes and conditioning history is safer than assuming every object has identical state. Padding memory requires correct attention masking; late-born and reset objects need their own histories; mixed precision and batched kernels may alter thresholded masks. The old batched/legacy predictor is not a direct replacement for the current dynamic-object API. No local batched implementation was benchmarked here.

### 5. Separate startup, long-video memory and repeated rendering

`max_frames` limits emitted frames after the entire JPEG directory has been passed to `init_state`. A short run can therefore preload a much longer video. Use a physically bounded cache for diagnostics and a streaming/chunked loader for long videos. The upstream float32 image allocation alone is about 12.6 MB per model-resolution frame. Moving it from GPU to CPU does not eliminate its physical footprint on unified memory. [Driver](../src/handball_cv/tracking/sam2_driver.py), lines 285–314; [loader](../sam2-upstream/sam2/utils/misc.py), line 267; [memory design notes](../TODO.md).

Frame-buffer streaming and predictor-output pruning require separate treatment: inference history and object-pointer selection can depend on older outputs. A seam that resets all players and asks appearance re-ID to repair them changes identity quality. Carry the persistent registry, number evidence and public identity mapping, and test survivor continuity across deliberately difficult seams.

For repeated overlay/label edits, cache per-frame geometry and packed masks once and redraw from them. This removes repeated SAM2 execution for subsequent renders without claiming to accelerate first-pass inference. The existing TODO already describes that design; it has not been implemented by this investigation.

## Alternatives and deployment paths

The following rankings are integration priorities, not an FPS leaderboard. “Parity unproven” means there is no matched handball result establishing preservation of the current trajectory and mask quality.

| Candidate | Evidence and practical role | Recommendation |
|---|---|---|
| Smaller SAM2.1 B+/Small | Same API; different weights; lower published segmentation accuracy | Cheapest model-size experiment, parity unproven |
| CUTIE with SAM image prompts | Mature mask propagation; explicit add/delete API | First independent propagator to test |
| DeepStream MaskTracker | Full SAM2 TensorRT path, object batching, DGX Spark container | Strong deployment experiment; integration is substantial |
| SAM3.1 Object Multiplex | Shared multi-object processing; local assets already present | Check tracker-only lifecycle and speed before full adapter |
| SAM-MT | Strong published dense-target scaling | Research candidate; individual lifecycle unresolved |
| EfficientTAM-S | Already measured 17–26% faster locally | Fails observed same-team continuity gate |
| EfficientTAM compressed-memory variants | Attack a more relevant cost than encoder substitution alone | Separate, lower-priority trained experiments |
| EdgeTAM | Efficient video model; incompatible released late-insertion pattern | Do not revisit reset/reseed integration under the current requirement |
| XMem / XMem++ | Mature memory propagation; annotation-focused extensions | Secondary baseline after CUTIE |
| DEVA | Detector/segmenter plus temporal fusion | Consider only if changing association ownership is acceptable |
| Efficient-SAM2 / Lean-SAM2 | Sparse encoder and memory computation | BF16 speed and quality evidence insufficient for adoption |
| SAMURAI / DAM / SAMWISE | Motion/distractor or language extensions | Relevant to other problems; no established dense-tracking speed benefit here |
| TinySAM 2 | Memory-compression research | Watchlist pending verified deployable assets and lifecycle proof |
| MobileSAM / FastSAM; box-only trackers | Image masks or box tracking alone | Components/fallbacks, not equivalent temporal replacements |

**Smaller SAM2.1.** Meta reports Large/B+/Small at 39.5/64.1/84.8 FPS in its A100 benchmark, with SA-V test J&F 79.5/78.2/76.6. This supports a speed–quality tradeoff, not guaranteed parity. The official example benchmark is single-object, BF16 and compiled, unlike this eager multi-player loop. Start a controlled B+ experiment eager; the previously rejected full-compilation recipe should not become a prerequisite. [Meta model table](https://github.com/facebookresearch/sam2#model-description).[^1]

**CUTIE.** The paper reports 36.4 FPS for base and 45.5 for small on its V100/YouTubeVOS protocol; these are not fourteen-player GB10 rates. Its object-level representation makes it a credible distractor-handling candidate. [Paper, table 1 and implementation details](https://arxiv.org/pdf/2310.12982).[^2] `InferenceCore.step` accepts partial masks and new IDs, and `delete_objects` purges removed objects' memory. Output tensor indices must be remapped to public IDs. [Official implementation](https://raw.githubusercontent.com/hkchengrex/Cutie/main/cutie/inference/inference_core.py).[^3]

The integration should retain RF-DETR and the existing manager, convert only initialization/correction boxes to masks using a shared image predictor, and include that conversion in timing. Test indexed-mask versus per-object-probability output carefully: CUTIE's mutual-exclusion behavior can differ from SAM2's independently overlapping masks. Reprocessing the checkpoint frame must not increment CUTIE's temporal state twice. The repository's SAM+Cutie-derived reference masks make CUTIE operationally relevant, but they are not a quality evaluation of this proposed adapter. Independent dense overlap labels are especially important for this candidate.

**DeepStream MaskTracker.** NVIDIA's 9.1 documentation exposes TensorRT image encoder, mask decoder, memory attention and memory encoder; the sample decoder supports batch size up to 20. It supports automatic target lifecycle and periodic detections but remains a developer preview. [Tracker documentation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvtracker.html#masktracker-developer-preview).[^4] There is an official DGX Spark/aarch64 container route; native Spark installation is unsupported. [Installation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Installation.html#dgx-spark-setup-for-ubuntu).[^5]

This is the strongest vendor-supported route targeting the expensive temporal modules. It is not merely an encoder export, and it is not a drop-in Python predictor: association, lifecycle, preprocessing, precision and mask outputs must be reconciled. First verify that the required full temporal mode works on the local stack, preserves all player targets and can export masks and IDs into the project. Do not quote other DeepStream trackers' FPS as MaskTracker performance. [Reference application/export tooling](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker).[^6]

**SAM3.1.** Object Multiplex is specifically designed to share multi-object work. Meta's H100 release graph reports 30.2 FPS at sixteen objects versus 9.8 for its original SAM3 release; there is no SAM2 curve. Its release table reports DAVIS17 J&F 92.7, but that is not handball identity continuity. [Release notes and graph](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).[^7] The official checkpoint is available under its model-access terms, and assets already exist locally. [Model card](https://huggingface.co/facebook/sam3.1).[^8]

Do not replace RF-DETR with a text-prompted person detector during the initial comparison. Verify correction, deletion and same-ID reset inside shared state before a substantial adapter. Then measure tracker-only scaling at 1, 8, 14, 16 and 17 objects to expose bucket boundaries. Original SAM3 is not an established speed upgrade; the existing [SAM2/SAM3 comparison](sam2-vs-sam3-speed.md) already separates direct comparisons from cross-hardware headlines.

**SAM-MT.** On the authors' A6000 synthetic 100-frame benchmark, fifteen targets yield 35.7 FPS versus SAM2.1-B+'s 4.9; both use the paper's 1024p setting. At the same target count, CUTIE is 18.4 FPS at 480p and 7.3 at 1024p. SAM-MT's LVOSv2 J&F is 76.6 versus 74.6 for its B+ comparator. These support investigating dense-target scaling, not parity with this project's Large checkpoint. [Paper, tables 1 and 3](https://arxiv.org/html/2607.08688v1).[^9]

The released timing script uses BF16 and discards propagation outputs. The demo places multiple target queries under one `obj_id` with `points_per_object`. A direct check of revision `55952679e062feb05004b598bb835d4a7a8205a1` corrects an earlier cached-source finding: `_obj_id_to_idx` allows late ID registration, despite retaining an unreachable rejection branch. However, consolidation still assumes object slot zero and a temporary frame-zero conditioning output; late arrivals in the LVOS evaluator use separate states grouped by arrival time. Registration alone therefore does not establish working individual insertion, correction or reset. See the [focused feasibility assessment](sam-mt-feasibility.md). [Demo](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/inference.py), [predictor](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/sam2/sam2_video_predictor.py), [benchmark](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/evaluation/evaluate_efficiency.py).[^10] Official checkpoint files exist; the Hub metadata labels them CC-BY-NC-SA-4.0. [Checkpoint metadata](https://huggingface.co/FudanCVL/SAM-MT).[^11]

**EfficientTAM and EdgeTAM.** EfficientTAM's paper reports S at 85.0 FPS and pooled-memory S/2 at 109.4 on A100, batch one. They are different memory variants. [Paper](https://arxiv.org/html/2411.18933v1).[^12] The locally measured plain S keeps the original memory block, so its modest gain is consistent with substantial remaining temporal cost. Its confirmed continuity failure prevents adoption; compressed-memory variants would require fresh matching weights and the same failure-window tests.

EdgeTAM's prominent 16 FPS result is on iPhone 15 Pro Max, not GB10. [Paper](https://arxiv.org/html/2501.07256v1).[^13] Its released predictor rejects new IDs once tracking has started. [Implementation](https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/sam2/sam2_video_predictor.py).[^14] The project's reset/reseed surrogate experiment already found a continuity cost. This rejects that integration pattern; it does not prove an intrinsic quality failure of every possible EdgeTAM port. A state-preserving port would be a separate engineering project.

**XMem, XMem++ and DEVA.** XMem remains a useful long-memory baseline; its official results distinguish 22.6 FPS without AMP from 33.9 with AMP on DAVIS17. [Results](https://github.com/hkchengrex/XMem/blob/main/docs/RESULTS.md).[^15] XMem++ adds permanent annotation memory and reports 30+ FPS on 480p RTX 3090 footage, emphasizing improved use of supplied annotations. This is attractive for reference annotation, not proof of automatic online parity. [Repository](https://github.com/mbzuai-metaverse/XMem2).[^16]

DEVA decouples image segmentation from temporal propagation and supports custom image models, but its semi-online fusion introduces different scheduling and association decisions. [Official repository](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).[^17] For the present scope, CUTIE offers the clearer first experiment. All three need measured prompt-to-mask cost and explicit identity-lifecycle integration; none should inherit an equal-quality claim from generic VOS scores.

**Sparse/compressed variants.** Efficient-SAM2 reports 1.68× acceleration with a one-point SA-V accuracy drop, but Appendix B says its speed measurements use FP32 and that aggressive sparsification produced little or no BF16 end-to-end gain on its tested stack. This is a material mismatch with the current baseline. [Paper, Appendix B](https://arxiv.org/html/2602.08224v1#A2).[^18]

Lean-SAM2 reports Large at 1.412× on RTX 3090 under strict FP32 timing. Its current repository table shows Large J&F 84.2→83.6 and B+ 83.6→82.4, whereas the preprint abstract advertises gains under its comparisons. The discrepancy should be resolved, not averaged away; the repository table does not establish equal quality. [Code/results](https://github.com/DeawhaleQwQ/Lean-SAM2), [preprint](https://arxiv.org/abs/2607.19811).[^19] TinySAM 2 addresses spatial/temporal memory compression, but a deployable official code/checkpoint combination was not verified in this review. [Paper](https://arxiv.org/html/2605.18013v1).[^20]

**Other task families.** SAMURAI and distractor-aware memory modify tracking/memory selection; they do not establish a faster fourteen-object replacement. [SAMURAI](https://github.com/yangchris11/samurai), [DAM paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf).[^21] SAMWISE adds text-driven segmentation, which is not a current requirement. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Cuttano_SAMWISE_Infusing_Wisdom_in_SAM2_for_Text-Driven_Video_Segmentation_CVPR_2025_paper.pdf).[^22] MobileSAM and FastSAM are useful image-segmentation components, but require a separate temporal tracker to meet this contract. [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM).[^23] MCByte and other box trackers remain speed fallbacks with already measured quality differences, not same-quality substitutes.

## Recommended sequence and acceptance gates

1. **Prototype removal-output suppression and action-free boundary reuse.** Keep the Large checkpoint, precision, cadence, classes and post-processing fixed. Prove identical masks and events on all three complete clips. Measure complete-loop wall time over at least three fresh-state warmed repetitions; compare checkpoint latency separately. A small validated gain is useful under a strict quality requirement.
2. **Handle correction overwrite as a separate correctness change.** Capture the preflight correction and following masks explicitly. Review every changed identity or mask at overlaps, and compare downstream jersey assignment. Establish a new reference implementation if this behavior changes before comparing replacement models against it.
3. **Measure a small same-weight batching prototype and an eager B+ control.** Batching targets the measured dominant cost; B+ estimates how much encoder size can realistically help. Neither should be promoted on predictor-only FPS.
4. **Test CUTIE behind the existing identity manager.** Reuse the current predictor-factory seam where practical, but build an explicit state adapter. Start with same-team overlap and late insertion/removal, then the three complete clips. Include seed/correction segmentation in the loop timer.
5. **Run lifecycle smoke tests before large-runtime migrations.** DeepStream and SAM3.1 are the next architecture candidates; SAM-MT follows if individual query-state operations can be demonstrated. Stop a candidate early if it needs global memory resets to emulate ordinary player lifecycle.

For same-computation optimizations, compare byte-identical masks, boxes, IDs and events, because tiny mask changes can cross retirement and fragment-filter thresholds. For a new model, byte equality is inappropriate; require no observed deterioration in per-clip detection coverage, wrong identities, fragmentation, uninterrupted continuity and downstream mask usability. A single new severe same-team collapse is disqualifying even if aggregate correctness improves.

The current scorer assigns each predicted tracklet its dominant reference identity. Clean fragmentation and errors outside trusted spans can escape that aggregate measure. Extend evaluation with a small densely checked event ledger: same-team crossing, opposite-team overlap, full occlusion/reappearance, late entry, exit, individual reset and scene transition. Record the actual person followed before, during and after each event. Do not let jersey-based identity repair conceal raw tracker swaps when assessing tracker quality.

Use fixed cached detections, original timestamps, the same trusted spans and the same inclusion classes. Preserve all targets; a lower target cap is a task change. Distinguish model-input resolution from final mask dimensions, and include box-to-mask prompts, CPU transfer/filtering, lifecycle decisions and complete generator exhaustion in the tracking timer. Report build/startup separately; report full-application runtime separately again.

For long clips, record process RSS, CUDA allocated/reserved memory and system memory pressure over time. On unified memory these measurements describe different accounting domains and must not be added indiscriminately. Preserve any resident services and record their activity rather than silently changing the machine workload.

These gates can establish a better measured configuration for the current clips. Three short clips cannot prove population-wide equivalence across every match, camera or occlusion. The appropriate deployment claim remains bounded by the tested footage and reviewed events.

Verification for this investigation: **36 existing tests passed** across edge-fragment filtering, mask moments, predictor construction, retirement and pending-new-object behavior. Report links and footnotes were checked, and `git diff --check` passed. These checks validate the inspected baseline and research artifacts; no new optimized production implementation is claimed.

## Sources

Public sources were checked September 10, 2026. Versioned paper links identify the versions used. Repository links describe the inspected public implementation and may change. Local evidence comes from the linked manifests, source files and research documents.

[^1]: Meta. [SAM2 repository and model description](https://github.com/facebookresearch/sam2#model-description). SAM2.1 checkpoint release September 2024; multi-object predictor update December 2024.
[^2]: Cheng et al. [Putting the Object Back into Video Object Segmentation](https://arxiv.org/pdf/2310.12982). CVPR 2024; table 1 and supplementary efficiency analysis.
[^3]: CUTIE authors. [InferenceCore implementation](https://raw.githubusercontent.com/hkchengrex/Cutie/main/cutie/inference/inference_core.py). Current add, partial-mask, deletion and ID-remapping behavior.
[^4]: NVIDIA. [DeepStream 9.1 Gst-nvtracker / MaskTracker](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvtracker.html#masktracker-developer-preview). Full temporal networks, batching and preview status.
[^5]: NVIDIA. [DeepStream 9.1 DGX Spark installation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Installation.html#dgx-spark-setup-for-ubuntu). Supported container/platform route.
[^6]: NVIDIA-AI-IOT. [DeepStream MaskTracker reference application](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker). Example/export integration.
[^7]: Meta. [SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), March 2026; [efficiency graph](https://github.com/facebookresearch/sam3/blob/main/assets/sam3.1_efficiency.png).
[^8]: Meta. [SAM3.1 model card](https://huggingface.co/facebook/sam3.1). Checkpoint availability and access conditions.
[^9]: Shen, Liu and Ding. [SAM-MT: Real-Time Interactive Multi-Target Video Segmentation](https://arxiv.org/html/2607.08688v1), July 9, 2026. Tables 1 and 3; synthetic scaling protocol.
[^10]: FudanCVL. [Inference demo](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/inference.py), [video predictor](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/sam2/sam2_video_predictor.py), [efficiency harness](https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/evaluation/evaluate_efficiency.py). Grouped queries, lifecycle restriction and timing scope.
[^11]: FudanCVL. [SAM-MT checkpoint repository](https://huggingface.co/FudanCVL/SAM-MT). Hub metadata; `checkpoints/sam-mt.pt` availability also verified through the Hub CLI.
[^12]: Xiong et al. [Efficient Track Anything](https://arxiv.org/html/2411.18933v1), November 28, 2024; ICCV 2025. Table 1 and memory-variant definitions.
[^13]: Zhou et al. [EdgeTAM: On-Device Track Anything Model](https://arxiv.org/html/2501.07256v1), January 2025; CVPR 2025.
[^14]: Meta. [EdgeTAM video predictor](https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/sam2/sam2_video_predictor.py). Released state/lifecycle API.
[^15]: Cheng and Schwing. [XMem official results](https://github.com/hkchengrex/XMem/blob/main/docs/RESULTS.md); ECCV 2022 model.
[^16]: XMem++ authors. [XMem2 repository](https://github.com/mbzuai-metaverse/XMem2). Permanent annotation memory and reported runtime.
[^17]: Cheng et al. [Tracking Anything with Decoupled Video Segmentation](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), ICCV 2023. Temporal fusion and custom image-model integration.
[^18]: Zhang et al. [Efficient-SAM2](https://arxiv.org/html/2602.08224v1), February 9, 2026; ICLR 2026. Appendix B precision limitations.
[^19]: Ouyang et al. [Lean-SAM2 repository](https://github.com/DeawhaleQwQ/Lean-SAM2) and [preprint](https://arxiv.org/abs/2607.19811), July 22, 2026. Timing protocol and conflicting quality summaries.
[^20]: Ding et al. [TinySAM 2](https://arxiv.org/html/2605.18013v1), May 18, 2026. Memory-compression research.
[^21]: Yang et al. [SAMURAI](https://github.com/yangchris11/samurai), 2024; Videnovic et al. [A Distractor-Aware Memory for Visual Object Tracking with SAM2](https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf), CVPR 2025.
[^22]: Cuttano et al. [SAMWISE](https://openaccess.thecvf.com/content/CVPR2025/papers/Cuttano_SAMWISE_Infusing_Wisdom_in_SAM2_for_Text-Driven_Video_Segmentation_CVPR_2025_paper.pdf), CVPR 2025.
[^23]: [MobileSAM official repository](https://github.com/ChaoningZhang/MobileSAM); [FastSAM official repository](https://github.com/CASIA-LMC-Lab/FastSAM). Image-segmentation scope.
