# Inference-time speedups for detection, tracking and their downstream stages: verdict

> Produced 2026-10-15 by a read-only multi-agent review (code maps of detection, SAM2 tracking and
> downstream stages, each adversarially checked against the source; engine research for GB10).
> Nothing was run or timed: every ms figure not cited to a repo measurement is an estimate,
> to be confirmed by the benchmark plan in section 4. Court keypoints were out of scope.
> Sections 1-4 are that review, unchanged. **Section 0 is what was then built and
> measured (2026-10-16) and supersedes them where they disagree.**

## 0. Measured results (2026-10-16)

Fixed workload, GB10 with no other GPU work running, one run per configuration:
SAM2 tracking-only on Han-Ber4 (198 frames), FelixClaar (248) and the BHC-FAG
window (499), plus the full `render_full_pipeline --tracker sam2` on the
1499-frame Melsungen window with live PARSeq reads. Runner, hashing and stage
profiler were scratch scripts; outputs are under `runs/perf_speedups/`.
Every kept change was accepted only when all outputs were identical to the
baseline: replay boxes and lifecycle events for the three tracking clips; run
JSON, reads JSON, geometry cache and the mp4's bytes for the render.

| Run | Baseline (`493f662`) | Now (`af1c375`) | Change |
|---|---:|---:|---:|
| Han-Ber4 tracking | 120.2 s | 100.1 s | -16.7% |
| FelixClaar tracking | 125.6 s | 111.3 s | -11.4% |
| BHC-FAG tracking | 242.8 s | 218.9 s | -9.8% |
| Melsungen full render | 800.2 s | 655.5 s | -18.1% (534 -> 437 ms/frame) |

Kept, in commit order (all output-identical):

- `1e4410c` -- render overlap and mask scans (items 2, 3a, 4). Frames decoded
  ahead on a thread; drawing, encoding and geometry packing on an ordered
  worker so propagation continues meanwhile; masks no longer restacked;
  checkpoint mask counts and boxes computed once with disjoint pairs skipped;
  number-to-player IoS on box-local slices. Render 800 -> 723 s.
- `a54ab7a` -- **the largest single find, which the review missed**:
  `masks_from_logits` called `.cpu()` on the thresholded mask stack, and on
  GB10 that allocates and faults in fresh pageable memory, ~67 ms for 13 masks
  at 1080p against ~1 ms into a reused pinned buffer. The stage fell from
  30.3 to 8.0 ms/frame. Same commit: `LazyFrames` replaces `init_state`'s
  whole-clip preload (item 5; bit-identical tensors, no up-front decode,
  memory no longer grows with clip length), and `fill_hole_area` pinned to the
  0 it already effectively was.
- `af1c375` -- the repeated RoPE frequency tensor built once per layer instead
  of per call (part of item 6).

Tried and dropped:

- **Item 1, skipping the repeated boundary frame, changes output.** The
  recomputed frame-t output is not bit-identical to the stored one; skipping
  it moved 5-45 boxes per clip by up to 12 px and cost FelixClaar 2
  wrong-identity frames (95.01% -> 94.92%; Han-Ber4 and BHC-FAG unchanged).
- **Item 7, batching objects through memory attention, is slower on GB10.**
  At the steady-state shapes, 13 calls at batch 1 took 212 ms and one call at
  batch 13 took 242 ms (groups of 2, 4 and 7: 235-240 ms), and no batched
  slice was bit-identical (max abs diff 0.012). Do not build it.
- Choosing another attention kernel: the default already is FlashAttention
  (identical output when forced); memory-efficient is 1.9x and math 9.2x
  slower. `torch.profiler` records no CUDA kernels on this sm_121 setup, so
  stage timing has to use synchronized timers.

Where the time goes now (Han-Ber4, synchronized timers, ms/frame; measured
before the RoPE commit): memory attention 238 (RoPE ~30% of it), image encoder
92, memory encoder 34, mask decoder 25, `masks_from_logits` 8, checkpoint
decisions 7 (mostly the team model), prompt encoder 3, propagation bookkeeping
2. The model is now ~95% of the tracking loop, and its remaining large terms
are not reachable without changing output: RoPE's float32 copies and complex
multiply could be fused, but a fused kernel would round differently.
**Corrected later on 2026-10-16:** a Triton kernel with the matching FMA order is
bit-identical and makes Han-Ber4 tracking 1.12x faster with identical output;
see [sam2-gpu-bottleneck.md](sam2-gpu-bottleneck.md) (experiment, not yet in
production).

Confirmed from the code, not fixed (it changes reads): **PARSeq runs in BF16
during SAM2 renders.** `drive_sam2` yields each frame from inside
`with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)`,
and autocast state stays on while the generator is suspended, so the render
loop's reader calls run under it. The 0.858 reader accuracy was measured in
FP32, and every `*_reads.json` written by a SAM2 render holds BF16 reads.
Measured on the same labelled set run under that autocast: accuracy at the
shipping 0.5 confidence gate is 0.86 either way; 6 of 621 crops read
differently and 4 of them cross the gate. Worth fixing for consistency, not
for accuracy.

## 1. Per-frame time budget (default SAM2 render, Han-Ber4-class load of about 13.4 objects)

| Stage | Device | ms/frame | Source |
|---|---|---|---|
| RF-DETR-L detection (cached, run once per window) | GPU | 36 (about 58 amortised per rendered overnight frame) | measured (TODO.md:565) |
| JPEG frame-cache extraction (`sam2_driver.py:51-60`) | CPU | 10–30 | estimate |
| `init_state` preload, serial PIL decode (`misc.py:267-277`) | CPU | 25–35 | measured 8.2–8.6 s per 198 frames, including the model build |
| Image encoder (Hiera-L) | GPU | about 95 (115.6 instrumented) | measured |
| Memory attention, per object | GPU | about 240 (293 instrumented) | measured |
| Memory encoder + decoder + prompt encoder | GPU | about 70 (90 instrumented) | measured |
| **Propagation total** (includes about 26 of repeated boundary-frame work) | GPU | **405** | measured |
| `masks_from_logits` | mixed | 24.3 | measured |
| Box computation and `update_from_propagation` | CPU | 3.7 | measured |
| Checkpoint CPU (team observe, overlap scan, re-decode) | mixed | 26.6 | measured as one bucket |
| PRTReID at checkpoints | GPU | 1.5–4 | estimate |
| Render JPEG decode and the BGR→RGB→BGR round trip | CPU | 21–23 | 19 ms decode measured |
| Number-to-player IoS matching (`identity.py:60-80`) | CPU | 6–16 | estimate |
| PARSeq | GPU | 0.5–1.1 | measured |
| Annotate, legend, mp4v write, `geometry.add` | CPU | 25–35 | estimate; 55 ms/frame redraw is the upper bound |
| **End to end** | | **470–630** | measured (overnight queue.log) |

The downstream estimates sum to more than the measured end-to-end gap. Treat them as upper bounds. The ~1.07 s/frame figure in `run_overnight_batch.py:8` is stale: it was written before the 2026-09-09 mask fixes.

## 2. Ranked recommendations

Items 1–6 are output-preserving. Each must still pass a byte-identity check on the geometry npz, the run JSON and the lifecycle events for Han-Ber4, FelixClaar and the BHC-FAG window. Where an item has a file:line and no source note, the location comes from the verified code maps.

**1. Skip the repeated boundary inference for objects that were not reprompted.**
- **Change:** subclass `propagate_in_video` (`sam2_driver.py:330-338`, `sam2_video_predictor.py:583-630`). On a resumed chunk's start frame, re-run only objects that have a new non-cond input at that frame. Skip that frame's video-resolution upsample.
- **Why it is safe:** the code bounds what the recompute reads. The untouched objects were already shown to be bit-equal (docs/sam2-performance-and-alternatives.md §2).
- **Gain:** about 20–27 ms/frame (about 5%). **Effort:** small.
- **Benchmark:** a Han-Ber4 tracking-only run, comparing replay hashes and ms/frame.

**2. Stop the render consumer from stalling the GPU.**
- **Change, part a:** decode the JPEG ahead of time in one prefetch thread. Keep the BGR array and convert to RGB only on OCR and checkpoint frames (`sam2_driver.py:293-294`, `render_full_pipeline.py:904-905`).
- **Change, part b:** reuse the cached decode for `last_frame` (`sam2_driver.py:355`).
- **Change, part c:** pass `result.masks` through directly instead of `list` followed by `np.stack` (`render_full_pipeline.py:906,920,433`).
- **Change, part d:** move `annotate_frame`, `draw_legend`, `writer.write` and `geometry.add` to a bounded writer thread (`render_full_pipeline.py:957-972`). Labels and colours must be computed on the main thread first.
- **Gain:** about 35–50 ms/frame (7–10%). Part d is capped by GIL contention. **Effort:** small.
- **Benchmark:** a 1499-frame render, comparing the mp4 bytes (same codec), the geometry npz and ms/frame.

**3. Make checkpoint CPU work cheaper.**
- **Change, part a:** gate the overlap scan with bounding boxes and count each mask's area once with `count_nonzero` (`sam2_manager.py:343-347,404-415,447,482`).
- **Change, part b:** compute the team observation and PRTReID for the `chunk_end` detections speculatively while the chunk propagates (`sam2_manager.py:364-368`). They depend only on the cached boxes and the frame.
- **Gain:** 5–20 ms/frame. **Effort:** small for part a, medium for part b.
- **Prerequisite:** cProfile `checkpoint()` first, so the scan's share can be separated from SigLIP and UMAP. Part b is output-preserving only if UMAP transform is deterministic; verify that.

**4. Compute IoS on box-local slices.**
- **Change:** replace the dense float32 matmul in `identity.py:60-80` with per-box slices. Keep supervision's inclusive box convention, the float32 operands, the float64 output and the sort order.
- **Gain:** 6–16 ms/frame. **Effort:** small.
- **Gate:** a unit equivalence test over the cached overnight masks and number boxes.

**5. Replace the image preload with a lazy, exact-arithmetic store, prune old outputs, and prefetch the next window's frames.**
- **Change, store:** substitute a lazy `images` object for `misc.py:267-277`. It must apply the same operations in the same order: u8 → f64/255 → f32 → cuda → −mean → /std.
- **Change, pruning:** after each frame, drop `non_cond_frame_outputs` older than t−16 (`sam2_base.py:539-621`).
- **Change, overnight queue:** build clip k+1's JPEG cache while clip k renders (`overnight_render_queue.py:201-207`).
- **Gain:** removes 40–90 s of serial startup per window, and memory becomes independent of clip length. Today the footprint is 12.58 MB/frame preload, about a 2× transient peak during the copy, and about 10 MB/frame of retained state. That removes the memory reason for the chunk-seam plan in TODO §1. **Effort:** medium.
- **Benchmark:** first compare image tensors byte for byte against the preload (no model needed). Then run a full clip, compare the replay hash, and sample RSS and `cudaMemGetInfo`.

**6. Hygiene batch.**
- Cache `freqs_cis.repeat` (`position_encoding.py:230-233`): about 5–7 ms/frame.
- Drop the extra `astype` and copy in `sam2_driver.py:106,136-139`: 3–6 ms/frame.
- Pin `fill_hole_area=0`. This guards against a silent output change if `SAM2_UPSTREAM_DIR` ever points at a fork that has `_C`.
- Call `remove_object(need_output=False)`.
- Load the reader lazily when a reads cache is present.
- In PRTReID, run only the backbone, the global pooling and the dim-reduce (`embeddings/prtreid.py:159-187`). Keep the team head for `team_logits`.

Together these are about 2–3%.

**7. Shape-grouped batching of memory attention, memory encoder and mask decoder.**
- **This is the largest same-model lever.** `sam2_video_predictor.py:585-614` runs objects at B=1. Group objects by the number of memory frames and object pointers they hold.
- **Output:** not guaranteed bit-identical, because batching changes GEMM shapes. Test byte equality first. If outputs differ, re-score all three clips against docs/tracking-evaluation.md §8.
- **Gain:** about 1.25–1.4× on the loop, but this is an estimate with no profiler behind it. **Effort:** large.
- **Prerequisite:** a 1-frame `torch.profiler` run to confirm `flash_fwd` kernels and to see whether there are launch gaps.

**8. Lifecycle constants (output-changing).** Each live object costs about 20–25 ms/frame. `UNMATCHED_CHECKPOINTS_MAX=20` and `MAX_LIVE_OBJECTS=20` (`sam2_manager.py:58,76`) are cheap knobs. Any change needs a full three-clip re-score.

**Detector.** Detection is about 6–10% of a rendered frame and is cached, so it is low priority. Note a provenance gap first: every scored tracker number used the hosted detector caches `3z9xf/3`, not the local fine-tune. Any detector-engine change has no scored baseline until the fine-tune caches are scored. If you do work on detection:
- In `cache_number_detections.py:238-242`, pass `include_source_image=False` and add a decode prefetch thread. Both are output-preserving.
- Do not do a naive GPU `/255`. It can differ by 1 ulp; use a lookup table instead.
- MCByte only: use the cached number boxes (`render_full_pipeline.py:719-722`). This changes MCByte's outputs.

## 3. What not to do

- **`vos_optimized` torch.compile:** measured slower and output-changing (docs/sam2-speed-research.md).
- **`async_loading_frames=True`:** refuted. It normalises in float64, which changes the model input, and holds 25 MB/frame (`misc.py:146-164`).
- **Smaller SAM2.1 checkpoints:** T, S, B+ and L share the same temporal path, so the ceiling is about 1.15–1.2× ([SAM2 README](https://github.com/facebookresearch/sam2#model-description)).
- **Rejected trackers:** EfficientTAM-S (same-team mask collapse), EdgeTAM, SAM-MT and ByteTrack.
- **Forcing cuDNN SDPA:** broken on SM121 ([NIM notes](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/2.3/release-notes.html)).
- **Global `cudnn.benchmark`:** non-deterministic.
- **Standalone flash-attn, FA3 or FA4:** no gain over torch's built-in FA2 for sm_121 (v2.9.1 `sdp_utils.cpp`).
- **FP8, or FP16 purely for speed:** same peak rate as BF16 on GB10 ([forum](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993)).
- **Inference at 512/768 resolution or with 4 memories:** out of distribution, with high quality risk.
- **DeepStream MaskTracker:** it owns association and lifecycle, which conflicts with our manager.
- **TensorRT 10.14.x:** silently wrong DETR scores on sm_12x ([forum](https://forums.developer.nvidia.com/t/tensorrt-10-14-silently-produces-wrong-detection-scores-for-d-fine-detr-style-models-on-rtx-5090-sm-120-default-build-drops-12-detections-to-0/380512)).
- **Unpinned `rfdetr[tensorrt]` on TensorRT 11:** silently builds an FP32 engine.
- **RF-DETR `optimize_for_inference`:** measured slower (TODO.md:557).
- **nvJPEG for SAM2 frames:** not bit-exact with libjpeg-turbo.
- **Installs or a torch 2.10 upgrade in the main venv:** the torch install is fragile, and a new version changes numerics.
- **TensorRT module swap for SAM2** (about 1.4–1.7×, needs a full re-score): only if item 7 falls short.
- **`run_overnight_batch --render` at its default 10 minutes:** a 15k-frame `init_state` is an OOM, not a slow run.

## 4. Benchmark plan after the fine-tune finishes

1. **Preconditions:** `nvidia-smi` shows no other process. Run each configuration 3 times, because power-limited clocks add variance. Workloads are fixed: Han-Ber4 (198 frames, tracking-only, `CHECK_EVERY=10`) and FelixClaar (full `render_full_pipeline --tracker sam2`, `--reads-cache`, `--geometry-cache`).
2. **Baseline:** a synchronised per-stage timer profile of the full render, a 1-frame `torch.profiler` for SDPA kernel names, and a cProfile of `checkpoint()`. Hash the geometry npz, run JSON, reads JSON, lifecycle events and mp4.
3. **Apply items 1→6 one at a time.** Accept an item only when all hashes on the three clips are identical, and when the median ms/frame improves by more than 2× the run-to-run spread.
4. **Item 7:** run the byte-equality check first. If outputs differ, do the full §8 re-score, with `*_reads.json` regenerated deliberately (stale keys silently replay as `''`). Accept only if no clip regresses from 95.0 / 99.91 (players only) / 98.9.
