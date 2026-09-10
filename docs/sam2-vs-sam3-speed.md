# SAM2 versus SAM3 speed — 2026-09-10

SAM3 is not an established speed upgrade over SAM2. SAM3.1 greatly improves
multi-object throughput over SAM3, but no verified comparison found here
establishes a gain over SAM2.1-L for 12–16 players on GB10. Treat the earlier
SAM3.1 recommendation as an architecture candidate, not a measured speed ranking.
This pass researched sources and inspected code; it ran no inference and left
the llama server untouched.

## Follow-up: direct comparisons found after widening the search

Useful direct comparisons do exist. Requiring an exact GB10/14-player match
before treating them as evidence was too restrictive. These results strengthen
the conclusion that original SAM3 commonly costs more than SAM2.

| Comparison | SAM2 | SAM3 | Scope |
|---|---:|---:|---|
| SurgSLOT paper, vanilla models | Hiera-B+: 26 FPS | 9 FPS | Surgical video, same A6000, default settings/resolutions |
| Medical-image study | Large: 42.7 FPS | 23.0 FPS | Same images/platform, FP16, batch 1, 1024 inputs |
| sam3.cpp video benchmark | 2.1-L: 8.5 s/frame | Visual-only: 22.6 s/frame | M4 Pro CPU, F16 weights, 4 threads, 5-frame test |

SurgSLOT section 5.3.2 specifies end-to-end timing on the same A6000. Vanilla
SAM2 uses B+ at 1024; SAM3 uses its default resolution. This is a direct video
comparison, about 2.9x throughput for SAM2 in that setup, but neither a Large
comparison nor a dense sports sweep. Do not substitute the fine-tuned 68-versus-20
FPS pair: those use different reduced resolutions, 512 versus 672.
[SurgSLOT v2, table 6](https://arxiv.org/html/2511.16618v2)

The medical study identifies an Ubuntu platform with two RTX 4090 GPUs; its
table does not establish two-GPU scaling. This corroborates the static-image
speed direction, not video throughput.
[Medical comparison, table 7](https://onlinelibrary.wiley.com/doi/10.1111/exsy.70383)

The C++ project publishes a harness and CPU/Metal results. The Large Metal
entry is missing, so the Large comparison above uses CPU only. It establishes
a result for that implementation, not PyTorch/CUDA performance.
[sam3.cpp benchmark](https://github.com/PABannier/sam3.cpp#benchmarks),
[harness](https://github.com/PABannier/sam3.cpp/blob/main/examples/benchmark.cpp)

A first-hand H200 report gives SAM2 above 30 FPS versus SAM3 at 5–6 FPS on the
same 1080p video. Object count, checkpoint and settings are insufficiently
specified: corroborating experience, not a controlled benchmark.
[Upstream issue 425](https://github.com/facebookresearch/sam3/issues/425)

For sports, Holmberg reports SAM3.1 dense tracking at 7.3 FPS and 27.2 GB peak
on four RTX 5090 broadcast clips. Uniform SAM3 propagation averages 5.5 FPS on
those clips. SAM3.1 uses its own detector with a player text prompt; uniform
SAM3 is seeded from external tracks. This practical system comparison has
different detection workloads and is not an isolated model ratio. It compares
SAM3.1 with SAM3, not SAM2.
[Author's sports benchmark](https://www.holma.io/blog/posts/selective-mask-propagation)

## Direct image comparison

The SAM3 paper reports 93.0 FPS for SAM2.1-L and 43.5 for SAM3 in its interactive
image-segmentation table: SAM2 has 2.14 times the throughput. This is image
prompting, not temporal video tracking; do not infer a 14-player video ratio.
The paper separately quotes about 30 ms for concept segmentation of an image
with over 100 objects on H200. That does not mean tracking 100 objects at 33 FPS.
[Primary paper, tables and introduction](https://arxiv.org/html/2511.16719v2)

## SAM3.1 multi-object scaling

Values transcribed from Meta's labeled release graph, single H100:

| Objects | SAM3 November release FPS | SAM3 + new optimization FPS | SAM3.1 FPS |
|---|---:|---:|---:|
| 1 | 26.5 | 36.4 | 33.8 |
| 4 | 19.7 | 31.5 | 32.5 |
| 8 | 14.6 | 19.2 | 31.6 |
| 16 | 9.8 | 12.2 | 30.2 |
| 32 | 5.8 | 7.0 | 22.1 |
| 128 | 1.6 | 2.2 | 11.5 |

At 16 objects the ratios are 3.08x versus original SAM3 and 2.48x versus
optimized SAM3. At one object multiplex is slower than optimized SAM3.
The 7x headline concerns 128 objects versus the November release. There is
no SAM2 curve. [Official graph](https://github.com/facebookresearch/sam3/blob/main/assets/sam3.1_efficiency.png)

Meta's blog separately summarizes roughly 16 to 32 FPS at medium object counts,
without fixing that count in the sentence. Use the graph for explicit counts.
[Release announcement](https://ai.meta.com/blog/segment-anything-model-3/)

Appendix H compares multiplex with an optimized SAM3 baseline and reports 5.2x
at 128 objects. This is consistent with 11.5/2.2 in the graph; the 7.2x label
uses the older 1.6 FPS baseline. It describes optimized internal implementations,
including batching, fewer synchronizations and compilation support.
[Appendix H](https://arxiv.org/html/2511.16719v2#A8)

## Why the published FPS cannot be compared directly

Official SAM2.1-L: 39.5 FPS; B+: 64.1 FPS, on A100 with PyTorch 2.5.1/CUDA 12.4.
[Official SAM2 model table](https://github.com/facebookresearch/sam2#model-description)
Its example benchmark uses one point-prompted object, BF16, full compilation,
warmed repeated propagation and discarded outputs. It has no application
manager, dynamic detector prompts or CPU mask processing, and its timing loop
has no explicit CUDA synchronization.
[SAM2 benchmark](https://github.com/facebookresearch/sam2/blob/main/sam2/benchmark.py)

SAM3's inspected speed script builds the high-level video predictor, uses a
synthetic moving-circle video and a text prompt, and times propagation after
reset/prompt setup. BF16 and compilation are used by default. It selects the
best throughput across rounds, including warmup, rather than reporting their
mean. It synchronizes at the end, not explicitly just before the interval.
Generated circle count is not independently verified as active tracked-object
count. This is not our RF-DETR-driven, tracker-only lifecycle loop.
[SAM3 benchmark](https://github.com/facebookresearch/sam3/blob/main/scripts/measure_speed.py)

A100 SAM2 single-object FPS and H100 SAM3.1 multi-object FPS cannot be divided
to estimate a GB10 speedup. Our measured SAM2 loop uses eager BF16 with roughly
12–14 live objects and includes manager/checkpoint/mask work. Same-weight SAM2
compilation already failed locally; this research does not reopen that experiment.
[Local baseline and rejected compilation](sam2-speed-research.md)

## Independent comparison found, with unresolved protocol limitations

The August RS3-Prune preprint reports unmodified SAM2 versus SAM3.1 at 9.6 versus
8.7 FPS on DAVIS17, and 9.1 versus 3.7 on SA-V, on a stated RTX-4090-class GPU,
batch size one. However the inspected methods do not identify enough checkpoint,
precision, compilation and multiplex/active-object details for our comparison.
Its reported SAM3.1 DAVIS J&F of 56.0 also differs greatly from Meta's 92.7.
The discrepancy needs investigation before attributing it to model quality.
This is evidence that a particular evaluated implementation was slower, not a
reliable model ranking or prediction for our application.
[RS3-Prune, section 4 and table 1](https://arxiv.org/html/2608.22526v1)

## Decision for this repository

Original SAM3 has no demonstrated speed advantage that justifies a migration.
SAM3.1 remains a plausible scaling experiment, with unmeasured GB10 gain and
unproven preservation of object-local reset semantics. It cannot yet be promised
to double our tracking-loop throughput.

The next useful speed evidence would be a tracker-only 1/8/14/16-object sweep
with native resolutions reported, common BF16 precision and complete mask outputs.
Before a full adapter, check whether shared state can preserve each survivor's
history through one player's reset. If promising, use the existing paired clips,
including the verified Felix same-team failure, with RF-DETR and TrackManager
unchanged. Benchmark service concurrency must be recorded; do not stop the
llama server to obtain an isolated run.
