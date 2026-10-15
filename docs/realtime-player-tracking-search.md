# Real-time player tracking search — 2026-10-15

**Follow-up:** Further work is on branch `experiment/sam31-multiplex` in the
new sibling worktree. The user considers the measured 2-to-3.3 FPS gain worthwhile;
30 FPS is a longer-term target. See [the local results](sam31-multiplex-results.md).

**Superseded recommendation:** The user has clarified that matching production
SAM2 tracking accuracy is a hard gate; speed does not compensate for regression.
The [quality-first follow-up](quality-first-tracker-research.md) now prioritizes
SAM3.1 Object Multiplex for investigation and DeepStream MaskTracker as a
secondary feasibility lead. The box-tracker comparisons below are retained as
research history, not the current next-step ranking. No replacement has passed
the local quality or 30 FPS gates.

## Updated requirement

The user explicitly accepts **30 FPS player IDs, with masks generated when
needed** for jersey reading and overlap analysis. Fresh masks for every player
on every frame are no longer a replacement requirement. Retain local RF-DETR,
tracker-independent team observations, reversible identity evidence and the
existing jersey reader. Production SAM2 remains the reference and default until
a replacement matches its tracking accuracy on each clip, including difficult
overlap continuity. Any accuracy regression disqualifies a faster candidate.

The SAM-MT comparison was explicitly stopped; this search does not restart it.
This was research and source inspection only: no new models, containers or
inference jobs were installed or run.

## Candidate evidence, separated by quality and speed

| Candidate | Public quality evidence | Speed evidence / remaining gap |
|---|---|---|
| TrackTrack | DanceTrack test: 66.5 HOTA / 67.8 IDF1 / 52.9 AssA | No verified GB10 FPS; published pipeline uses extra training data and post-processing |
| DiffMOT | DanceTrack test: 62.3 HOTA / 63.0 IDF1; SportsMOT test: 76.2 / 76.1 with train+val-trained detector | Better DanceTrack configuration is 22.7 FPS; 30.3 FPS uses a weaker detector and loses quality |
| NvDCF / DeepStream 9.1 | NVIDIA shows pedestrian occlusion recovery and cites MOT17 submission NvMOT_DSv62; comparable sports identity numbers were not established | 190 FPS RT-DETR pipeline on Spark uses max_perf; does not establish accuracy-config speed or handball quality |

## Public quality comparisons checked

Higher is better. HOTA combines detection and association; IDF1 measures identity
consistency. Neither is our local percent-correct score. These are published
pipeline results, not a controlled tracker-only comparison using our RF-DETR.

### DanceTrack test: similar appearance and nonlinear motion

| Published pipeline | HOTA | IDF1 |
|---|---:|---:|
| OC-SORT (TrackTrack paper Table 3) | 55.1 | 54.6 |
| CBIoU | 60.6 | 61.6 |
| Deep OC-SORT | 61.3 | 61.5 |
| DiffMOT + YOLOX-X | 62.3 | 63.0 |
| Hybrid-SORT-ReID | 65.7 | 67.4 |
| TrackTrack | 66.5 | 67.8 |

Sources: [TrackTrack, Table 3 and implementation details](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf),
[DiffMOT, Tables 1 and 4](https://diffmot.github.io/static/pdfs/paper.pdf).

TrackTrack is better than DiffMOT on this benchmark, but its implementation uses
YOLOX-X trained with extra CrowdHuman/Widerperson data, dataset-trained FastReID,
and post-processing. DiffMOT's DanceTrack comparison explicitly excludes extra
training data. Thus the 4.2 HOTA / 4.8 IDF1 difference cannot all be attributed to
association. TrackTrack's own component ablation (Table 4) is more controlled:
adding TPA+TAI raises DanceTrack-val HOTA from 61.9 to 63.3 and AssA from 47.7 to
49.7 within its pipeline. Table 5 changes the assignment method and improves
61.6 to 63.3 HOTA. A strict live implementation must report quality without any
future-frame post-processing before using these scores as a real-time claim.

### DiffMOT's speed-quality tradeoff: same DanceTrack test

| Detector | HOTA | IDF1 | FPS on RTX 3090 |
|---|---:|---:|---:|
| YOLOX-S | 53.3 | 56.6 | 30.3 |
| YOLOX-M | 57.2 | 58.6 | 25.4 |
| YOLOX-L | 61.5 | 61.7 | 24.2 |
| YOLOX-X | 62.3 | 63.0 | 22.7 |

The 30.3 FPS headline comes with a **9.0 HOTA / 6.4 IDF1 point loss** relative
to its stronger detector. It is not evidence of 30 FPS at the best published
quality, and it cannot be combined with the SportsMOT best score as one tested
configuration. [Authors' paper, Table 4](https://diffmot.github.io/static/pdfs/paper.pdf).

### SportsMOT test: detector training protocol matters

| Detector training split | Tracker | HOTA | IDF1 |
|---|---|---:|---:|
| Train only | OC-SORT | 71.9 | 72.2 |
| Train only | DiffMOT | 72.1 | 72.8 |
| Train + validation | OC-SORT | 73.7 | 74.0 |
| Train + validation | DiffMOT | 76.2 | 76.1 |

DiffMOT's gain over OC-SORT is modest, especially in the train-only setting:
0.2 HOTA and 0.6 IDF1. Even the stronger setting gains 2.5 HOTA and 2.1 IDF1.
These numbers do not establish SAM2-like handball continuity. SportsMOT covers
basketball, soccer and volleyball, not handball.
[Authors' paper, Table 2 and dataset description](https://diffmot.github.io/static/pdfs/paper.pdf).

### NvDCF: public sports quality remains a gap

NVIDIA's primary article demonstrates pedestrian occlusions and names the MOT17
submission NvMOT_DSv62. It does not give a SportsMOT/DanceTrack comparison or a
controlled comparison against SAM2. The speed and quality examples use different
settings and workloads; neither establishes that the 190 FPS max_perf pipeline
preserves the identity quality shown in its Re-ID demonstrations. NvDCF remains
a throughput experiment, not an established quality winner.
[NVIDIA's quality demonstrations and benchmark claim](https://developer.nvidia.com/blog/state-of-the-art-real-time-multi-object-trackers-with-nvidia-deepstream-sdk-6-2/).

### 1. NvDCF: strongest hardware-relevant lead

NVIDIA's DeepStream 9.1 table lists **190 FPS for RT-DETR + NvDCF on DGX Spark**,
FP16, 640×640 inference, using the supplied 1080p sample video. Their timing
includes decoding, preprocessing, inference and metadata generation, with
rendering disabled. It uses the speed-focused `max_perf` tracker configuration.
**RT-DETR is not our RF-DETR**, the sample is not handball, and the target count
is not specified. This proves a fast implementation exists on the right
hardware class, not that our complete application runs at 190 FPS.
[NVIDIA performance table and measurement settings](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html).

NvDCF maintains an online visual correlation filter per target and can localize
between detections. NVIDIA batches cropping, feature extraction, filter learning
and localization on GPU. Re-association can use motion and Re-ID. This offers
visual evidence beyond box motion at much lower complexity than dense VOS.
The docs distinguish `max_perf`, `perf` and `accuracy` configurations; benchmark
the latter two as well rather than claiming the speed configuration's result
also proves identity quality.
[Official tracker design and settings](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#nvdcf).

DGX Spark uses the dedicated
`nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark` container. NVIDIA does not
support native DeepStream installation on Spark. Docker exists locally;
`deepstream-app` and an existing DeepStream container were not found in this
inspection. Setup should stay isolated from the working Python/CUDA environment.
[Official Spark container instructions](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html#a-docker-container-for-dgx-spark).

### 2. DiffMOT: released sports-trained motion alternative

DiffMOT learns nonlinear motion rather than using only constant-velocity
prediction. Its official repository supplies motion weights and SportsMOT Re-ID
resources. It reports **30.3 FPS including detection and tracking with YOLOX-S**
on RTX 3090, but that DanceTrack configuration scores only 53.3 HOTA / 56.6
IDF1. YOLOX-X achieves 22.7 FPS at 62.3 HOTA / 63.0 IDF1. The separate best
SportsMOT result is 76.2 HOTA / 76.1 IDF1 with a train+val-trained detector. These metrics cannot be compared numerically with our percent-correct
scorer, and the small-detector timing does not transfer to RF-DETR automatically.
Its evaluation accepts prepared detections and embeddings, making a cached
RF-DETR comparison a reasonable first adaptation. The older Torch 2.0.1 setup
must not replace our GB10 CUDA packages.
[Official code, results and weights](https://github.com/ShuCvlab/DiffMOT),
[paper](https://arxiv.org/abs/2403.02075).

### 3. TrackTrack: cheaper association experiment

TrackTrack changes association to consider detections together and suppress
redundant track initialization. The authors show overlap/occlusion improvements
and also document remaining failures under severe occlusion and rapid motion.
Its official release separates YOLOX, FastReID and tracker stages; our detector
would need an adapter. It is a candidate for testing, with no established local
30 FPS or handball-quality claim.
[Official code](https://github.com/kamkyu94/TrackTrack),
[authors' qualitative results and failure cases](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Shim_Focusing_on_Tracks_CVPR_2025_supplemental.pdf).

## Other leads checked

- **SAM3.1 Object Multiplex:** Meta's own graph shows 30.2 FPS at 16 objects on
  H100, not GB10. A released sports project reports only 7.3 FPS aggregate for
  its full SAM3.1 text-detection pipeline on RTX 5090; that also changes the
  detector. These are different protocols, but neither establishes our desired
  local speed. The later quality-first follow-up elevates this architecture above box-ID
  candidates; optional output masks do not waive tracking quality.
  [Meta release and graph](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md),
  [sports project's measured native pipeline](https://github.com/holma91/selective-mask-propagation#postscript-sam-31).
- **CUTIE:** the multi-target scaling table in SAM-MT's paper measures 18.4 FPS
  at 15 objects at 480p and 7.3 at 1024p on A6000. This is more relevant than a
  generic single-object real-time claim. It remains useful as an optional mask
  propagator rather than a demonstrated 30 FPS dense replacement.
  [Primary scaling table](https://arxiv.org/html/2607.08688v1#S5.T3).
- **LiVOS:** released CVPR 2025 model with a compact recurrent linear-memory
  state. Its paper measures 47.3 FPS on 480p LVOS validation, but 12.8 FPS on
  upsampled 1024p DAVIS, on A6000. Neither establishes fourteen-player GB10
  throughput or handball continuity. Memory efficiency is credible; 30 FPS for
  our workload remains unproven.
  [Paper, Tables 2–3](https://arxiv.org/pdf/2411.02818),
  [official code and weights](https://github.com/uncbiag/LiVOS).
- **MAVOS:** published 37 FPS on V100 for LVOS, with released weights, but no
  verified fourteen-player GB10 result. The supplement explicitly shows failure
  on near-identical objects after severe occlusion. Lower-priority mask option.
  [Official release](https://github.com/Amshaker/MAVOS),
  [authors' failure example](https://openaccess.thecvf.com/content/WACV2025/supplemental/Shaker_Efficient_Video_Object_WACV_2025_supplemental.pdf).
- **StreamDAM:** an August 2026 paper whose real-time evaluation is explicitly
  single-object on RTX 6000D. It leaves multi-object streaming open. It does not
  solve our demonstrated scaling requirement.
  [Paper, experiment protocol and limitations](https://arxiv.org/html/2608.03912v1).
- **SegFS:** genuine fast/slow segmentation design, but its headline 38.2 FPS is
  amortized over six frames at 480×480 on Galaxy S25 Ultra. I did not establish
  released runnable code/checkpoints from the inspected paper and search. Not
  the next ready-to-run local candidate.
  [Paper and timing protocol](https://arxiv.org/html/2607.00124v1).
- **SportMamba:** sports-specific nonlinear motion with a roughly 30 FPS claim;
  public code/checkpoint availability was not established in this search.
  DiffMOT has a more concrete released experiment path.
  [Authors' paper](https://arxiv.org/html/2506.03335v1).

## Earlier experiment proposal (superseded)

The quality evidence now favors inspecting a strict online TrackTrack adaptation
first, including which published gains survive without post-processing or its
trained detector. NvDCF remains an alternative throughput experiment. The
following NvDCF procedure is retained as a concrete option, not an adoption
decision or an already-authorized installation.

1. Prepare isolated NvDCF on the Spark container. Feed the existing RF-DETR
   detections and unchanged JPEG frames; preserve all relevant people and the
   original frame spacing. Test `perf` and `accuracy`, including Re-ID cost.
2. Score FelixClaar first against the existing trusted reference, and inspect
   the known teammate overlap densely. Continue to Han-Ber4 and BHC-FAG if the
   result is competitive with SAM2. Keep raw tracker output separate from
   downstream identity repairs.
3. Measure real tracker latency with fresh frames at 12–14 targets, then
   end-to-end RF-DETR + tracker latency. Cached detection timing cannot establish
   a 30 FPS deployed pipeline. Report p50/p95 and deadline misses, not only
   aggregate throughput.
4. Schedule mask generation on read/overlap events and account for that workload
   separately. Delayed masks must remain attached to their original frame and
   identity; do not present stale masks as newly inferred masks.
5. Use bounded/streaming inputs and check system memory before GPU runs. Earlier
   severe shared-memory/swap pressure invalidates clean speed conclusions.

Do not change production defaults or retire SAM2 from research claims alone.
The next adoption decision needs measured local identity quality and speed.
