# Faster tracking alternatives beyond SAM3.1 — 2026-10-16

## Finding and scope

**No inspected alternative has public proof of preserving our SAM2.1-L handball
identity accuracy while running faster with 12–14 players on GB10.** There are
credible experiments to run, but no evidence-backed drop-in replacement yet.

The most useful additions to the previous search are **CAMELTrack and TDLP**
(released, learned online association), a closer examination of **Selective Mask
Propagation** (strong sports quality, but retrospective corrections), and a
check of **RDS, OASIS, TinySAM 2 and EdgeDAM** against their actual limitations.
The distinction matters: stronger public quality justifies a local experiment;
it does not demonstrate equality to our production tracker.

HOTA and IDF1 describe multi-object detection/association; J&F measures mask
region/boundary quality. None converts directly to our local correct-observation
percentage. FPS figures below retain their published hardware and workload
scope; none is a measurement on this machine.

This research lives on branch `research/quality-tracker-alternatives-20261016`,
based on `370f667`, in:
`/home/valentinweyer/projects/handball-computer-vision-tracker-research`.
The user's SAM3.1 investigation is in a different worktree and was not modified.
No inference, model installation, weight download or production change was run.
Only public paper/source pages, research notes and release metadata were read.

Requirements carried forward: retain RF-DETR; accuracy takes precedence over
speed; incremental improvements around 2→3.3 FPS are worthwhile; 30 FPS remains
the longer-term target. IDs should update every frame; output masks may be
requested selectively. Optional mask output does not waive tracking quality.
SAM-MT and EfficientTAM remain locally rejected; this research does not rerun them.

## What is worth testing

| Direction | Why it is relevant | What prevents an adoption claim |
|---|---|---|
| **CAMELTrack** | Released online learned association with sports checkpoints; stronger public identity scores than older heuristic trackers | Uses learned appearance/pose cues; no matched comparison with our SAM2; full cue extraction can dominate runtime |
| **Cutie** | Released mask-memory tracker with addition, mask corrections and removal APIs; a relatively small integration experiment | Public generalization gap against SAM2; multi-object speed gain is much smaller than a generic real-time headline |
| **Selective Mask Propagation (SMP)** | Retains SAM for the ambiguous crossings where masks help; strong published sports identity results | Released augmentation is retrospective, and top scores include offline tracklet linking; immediate live-ID equivalence is unproven |
| **TDLP** | Newer learned association with stronger sports results than CAMELTrack in its paper | Its fast box-only and stronger multimodal configurations have substantially different quality and cost |

This ordering is an engineering judgment, not a quality ranking across unrelated
metrics. For a strictly online non-SAM experiment I would screen CAMELTrack
first. For an experiment retaining internal mask memory I would screen Cutie.
SMP deserves attention if delayed corrections can be accommodated. TDLP is a
comparison candidate rather than an established speed upgrade.

## Learned online tracking: public quality and actual cost

### CAMELTrack

CAMEL learns how to combine temporal box, appearance and optional pose evidence.
It remains modular enough to accept external detections. Its paper reports
SportsMOT test **80.3 HOTA / 84.8 IDF1 with pose**, versus Deep-EIoU's
77.2 / 79.8; without pose it reports 80.4 / 84.8. These are published pipelines,
not matched RF-DETR or SAM2 comparisons. The reported **13 FPS is on MOT17**;
18 ms is association alone, with detection, pose and appearance adding cost.
[Authors' paper, Tables 3–4 and §4.2](https://arxiv.org/html/2505.01257).

Code and sports/global checkpoints are released. Hub metadata confirms those
files exist; no checkpoint was downloaded. The released sports checkpoint uses
pose. Its BPBReID input is not automatically interchangeable with our PRTReID
features: preserving tensor dimensions alone would not preserve learned meaning.
[Official implementation and weights](https://github.com/TrackingLaboratory/CAMELTrack).

### TDLP

The June 2026 v2 paper reports SportsMOT test **81.9 HOTA / 87.5 IDF1** for full
TDLP versus **74.8 / 79.0** for its box-only version. Its same-setup SportsMOT
profiling separates costs:

| Configuration | Association FPS | Complete pipeline FPS |
|---|---:|---:|
| TDLP, boxes only | 60.4 | 7.9 |
| TDLP, pose and appearance | 11.4 | 2.0 |
| CAMELTrack | 20.1 | 2.2 |

Thus 60.4 FPS cannot be paired with the full model's 87.5 IDF1. Hardware is
described only as a consumer GPU in that section; these are not GB10 predictions.
[Paper v2, Tables 2 and 13](https://arxiv.org/html/2512.22105).

Code and links to both checkpoint variants are published; linked weight contents
were not downloaded or validated.
[Official release](https://github.com/Robotmurlock/TDLP).

## Mask-memory alternatives

### Cutie: practical, but public accuracy is not equal

Cutie retains both pixel memory and object queries, with foreground/background
attention to reduce distractor confusion. Its public implementation exposes
`step(image, mask, objects)` for mask observations and `delete_objects` for
removal. A box-to-mask seed is still needed; preserving the detector correction
policy requires an adapter and local lifecycle tests.
[Official code](https://github.com/hkchengrex/Cutie),
[inference API](https://github.com/hkchengrex/Cutie/blob/main/cutie/inference/inference_core.py).

A useful direct comparison in the original SAM2 paper is:

| First-frame mask prompt | MOSE val J&F | DAVIS17 val J&F | SA-V test J&F |
|---|---:|---:|---:|
| Cutie-base+ | 71.7 | 88.1 | 62.8 |
| SAM2 Hiera-L | 77.9 | 90.7 | 78.4 |

These rows use **original SAM2**, not our production SAM2.1 checkpoint. They
show why being close on DAVIS is insufficient evidence of broad equivalence.
[Meta's paper, Table 6](https://arxiv.org/html/2408.00714).

A later authors' scaling experiment measures Cutie-base at **18.4 FPS at 480p**
and **7.3 at 1024p for 15 targets**, versus SAM2.1-B+ at **4.9 FPS at 1024p**,
on A6000 with synthetic videos. That supports a modest multi-object speed lead,
not a 30 FPS GB10 promise or a quality conclusion on those synthetic inputs.
[SAM-MT paper, §5.2/Table 3; used only as a baseline measurement](https://arxiv.org/html/2607.08688v1#S5.T3).

### OASIS: a newer Cutie-family option

OASIS adds edge/structure reasoning for occluded, similar objects. Its ordinary
training setting reports DAVIS17-val **88.3 J&F**, and its MOSE-trained setting
reports MOSE **67.5 J&F**. The **48 FPS** result is not a measured fourteen-player
GB10 throughput, and these tables do not establish SAM2-level generalization.
[Paper, Tables 1–2 and 5](https://arxiv.org/html/2507.18944).

Code is published and a checkpoint folder is linked. The folder's exact trained
weights and correspondence to benchmark variants still need verification.
[Official checkpoint instructions](https://github.com/jinlab-imvr/OASIS/blob/main/ckpts/README.md).

### LiVOS and MAVOS: keep as secondary leads

LiVOS's recurrent memory is appealing for long clips. Its paper reports
**47.3 FPS at 480p** on LVOS validation, but **12.8 FPS at 1024p** on upsampled
DAVIS, using A6000. MOSE accuracy is **64.8 J&F**; a smaller memory bank is not
proof of equal identity recovery after a player overlap.
[LiVOS paper, Tables 1–3](https://arxiv.org/html/2411.02818),
[official release](https://github.com/uncbiag/LiVOS).

The prior MAVOS review found a severe-occlusion failure on near-identical objects,
which is directly relevant to teammates. No new controlled SAM2-quality,
fourteen-player speed result was established here. Keep its 37-FPS LVOS/V100
claim separate from our workload.
[Authors' implementation](https://github.com/Amshaker/MAVOS),
[authors' failure analysis](https://openaccess.thecvf.com/content/WACV2025/supplemental/Shaker_Efficient_Video_Object_WACV_2025_supplemental.pdf).

## Hybrid: spend SAM computation on ambiguous crossings

SMP runs a cheap association tracker and invokes SAM in selected ambiguity
windows, seeding from an earlier clear frame. The public SportsMOT test results
are **85.5 HOTA / 91.2 IDF1 with SAM2**, and **87.2 / 93.6 with original SAM3**,
both including global track association. These are not results against our
periodically corrected SAM2 baseline.
[Authors' paper, Table 3 and §3](https://arxiv.org/html/2606.13033).

The released code consumes tracks across frames and applies identity corrections
after propagating windows; the paper's swap decision can rename earlier IDs.
Consequently, this implementation does not demonstrate immutable correct IDs at
each live frame's deadline. The top sports scores also include linking finished
tracklets, an offline operation.
[Released augmentation API](https://github.com/holma91/selective-mask-propagation/blob/main/selective_mask_propagation/augment.py).

The authors report **13 FPS for SAM+merge only** across SportsMOT on RTX PRO
6000. Separate four-clip RTX5090 timings report **15.8 FPS aggregate**, with the
two basketball clips at **8.7 and 9.5 FPS**. Those throughput measurements are
not evidence of a fixed live latency or 30 FPS on GB10.
[Official results, efficiency and GTA sections](https://github.com/holma91/selective-mask-propagation).

**Inference:** this is a more relevant architecture experiment than merely
requesting fewer output masks from a dense tracker. It reduces expensive
tracking work itself. A causal implementation would still need to prove that
its dispatch signal catches our difficult crossings early enough; late repairs
cannot silently be counted as immediate correct live IDs.

## Attractive headlines screened out

| Candidate | Checked evidence | Assessment for this requirement |
|---|---|---|
| RDS, activation 0.7 | SAM2 DAVIS 90.0→90.8, 29.8→35.1 FPS; SA-V 78.5→75.6; YTVOS19 88.5→87.3 | Not a general no-loss result; public code link on author's page is empty |
| RDS, activation 0.6 | 38.2 FPS with DAVIS 89.6; SA-V falls to 74.3 | Faster configuration loses more quality |
| TinySAM 2 | V100 25.6 FPS; average score 73.8 versus SAM2.1-T's 82.1, SA-V test 65.3 versus 76.5 | Its “90% of performance” claim is a sizeable loss, even against Tiny |
| EdgeDAM | 25 FPS on iPhone and 88.2 DiDi accuracy; single-object tracking | Author repo still says code coming; neither multi-player speed nor handball quality shown |
| SeC | SA-V throughput 18.1 versus SAM2's 22.0; stronger quality | Quality lead, not an acceleration |
| StreamDAM | Single-object streamed evaluation on RTX6000D; authors explicitly leave multi-object streaming open | Credible single-object lead, no proof for fourteen players |
| FECNet | Search surfaces a favourable comparison table; paper URL is `xxxx.xxxxx`, model URL is a placeholder | Insufficient auditable evidence to recommend |

Sources, respectively:
[RDS paper, Table 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Tang_Efficient_Video_Object_Segmentation_and_Tracking_with_Recurrent_Dynamic_Submodel_CVPR_2026_paper.pdf),
[RDS author's publication links](https://wd7ang.github.io/),
[TinySAM 2, Table 1](https://arxiv.org/html/2605.18013),
[EdgeDAM paper](https://arxiv.org/abs/2603.05463),
[EdgeDAM code status](https://github.com/SMRaza1009/EdgeDAM),
[SeC, Table 2](https://arxiv.org/html/2507.15852),
[StreamDAM setup and limitations](https://arxiv.org/html/2608.03912v1),
[FECNet release page](https://github.com/tomato233144/FECNet).

## Same-model engineering fallback

DeepStream MaskTracker retains SAM2's mask/temporal-memory mechanism and
accelerates all four networks with TensorRT FP16. It is distinct from NvDCF.
Its association, box fusion and memory updates differ from our driver, so same
weights do not establish identical IDs. The inspected NVIDIA sample asks for
x86 dGPU, leaving the full temporal mode on Spark/aarch64 unverified; no matched
12–14-player speed/accuracy figure was established. This remains a feasibility
lead rather than a new model recommendation.
[NVIDIA tracker documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#masktracker-developer-preview),
[official sample prerequisites](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker).

## Smallest useful next experiment

1. Test one released candidate against **production SAM2.1-L with its periodic
   corrections**, using existing RF-DETR detections and trusted player-only
   reference spans. Do not replace detector data or discard hard players to
   improve a score. Preserve the native feature heads required by its checkpoint.
2. Start with the Felix overlap and full annotated Felix clip; inspect masks or
   IDs densely and score misses, wrong identities and continuity separately.
   Then compare Han and BHC individually. A one-point gap on one correlated clip
   neither proves inferiority nor establishes an acceptable loss allowance.
3. Exercise target entry, removal, reprompt and reset before treating the adapter
   as usable. Inspect per-player history and survivor state, including capacity
   changes. This research makes no edits to the SAM3.1 fix the user is developing.
4. Only compare speed for configurations that remain quality candidates. Count
   pose, appearance, initialization, correction and requested masks. Measure
   both cached-detection tracker cost and live-pipeline throughput; separate
   immediate outputs from delayed revisions. Report latency and memory alongside
   FPS under comparable GPU load.

Paper/source snapshots and public Hub listing metadata are retained under
`runs/tracker_research_20261016/` in this worktree (gitignored). These are research
artifacts, not locally reproduced benchmark results. Code/weight availability
means links or file metadata were checked, not that a complete environment ran.
