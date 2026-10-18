# Tracking candidates after Cutie — 2026-10-16

Research only. Branch `research/post-cutie-20261016`, separate worktree
`handball-computer-vision-tracker-research-next`. No installations, weights,
inference, GPU work, or production changes.

## Decision and constraints

**Test Selective Mask Propagation (SMP) next; retain Lean-SAM2 as the next
incremental-speed candidate. Neither has established equivalence to our SAM2
tracker on handball.** User confirmed that later frames may correct earlier IDs.
This makes retrospective association usable. Quality remains the acceptance
condition; 30 FPS IDs with masks when needed remains the longer-term target.

SAM-MT, CAMELTrack, EfficientTAM and Cutie have failed the local quality
requirement. In particular, user review of unseen Hannover and VfL clips rejected
Cutie despite attractive older aggregate scores. Do not reuse those scores as
proof of SAM-level identity preservation. Preserve production SAM2, the existing
SAM3.1 experiment, RF-DETR detections, and independent/reversible team evidence.

## 1. Selective Mask Propagation: best next experiment

A cheap base tracker supplies trajectories. SAM handles selected uncertain
intervals, seeded from earlier clear observations, and later evidence can correct
IDs. This preserves SAM's tracking mechanism in the difficult intervals while
avoiding its cost for every player throughout the video.

Public SportsMOT test results, including global track association (GTA):

| Configuration | HOTA | IDF1 |
| --- | ---: | ---: |
| Deep-EIoU | 77.2 | 79.8 |
| SMP + SAM2 + GTA | 85.5 | 91.2 |
| SMP + original SAM3 + GTA | 87.2 | 93.6 |

**The ablation is more informative than the headline:** on SportsMOT validation,
Deep-EIoU scores 78.9 HOTA, selective SAM3 alone 80.4, GTA alone 86.7, and both
88.5. Most improvement comes from linking tracklets after disappearance, rather
than selective propagation alone. These are box-tracking metrics, not comparisons
against our periodically reprompted SAM2 system.
[Paper, Tables 3–4](https://arxiv.org/html/2606.13033).

Reported speed is **SAM + merge only**, excluding the base tracker, detector and
GTA: 13 FPS overall SportsMOT on RTX PRO 6000. On two bundled basketball clips,
RTX 5090 selective propagation achieves 8.7/9.5 FPS versus uniform SAM3's 6.0/6.1.
Football gains are much larger; basketball is my more relevant proxy for indoor
handball. These figures do not predict our GB10 throughput.
[Official efficiency results](https://github.com/holma91/selective-mask-propagation#efficiency).

Source audit: revision `6472b52e79c47a7793cd4c6cabd64e812a8e5852`.
`augment()` accepts external boxes and assignment margins, so RF-DETR can stay.
The SAM2 path uses SAM2.1-Large. Dispatch includes low-margin, gap and overlapping
witness intervals. The default output preserves base box geometry and corrects
IDs; `mode="prod"` makes masks authoritative in healthy windows and needs separate
evaluation. The public helper returns boxes; mask-dependent consumers need the
lower-level mask outputs. GTA additionally uses pose, PARSeq and Gemini team
labels; replacing its team classifier means a different configuration from the
reported results.
[Augmentation API](https://github.com/holma91/selective-mask-propagation/blob/6472b52e79c47a7793cd4c6cabd64e812a8e5852/selective_mask_propagation/augment.py),
[SAM2 implementation](https://github.com/holma91/selective-mask-propagation/blob/6472b52e79c47a7793cd4c6cabd64e812a8e5852/selective_mask_propagation/core/sam2.py),
[GTA dependencies](https://github.com/holma91/selective-mask-propagation#global-track-association-gta).

Remaining risks: confident wrong associations can escape dispatch; missing
observations remain a concern; crowded handball may keep SAM active much of the
time. Retaining SAM selectively does not guarantee retaining dense SAM2 quality.

## 2. Lean-SAM2: credible smaller speed gain

Keeps SAM2.1 weights and reduces memory/encoder work. Its Large model reports
1.412× overall model speedup on RTX 3090, strict FP32. LVOSv2 J&F changes
84.2 → 83.6. This is segmentation quality, not a player-ID metric, and our BF16
GB10 pipeline may gain differently.
[Official results and checkpoints](https://github.com/DeawhaleQwQ/Lean-SAM2).

Paper Table I: full model averages 73.1 versus SAM2.1-Large 73.3 across five
segmentation evaluations. Memory-only TAMP+TCIM averages 73.4, but its **1.679×
speedup measures memory attention only**, not the complete model. Dataset changes
include LVOSv2 84.2 → 84.0 and MOSEv2 49.4 → 48.9. Close averages do not establish
identity equivalence during rare occlusions. The abstract's 5-point gain compares
with Efficient-SAM2, not baseline SAM2.
[Paper, Table I and timing protocol](https://arxiv.org/html/2607.19811).

Source audit: revision `8d1cf2618519d291905d22fe859b8302476d7f50`.
The predictor allows adding objects after propagation starts, exposes reprompting
and removal, and remaps insurance memory when an object is removed. These are
source observations, not lifecycle tests. The released dataset runner requires
all three acceleration modules and bypass weights; a memory-only experiment
needs an explicit adapter/configuration rather than assuming an existing CLI
switch. Anchor refresh after corrections and cached encoder routing across
multiple players still need verification.
[Predictor](https://github.com/DeawhaleQwQ/Lean-SAM2/blob/8d1cf2618519d291905d22fe859b8302476d7f50/sam2/sam2_video_predictor.py),
[Configuration](https://github.com/DeawhaleQwQ/Lean-SAM2/blob/8d1cf2618519d291905d22fe859b8302476d7f50/CONFIGURATION.md).

## Additional leads screened

- **LiAM-SAM:** RF-DETR + SAM2 lifecycle/memory method. Attention pruning reports
  4-object speed 5 → 10 FPS and 30-object speed 2 → 5 on A100, with DanceTrack
  HOTA 78.2 → 76.3. Main accuracy results disable pruning. No official runnable
  implementation located in the paper or GitHub search on this date. Research
  lead, not an immediately reproducible next model.
  [Paper, Table 5 and implementation section](https://arxiv.org/html/2609.28078).
- **NOOUGAT:** learned graph association becomes relevant now that look-ahead is
  allowed: SportsMOT offline 85.6 HOTA / 92.3 IDF1. Its 340 FPS DanceTrack figure
  excludes detection and ReID extraction (V100); no official code/checkpoints
  located. It still relies on detections, so missing-player coverage remains
  unproven. Do not promise a runnable fast replacement.
  [Paper, Tables 2 and runtime analysis](https://arxiv.org/html/2509.02111).
- **Efficient-SAM2:** distinct from rejected EfficientTAM, but Lean-SAM2's matched
  comparison shows larger quality losses. Lower priority than Lean-SAM2.

## Bounded next experiment

If proceeding, create a fresh experiment branch/worktree. Use cached RF-DETR
boxes, the released Deep-EIoU + OSNet base association, and SMP with our existing
SAM2.1-Large checkpoint first. Score the base, SMP box output, SMP mask output,
and production SAM2 on the same clips and frozen references. Keep GTA disabled
for the first isolated propagation comparison and state this explicitly; those
results cannot be represented as reproduction of the headline SportsMOT system.

Prioritize the unseen Hannover/VfL passages where Cutie failed. Inspect same-team
crossings, full occlusion recovery, exits/re-entry, fragmentation and confident
wrong identities before broad speed benchmarking. Measure complete stages,
including appearance features, dispatch, propagation, merge and any later
linking. No claim of SAM2 equivalence from one aggregate percentage.
