# First CAMELTrack comparison — 2026-10-16

FelixClaar frames 0–140, NVIDIA GB10, existing cached RF-DETR detections.
CAMELTrack uses the released SportsMOT checkpoint and native GPU KPR/RTMPose cues,
FP32, with the prepared SportsMOT thresholds. No local tuning was applied.

| Metric | CAMELTrack | SAM2 |
|---|---:|---:|
| Correct identity, including missing observations | 83.70% | 96.84% |
| Reference coverage | 90.09% | 97.45% |
| ID switches | 26 | 7 |
| Wrong identity observations | 105 | 10 |
| Missing player observations | 163 | 42 |

Quality uses the repository's IoU>=0.5 box matching and human-verified reference
spans, with all 1644 trusted player observations as the denominator.
Both replays were scored over exactly the same 141 frames. SAM2 is the existing
full-run baseline, restricted to this prefix. This local reference metric is not
HOTA/IDF1 and is not independent ground truth for SAM2.

CAMELTrack loop: 5.47 FPS, 25.78 seconds for 141 frames.
Includes JPEG decoding, pose, appearance and association/lifecycle; excludes model
loading, synthetic warmup, live detection, OCR, team classification and encoding.

Stage totals: pose 5.14s,
appearance 17.25s,
association/lifecycle 2.80s.
Appearance extraction accounts for about 67% of the loop time.

Historical SAM2 speed was 2.38 FPS on the full 249-frame clip with a different
loop/initialization span; it was not rerun for this experiment. Treat that speed
comparison as indicative rather than a controlled same-window timing comparison.

CAMELTrack loses 13.14 percentage points
of correct-identity score here and does not meet the user's quality requirement.
This first clip does not establish performance on the remaining clips.

Artifacts: `runs/cameltrack/felix141/manifest.json`, `replay.npz`,
`identity_score.json`, and sibling run/score logs. Source and checkpoint hashes
are recorded in the manifest. Worktree branch: `experiment/cameltrack`.
