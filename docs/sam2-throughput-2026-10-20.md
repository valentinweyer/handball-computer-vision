# SAM2 throughput: concurrency and reduced frame rate, 2026-10-20

Two "free" speedups for the production SAM2 tracker, measured on the three
scored clips with the §8 configuration (`scripts.run_sam2_reprompt_tracker`,
JPEG frames, `_v1` detections, cached team models, checkpoint every 10
frames). No production code or default changed. Code:
`experiments/sam2_throughput/`. Artifacts: `runs/sam2_throughput/` (dumps,
logs, scores, review renders).

## Short answer

- **Running two SAM2 processes side by side: 1.07x.** Not worth it. One
  process already keeps the GB10 busy. Outputs stay byte-identical.
- **Propagating every second frame: ~1.86x faster tracking loop, identity
  quality unchanged on all three clips.** FelixClaar 95.01 -> 96.63%,
  Han-Ber4 99.91 -> 99.57%, BHC-FAG 98.88 -> 98.86%. The one new error is the
  same scrum where full rate also errs, and neither run can be judged there.
- **Every third frame: ~2.5x, but Han-Ber4 gets its first identity errors**
  (99.91 -> 98.31%, 4 wrong frames, 2 mixed tracklets). Every second frame
  is the setting to take forward.

## Baseline

The new full-rate runs reproduce the stored
`runs/sam2_reprompt/*_policy_reprompt.npz` dumps byte for byte and score
95.01 / 99.91 / 98.88% (FelixClaar / Han-Ber4 player ids / BHC-FAG) -- the
production figures in `docs/tracking-evaluation.md` §8.10-8.12.

`run_sam2_reprompt_tracker.py` crashed after tracking when printing a
lifecycle event without an `obj_id`; fixed with `.get` (print only).

## 1. Concurrency

FelixClaar and Han-Ber4, one after the other: 99.6 + 90.4 = 190 s wall
(model load included). The same two started together: both finished in
177 s. **1.07x throughput.** Both concurrent outputs are byte-identical to
the solo runs. SAM2 at this configuration saturates the GPU; a second
process only queues behind the first.

Corollary: two pipeline runs at once each slow to about half speed. The
2026-10-19 `BAL_THW` and `TBV_SCM` 10-minute runs show a track loop at
1.51 it/s (0.66 s/frame) against the 0.32-0.33 s/frame documented for
Kiel-Lemgo; concurrent runs would explain that, but the logs do not show it.

## 2. Reduced frame rate

`half_rate.py prepare` builds a JPEG folder of every `stride`-th frame
(symlinks, renumbered) and the matching detection cache, so the unchanged
runner propagates at 25/stride fps. Checkpoints are kept at the same real-time
interval (`--check-every 10/stride`: 5 for stride 2, 3 -- i.e. every 9 frames
-- for stride 3). `half_rate.py expand` maps results back and fills skipped
frames by linear interpolation of each player's box between neighbouring
processed frames (only when the player is present on both). Scoring is the
unchanged §8 scorer on the full-rate frame sequence.

| clip | arm | correct % | recall % | wrong % | wrong frames | mixed | switches | frags/id | tracking loop |
|---|---|---|---|---|---|---|---|---|---|
| FelixClaar | full | 95.01 | 95.76 | 0.78 | 18 | 5 | 8 | 1.38 | 80 s |
| | stride 2 | 96.63 | 96.96 | 0.34 | 8 | 3 | 6 | 1.31 | 43 s |
| | stride 3 | 95.92 | 96.30 | 0.39 | 9 | 4 | 7 | 1.31 | 31 s |
| Han-Ber4 | full | 99.91 | 99.91 | 0.00 | 0 | 0 | 0 | 1.00 | 71 s |
| | stride 2 | 99.57 | 99.57 | 0.00 | 0 | 0 | 0 | 1.00 | 38 s |
| | stride 3 | 98.31 | 98.48 | 0.18 | 4 | 2 | 4 | 1.25 | 28 s |
| BHC-FAG | full | 98.88 | 98.99 | 0.12 | 6 | 3 | 6 | 1.42 | 171 s |
| | stride 2 | 98.86 | 98.99 | 0.14 | 7 | 3 | 6 | 1.42 | 92 s |
| | stride 3 | 98.88 | 98.92 | 0.04 | 2 | 1 | 4 | 1.17 | 68 s |

Tracking loop = the runner's propagation loop, model load excluded:
stride 2 is 1.86 / 1.87 / 1.86x faster, stride 3 2.58 / 2.54 / 2.51x. Per
processed frame it is slightly slower (checkpoints come every 5 processed
frames instead of 10).

### Every difference, stride 2 vs full (`diff.py`)

- **FelixClaar.** Full rate has wrong runs 8->1 (f134-148, 8 frames) and
  7->11 (f180-188, 5); stride 2 has neither, but has 8->11 (f140-145, 4) in
  the same scrum. Review render `review/felix_132_148.jpg`: in both arms track
  8 drifts across the f137-148 scrum, onto reference 1 at full rate and onto
  reference 11 at stride 2. Reference 8 itself is only verified to f90, so
  neither arm can be called right there. Same failure, same place -- not a
  new failure mode. Stride 2 also keeps reference 1 through f154-180, which
  full rate loses (the recall gain).
- **Han-Ber4.** No wrong runs in either arm. Stride 2 loses 9 isolated
  single-frame matches (f86, f101, f175-194): interpolated boxes on skipped
  frames during fast motion that fall just under IoU 0.5.
- **BHC-FAG.** Wrong runs nearly identical (5->6 f449-452 vs f449-453, plus
  one single frame). Single-frame match differences both ways, and the clip's
  last frame (f499, odd) is never processed at stride 2.

## What it would take to adopt stride 2

The scored output is boxes. The pipeline also uses per-frame masks and reads
jerseys every `OCR_EVERY_N_FRAMES = 5` frames, so:

- Jersey reads and anything else on a fixed cadence must land on processed
  frames: every 4 or 10 frames instead of 5 (`detection_stride_guard` style
  check), with the read-count change measured.
- Skipped frames get interpolated boxes but no SAM2 mask. Renders can reuse
  the nearest processed mask; anything that consumes masks for evidence
  (team, overlap guards) should only sample processed frames.
- The detector already runs every 5th frame in `process_video`, and SAM2
  checkpoints every 10, both even: compatible with stride 2.
- Re-check on the unseen clips (Hannover, VfL) by eye before switching, as for
  CUTIE: three clips are enough to rule out an obvious regression, not to
  promise none.

## Effect on a full match

At the documented 0.32-0.33 s/frame, a 60-minute match at 25 fps (90,000
frames) is ~8 h of tracking plus ~1-2 h for scan, teams, bundle and render.
Stride 2 would bring tracking to ~4.4 h, so roughly 10 h -> 6 h end to end.
Running a second match concurrently does not help.

## Reproduce

```bash
S=experiments/sam2_throughput
python -m $S.half_rate prepare felix han bhc --stride 2        # and --stride 3
# run scripts.run_sam2_reprompt_tracker per clip: full rate with the original
# frame folder and detections (--check-every 10); stride N with
# runs/sam2_throughput/strideN/frames/<clip> and <clip>_detections.npz
# (--check-every 10/N), output runs/sam2_throughput/strideN/<clip>_sub.npz
python -m $S.half_rate expand felix han bhc --stride 2
python -m $S.score full stride2 stride3
python -m $S.diff full stride2
python -m $S.render_compare felix 132 137 141 144 148 --refs 1 8 11 --out felix.jpg
```
