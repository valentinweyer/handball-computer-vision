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
- **Built into the pipeline as `--sam2-stride 2` (opt-in, default unchanged):**
  the track stage of a 91 s clip runs 1.52x faster with jersey reads kept on
  real masks; numbers resolved, reads and switch counts unchanged. On the
  unseen review clips it made one identity error full rate did not (VfL
  f1480: a goalkeeper re-identified as an earlier floor-wiper track) and
  one extra fragment (Hannover).

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

## Built into the pipeline (opt-in)

`drive_sam2(..., sam2_stride=N, real_mask_every=(...))` runs SAM2 on
`track_frame_schedule(num_frames, N, check_every, *real_mask_every)`: every
Nth frame, every checkpoint frame, every frame on a `real_mask_every`
cadence, and the last frame. SAM2 sees only those (`LazyFrames(frames=...)`);
TrackManager, detections, court and the consumer keep video frame numbers.
Skipped frames are yielded with `Sam2FrameResult.interpolated=True`:
interpolated boxes, earlier mask shifted by the box-centre displacement,
players with an empty mask listed empty as on processed frames. They are
display geometry, never evidence.

- `render_full_pipeline.py --sam2-stride N` passes `real_mask_every=(ocr_every,)`,
  so every jersey-read frame keeps a real SAM2 mask (reads unchanged), and it
  skips reads on interpolated frames as a guard. Run summary records
  `sam2_stride`.
- `process_video.py --sam2-stride N` (default 1) forwards it and refuses to
  reuse a track pass made with a different stride.
- With reads every 5 frames, stride 2 processes 60% of frames (even frames
  plus 5, 15, 25, ...), so expect ~1.6x on the track loop rather than 1.86x.
- `run_sam2_reprompt_tracker.py` and `experiments/cutie/render_review.py`
  take `--sam2-stride` for scoring and review.

**Default path unchanged:** FelixClaar at stride 1 through the modified
driver is byte-identical to the stored baseline. **Built-in stride 2 on the
references** (runner path, no read cadence): 96.67 / 99.57 / 98.97% against
95.01 / 99.91 / 98.88% at full rate, same loop times as the prototype.

## Unseen clips (Hannover, VfL 60 s)

Rendered with the CUTIE review harness, same inputs as the 2026-10-16
review, full rate and stride 2 with current code
(`runs/sam2_throughput/review/`: `<clip>_s1`, `<clip>_s2`, and
`<clip>_left-full_right-stride2.mp4` side by side). Wall time including
drawing and encoding: Hannover 408 -> 244 s (1.68x), VfL 375 -> 220 s (1.70x).

No reference exists, so `agreement.py` matches each run's boxes to the other's
and lists where identities disagree; every disagreement was checked in the
frames:

- **Hannover f302-330**: both runs open a duplicate track (15) on the same
  player as track 7; at full rate the duplicate is removed and 7 continues,
  at stride 2 track 7 is removed and 15 continues. Same player throughout --
  one extra fragment at stride 2, not a swap.
- **VfL f0-36**: two tracks (8, 10) with identical boxes on one player in
  both runs; an artefact of the comparison, not a difference.
- **VfL f1480 -- a real stride-2 error.** At stride 2 the re-ID layer reuses
  identity 30 for a goalkeeper entering at the left edge. Track 30 was last
  seen at f1190 and was not a player but a floor wiper
  (`review/vfl_reid_1480.jpg`). At full rate the same wiper is tracked too
  (track 29), but the goalkeeper correctly gets a new track. The weakness is
  the re-ID layer (it tracks and later reuses a non-player identity); stride 2
  changed the masks enough to tip the decision on this clip.

Overall 97.4-99.8% of matched boxes agree in each direction, and every
disagreement of three or more frames is in one of these spans.

## End to end through `process_video` (BHC-FAG, 91 s)

Same prepare/scan/teams outputs (hard-linked), track stage at stride 1 and 2
(`runs/sam2_throughput/pipeline/BHC-FAG_s{1,2}`, bundles alongside):

| | stride 1 | stride 2 |
|---|---|---|
| track stage | 716 s | 470 s (**1.52x**) |
| jersey-read frames / reads | 434 / 793 | 434 / 800 |
| resolved numbers | 15 | the same 15 |
| team switches / suspected ID switches | 6 / 2 | 6 / 2 |
| new allocations / distinct players after linking | 39 / 29 | 36 / 26 |
| lifecycle events | 182 | 67 |

The track stage gains 1.52x rather than the loop's 1.86x: 60% of frames are
still processed (reads every 5 frames), and reads, re-ID and team work do not
shrink.

The lifecycle gap is one track. At stride 1 the #22 orange goalkeeper
(track 13) is masked over shirt and shorts only, so his mask box overlaps the
detector's full-body box at IoU 0.2-0.45 -- the drift band -- and he is
reprompted at **every checkpoint, 103 times**, without the mask ever
recovering (`pipeline/track13_s1.jpg`). At stride 2 the same goalkeeper
(track 10) is masked whole and never needs a reprompt. Not a stride
regression; a separate production finding: a reprompt that does not fix the
mask is repeated forever.

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
