# Overnight 2026-10-17: the full pipeline, from any video to a result bundle

Branch `feat/video-frame-source`, eight commits on top of `f44956b`, nothing on
`main`. All six planned steps are done and verified, plus one fix that the first
10-minute run exposed. Each section below gives
the check it passed and the numbers behind it.

## What you can run now

```bash
python -m scripts.process_video data/raw/<video>.mp4   # -> runs/bundles/<video>
python -m scripts.render_result runs/bundles/<video>    # -> runs/bundles/<video>/render.mp4
```

Nothing has to exist beforehand, and it runs in this venv only. The one
exception is the detector's TorchScript file, which is exported once in the
finetune venv (`scripts/export_detector.py`); that is already done
(`rfdetr-handball-finetune/exports/detector-local/`). Stages resume when
interrupted, and `--force STAGE` reruns a stage and everything after it.

```python
from handball_cv.pipeline.result import load
r = load("runs/bundles/BHC-FAG")
r.tracks          # one row per player per frame, as a pandas DataFrame
r.players, r.court, r.reads, r.events, r.detections, r.keypoints
r.frame(120)      # RGB frame;  r.mask(row.mask_row) -> that player's mask
```

`render_result` options: `--start/--end` (seconds) for highlight clips,
`--live-numbers` to label with what the run knew at each frame instead of the
backfilled verdict, `--minimap-corner` to move the minimap away from the
broadcast's score graphic, `--no-minimap`.

Result bundles to look at:

- `runs/bundles/Melsungen_window/render.mp4`: built from the existing 60 s run.
- `runs/bundles/BHC-FAG/render.mp4`: the full 91 s BHC-FAG clip at 50 fps,
  processed from the bare mp4.
- `runs/bundles/Kiel_Lemgo_66m00_10min/render.mp4`: a 10-minute window of
  Kiel–Lemgo (broadcast minutes 66 to 76, game clock about 39:30 to 46:30),
  cut tonight and processed from the bare mp4.

## The bundle

One directory per video:

- `manifest.json`: the source, models, parameters, provenance and timings. It
  also gives each table's own schema version, so a later stage (ball, actions)
  adds a table without invalidating existing bundles.
- One Parquet file per table:

| Table | One row per | Main columns |
|---|---|---|
| `tracks` | player × frame | `frame`, `time_s`, `player_id`, `track_id`, `team`, `team_state`, `team_confidence`, `is_goalkeeper`, `number` (backfilled), `number_live`, box, `foot_x/y`, `court_x/y` (cm), `court_fit_source`, `mask_row` |
| `players` | identity | team, final number, first/last frame, frames visible, re-ID count, reads, vote tally |
| `court` | frame | court→image homography and its inverse, source, age, inliers |
| `detections` | detection | every class, confidence, box (detector stride is in the manifest) |
| `reads` | jersey read | value, reader confidence, number box |
| `events` | identity event | re-ID, link, team switch, suspected ID switch |
| `keypoints` | frame × court slot | all 37, unfiltered |

- Alongside the tables: `masks.npz`, `team_model.pkl`, and `video.mp4` (the
  video the rows refer to).

Court coordinates are in centimetres: x runs along the 4000 cm length, y across
the 2000 cm width, with the origin at a court corner.

## What each step built, and how it was checked

**0. Frames straight from the video** (`d62b763`). SAM2 can decode the video
instead of reading a JPEG folder. Re-scored on the three references
(tracking-evaluation §8.12): 96.21 / 99.91 / 98.96% against JPEG's
95.01 / 99.91 / 98.88%. Wall time is the same.

**1. Bundle format** (`abe8324`). Built from the existing Melsungen run.
- All 18,867 player-frames carry exactly the label `--redraw` draws, in both
  the backfilled and the live column.
- 94.3% of foot points land inside the court.

**2. Render from the bundle** (`05a7b86`). Overlay plus court minimap, encoded
once to x264 (CRF 18). It plays in a browser with no transcode step, at about
24 frames per second.

**3. Detector in this venv** (`3147af5`). TorchScript export of the RF-DETR
Large fine-tune. Against `rfdetr predict()` on 100 frames each from two
matches:
- 2439/2441 and 1963/1963 boxes matched, every one the same class.
- Median confidence difference 0.0003; 98.6–98.8% of boxes at IoU ≥ 0.99.
- 12–24 ms/frame.

It is close, not bit-identical. On an identical input tensor, trace and eager
agree to 2.4e-6 on logits, and my preprocessing matches rfdetr's exactly. The
remaining 1–2% of slightly moved boxes come from RF-DETR's two-stage query
selection: a discrete top-k inside the network, which a 1e-6 difference can
occasionally flip.

**4. `process_video`** (`3c7055f`). The five stages:
- **prepare**: resample to 25 fps when the source is faster than 30 fps.
- **scan**: one decode feeds the detector (every 5th frame), the court model
  and camera motion.
- **teams**: crops spread over the whole video. The old fit stopped after its
  first ~80 s, which on a full match is the warm-up.
- **track**: SAM2, reading the video directly, writing no frames.
- **bundle**: joins everything into the tables.

The detector runs only on every 5th frame because tracking checkpoints and
number reads look at no other frame. A guard fails if their cadences ever stop
being multiples of that stride.

**5. SAM2 memory stays flat** (`d4b79f2`). Output is byte-identical: FelixClaar,
Han-Ber4 and BHC-FAG reproduce their stored baselines exactly, and Melsungen
gives identical output with pruning on and off. GPU memory on Melsungen:

| | Without pruning | With pruning |
|---|---|---|
| Memory over the clip | 3.0 → 9.4 GiB (~5 MB/frame) | flat at 2.3–2.4 GiB |
| Peak | 10.4 GiB | 2.7 GiB |
| Time | 529 s | 526 s |

Clip length is no longer capped by memory (TODO §1 resolved).

**6. End to end on unprocessed footage:**

| Video | Frames at 25 fps | prepare | scan | teams | track | bundle | render | identities | labelled player-frames | on court |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BHC-FAG (91 s, 50 fps) | 2270 | 10 s | 74 s | 24 s | 728 s | 2 s | 95 s | 24 | 72.4% | 96.9% |
| Kiel–Lemgo 10 min (50 fps) | 15000 | 66 s | 477 s | 40 s | 4898 s | 24 s | 653 s | 138 | 62.1% | 89.7% |

Tracking runs at 0.32–0.33 s per frame on both videos. The 10-minute run held
GPU memory steady:

- 4.1 GiB at the start.
- Rising to 6.2 GiB by about frame 7,000. This is the tracking process's total,
  including PRTReID, PARSeq, SigLIP and the allocator's cache.
- Then exactly 6,243 MiB from frame ~7,300 to the end.

Without pruning, SAM2's stored outputs alone would have grown by ~5 MB per
frame, about 75 GB over this clip (`runs/pipeline_e2e/gpu_memory.log`).

**138 identities in 10 minutes is fragmentation**, the failure the handoff
already calls dominant. It is just more visible over a long clip: the 39
identities visible for at least 1000 frames (17 of them for over 3000) carry
80% of all player-frames, and
the run logged 136 re-IDs and 79 number links. Nothing tonight changes
tracking or re-ID; the re-ID spatial gate from the review targets exactly this.

**Tracking quality with the new pipeline, scored.** The three reference clips
were run through `process_video` from their bare mp4s, then scored with
`evaluate_tracker_identity` against the section 8 references
(`runs/pipeline_e2e/reference_scores.json`). The new pipeline is the local
detector, video frames, the new team fit, PRTReID re-ID and pruning:

| Clip | Scored setup (hosted detector, JPEG) | New pipeline | Mixed / switches / frags |
|---|---:|---:|---|
| FelixClaar | 95.01% | **95.38%** | 5 / 8 / 1.38, unchanged |
| Han-Ber4 (players) | 99.91% | **99.91%** | 0 / 0 / 1.00, unchanged |
| BHC-FAG window | 98.88% | **99.07%** | 3 / 6 / 1.42 → 1.50 |

No clip regresses. The small gains are within what an input change moves, as in
section 8.12.

Spot check on the BHC-FAG render at 30 s: where the shirts are legible the
labels match them (#25, #22, #2), the goalkeeper (#12) is grey, and the minimap
places every player around the left goal area as on screen. The numbers this
clip's window resolved earlier (22, 53, 49) resolve again on the full clip.

Spot check on Kiel–Lemgo at 6:10:
- **Right:** #24, #23 and the goalkeeper #1 match their shirts, and the
  minimap matches the frame. #28 is on the minimap; its frame label falls just
  below the frame edge.
- **Wrong:** a floor wiper with a mop beside the goal is drawn as an orange
  #22. An identity moved onto a person off the court. That is the weakness the
  review named, re-ID with no spatial term and a court test that is still a
  no-op, and nothing tonight changes it.

## A fix the 10-minute run forced (`0df594d`)

The first 10-minute bundle labelled only **11.6%** of player-frames with a
number, against 68–74% on the short clips. Within identities that did resolve
a number, 81% of their frames came from fragments linked into them by number,
and none of those frames was labelled. The cause is the segment backfill
(shared with `--redraw`): it keys frames by tracker id, so a linked fragment's
frames sit under an id with no reads after the link. Short clips have no links,
which is why it never showed.

`tracks.number` now folds number links first: a linked fragment counts as the
identity it was linked into. Two per-frame rules then withhold rather than
guess:

- **An identity is on at most one box.** A link is checked when it is made but
  never undone, so two linked ids later on screen together would have put one
  identity in two places (690 frames on Kiel–Lemgo).
- **A number is on at most one field player per team.** The backfill can reach
  over frames where the run's arbitration had withheld a duplicate. The old
  `--redraw` labels have 49 such frames on this clip.

Results:

| Bundle | Labelled before | Labelled after |
|---|---:|---:|
| Kiel–Lemgo | 11.6% | 62.1% |
| BHC-FAG | 68.0% | 72.4% |
| Melsungen | 73.9% | 73.9% (no links; still identical to `--redraw` on every player-frame) |

No duplicates of either kind remain in any bundle. `--redraw` itself is
unchanged and still has the gap on runs with links.

## Decisions I made without you

- **Bundles go under `runs/bundles/`, not `results/`.** `results/` was not
  gitignored, and `docs/architecture.md` puts generated results under `runs/`.
- **The orchestrator is `scripts/process_video.py`, not
  `handball_cv.pipeline.process`.** The tracking and number logic
  (`NumberIdentityResolver`, matching, reader setup) lives in
  `scripts/render_full_pipeline.py`, and moving it into the package is a
  refactor of its own. The track stage runs that script with
  `--frame-source video --no-video` in its own process, so each stage's GPU
  models are freed.
- **Fixed the reader's BF16 leak** (TODO §7). The SAM2 render runs PARSeq
  outside the autocast now. New reads therefore differ from earlier SAM2
  renders on the few crops measured to change (6 of 621); cached reads replay
  unchanged.
- **The `OfflineRFDETR` ONNX decode was left alone.** Only an experiment uses
  it, and sharing a decode with it would have changed that experiment.
- **The detector class map is duplicated** in `scripts/cache_number_detections.py`.
  That script runs in the finetune venv, where `handball_cv` is not importable.

## Things to know

- **Scored through the new pipeline: 95.38 / 99.91 / 99.07%** (table above).
  The reference clips are short, so this does not cover how identity holds up
  over 10 minutes, where fragmentation dominates.
- **Number boxes are still taken at the cache threshold (0.3)** on the SAM2
  path, as before. TODO §5 (raise to 0.5) is still open and is now cheap to
  test: `detections.parquet` holds every confidence.
- **Court positions are raw.** They jitter a few px per frame and sometimes
  spike toward the far sideline when a player's feet are hidden. No smoothing
  was applied; that is a decision for whatever uses the positions.
- **Frame 0 has no `tracks` rows.** SAM2 uses it as the seed frame and does not
  output it, which is existing behaviour.
- **Old runs have two empty columns.** `team_confidence` and the reader's
  confidence are filled only for runs made after `3c7055f`; the Melsungen
  bundle, converted from an older run, has them empty.
- **`render_court_view` now calls the shared court functions.** I checked it
  imports and that the fitting code is identical, but did not re-render a video
  with it.

## Where the evidence is

- `runs/frame_source_rescore/`: JPEG vs video re-score.
- `runs/detector_export_parity/`: the parity scripts.
- `runs/sam2_memory_prune/`: byte checks and the memory curve.
- `runs/pipeline_e2e/`: end-to-end logs and GPU memory samples.
- 488 unit tests pass (`uv run --no-sync pytest -q tests`).
