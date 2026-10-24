# Handball Computer Vision

Research code that turns a handball broadcast into per-frame player data:
detection, per-video team classification, tracking, identity and jersey
numbers, and court coordinates.

## Setup

The pipeline runs in the project's `.venv`. `torch` and `onnxruntime` are
CUDA-13 builds for aarch64/GB10, recorded in `requirements.txt` rather than in
`pyproject.toml`. That file records a working machine, not a from-scratch
install recipe. Install the package itself in editable mode, with the pipeline
extras:

```bash
pip install -e '.[pipeline,jersey-parseq,dev]'
```

`environment.yml` is a Conda equivalent. Roboflow access, needed only by some
older scripts, is read from `ROBOFLOW_API_KEY`; copy `.env.example` to `.env`
and fill it in locally.

The default configuration is the most accurate one measured, and it needs three
more things before `process_video` will run:

- **External research checkouts**, cloned beside the project and gitignored:
  `sam2-upstream/` (`--tracker sam2`), `prtreid-upstream/` (`--reid-embedding
  prtreid`) and `parseq-upstream/` (`--reader parseq`). Their locations can be
  overridden with `SAM2_UPSTREAM_DIR`, `--prtreid-root` and
  `PARSEQ_UPSTREAM_DIR`.
- **A downloaded PARSeq checkpoint** for the default jersey reader.
- **The detector's TorchScript export**, made once in the sibling finetune venv
  (`scripts/export_detector.py`).

Every missing piece fails loudly and names its fix. The fully self-contained
fallback is `--tracker mcbyte --reader easyocr --reid-embedding team-model`, at
the accuracies listed under *Current status*.

## Quick start

```bash
python -m scripts.process_video data/raw/<video>.mp4   # -> runs/bundles/<video>
python -m scripts.render_result runs/bundles/<video>    # -> runs/bundles/<video>/render.mp4
```

With the setup above in place, `process_video` needs nothing pre-computed. It
has five stages: `prepare` (resample to 25 fps), `scan` (detector, court
keypoints, camera motion), `teams`, `track` and `bundle`. An interrupted run
resumes, and `--force STAGE` reruns that stage and every one after it.

The result bundle is one Parquet table per kind of row (`tracks`, `players`,
`court`, `detections`, `reads`, `events`, `keypoints`), plus `masks.npz`, the
team model and the video the rows refer to:

```python
from handball_cv.pipeline.result import load
r = load("runs/bundles/<video>")
r.tracks          # one row per player per frame: team, number, box, court_x/y (cm), ...
r.frame(120)      # RGB frame;  r.mask(row.mask_row) -> that player's mask
```

`render_result` draws the overlay and a court minimap. It takes `--start/--end`
(seconds) for highlight clips, `--live-numbers`, `--minimap-corner` and
`--no-minimap`. The format is documented in `handball_cv/pipeline/result.py` and
`docs/pipeline-overnight-2026-10-17.md`.

## Current status

**Detection.** A fine-tuned RF-DETR produces goalkeeper, player, referee and
jersey-number boxes, and runs locally from a TorchScript export (12–24 ms/frame,
matching `rfdetr predict()` on 99.9% of boxes). Its output is cached per video
with every class kept. Each consumer filters for the classes it wants, because
a cache that drops classes forces a re-detection later.

**Team classification** finds two anonymous field-player teams separately in
each video, from clean torso colour with a guarded visual fallback. It stays
frame-local: a tracker never supplies or freezes a team prediction. Mask-derived
colour is only a fallback for overlapping players and cannot override a valid
clean-crop observation.

**Tracking.** The default is SAM2 with periodic detector reprompting. Scored
against ground truth on three clips, it wins on each: 95.0 / 89.2 / 98.9%
correct, against 81.5 / 86.7 / 95.9% for the best box tracker on each clip
(`docs/tracking-evaluation.md` §8). The Han-Ber4 pair is scored against every
reference identity, including bench and referee ids. That ranks the trackers
fairly but understates both; on players alone SAM2 scores 99.91% (§8.11).
Through `process_video` the three clips score 95.38 / 99.91 / 99.07%, and SAM2
memory stays flat over a full match. SAM2 tracks at about 0.33 s/frame against
MCByte's 0.1 s, so `--tracker mcbyte` stays supported for when speed matters.
`--sam2-stride` (reduced frame rate) and ARC team repair are opt-in.

**Jersey numbers.** Number boxes come from the detector and are matched to
players by mask intersection-over-smaller. A scene-text recogniser reads them,
and the reads are voted per identity. Reader accuracy on 323 human-labelled
1080p crops:

| reader | accuracy | selective | abstention |
| --- | ---: | ---: | ---: |
| `--reader parseq` (baudm original, **default**) | **0.858** | 0.958 | 0.89 |
| `--reader qwen` (Qwen3.8, no-think) | 0.628 | 0.736 | 0.81 |
| `--reader doctr --doctr-arch parseq` | 0.622 | 0.817 | 0.88 |
| `--reader easyocr` | 0.365 | 0.641 | 0.84 |

All four were scored on the same crops with the same rule, at confidence 0.5.
*Selective* is accuracy over the crops a reader chose to answer; *abstention* is
how often it correctly stays silent on a crop a human called unreadable. The two
`parseq` rows are the same architecture with different weights: docTR trains its
own on document text. The gap between them is the largest single measured
improvement in the pipeline (paired McNemar p=2e-19).

**Identity** does not depend on which tracker is used. `PlayerRegistry` holds
reversible team labels, so an early mistake is not frozen, and re-ID matches a
returning player by appearance. That appearance signal is weak within a team:
0.55 rank-1 against a 0.19 chance floor with `--reid-embedding prtreid` (the
default), and 0.35 with the team model's own features. Jersey numbers
therefore arbitrate afterwards:

- A number verdict is suspended when re-ID moves an identity.
- It is withheld when two identities on one team claim the same number.
- It is used to merge identities that are provably the same player.

A number is backfilled to the start of the segment that earned it. In the
bundle, an identity is on at most one box, and a number on at most one field
player per team. Fragmentation remains the dominant failure: a 10-minute
window produces 138 identities, though 39 of them carry 80% of player-frames.

**Court.** A court-keypoint model (our own RF-DETR fine-tune) fits a homography
for each frame. The fit is carried between frames, and history resolves the
mirror ambiguity. Foot points become court coordinates in centimetres, which
feed the minimap. Where no court is in view, for example a timeout huddle, the
fit abstains and `process_video` pauses tracking until play returns. The
keypoint model fails in many lower-league (non-HBF) halls, where bundles come
out nearly empty; a fine-tune is planned (`TODO.md`).

**Ball.** Tracking the ball is a separate experiment, not a `process_video` stage
(`scripts/run_ball_tracker.py`, `scripts/render_ball_demo.py`,
`docs/ball-tracking.md`).

**Not defaults:** team-gated association, and the court test, which is still a
no-op (`docs/architecture.md`, `TODO.md`).

Open work, each item with the measurement behind it, is in `TODO.md`.

## Repository layout

```text
src/handball_cv/    reusable project code (detection, teams, tracking, jersey,
                    court, ball, pipeline, visualization)
scripts/            processing, labeling, evaluation, and rendering commands
experiments/        rejected or non-default approaches, kept with their results
tests/unit/         package-level tests
configs/            model registry and reference pipeline configuration
data/annotations/   tracked human labels
data/raw/           ignored source media
data/cache/         ignored caches (SAM2 JPEG frames)
data/interim/       ignored resumable stage outputs of process_video
models/             ignored local checkpoints
runs/               ignored generated results; runs/bundles/ holds result bundles
notebooks/          the .ipynb files, fonts, and their local assets
docs/               architecture, handoff, and per-experiment findings
TODO.md             open and finished work, each entry with its evidence
```

The ignored directories hold about 20 GB of artifacts, most of which took GPU
hours to build. Do not `git clean` the tree. See `CLAUDE.md`.

`outputs/`, model duplicates and nested upstream repositories remain from the
first migration pass; see `docs/legacy-layout.md`.

## Tests

```bash
uv run --no-sync pytest -q tests
```

`--no-sync` is required. There is no `uv.lock`, and a plain `uv run` would
resolve from `pyproject.toml` alone and can uninstall the CUDA `torch`.
`testpaths` is scoped to `tests/unit/`. Do not run bare `pytest` from the
repository root: vendored ONNX Runtime and model trees carry unrelated test
entry points that break collection.

## Diagnostics and measurement

`render_full_pipeline` is the older single-command run. It writes an overlay
video and a run summary from cached detections. It caches reader output
(`<name>_reads.json`) and tracking geometry, so `--redraw` re-renders without a
tracking pass:

```bash
python -m scripts.render_full_pipeline data/raw/<clip>.mp4 \
    --detections <cache>.npz --number-detections <cache>.npz \
    --team-model <model>.pkl \
    --output runs/full_pipeline/<name>.mp4
```

The claims in *Current status* are reproducible:

```bash
python -m scripts.evaluate_tracker_identity      --help   # tracking accuracy against references
python -m scripts.benchmark_jersey_parseq        --help   # reader accuracy table
python -m scripts.benchmark_doctr_readers        --help
python -m scripts.measure_reid_discriminability  --help   # can re-ID separate teammates?
python -m scripts.compare_court_view             --help   # courtless share per video
python -m scripts.sweep_number_crop_padding      --help
```

Team-classification diagnostics:

```bash
python -m scripts.render_raw_team_classification --help
python -m scripts.render_mcbyte_team_correction  --help
python -m scripts.render_mask_team_comparison    --help
python -m scripts.evaluate_team_embeddings       --help
```

`docs/architecture.md` documents the dependency rules and artifact boundaries.
`docs/team-classification-handoff.md` holds the detailed findings, rejected
approaches and implementation status.

<details>
<summary>Historical pre-migration README</summary>

# Handball AI: Player Detection, Tracking, and Identification

**Detect, track, and identify handball players in videos using computer vision.**

This project is **adapted from [Roboflow's Basketball Player Detection Notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb)** and repurposed for handball. It demonstrates a complete pipeline for detecting, tracking, and identifying handball players in videos using **RF-DETR** for object detection, **SAM2** for real-time player tracking, **SigLIP** for team classification, and **SmolVLM2** for jersey number recognition. The pipeline also maps player positions to court coordinates for advanced analytics.

It is in active development. Current status is shown below.

I have already fine-tuned RF-DETR on a handball-player detection dataset. The detection of players already works quite well.
However I still need to adapt some features, which will include fine-tuning other models to handball scenarios.

---

## Current Features

- **Player Detection**: Detect players, referees, and the ball using a fine-tuned RF-DETR model.
- **Player Tracking**: Track players across frames with stable IDs and masks using SAM2.

## Future Features

- **Team Classification**: Automatically cluster players into teams using SigLIP embeddings and K-means.
- **Jersey Number Recognition**: Recognize and validate player numbers using SmolVLM2 OCR.
- **Court Mapping**: Map player positions to real-world court coordinates.
- **Visualization**: Overlay player names, numbers, team colors, and movement paths on video.

## Project Status

- **RF-DETR** has been fine-tuned on a handball-player detection dataset.
- SAM2 player tracking works using the prompt from the RF-DETR detection.
- Additional features are being adapted for handball scenarios.


---

## References and Acknowledgements

- **[Roboflow Basketball Player Detection Notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb)**: This project is based on Roboflow's basketball pipeline, adapted for handball.
- [Roboflow Universe](https://universe.roboflow.com/) for pre-trained models, fine-tuning models and tools useful for dataset creation.
- [SAM2 Real-Time](https://github.com/Gy920/segment-anything-2-real-time) for real-time segmentation.
- [Roboflow Sports](https://github.com/roboflow/sports) for team classification and court mapping tools.

---

## 🤝 Contributing

Found a bug or have an idea for improvement? Open an issue or submit a pull request!

</details>
