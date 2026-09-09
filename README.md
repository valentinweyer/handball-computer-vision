# Handball Computer Vision

Research code for handball player detection, per-video team classification,
tracking, identity correction, and court mapping.

## Current status

**Detection.** A fine-tuned RF-DETR is the baseline, producing goalkeeper,
player, referee and jersey-number boxes. Its output is cached per video; every
consumer filters the classes it wants, because a cache that drops classes forces
a re-detection later.

**Team classification** discovers two anonymous field-player teams separately for
each video from clean torso colour, with a guarded visual fallback. It stays
frame-local: a tracker never supplies or freezes a team prediction, and
mask-derived colour is an overlap-only fallback that cannot override a valid
clean-crop observation.

**Tracking.** SAM2 with periodic detector reprompting is the default. Scored
against ground truth on three clips it wins each one -- 93.8 / 89.2 / 98.9%
correct, against 81.5 / 86.7 / 95.9% for the best box tracker on each
(`docs/tracking-evaluation.md` §8). It costs about 1 s/frame against MCByte's
0.1 s, so `--tracker mcbyte` stays supported and is the right pick when that
matters.

**Jersey numbers.** Number boxes come from the detector, are matched to players
by mask intersection-over-smaller, read by a scene-text recogniser, and voted per
identity. Reader accuracy on 323 human-labelled 1080p crops:

| reader | accuracy | selective | abstention |
| --- | ---: | ---: | ---: |
| `--reader parseq` (baudm original, **default**) | **0.858** | 0.958 | 0.89 |
| `--reader qwen` (Qwen3.8, no-think) | 0.628 | 0.736 | 0.81 |
| `--reader doctr --doctr-arch parseq` | 0.622 | 0.817 | 0.88 |
| `--reader easyocr` | 0.365 | 0.641 | 0.84 |

All four scored on the same crops with the same rule, at confidence 0.5.
*Selective* is accuracy over the crops a reader chose to answer; *abstention* is
how often it correctly stays silent on a crop a human called unreadable. The two
`parseq` rows are the same architecture with different weights -- docTR trains its
own on document text, and the gap between them is the largest single measured
improvement in the pipeline (paired McNemar p=2e-19).

**Identity** is tracker-agnostic. `PlayerRegistry` holds reversible team labels
so an early mistake is not frozen, and re-ID matches a returning player by
appearance. That appearance signal is weak within a team -- 0.55 rank-1 against
a 0.19 chance floor with `--reid-embedding prtreid` (the default), 0.35 with the
team model's own features -- so jersey numbers arbitrate afterwards: a verdict is suspended
when re-ID moves an identity, withheld when two identities on one team claim the
same number, and used to fold identities that are provably the same player.

**Not defaults:** team-gated association, and the court test, which is still a
no-op (`docs/architecture.md`, `TODO.md`).

Open work, each with the measurement that motivates it, is in `TODO.md`.

## Repository layout

```text
src/handball_cv/    reusable project code
scripts/            labeling, evaluation, and rendering commands
experiments/        SAM2 and team-gated association baselines
tests/unit/         package-level tests
configs/            model registry and reference pipeline configuration
data/annotations/   tracked human labels
data/raw/           ignored source media
data/cache/         ignored reproducible caches
models/             ignored local checkpoints
runs/               ignored generated results
notebooks/          the .ipynb files, fonts, and their local assets
docs/               architecture, handoff, and per-experiment findings
TODO.md             deferred work, each entry with its evidence
```

The default configuration is the most accurate one measured, which means it
depends on all three external checkouts below plus a downloaded PARSeq
checkpoint. Every failure is loud and names its fix, and
`--tracker mcbyte --reader easyocr --reid-embedding team-model` is the fully
self-contained fallback, at the accuracies shown above.

External research checkouts are cloned beside the project and gitignored:
`sam2-upstream/` (`--tracker sam2`), `prtreid-upstream/` (`--reid-embedding
prtreid`) and `parseq-upstream/` (`--reader parseq`). Each is overridable by
`SAM2_UPSTREAM_DIR`, `--prtreid-root` and `PARSEQ_UPSTREAM_DIR`.

`outputs/`, model duplicates and nested upstream repositories are still present
from the first migration pass; see `docs/legacy-layout.md`. Source clips have
since moved to `data/raw/`, and the 29 legacy `notebooks/*.py` module copies were
deleted on 2026-09-09 once they had drifted from the modules that replaced them
(`TODO.md`, Done section). `git log` is the fallback for those.

## Setup

Use the existing CUDA-capable environment or create the documented Conda
environment, then install the package itself in editable mode:

```bash
conda env create -f environment.yml
conda activate handball-cv
pip install -e .
```

The current machine snapshot remains in `requirements.txt`. Roboflow access is
read from `ROBOFLOW_API_KEY`; copy `.env.example` to `.env` and fill it locally.

## Tests

```bash
conda run -n NewEnv pytest -q tests
```

`testpaths` is scoped to `tests/unit/`, so vendored ONNX Runtime and model
repositories are never collected. Do not run bare `pytest` from the repository
root -- those trees carry unrelated test entry points that break collection.

## Team-classification diagnostics

Run tools as modules after the editable install:

```bash
python -m scripts.render_raw_team_classification --help
python -m scripts.render_mcbyte_team_correction --help
python -m scripts.render_mask_team_comparison --help
python -m scripts.evaluate_team_embeddings --help
```

## End-to-end and measurement

`render_full_pipeline` is the one command that runs detection, tracking, team
classification and number reading together and writes both an overlay video and
a run summary (per-frame identities, votes, reads, teams and events):

```bash
python -m scripts.render_full_pipeline data/raw/<clip>.mp4 \
    --detections <cache>.npz --number-detections <cache>.npz \
    --team-model <model>.pkl \
    --output runs/full_pipeline/<name>.mp4
```

Reader output is cached beside the run as `<name>_reads.json` and replayed on
re-render, so iterating on voting rules costs one tracking pass rather than a
full reader pass.

The claims in *Current status* are reproducible:

```bash
python -m scripts.benchmark_jersey_parseq        --help   # reader accuracy table
python -m scripts.benchmark_doctr_readers        --help
python -m scripts.measure_reid_discriminability  --help   # can re-ID separate teammates?
python -m scripts.sweep_number_crop_padding      --help
python -m scripts.compare_read_modes             --help
```

The architectural dependency rules and artifact boundaries are documented in
`docs/architecture.md`. Detailed experimental findings remain in `CLAUDE.md`.

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
