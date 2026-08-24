# Handball Computer Vision

Research code for handball player detection, per-video team classification,
tracking, identity correction, and court mapping.

## Current status

- RF-DETR player and goalkeeper detection is the detector baseline.
- Team classification discovers two anonymous field-player teams separately for
  each video using clean torso color evidence and guarded visual fallback.
- Raw classification can be evaluated without a tracker.
- MCByte is the preferred tracker. `IdentityManager` adds reversible temporal
  team labels so an early mistake is not frozen for the rest of a track.
- Mask-derived jersey color is an overlap-only fallback and cannot override a
  valid clean-crop observation by itself.
- Team-gated tracking and the SAM2 tracker remain experiments, not defaults.

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
notebooks/          actual notebooks plus untouched migration originals
```

The old `notebooks/*.py`, `outputs/`, `source/`, model duplicates, and nested
upstream repositories are still present. Nothing was deleted during the first
migration pass; see `docs/legacy-layout.md`.

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
conda run -n NewEnv pytest -q
```

Pytest is scoped to `tests/unit/`, so vendored ONNX Runtime and model repositories
are no longer collected.

## Team-classification diagnostics

Run tools as modules after the editable install:

```bash
python -m scripts.render_raw_team_classification --help
python -m scripts.render_mcbyte_team_correction --help
python -m scripts.render_mask_team_comparison --help
python -m scripts.evaluate_team_embeddings --help
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
