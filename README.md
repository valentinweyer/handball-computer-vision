# Handball Computer Vision

Turns tactical handball footage into per-frame data about the players, and
renders it back onto the video with a court minimap:

- **players**: detected with an RF-DETR model, then tracked with SAM2 and
  regular re-prompting from the detector;
- **teams**: discovered anonymously in each video, with no team catalogue and no
  labels;
- **jersey numbers**: read with parseq and voted per player, so identities
  survive the player leaving and re-entering the picture;
- **court positions**: court keypoints give a homography for every frame, which
  maps each player's feet onto the court.

The output is a result bundle of Parquet tables, plus a rendered video.

> **Status:** the code is complete, but the handball models it needs (player
> detector, court keypoints, jersey-number detector) are **not published yet**.
> They will follow on Hugging Face under their own licence. Until then the
> pipeline cannot run end to end.

## Requirements

- Linux and an NVIDIA GPU with CUDA. Developed on Python 3.11 and PyTorch 2.9.
- About 2 GB of third-party weights (SAM2, parseq, prtreid), plus the handball
  models.

### Input video

Plain tactical footage: one continuous shot from a single camera that pans
to follow play, with no replays, close-ups, zooms or cuts to other cameras.
1080p works best; at lower resolutions far fewer jersey numbers can be read.
Where the court is not in view, tracking pauses until it is.

## Installation

```bash
git clone https://github.com/valentinweyer/handball-computer-vision.git
cd handball-computer-vision
python3.11 -m venv .venv && source .venv/bin/activate

# 1. PyTorch with CUDA for your system: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. This package and its dependencies
pip install -e ".[dev]"
```

### Upstream checkouts

SAM2, parseq and prtreid are not installed as packages. The pipeline imports
them from checkouts in the repository root, at the commits it was tested with:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2-upstream
git -C sam2-upstream checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4

git clone https://github.com/baudm/parseq.git parseq-upstream
git -C parseq-upstream checkout 1902db043c029a7e03a3818c616c06600af574be

git clone https://github.com/VlSomers/prtreid.git prtreid-upstream
git -C prtreid-upstream checkout 30617a75967e84d5d516959c4b84cbeea6f56493
```

`SAM2_UPSTREAM_DIR` and `PARSEQ_UPSTREAM_DIR` point elsewhere if you keep them
outside the repository, and `--prtreid-root` does the same for prtreid.

### Third-party weights

```bash
mkdir -p models/sam2 models/jersey_parseq models/prtreid
curl -L -o models/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
curl -L -o models/jersey_parseq/parseq_original.ckpt \
  https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt
curl -L -o models/prtreid/prtreid-soccernet-baseline.pth.tar \
  "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
```

| File | sha256 |
|---|---|
| `sam2.1_hiera_large.pt` | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| `parseq_original.ckpt` | `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00` |
| `prtreid-soccernet-baseline.pth.tar` | `1304562c4c930a4a54bbf2d44e221eb911fe31408f6875c5058b2d6ed621cf3b` |

The team model's image features (`google/siglip-base-patch16-224`) download
from Hugging Face on first use.

### Handball models

`configs/models.yaml` names the three default models by version and checksum,
and where they are expected:

```
models/handball_cv/detector.ts          player detector
models/handball_cv/court-keypoints.ts   court keypoints
models/handball_cv/number-sheets.ts     jersey-number detector (--sheet-numbers)
```

Each run records the version it used when the file's checksum matches, and the
file's path otherwise. Other files can be passed with `--detector`,
`--court-weights` and `--sheet-model`.

## Usage

```bash
python -m scripts.process_video path/to/match.mp4   # -> runs/bundles/match/
python -m scripts.render_result runs/bundles/match  # -> runs/bundles/match/render.mp4
```

`process_video` runs in stages (`prepare`, `scan`, `teams`, `numbers`, `track`,
`bundle`) and keeps each stage's output under `runs/interim/<video>/`. An
interrupted run resumes where it stopped, and `--force STAGE` reruns that stage
and everything after it. Both commands list their options with `--help`.

### Result bundle

| File | Contents |
|---|---|
| `manifest.json` | what is in the bundle, how it was made, the version of each table |
| `tracks.parquet` | one row per tracked player per frame: identity, team, jersey number, box, foot point, court position |
| `players.parquet` | one row per player identity |
| `court.parquet` | one row per frame: the court homography and whether a court is in view |
| `detections.parquet` | raw detector output |
| `reads.parquet` | the jersey-number reads |
| `events.parquet` | identity events: re-identification, linking, team changes |
| `keypoints.parquet` | raw court-keypoint output |
| `masks.npz`, `team_model.pkl`, `video.mp4` | player masks, the team model, and the video the rows refer to |

`handball_cv.pipeline.result.load(bundle)` reads a bundle back.

## Tests

```bash
pytest -q
```

The unit tests need neither weights nor a GPU.

## Licence

Copyright (C) 2026 Valentin Weyer.

The code is licensed under the **GNU Affero General Public License v3.0**
([LICENSE](LICENSE)). You may use, study and modify it freely. If you
distribute it, or let others use a modified version over a network, you must
publish your source under the same licence. For use on other terms, such as in
a closed-source product, open an issue to ask about a commercial licence.

The handball models will be published under a separate licence.

### Third-party components

| Component | Licence |
|---|---|
| [SAM2](https://github.com/facebookresearch/sam2) | Apache 2.0 |
| [parseq](https://github.com/baudm/parseq) | Apache 2.0 |
| [prtreid](https://github.com/VlSomers/prtreid) | [Hippocratic License 3.0](https://firstdonoharm.dev/) |
| [sports](https://github.com/roboflow/sports) (handball fork) | MIT |
| [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) | Apache 2.0 |
| Concert One font (`assets/fonts/`) | SIL Open Font License 1.1 |

prtreid is the default re-identification model. You install it yourself, and the
pipeline only calls it. Its licence permits commercial use under ethical
conditions, which bind anyone who uses or distributes it. Read them before
using this pipeline commercially or shipping it to others.

## Contributing

Issues and bug reports are welcome. Pull requests are not accepted at the
moment.
