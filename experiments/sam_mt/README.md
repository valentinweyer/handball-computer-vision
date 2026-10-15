# SAM-MT setup

Experimental source and checkpoint setup, with a successful GPU inference smoke
test. SAM-MT is not integrated into the production tracker. The current SAM2
default remains in place. See the [feasibility assessment](../../docs/sam-mt-feasibility.md) for the
separate speed, overlap-continuity and individual-player lifecycle experiments.

The [native benchmark comparison](../../docs/sam-mt-benchmark-results.md) was
stopped at the user's request after the first two clips showed substantial
quality regressions; the third process was killed under severe system memory
pressure. Only single-run results exist. Keep SAM2 as the production tracker;
do not automatically resume this suite.

## Local assets

- Official source: `SAM-MT/`, pinned to
  `55952679e062feb05004b598bb835d4a7a8205a1`.
- Official weights: `models/sam_mt/checkpoints/sam-mt.pt`, downloaded from
  `FudanCVL/SAM-MT` at `253e14a35585de8a662dba3c3e19096c7644c5a4`.
- SHA-256:
  `8646f62975a5d3692eabb883003b8ca46160953d77c6a4c7310feccd39010f48`.
- Model configuration: `configs/sam2.1/sam2.1_hiera_b+.yaml` inside the
  SAM-MT checkout. Despite the filename, this is SAM-MT's modified model.

The source and weights are gitignored. The checkpoint provenance is also in
[`configs/models.yaml`](../../configs/models.yaml). The Hub labels the weights
CC-BY-NC-SA-4.0.

## Verify setup without inference

From the project root:

```bash
.venv/bin/python -m experiments.sam_mt.check_setup
```

This verifies the source revision, checkpoint checksum, imports, model
construction and strict weight loading on **CPU**. It disables compilation and
GPU visibility. It does not read video, initialize a video state, add prompts,
propagate masks or execute a model forward pass.

Use the existing Python 3.11 CUDA environment with Torch `2.9.1+cu130` and
torchvision `0.24.1`. Upstream pins Torch 2.8.0 and torchvision 0.23.0; its blanket
requirements installation would replace the project's GB10 CUDA packages.
The core model dependencies are already available in the project environment.
Gradio (required by upstream's demo utilities, including its CLI demo) and the
Decord-based video loader are not installed. The core predictor API and its
JPEG-frame loading path are prepared for the later handball experiment.

SAM-MT and production SAM2 both use the package name `sam2`. Keep SAM-MT out of
the shared environment's installed packages. The setup checker selects its
source in a fresh process, leaving other processes' imports unchanged. For
future SAM-MT-only Python commands, explicitly select the source:

```bash
PYTHONPATH="$PWD/SAM-MT" .venv/bin/python <sam-mt-experiment.py>
```

This command pattern is documentation only; no inference is started by setup.

Verified locally on 2026-10-15: the pinned source imported successfully and all
85,727,843 model parameters loaded strictly on CPU with the versions above.
The checkpoint checksum matched the Hub LFS SHA-256. A separate process still
imports production SAM2 from `sam2-upstream/`. That setup check runs no inference.

## GPU inference verification

A separately authorized GPU smoke test passed on 2026-10-15 using six consecutive
cached `Han-Ber4_cached` frames and 14 targets. Positive point prompts came from
the first-frame RF-DETR player/goalkeeper boxes. The model, loaded frames,
tracking-state storage, and output masks were on `cuda:0` (NVIDIA GB10), using
BF16 autocast with compilation disabled. Strict weight loading passed, and all
84 per-target masks were finite and nonempty. Peak CUDA allocation was 1.22 GiB.
No CPU inference fallback was used. The existing PyTorch capability warning
remained visible but did not prevent this run.

Evidence and reproduction script:
[`report.json`](../../runs/sam_mt_gpu_smoke_20261015/report.json),
[`gpu_smoke.py`](../../runs/sam_mt_gpu_smoke_20261015/gpu_smoke.py), and
[`run.log`](../../runs/sam_mt_gpu_smoke_20261015/run.log). Packed masks are saved
alongside them as `masks.npz`. These generated artifacts are gitignored.

This establishes working GPU prompting and temporal propagation, not measured
tracking accuracy, long-video stability, dynamic-player lifecycle support, or
benchmark throughput.

## Reproduce the downloads

Run from the project root, with `SAM-MT/` absent:

```bash
git clone --filter=blob:none --no-checkout https://github.com/FudanCVL/SAM-MT.git SAM-MT
git -C SAM-MT checkout --detach 55952679e062feb05004b598bb835d4a7a8205a1
HF_HOME="$PWD/models/sam_mt/.hf-home" hf download FudanCVL/SAM-MT \
    checkpoints/sam-mt.pt \
    --revision 253e14a35585de8a662dba3c3e19096c7644c5a4 \
    --local-dir models/sam_mt
```

Upstream references:
[source and installation](https://github.com/FudanCVL/SAM-MT),
[checkpoint](https://huggingface.co/FudanCVL/SAM-MT/tree/253e14a35585de8a662dba3c3e19096c7644c5a4).
