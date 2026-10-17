"""Export the RF-DETR Large player/number detector as TorchScript, for this project's venv.

`rfdetr` lives only in the sibling fine-tuning venv (it needs a transformers
this project pins below), so detection used to be a separate subprocess with
its own interpreter. Tracing the export-mode network once, there, lets
`handball_cv.detection.rfdetr_torchscript` run it here with plain torch -- the
same move `scripts/export_court_keypoints.py` made for the court model, and it
reuses that script's `trace`.

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" -m scripts.export_detector

Check the trace against `rfdetr predict()` before trusting it (see
`docs/pipeline-overnight-2026-10-17.md`).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from scripts.export_court_keypoints import trace

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_ROOT = Path(os.getenv("RFDETR_FINETUNE_DIR", ROOT.parent / "rfdetr-handball-finetune"))
DEFAULT_CHECKPOINT = FINETUNE_ROOT / "runs/detector-retrain-2026-10-17/expanded/checkpoint_best_ema.pth"
DEFAULT_OUTPUT = FINETUNE_ROOT / "exports/detector-expanded-v1/rfdetr-large-handball.ts"
RESOLUTION = 704  # the checkpoint's training resolution, as cache_number_detections runs it


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(pretrain_weights=str(args.checkpoint))
    expected = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    loaded = model.model.model.state_dict()
    if not all(torch.equal(loaded[k].cpu(), v) for k, v in expected.items() if k in loaded):
        raise RuntimeError(f"rfdetr did not load the weights in {args.checkpoint}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace(model, RESOLUTION, args.output)
    print(f"TorchScript -> {args.output}  (classes: {model.class_names})")


if __name__ == "__main__":
    main()
