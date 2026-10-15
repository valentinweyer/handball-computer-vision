"""Verify SAM-MT source and weights on CPU, without running inference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "SAM-MT"
REVISION = "55952679e062feb05004b598bb835d4a7a8205a1"
CHECKPOINT = ROOT / "models/sam_mt/checkpoints/sam-mt.pt"
CHECKPOINT_SHA256 = "8646f62975a5d3692eabb883003b8ca46160953d77c6a4c7310feccd39010f48"
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def main() -> None:
    if "torch" in sys.modules or "sam2" in sys.modules:
        raise RuntimeError("Run this check in a fresh Python process.")
    if not (SOURCE / "sam2/build_sam.py").is_file():
        raise FileNotFoundError(f"Clone the pinned SAM-MT source into {SOURCE}")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected SAM-MT {REVISION}, found {revision}")
    with CHECKPOINT.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint checksum mismatch: {digest}")

    # SAM-MT and production SAM2 both import as `sam2`. Select the experiment
    # only in this fresh process; do not install it over the production package.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(SOURCE))
    import sam2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    imported_from = Path(sam2.__file__).resolve()
    if not imported_from.is_relative_to(SOURCE.resolve()):
        raise RuntimeError(f"Wrong sam2 package imported: {imported_from}")

    # Construction and strict CPU weight loading only. Do not initialize a
    # video state, add prompts, propagate, or call the model on any input.
    model = build_sam2_video_predictor(
        CONFIG,
        ckpt_path=None,
        device="cpu",
        vos_optimized=False,
        hydra_overrides_extra=["++model.compile_image_encoder=false"],
    )
    weights = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(weights, strict=True)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Setup verification must keep all parameters on CPU")
    print(json.dumps({
        "source_revision": revision,
        "sam2_import": str(imported_from),
        "config": CONFIG,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": digest,
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "strict_weight_load": "passed",
        "device": "cpu",
        "inference_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
