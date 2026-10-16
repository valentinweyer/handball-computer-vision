"""Export a court-keypoint checkpoint as ONNX (CPU) and TorchScript (GPU).

Runs in the rfdetr-handball-finetune venv -- the only interpreter with
`rfdetr` -- and writes into that checkout's exports/:

    "$RFDETR_FINETUNE_DIR/.venv/bin/python3" -m scripts.export_court_keypoints \\
        --checkpoint "$RFDETR_FINETUNE_DIR/runs/court-keypoints-v4-200/checkpoint_139.ckpt" \\
        --output-dir "$RFDETR_FINETUNE_DIR/exports/court-keypoints-local-ep139"

`--checkpoint` is a `.pth` export-style checkpoint (`checkpoint_best_ema.pth`)
or a Lightning `checkpoint_N.ckpt`, whose EMA weights are taken: the training
evaluations name those `emaN`, and they are what the periodic checkpoints save
beside the raw weights. The `.ckpt` carries no model config, so the run's own
`checkpoint_best_ema.pth` from the same directory supplies it.

The TorchScript file is what `handball_cv.court.rfdetr_keypoints` runs on the
GPU in this project's venv with plain torch -- rfdetr needs transformers>=5.1,
which this project pins below. It is traced from the same export-mode graph the
ONNX file comes from: DINOv2 frozen at the export shape, then `model.export()`.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


def ema_checkpoint(ckpt_path: Path, out_path: Path) -> Path:
    """Rewrite a Lightning checkpoint's EMA weights in the `.pth` layout rfdetr loads."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    averaged = ckpt["callbacks"]["RFDETREMACallback"]["average_model_state_dict"]
    template = torch.load(ckpt_path.parent / "checkpoint_best_ema.pth", map_location="cpu",
                          weights_only=False)
    state = {k.removeprefix("module."): v for k, v in averaged.items() if k != "n_averaged"}
    if set(state) != set(template["state_dict"]):
        raise ValueError("EMA state does not match the run's checkpoint_best_ema.pth parameters")
    # rfdetr loads `model` (unprefixed); `state_dict` is the Lightning-prefixed
    # copy. Both must be this epoch's: a template's own `model` would otherwise
    # load silently in place of the weights asked for.
    exported = {k: template[k] for k in ("args", "model_name", "model_config", "rfdetr_version")
                if k in template}
    exported.update(model={k.removeprefix("model."): v for k, v in state.items()}, state_dict=state,
                    epoch=ckpt["epoch"], global_step=ckpt["global_step"])
    torch.save(exported, out_path)
    return out_path


def trace(model, resolution: int, out_path: Path) -> Path:
    """TorchScript of the export-mode network, as RFDETR.export() prepares it for ONNX."""
    from rfdetr.models.backbone.dinov2 import DinoV2

    net = deepcopy(model.model.model).to("cuda").eval()
    for module in net.modules():
        if isinstance(module, DinoV2):
            module.shape = (resolution, resolution)
            module.export()
    net.export()
    example = torch.zeros(1, 3, resolution, resolution, device="cuda")
    with torch.no_grad():
        traced = torch.jit.trace(net, example, check_trace=False)
    traced.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETRKeypointPreview

    weights = args.checkpoint
    if weights.suffix == ".ckpt":
        weights = ema_checkpoint(weights, args.output_dir / f"court_keypoints_{weights.stem}_ema.pth")
        print(f"EMA weights -> {weights}")
    model = RFDETRKeypointPreview(pretrain_weights=str(weights))
    expected = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    loaded = model.model.model.state_dict()
    if not all(torch.equal(loaded[k].cpu(), v) for k, v in expected.items() if k in loaded):
        raise RuntimeError(f"rfdetr did not load the weights in {weights}")
    onnx_path = model.export(output_dir=str(args.output_dir))
    print(f"ONNX -> {onnx_path}")
    ts_path = trace(model, model.model.resolution, args.output_dir / "rfdetr-keypoint-preview.ts")
    print(f"TorchScript -> {ts_path}")


if __name__ == "__main__":
    main()
