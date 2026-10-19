"""Frozen alternative appearance encoders for the matched comparison (research note, step C).

Each encoder embeds the same saved crops (`<results>/crops/<crop_id>.jpg`, the raw
player boxes) into `<results>/features_<name>.npz` (key `raw`, one L2-normalised row
per crop id). Matching, gallery budget, query aggregation and the leave-one-match-out
protocol stay exactly as for PRTReID; only the embedding changes, and each encoder
gets its own gates chosen on development folds (raw cosines are not comparable
across models).

Weights load strictly: every backbone tensor in the checkpoint must land in the
model and every model tensor must come from the checkpoint, so a loader cannot
silently skip most of the weights.

    python -m experiments.teammate_reid.encoders personvit_b_msmt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")


class PersonViT:
    """PersonViT ViT-B/16 (LUPerson self-supervised, ReID fine-tuned on MSMT17), 256x128.

    TransReID-style checkpoint, no SIE/JPM, BN-neck with `NECK_FEAT: before`:
    the embedding is the final-norm CLS token, before the bottleneck.
    """
    name = "personvit_b_msmt"
    checkpoint = REPO / "models/personvit/msmt.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth"
    size = (256, 128)
    mean = std = (0.5, 0.5, 0.5)

    def __init__(self, device: str = "cuda"):
        import timm
        self.device = device
        model = timm.models.vision_transformer.VisionTransformer(
            img_size=self.size, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
            num_classes=0, global_pool="token", class_token=True)
        sd = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        backbone = {k[len("base."):]: v for k, v in sd.items()
                    if k.startswith("base.") and not k.startswith("base.fc.")}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"PersonViT load mismatch: missing={missing} unexpected={unexpected}")
        self.loaded = len(backbone)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def encode(self, crops_rgb: list) -> np.ndarray:
        h, w = self.size
        batch = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in crops_rgb])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        feats = self.model((x - mean) / std)
        return torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float32)


class SportsReIDViT:
    """SportsReID ViT-B/16 (torchvision ViT, SoccerNet-v3 ReID, hierarchical sampler), 224x224.

    The author's wrapper replaces the head with identity and adds Linear(768,512) +
    BatchNorm1d; in eval it returns that 512-d vector. ImageNet normalisation
    (torchreid default).
    """
    name = "sportsreid_vitb16"
    checkpoint = REPO / "models/sportsreid/vit_b16.bin"
    size = (224, 224)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cuda"):
        import torchvision
        self.device = device
        vit = torchvision.models.vit_b_16(weights=None)
        vit.heads = torch.nn.Identity()
        fc = torch.nn.Sequential(torch.nn.Linear(768, 512), torch.nn.BatchNorm1d(512))
        sd = torch.load(self.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        vit_sd = {k[len("module.model."):]: v for k, v in sd.items() if k.startswith("module.model.")}
        fc_sd = {k[len("module.fc."):]: v for k, v in sd.items() if k.startswith("module.fc.")}
        for module, part, label in ((vit, vit_sd, "vit"), (fc, fc_sd, "fc")):
            missing, unexpected = module.load_state_dict(part, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"SportsReID {label} load mismatch: missing={missing} unexpected={unexpected}")
        self.loaded = len(vit_sd) + len(fc_sd)
        self.model = torch.nn.Sequential(vit, fc).eval().to(device)

    encode = PersonViT.encode


ENCODERS = {PersonViT.name: PersonViT, SportsReIDViT.name: SportsReIDViT}


def local_digit_boxes(record: dict, crop_shape: tuple) -> list:
    """The record's digit boxes in the saved crop's pixels (as `masking.crop_with_digits`)."""
    h, w = crop_shape[:2]
    x1, y1 = (max(0, int(v)) for v in np.asarray(record["box"][:2], dtype=float).round())
    local = []
    for dx1, dy1, dx2, dy2 in record["digit_boxes"]:
        lx1, ly1 = max(0.0, dx1 - x1), max(0.0, dy1 - y1)
        lx2, ly2 = min(float(w), dx2 - x1), min(float(h), dy2 - y1)
        if lx2 > lx1 and ly2 > ly1:
            local.append((lx1, ly1, lx2, ly2))
    return local


def extract(results: Path, encoder, modes: tuple = ("raw",), batch: int = 128) -> Path:
    """Embed every saved crop under each masking mode (same masks and seeds as the
    PRTReID features), adding keys to `features_<name>.npz`."""
    from experiments.teammate_reid.masking import apply_mask
    crops = sorted(json.loads((results / "crops.json").read_text()), key=lambda c: c["crop_id"])
    n = crops[-1]["crop_id"] + 1
    path = results / f"features_{encoder.name}.npz"
    stored = dict(np.load(path)) if path.is_file() else {}
    for mode in modes:
        out = np.zeros((n, encoder.encode([np.zeros((8, 4, 3), np.uint8)]).shape[1]), np.float32)
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            imgs = []
            for r in chunk:
                img = cv2.cvtColor(cv2.imread(str(results / "crops" / f"{r['crop_id']}.jpg")), cv2.COLOR_BGR2RGB)
                imgs.append(apply_mask(img, local_digit_boxes(r, img.shape), mode, seed_key=str(r["crop_id"]))[0])
            out[[r["crop_id"] for r in chunk]] = encoder.encode(imgs)
        stored[mode] = out
    np.savez(path, **stored)
    return path


def main() -> None:
    from experiments.teammate_reid.openset import DEV_SOURCES
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("encoder", choices=tuple(ENCODERS))
    parser.add_argument("--modes", nargs="+", default=["raw"], choices=("raw", "digits", "strict"))
    args = parser.parse_args()
    encoder = ENCODERS[args.encoder]()
    print(f"{encoder.name}: loaded {encoder.loaded} backbone tensors strictly", flush=True)
    for results in DEV_SOURCES:
        print(extract(results, encoder, tuple(args.modes)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
