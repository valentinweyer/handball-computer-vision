"""Compare team embeddings on exact RF-DETR crops, without a tracker.

This is deliberately a representation benchmark.  It clusters each video's
embeddings into two anonymous teams and uses human labels only to score the
resulting partition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from handball_cv.teams.calibration import spatial_jersey_features
from handball_cv.teams.dataset import read_manifest
from handball_cv.teams.evaluation import evaluate_manifest_embeddings


class ColorBackend:
    name = "color"

    def encode(self, manifest_path: Path, manifest: dict, batch_size: int) -> np.ndarray:
        del batch_size
        root = manifest_path.parent
        crops = []
        for sample in manifest["samples"]:
            path = root / sample["torso_path"]
            crop_bgr = cv2.imread(str(path))
            if crop_bgr is None:
                raise FileNotFoundError(path)
            crops.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return spatial_jersey_features(crops)


class ResNetPartBackend:
    """Generic ImageNet baseline with global plus horizontal-part pooling."""

    name = "resnet50_parts"

    def __init__(self, device: str):
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        network = resnet50(weights=weights)
        self.trunk = torch.nn.Sequential(*list(network.children())[:-2]).eval().to(device)
        self.device = torch.device(device)
        self.mean = torch.tensor(weights.transforms().mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(weights.transforms().std, device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def encode(self, manifest_path: Path, manifest: dict, batch_size: int) -> np.ndarray:
        root = manifest_path.parent
        paths = [root / sample["crop_path"] for sample in manifest["samples"]]
        features = []
        for start in tqdm(range(0, len(paths), batch_size), desc="resnet50 parts"):
            batch = []
            for path in paths[start:start + batch_size]:
                image_bgr = cv2.imread(str(path))
                if image_bgr is None:
                    raise FileNotFoundError(path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                tensor = torch.nn.functional.interpolate(
                    tensor[None], size=(256, 128), mode="bilinear", align_corners=False,
                )[0]
                batch.append(tensor)
            images = torch.stack(batch).to(self.device)
            images = (images - self.mean) / self.std
            maps = self.trunk(images)
            pooled = [torch.nn.functional.adaptive_avg_pool2d(maps, 1).flatten(1)]
            for stripe in torch.tensor_split(maps, 3, dim=2):
                pooled.append(torch.nn.functional.adaptive_avg_pool2d(stripe, 1).flatten(1))
            features.append(torch.cat(pooled, dim=1).cpu().numpy())
        return np.concatenate(features).astype(np.float32)


class SiglipBackend:
    name = "siglip"

    def __init__(self, device: str):
        from sports import TeamClassifier

        self.model = TeamClassifier(device=device)

    def encode(self, manifest_path: Path, manifest: dict, batch_size: int) -> np.ndarray:
        root = manifest_path.parent
        outputs = []
        samples = manifest["samples"]
        for start in tqdm(range(0, len(samples), batch_size), desc="siglip"):
            crops_bgr = []
            for sample in samples[start:start + batch_size]:
                path = root / sample["crop_path"]
                crop = cv2.imread(str(path))
                if crop is None:
                    raise FileNotFoundError(path)
                # sports.TeamClassifier calls cv2_to_pillow and therefore
                # expects BGR. Passing our RGB crops here would swap red/blue.
                crops_bgr.append(crop)
            outputs.append(np.asarray(self.model.extract_features(crops_bgr)))
        return np.concatenate(outputs).astype(np.float32)


def build_backend(args):
    if args.backend == "color":
        return ColorBackend()
    if args.backend == "resnet50_parts":
        return ResNetPartBackend(args.device)
    if args.backend == "siglip":
        return SiglipBackend(args.device)
    if args.backend in {"prtreid", "prtreid_team", "prtreid_team_logits"}:
        from handball_cv.embeddings.prtreid import PRTReIDBackend

        return PRTReIDBackend(
            source_root=args.prtreid_root,
            checkpoint=args.prtreid_checkpoint,
            device=args.device,
            feature_kind={
                "prtreid": "global",
                "prtreid_team": "team",
                "prtreid_team_logits": "team_logits",
            }[args.backend],
        )
    raise ValueError(args.backend)


def cached_features(
    backend, manifest_path: Path, manifest: dict, batch_size: int, recompute: bool,
) -> np.ndarray:
    cache_path = manifest_path.parent / f"features_{backend.name}.npz"
    sample_ids = np.array([sample["sample_id"] for sample in manifest["samples"]])
    if cache_path.exists() and not recompute:
        with np.load(cache_path, allow_pickle=False) as raw:
            if np.array_equal(raw["sample_ids"], sample_ids):
                print(f"reuse {cache_path}")
                return raw["features"]
    features = backend.encode(manifest_path, manifest, batch_size)
    if len(features) != len(sample_ids):
        raise RuntimeError("backend returned the wrong number of embeddings")
    payload = {"sample_ids": sample_ids, "features": features}
    auxiliary = getattr(backend, "last_auxiliary", None)
    if auxiliary:
        payload.update(auxiliary)
    np.savez_compressed(cache_path, **payload)
    print(f"wrote {cache_path}: {features.shape}")
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument(
        "--backend",
        choices=(
            "color", "resnet50_parts", "siglip", "prtreid",
            "prtreid_team", "prtreid_team_logits",
        ),
        default="color",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-fit-overlap", type=float, default=0.25)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prtreid-root", type=Path, default=Path("/tmp/handball-prtreid"))
    parser.add_argument(
        "--prtreid-checkpoint", type=Path,
        default=Path("models/prtreid/prtreid-soccernet-baseline.pth.tar"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = build_backend(args)
    results = []
    for manifest_path in args.manifests:
        manifest_path = manifest_path.resolve()
        manifest = read_manifest(manifest_path)
        features = cached_features(
            backend, manifest_path, manifest, args.batch_size, args.recompute
        )
        result = evaluate_manifest_embeddings(
            manifest, features, max_fit_overlap=args.max_fit_overlap
        )
        results.append(result)
        print(
            f"{result['video_id']}: accuracy={100 * result['accuracy']:.2f}%  "
            f"balanced={100 * result['balanced_accuracy']:.2f}%  "
            f"ARI={result['ari']:.3f}  labels={result['count']}  "
            f"fit={result['fit_count']}"
        )
    summary = {
        "backend": backend.name,
        "videos": results,
        "macro_accuracy": float(np.mean([result["accuracy"] for result in results])),
        "macro_balanced_accuracy": float(
            np.mean([result["balanced_accuracy"] for result in results])
        ),
        "macro_ari": float(np.mean([result["ari"] for result in results])),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
