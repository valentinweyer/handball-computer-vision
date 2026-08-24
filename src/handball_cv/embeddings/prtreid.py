"""Inference-only adapter for the SoccerNet PRTReID checkpoint.

PRTReID is used only to describe independent person detections. It does not
assign track IDs or permanent player identities; its embeddings are clustered
into two anonymous teams separately for each video.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


GLOBAL = "globl"
ROLE_NAMES = np.asarray(("ball", "goalkeeper", "other", "player", "referee"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _isolated_bpbreid_class(source_root: Path):
    """Load only BPBreID and HRNet, avoiding the upstream training stack."""
    package_root = source_root / "prtreid"
    paths = {
        "hrnet": package_root / "models" / "hrnet.py",
        "bpbreid": package_root / "models" / "bpbreid.py",
        "constants": package_root / "utils" / "constants.py",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "PRTReID source files not found: " + ", ".join(map(str, missing))
        )

    package = types.ModuleType("prtreid")
    package.__path__ = [str(package_root)]
    models_package = types.ModuleType("prtreid.models")
    models_package.__path__ = [str(package_root / "models")]
    utils_package = types.ModuleType("prtreid.utils")
    utils_package.__path__ = [str(package_root / "utils")]
    package.models = models_package
    for name, module in (
        ("prtreid", package),
        ("prtreid.models", models_package),
        ("prtreid.utils", utils_package),
    ):
        sys.modules[name] = module

    _load_module("prtreid.utils.constants", paths["constants"])
    hrnet = _load_module("prtreid.models.hrnet", paths["hrnet"])

    def build_model(name, num_classes, loss="part_based", pretrained=False, **kwargs):
        if name != "hrnet32":
            raise ValueError(f"isolated adapter supports hrnet32, not {name!r}")
        return hrnet.hrnet32(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            **kwargs,
        )

    models_package.build_model = build_model
    return _load_module("prtreid.models.bpbreid", paths["bpbreid"]).BPBreID


def _classifier_count(state: dict[str, torch.Tensor], key: str) -> int:
    weight = state.get(key)
    if weight is None or weight.ndim != 2:
        raise RuntimeError(f"checkpoint is missing classifier tensor {key!r}")
    return int(weight.shape[0])


class PRTReIDBackend:
    """SoccerNet-trained global embedding with auxiliary role probabilities."""

    def __init__(
        self, source_root: Path, checkpoint: Path, device: str,
        feature_kind: str = "global",
    ):
        if feature_kind not in {"global", "team", "team_logits"}:
            raise ValueError(feature_kind)
        self.feature_kind = feature_kind
        self.name = (
            "prtreid" if feature_kind == "global" else f"prtreid_{feature_kind}"
        )
        self.source_root = Path(source_root).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable; using CPU for PRTReID")
            device = "cpu"
        self.device = torch.device(device)

        # The official checkpoint contains a yacs config, requiring full pickle
        # loading. Only use the checksum-verified official SoccerNet artifact.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_state, config = saved.get("state_dict"), saved.get("config")
        if not isinstance(raw_state, dict) or config is None:
            raise RuntimeError("not a PRTReID training checkpoint")
        state = {key.removeprefix("module."): value for key, value in raw_state.items()}
        num_ids = _classifier_count(
            state, "global_identity_classifier.classifier.weight"
        )
        num_teams = _classifier_count(state, "global_team_classifier.classifier.weight")
        num_roles = _classifier_count(state, "global_Role_classifier.classifier.weight")
        if num_roles != len(ROLE_NAMES):
            raise RuntimeError(f"checkpoint has {num_roles} unexpected role classes")

        model_class = _isolated_bpbreid_class(self.source_root)
        self.model = model_class(
            num_classes=num_ids,
            pretrained=False,
            loss="part_based",
            model_cfg=config.model.bpbreid,
            num_teams=num_teams,
            num_roles=num_roles,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.height, self.width = int(config.data.height), int(config.data.width)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
        self.last_auxiliary: dict[str, np.ndarray] = {}

    def _read_tensor(self, path: Path) -> torch.Tensor:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise FileNotFoundError(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        return torch.nn.functional.interpolate(
            image[None], size=(self.height, self.width), mode="bilinear",
            align_corners=False,
        )[0]

    @torch.inference_mode()
    def encode(self, manifest_path: Path, manifest: dict, batch_size: int) -> np.ndarray:
        root = manifest_path.parent
        paths = [root / sample["crop_path"] for sample in manifest["samples"]]
        embedding_batches, role_batches = [], []
        for start in tqdm(range(0, len(paths), batch_size), desc="PRTReID"):
            images = torch.stack([
                self._read_tensor(path) for path in paths[start:start + batch_size]
            ]).to(self.device)
            output = self.model((images - self.mean) / self.std)
            embeddings, team_scores, role_scores = output[0], output[3], output[4]
            global_embedding = embeddings[GLOBAL]
            if self.feature_kind == "global":
                selected = global_embedding
            elif self.feature_kind == "team":
                selected = self.model.global_team_classifier.bn(global_embedding)
            else:
                selected = team_scores[GLOBAL]
            embedding_batches.append(selected.float().cpu().numpy())
            role_batches.append(torch.softmax(
                role_scores[GLOBAL].float(), dim=1
            ).cpu().numpy())

        features = np.concatenate(embedding_batches).astype(np.float32)
        probabilities = np.concatenate(role_batches).astype(np.float32)
        self.last_auxiliary = {
            "role_probabilities": probabilities,
            "role_indices": probabilities.argmax(axis=1).astype(np.int64),
            "role_names": ROLE_NAMES,
        }
        return features
