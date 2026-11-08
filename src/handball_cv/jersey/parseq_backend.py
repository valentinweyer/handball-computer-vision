"""Load the SoccerNet/hockey fine-tuned PARSeq checkpoints from Koshkina et al.

These come from `mkoshkina/jersey-number-pipeline` (CVPR 2024 CVSports) and are
plain `baudm/parseq` checkpoints -- the same architecture docTR's `parseq` mirrors,
but trained on *sports jerseys* instead of scene text, which is exactly the domain
gap this project measured: three scene-text architectures share a ~6% floor of
confidently-wrong reads, so the failure is the domain, not the model family.

Two mismatches have to be absorbed to run them here:

  - The checkpoints predate an upstream refactor that moved the network under a
    `model.` attribute, so their flat `encoder.*`/`head.*` keys need re-prefixing.
  - Torch >= 2.6 defaults `weights_only=True`, which these Lightning checkpoints
    cannot satisfy.

LICENCE: the source repository is CC BY-NC 3.0. Fine for measuring whether jersey
fine-tuning helps; not for shipping in a commercial product.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


# The three published checkpoints are the same architecture saved three ways:
# the fine-tuned ones are Lightning checkpoints carrying `hyper_parameters`, while
# the original is a bare state_dict with none. Identical key sets and shapes, so
# the fine-tuned hparams describe all of them.
FALLBACK_HPARAMS = {
    "charset_train": (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    ),
    "charset_test": "0123456789abcdefghijklmnopqrstuvwxyz",
    "max_label_length": 25, "batch_size": 384, "lr": 7e-4, "warmup_pct": 0.075,
    "weight_decay": 0.0, "img_size": [32, 128], "patch_size": [4, 8],
    "embed_dim": 384, "enc_num_heads": 6, "enc_mlp_ratio": 4, "enc_depth": 12,
    "dec_num_heads": 12, "dec_mlp_ratio": 4, "dec_depth": 1, "perm_num": 6,
    "perm_forward": True, "perm_mirrored": True, "decode_ar": True,
    "refine_iters": 1, "dropout": 0.1,
}


def _ensure_upstream() -> None:
    """Put an external `baudm/parseq` checkout on the path, as SAM2 does.

    The model code is not pip-installable, so it is vendored the same way the
    rest of this project handles upstream research repos.
    """
    import os
    import sys

    root = Path(os.getenv("PARSEQ_UPSTREAM_DIR",
                          Path(__file__).resolve().parents[3] / "parseq-upstream"))
    if not root.is_dir():
        raise FileNotFoundError(
            "Needs an external baudm/parseq checkout. Clone it to ./parseq-upstream "
            "or set PARSEQ_UPSTREAM_DIR."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_jersey_parseq(checkpoint: Path, device: str = "cuda"):
    """-> (model, transform). Mirrors `SceneTextDataModule.get_transform`."""
    _ensure_upstream()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "state_dict" not in state:
        state = {"state_dict": state}
    hparams = dict(state.get("hyper_parameters") or FALLBACK_HPARAMS)
    from strhub.models.parseq.system import PARSeq

    model = PARSeq(**hparams)
    weights = state["state_dict"]
    if not any(k.startswith("model.") for k in weights):
        weights = {f"model.{k}": v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{checkpoint.name}: {len(missing)} missing, {len(unexpected)} unexpected "
            f"state_dict keys -- refusing to score a partially loaded model"
        )
    model = model.eval()
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)
    transform = T.Compose([
        T.Resize(tuple(hparams["img_size"]), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    return model, transform


def read_crops(model, transform, crops_rgb, batch_size: int = 128) -> list:
    """-> [(text, confidence)] for a list of RGB numpy crops."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(crops_rgb), batch_size):
        batch = crops_rgb[start:start + batch_size]
        images = torch.stack([
            transform(Image.fromarray(np.ascontiguousarray(c))) for c in batch
        ]).to(device)
        with torch.inference_mode():
            probability = model(images).softmax(-1)
        texts, confidences = model.tokenizer.decode(probability)
        for text, confidence in zip(texts, confidences):
            score = float(confidence.prod()) if len(confidence) else 0.0
            out.append((text, score))
    return out
