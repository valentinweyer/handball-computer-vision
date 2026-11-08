"""The pipeline's default models, from the registry in `configs/models.yaml`.

Each default is one published version: a model repo at a tag, whose weights file is
expected at a local path. What a stage records about the model it ran is that
version's name, `id@revision`, when the file is the registered one (by sha256), and
the file's path otherwise -- so a result names the exact model behind it, and a file
swapped behind a path cannot pass for the registered version.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import yaml

__all__ = ["ModelRef", "default_model", "model_label", "same_model"]

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/models.yaml"


@dataclass(frozen=True)
class ModelRef:
    id: str
    revision: str
    file: str
    path: Path
    sha256: str

    @property
    def name(self) -> str:
        return f"{self.id}@{self.revision}"


def default_model(role: str) -> ModelRef:
    entry = yaml.safe_load(REGISTRY.read_text())["models"][role]
    return ModelRef(entry["id"], entry["revision"], entry["file"], ROOT / entry["path"],
                    entry["sha256"])


@lru_cache(maxsize=None)
def _sha256(path: Path, size: int, mtime: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_label(path: Path, role: str) -> str:
    """`id@revision` if `path` holds `role`'s registered file, else the path itself."""
    path = Path(path)
    ref = default_model(role)
    if path.is_file():
        stat = path.resolve().stat()
        if _sha256(path.resolve(), stat.st_size, stat.st_mtime) == ref.sha256:
            return ref.name
    return str(path)


def same_model(stored: str, path: Path, role: str) -> bool:
    """Whether a stage's recorded model is the one `path` would run now.

    A stage records `model_label`; one written before the registry recorded the
    file's path, which still matches when it resolves to the same file or holds the
    same registered version.
    """
    current = model_label(path, role)
    if stored == current:
        return True
    old = Path(stored)
    return old.is_absolute() and (old.resolve() == Path(path).resolve()
                                  or model_label(old, role) == current)
