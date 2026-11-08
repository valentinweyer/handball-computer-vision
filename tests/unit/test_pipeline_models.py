import hashlib

import pytest

from handball_cv.pipeline import models


@pytest.fixture
def registry(tmp_path, monkeypatch):
    weights = tmp_path / "detector.ts"
    weights.write_bytes(b"registered weights")
    (tmp_path / "models.yaml").write_text(f"""models:
  player_detector:
    id: owner/player-detector
    revision: expanded-v1
    file: model.ts
    path: {weights}
    sha256: {hashlib.sha256(b"registered weights").hexdigest()}
""")
    monkeypatch.setattr(models, "REGISTRY", tmp_path / "models.yaml")
    return weights


def test_the_registered_file_is_named_by_its_version(registry):
    assert models.model_label(registry, "player_detector") == "owner/player-detector@expanded-v1"


def test_a_copy_elsewhere_is_still_the_registered_version(registry, tmp_path):
    copy = tmp_path / "cache" / "model.ts"
    copy.parent.mkdir()
    copy.write_bytes(registry.read_bytes())
    assert models.model_label(copy, "player_detector") == "owner/player-detector@expanded-v1"


def test_other_weights_behind_the_registered_path_are_named_by_their_path(registry):
    registry.write_bytes(b"retrained weights")
    assert models.model_label(registry, "player_detector") == str(registry)


def test_a_missing_file_is_named_by_its_path(registry, tmp_path):
    assert models.model_label(tmp_path / "gone.ts", "player_detector") == str(tmp_path / "gone.ts")


def test_a_stage_recorded_by_version_matches_the_registered_file(registry):
    assert models.same_model("owner/player-detector@expanded-v1", registry, "player_detector")


def test_a_stage_recorded_by_path_before_the_registry_still_matches(registry, tmp_path):
    link = tmp_path / "link.ts"
    link.symlink_to(registry)
    assert models.same_model(str(registry), link, "player_detector")


def test_a_stage_recorded_by_an_old_path_holding_the_version_matches(registry, tmp_path):
    old = tmp_path / "exports" / "rfdetr-large-handball.ts"
    old.parent.mkdir()
    old.write_bytes(registry.read_bytes())
    assert models.same_model(str(old), registry, "player_detector")


def test_a_stage_made_by_another_model_does_not_match(registry, tmp_path):
    other = tmp_path / "other.ts"
    other.write_bytes(b"another model")
    assert not models.same_model(str(other), registry, "player_detector")
    assert not models.same_model("owner/player-detector@original-v1", registry, "player_detector")
