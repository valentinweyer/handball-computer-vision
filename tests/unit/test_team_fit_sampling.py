"""The detection-cache team fit draws its crops from the whole video, not its start."""
import cv2
import numpy as np
import pytest

from handball_cv.teams.model import TeamModel

FRAMES = 200


@pytest.fixture
def video(tmp_path):
    """Each frame's mean grey level is its index plus 30, so a crop says where it came from.

    A 4 px checkerboard of +-30 around it gives the contrast and sharpness the
    crop-quality gate asks for.
    """
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    yy, xx = np.mgrid[0:240, 0:320]
    checker = (((yy // 4) + (xx // 4)) % 2 * 60).astype(np.uint8)[..., None]
    for i in range(FRAMES):
        writer.write(np.repeat(checker, 3, axis=2) + np.uint8(i))
    writer.release()
    return path


def cache_with_one_player_per_frame(stride: int) -> dict:
    offsets, boxes, classes = [0], [], []
    for i in range(FRAMES):
        if i % stride == 0:
            boxes += [[100, 40, 180, 220], [10, 10, 40, 60]]  # a player, and a goalkeeper far away
            classes += [2, 1]
        offsets.append(len(boxes))
    return {"offsets": np.array(offsets), "boxes": np.array(boxes, dtype=float),
            "class_id": np.array(classes), "confidence": np.ones(len(boxes)), "stride": np.array(stride)}


def captured_crops(monkeypatch, video, cache, **options):
    seen = {}

    def fake_fit(cls, crops, seed=0, device="cuda", **_):
        seen["crops"] = crops
        return "model"

    monkeypatch.setattr(TeamModel, "fit_from_crops", classmethod(fake_fit))
    assert TeamModel.fit_from_detection_cache(video, cache, device="cpu", **options) == "model"
    return seen["crops"]


def test_crops_span_the_whole_video_even_when_capped(monkeypatch, video):
    crops = captured_crops(monkeypatch, video, cache_with_one_player_per_frame(5),
                           stride=10, max_frames=8, max_crops=6)
    assert len(crops) == 6
    levels = sorted(float(c.mean()) for c in crops)
    # Frame i is mean grey ~i + 30: the first crop is from the opening frames,
    # the last from the final ones -- not the first 6 frames fit_from_video would take.
    assert levels[0] < 50 and levels[-1] > FRAMES + 10


def test_only_field_players_are_cropped(monkeypatch, video):
    crops = captured_crops(monkeypatch, video, cache_with_one_player_per_frame(10), stride=10)
    assert len(crops) == FRAMES // 10
    assert all(c.shape[0] > 60 for c in crops)  # the goalkeeper box is far smaller


def test_a_stride_off_the_cache_is_refused(monkeypatch, video):
    with pytest.raises(ValueError, match="scored frames"):
        captured_crops(monkeypatch, video, cache_with_one_player_per_frame(5), stride=7)
