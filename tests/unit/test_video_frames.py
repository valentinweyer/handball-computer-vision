"""Decoding the video directly must hand SAM2 and the consumer the right frames."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from handball_cv.tracking.sam2_driver import rgb_to_sam2_tensor
from handball_cv.video import VideoFrames

FRAMES = 40


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(FRAMES):
        frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def decoded(video) -> list:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def test_counts_and_size_come_from_decoding(video, decoded):
    frames = VideoFrames(video)
    assert len(frames) == len(decoded) == FRAMES
    assert (frames.width, frames.height) == (64, 48)


def test_in_order_repeated_and_backward_access(video, decoded):
    frames = VideoFrames(video, keep=4)
    order = [0, 1, 1, 2, 3, 3, 10, 9, 10, 39, 5, 6, 0]
    for idx in order:
        assert np.array_equal(frames.rgb(idx), decoded[idx]), idx
    # 9 is still buffered after 10; 5 and 0 are older than a 4-frame buffer.
    assert frames.backward_seeks == 2


def test_returns_a_copy(video, decoded):
    frames = VideoFrames(video)
    first = frames.rgb(3)
    first[:] = 0
    assert np.array_equal(frames.rgb(3), decoded[3])


def test_bgr_is_the_same_frame_in_opencv_order_and_a_copy(video, decoded):
    frames = VideoFrames(video, keep=4)
    for idx in (0, 1, 2, 10, 5):
        bgr = frames.bgr(idx)
        assert np.array_equal(bgr[:, :, ::-1], decoded[idx]), idx
        bgr[:] = 0
        assert np.array_equal(frames.rgb(idx), decoded[idx]), idx


def test_concurrent_readers_get_the_right_frames(video, decoded):
    frames = VideoFrames(video, keep=8)
    indices = [i for i in range(FRAMES) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(frames.rgb, indices))
    assert all(np.array_equal(r, decoded[i]) for r, i in zip(results, indices))


def test_out_of_range_is_an_error(video):
    with pytest.raises(IndexError):
        VideoFrames(video).rgb(FRAMES)


def test_preprocessing_matches_upstream_on_identical_pixels(tmp_path, monkeypatch):
    """Same pixels in, same tensor out: the video path changes only the source.

    A lossless PNG gives upstream's file loader exactly the array the video
    path receives, so any difference would be our preprocessing, not JPEG loss.
    """
    upstream = Path(os.environ.get(
        "SAM2_UPSTREAM_DIR", Path(__file__).resolve().parents[2] / "sam2-upstream"))
    monkeypatch.syspath_prepend(str(upstream))
    misc = pytest.importorskip("sam2.utils.misc")
    rgb = np.random.default_rng(1).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    png = tmp_path / "frame.png"
    cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    expected, height, width = misc._load_img_as_tensor(str(png), 64)
    actual = rgb_to_sam2_tensor(rgb, 64)
    assert (height, width) == (120, 200)
    assert actual.dtype == expected.dtype == torch.float64
    assert torch.equal(actual, expected)
