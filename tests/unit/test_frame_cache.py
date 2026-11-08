"""The JPEG frame cache the SAM2 path and `--redraw` read instead of the video."""
from pathlib import Path
import subprocess

import pytest

from handball_cv.tracking.sam2_driver import ensure_frame_cache
from handball_cv.video import ffmpeg_exe


def make_video(path: Path, frames: int) -> Path:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x64:rate=25", "-frames:v", str(frames),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_extraction_is_complete_and_leaves_no_partial(tmp_path):
    video = make_video(tmp_path / "clip.mp4", 5)
    cache = tmp_path / "frames" / "clip"
    assert len(ensure_frame_cache(video, cache)) == 5
    assert (cache / "source.json").is_file()
    assert not cache.with_name("clip.partial").exists()


def test_an_interrupted_extraction_is_not_taken_for_the_clip(tmp_path):
    video = make_video(tmp_path / "clip.mp4", 5)
    cache = tmp_path / "clip"
    partial = tmp_path / "clip.partial"
    partial.mkdir()
    (partial / "00000.jpg").write_bytes(b"stale")
    assert len(ensure_frame_cache(video, cache)) == 5
    assert not partial.exists()


def test_a_different_video_under_the_same_cache_is_refused(tmp_path):
    cache = tmp_path / "clip"
    ensure_frame_cache(make_video(tmp_path / "a.mp4", 5), cache)
    with pytest.raises(RuntimeError, match="extracted from"):
        ensure_frame_cache(make_video(tmp_path / "b.mp4", 9), cache)


def test_a_cache_made_before_source_records_is_kept_as_it_is(tmp_path):
    video = make_video(tmp_path / "clip.mp4", 5)
    cache = tmp_path / "clip"
    cache.mkdir()
    for i in range(3):
        (cache / f"{i:05d}.jpg").write_bytes(b"legacy")
    assert [p.read_bytes() for p in ensure_frame_cache(video, cache)] == [b"legacy"] * 3
    assert not (cache / "source.json").exists()
