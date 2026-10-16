"""Stage bookkeeping for process_video, and the detection-stride guard."""
import pytest

from handball_cv.pipeline.scan import detection_stride_guard
from scripts.process_video import STAGES, clear_from, stage_outputs


def test_forcing_a_stage_clears_it_and_everything_after(tmp_path):
    outputs = stage_outputs(tmp_path)
    for paths in outputs.values():
        for path in paths:
            path.write_text("x")
    clear_from("teams", outputs)
    for stage in STAGES:
        if not outputs[stage]:
            continue  # the bundle stage always rebuilds; it has nothing to clear
        exists = all(p.is_file() for p in outputs[stage])
        assert exists == (STAGES.index(stage) < STAGES.index("teams")), stage


def test_a_forced_track_pass_does_not_replay_old_reads(tmp_path):
    assert tmp_path / "track_reads.json" in stage_outputs(tmp_path)["track"]


def test_cadences_must_land_on_detected_frames():
    detection_stride_guard(5, 10, 5)
    with pytest.raises(ValueError, match="never scored"):
        detection_stride_guard(5, 10, 3)
