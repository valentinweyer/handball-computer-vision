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


def _video(path, frames=3):
    import subprocess
    from handball_cv.video import ffmpeg_exe
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x64:rate=50", "-frames:v", str(frames),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


SETTINGS = {"sam2_stride": 1, "check_every": 10, "ocr_every": 5}


def _conflicts(work, source, **overrides):
    from scripts.process_video import resume_conflicts
    kwargs = {"detector": work / "det.ts", "court_weights": work / "court.pt",
              "track_settings": SETTINGS, **overrides}
    return resume_conflicts(work, source, **kwargs)


def test_a_work_directory_prepared_from_another_video_is_refused(tmp_path):
    import json
    a, b = _video(tmp_path / "a.mp4"), _video(tmp_path / "b.mp4")
    (tmp_path / "meta.json").write_text(json.dumps({"source": str(a), "source_bytes": a.stat().st_size}))
    assert _conflicts(tmp_path, a) == []
    assert "was prepared from" in _conflicts(tmp_path, b)[0]


def test_a_track_pass_with_other_cadences_is_refused_and_unrecorded_ones_are_not(tmp_path):
    import json
    source = _video(tmp_path / "a.mp4")
    (tmp_path / "track.json").write_text(json.dumps({"sam2_stride": 1, "check_every": 10, "ocr_every": 5}))
    assert _conflicts(tmp_path, source) == []
    changed = _conflicts(tmp_path, source, track_settings={**SETTINGS, "ocr_every": 10})
    assert len(changed) == 1 and "--ocr-every 5" in changed[0]
    # A pass made before cadences were recorded cannot be held to them.
    (tmp_path / "track.json").write_text(json.dumps({"sam2_stride": 1}))
    assert _conflicts(tmp_path, source, track_settings={**SETTINGS, "check_every": 20}) == []
    assert _conflicts(tmp_path, source, track_settings={**SETTINGS, "sam2_stride": 2})


def test_a_track_pass_from_before_small_number_crops_counts_as_without_them(tmp_path):
    import json
    source = _video(tmp_path / "a.mp4")
    (tmp_path / "track.json").write_text(json.dumps({"sam2_stride": 1}))
    assert _conflicts(tmp_path, source, track_settings={**SETTINGS, "small_number_crops": False}) == []
    changed = _conflicts(tmp_path, source, track_settings={**SETTINGS, "small_number_crops": True})
    assert len(changed) == 1 and "--small-number-crops False" in changed[0]


def test_detections_from_another_detector_are_refused(tmp_path):
    import numpy as np
    source = _video(tmp_path / "a.mp4")
    np.savez(tmp_path / "detections.npz", detector_id=np.asarray(str(tmp_path / "old.ts")))
    assert _conflicts(tmp_path, source, detector=tmp_path / "old.ts") == []
    assert "--detector" in _conflicts(tmp_path, source, detector=tmp_path / "new.ts")[0]


def test_detections_made_at_another_stride_are_refused(tmp_path):
    import numpy as np
    source = _video(tmp_path / "a.mp4")
    np.savez(tmp_path / "detections.npz", detector_id=np.asarray(str(tmp_path / "det.ts")), stride=5)
    assert _conflicts(tmp_path, source) == []
    assert "every 5 frames, not every 1" in _conflicts(tmp_path, source, detect_stride=1)[0]


def test_re_preparing_does_not_rewrite_a_bundled_video(tmp_path):
    import os
    from handball_cv.pipeline.result import _place, sha256
    from scripts.process_video import prepare
    source = _video(tmp_path / "fast.mp4", frames=10)   # 50 fps: resampled
    work = tmp_path / "work"
    work.mkdir()
    prepare(source, work)
    bundled = tmp_path / "bundle_video.mp4"
    _place(work / "video.mp4", bundled)
    before = sha256(bundled)
    _video(source, frames=20)
    prepare(source, work)
    assert sha256(bundled) == before
    assert os.stat(bundled).st_ino != os.stat(work / "video.mp4").st_ino
    assert not (work / "video.partial.mp4").exists()


def test_sheet_numbers_from_another_model_are_refused(tmp_path):
    import numpy as np
    source = _video(tmp_path / "a.mp4")
    np.savez(tmp_path / "numbers.npz",
             detector_id=np.asarray(f"det.ts + sheet numbers {(tmp_path / 'old.ts').resolve()}"))
    assert _conflicts(tmp_path, source, sheet_model=tmp_path / "old.ts") == []
    assert "--force numbers" in _conflicts(tmp_path, source, sheet_model=tmp_path / "new.ts")[0]
    # Without --sheet-numbers the file is not consulted.
    assert _conflicts(tmp_path, source) == []


def test_a_track_pass_that_read_the_detector_numbers_is_refused_for_sheet_numbers(tmp_path):
    import json
    source = _video(tmp_path / "a.mp4")
    (tmp_path / "track.json").write_text(json.dumps({"sam2_stride": 1}))   # recorded no source
    detector_numbers = {**SETTINGS, "number_detections": str(tmp_path / "detections.npz")}
    sheet_numbers = {**SETTINGS, "number_detections": str(tmp_path / "numbers.npz")}
    assert _conflicts(tmp_path, source, track_settings=detector_numbers) == []
    changed = _conflicts(tmp_path, source, track_settings=sheet_numbers)
    assert len(changed) == 1 and "--number-detections" in changed[0]


def test_forcing_numbers_clears_tracking_but_not_the_scan_or_teams(tmp_path):
    from scripts.process_video import clear_from, stage_outputs
    outputs = stage_outputs(tmp_path)
    for name in ("detections.npz", "numbers.npz", "team.pkl", "track.json"):
        (tmp_path / name).write_text("x")
    clear_from("numbers", outputs)
    assert (tmp_path / "detections.npz").exists() and (tmp_path / "team.pkl").exists()
    assert not any((tmp_path / n).exists() for n in ("numbers.npz", "track.json"))
