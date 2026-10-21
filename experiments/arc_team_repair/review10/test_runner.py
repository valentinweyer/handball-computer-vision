import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("review_runner", Path(__file__).with_name("runner.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_gate_requires_bundle_completion_and_no_active_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: "ARGS\n")
    work = tmp_path / "inference"
    work.mkdir()
    p = {"wait_for": [str(work)]}
    (work / "timings.json").write_text(json.dumps({"track": 20}))
    for name in ("track.json", "track_geometry.npz", "track_reads.json"):
        (work / name).touch()
    assert driver.gate(p)
    (work / "timings.json").write_text(json.dumps({"track": 20, "bundle": 1}))
    assert driver.gate(p) == []
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: "python -m scripts.process_video x.mp4\n")
    assert driver.gate(p)


def test_selection_is_repeatable_and_keeps_all_corrections():
    original = ["6001,1,1,1,10,20,1,-1,-1,-1"]
    repaired = ["6001,2,1,1,10,20,1,-1,-1,-1"]
    correction = {"pair": [1, 2], "start": 5900, "end": 5999,
                  "post": {"first": 6000, "last": 6009}, "decision": "repair"}
    events = [{"pair": [1, 2], "start": t, "end": t + 10, "decision": "third_participant"}
              for t in range(100, 12000, 500)]
    audit = {"corrections": [correction], "encounters": events}
    a = driver.selection(original, repaired, audit, 15000, 25)
    assert a == driver.selection(original, repaired, audit, 15000, 25)
    assert sum(v["kind"] == "correction" for v in a) == 1
    assert sum(v["kind"] == "unchanged_control" for v in a) == 12
    assert all(not v["start"] <= 6000 <= v["end"] for v in a if v["kind"] == "unchanged_control")


def test_existing_launch_never_starts_another_job(tmp_path, monkeypatch):
    script = tmp_path / "runner.py"
    script.write_text("stub")
    (tmp_path / "protocol.json").write_text(json.dumps({"runner_sha256": driver.sha(script)}))
    (tmp_path / "launched.json").write_text(json.dumps({"pid": 123}))
    monkeypatch.setattr(sys, "argv", ["runner.py", "--root", str(tmp_path)])
    def forbidden(*a, **k):
        raise AssertionError("must not launch or check dependencies again")
    monkeypatch.setattr(driver.subprocess, "Popen", forbidden)
    monkeypatch.setattr(driver, "gate", forbidden)
    driver.main()
