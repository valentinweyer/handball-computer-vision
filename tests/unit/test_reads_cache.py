"""The jersey reads cache: replayed only for the reader that made it, never blank."""
import json

import pytest

from handball_cv.pipeline.track import (
    cached_read, load_reads_cache, parseq_provenance, reads_provenance_path,
)


def test_a_row_the_cache_lacks_is_read_not_replayed_blank():
    known = {"10": {"0": "15"}}
    asked = []

    def read_rows(rows, confidences):
        asked.append(list(rows))
        confidences[:] = [0.9] * len(rows)
        return ["29"] * len(rows)

    texts, confidence, added = cached_read(known, 10, [0, 2], read_rows)
    assert texts == {0: "15", 2: "29"}
    assert asked == [[2]] and added == 1
    assert confidence == {2: 0.9}           # the replayed row never had one
    assert known["10"] == {"0": "15", "2": "29"}
    # A second pass over the same frame reads nothing.
    assert cached_read(known, 10, [0, 2], read_rows)[2] == 0 and asked == [[2]]


def test_a_reader_without_confidence_leaves_none_recorded():
    texts, confidence, added = cached_read({}, 5, [1], lambda rows, conf: ["7"])
    assert texts == {1: "7"} and confidence == {} and added == 1


def test_a_cache_from_another_reader_is_refused(tmp_path):
    cache = tmp_path / "run_reads.json"
    cache.write_text(json.dumps({"10": {"0": "15"}}))
    made_by = parseq_provenance(tmp_path / "a.ckpt")
    reads_provenance_path(cache).write_text(json.dumps(made_by))
    assert load_reads_cache(cache, made_by) == ({"10": {"0": "15"}}, True)
    with pytest.raises(SystemExit, match="was made by"):
        load_reads_cache(cache, parseq_provenance(tmp_path / "b.ckpt"))


def test_a_cache_without_provenance_replays_but_stays_unvouched(tmp_path):
    cache = tmp_path / "run_reads.json"
    cache.write_text(json.dumps({"10": {"0": "15"}}))
    made_by = parseq_provenance(tmp_path / "a.ckpt")
    assert load_reads_cache(cache, made_by) == ({"10": {"0": "15"}}, False)
    assert load_reads_cache(tmp_path / "none.json", made_by) == ({}, True)


def test_parseq_provenance_names_the_checkpoint(tmp_path):
    checkpoint = tmp_path / "a.ckpt"
    checkpoint.write_bytes(b"1234")
    recorded = parseq_provenance(checkpoint)
    assert recorded["parseq_checkpoint"] == str(checkpoint.resolve())
    assert recorded["parseq_checkpoint_bytes"] == 4
