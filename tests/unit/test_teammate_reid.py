"""Teammate re-ID prototype: mining provenance, leakage, masking, abstention, isolation."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.teammate_reid.evaluation import (
    build_queries, decide, operating_point, outcomes, runtime_point, score_query)
from experiments.teammate_reid.masking import FILL, apply_mask, crop_with_digits, digit_patches
from experiments.teammate_reid.mining import (
    attach_read_tracks, own_verdict, split_fragments, vouch_fragments)
from experiments.teammate_reid.train import batch_hard_triplet


def _tracks(spec):
    """spec: {track_id: (frames, team, box, goalkeeper)} -> tracks frame."""
    rows = []
    for track_id, (frames, team, box, keeper) in spec.items():
        for f in frames:
            rows.append({"frame": f, "track_id": track_id, "player_id": track_id, "team": team,
                         "team_state": "stable", "is_goalkeeper": keeper, "number": None,
                         "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
    return pd.DataFrame(rows)


def _reads(track_id, frames, value):
    return [{"frame": f, "player_id": track_id, "track_id": track_id, "value": value} for f in frames]


def _mine(tracks, reads, events=()):
    frags = split_fragments("v", "m", tracks, list(events))
    vouch_fragments(frags, reads, tracks, list(events))
    return frags


# --- mining / provenance ----------------------------------------------------

def test_fragment_is_labelled_only_by_its_own_reads():
    tracks = _tracks({1: (range(0, 100), 0, (0, 0, 50, 150), False),
                      2: (range(200, 300), 0, (100, 0, 150, 150), False)})
    tracks["number"] = "7"  # a displayed/backfilled family number on both tracks
    reads = _reads(1, range(0, 100, 5), "7")  # only track 1 was ever read
    frags = {f.track_id: f for f in _mine(tracks, reads)}
    assert frags[1].status == "accepted" and frags[1].number == "7"
    assert frags[2].status == "excluded" and frags[2].number is None
    assert "no_own_reads" in frags[2].reasons


def test_identity_break_and_gap_split_fragments_and_reads_do_not_cross():
    frames = list(range(0, 100)) + list(range(100, 200))
    tracks = _tracks({1: (frames, 0, (0, 0, 50, 150), False)})
    tracks["number"] = "7"
    events = [{"frame": 100, "type": "reid", "player_id": 1, "fragment_id": 1}]
    reads = _reads(1, range(0, 100, 5), "7")
    frags = sorted(_mine(tracks, reads, events), key=lambda f: f.start)
    assert [(f.start, f.end) for f in frags] == [(0, 99), (100, 199)]
    assert frags[0].status == "accepted"
    assert frags[1].opened_by == "reid" and frags[1].number is None  # earlier reads cannot vouch


def test_suspected_switch_rival_number_and_goalkeeper_are_excluded():
    tracks = _tracks({1: (range(0, 200), 0, (0, 0, 50, 150), False),
                      3: (range(0, 100), 1, (300, 0, 350, 150), True)})
    tracks["number"] = "7"
    events = [{"frame": 100, "type": "suspected_id_switch", "player_id": 1}]
    reads = _reads(1, range(0, 100, 5), "7") + _reads(3, range(0, 100, 5), "7")
    frags = _mine(tracks, reads, events)
    assert all(f.status == "excluded" for f in frags)
    assert any("suspected_id_switch" in f.reasons for f in frags if f.track_id == 1)
    assert "goalkeeper" in next(f for f in frags if f.track_id == 3).reasons
    verdict, reasons = own_verdict(["21"] * 6 + ["11"] * 2)
    assert verdict == "21" and any(r.startswith("rival_number") for r in reasons)
    # a trailing digit backs the full number rather than competing with it
    assert own_verdict(["21"] * 6 + ["1"] * 3)[1] == []


def test_withheld_display_and_co_live_duplicates_are_excluded():
    tracks = _tracks({1: (range(0, 100), 0, (0, 0, 50, 150), False),
                      2: (range(0, 100), 0, (100, 0, 150, 150), False)})
    tracks["number"] = "7"
    reads = _reads(1, range(0, 100, 5), "7") + _reads(2, range(0, 100, 5), "7")
    frags = _mine(tracks, reads)
    assert all("co_live_duplicate_number" in f.reasons for f in frags)
    tracks["number"] = None  # the final bundle withheld the number (arbitration)
    frags = _mine(tracks, _reads(1, range(0, 100, 5), "7"))
    assert "number_withheld" in next(f for f in frags if f.track_id == 1).reasons


def test_legacy_reads_attach_only_to_a_single_containing_track():
    tracks = _tracks({1: ([10], 0, (0, 0, 50, 150), False),
                      2: ([10], 0, (20, 0, 80, 150), False)})
    legacy = [{"frame": 10, "player_id": 1, "value": "7", "box": [5, 20, 15, 30]},   # only in 1
              {"frame": 10, "player_id": 1, "value": "7", "box": [25, 20, 45, 30]}]  # in both
    out, provenance, tally = attach_read_tracks(legacy, tracks, aliases={})
    assert provenance == "reconstructed"
    assert [r["track_id"] for r in out] == [1] and tally["ambiguous"] == 1
    # a box that belongs to a different identity than the read was filed under
    other = [{"frame": 10, "player_id": 2, "value": "7", "box": [5, 20, 15, 30]}]
    assert attach_read_tracks(other, tracks, aliases={})[0] == []
    assert attach_read_tracks(other, tracks, aliases={"1": 2})[0][0]["track_id"] == 1


# --- digit masking -----------------------------------------------------------

def test_digit_patch_covers_the_number_box_and_hides_digit_count():
    crop = np.full((160, 64, 3), 200, np.uint8)
    one, _ = digit_patches(64, 160, [(28, 40, 36, 52)], "a")
    two, _ = digit_patches(64, 160, [(24, 40, 40, 52)], "a")
    for (px1, py1, px2, py2), (x1, y1, x2, y2) in ((one[0], (28, 40, 36, 52)), (two[0], (24, 40, 40, 52))):
        assert px1 <= x1 and py1 <= y1 and px2 >= x2 and py2 >= y2
    size = lambda p: (p[2] - p[0], p[3] - p[1])
    assert size(one[0]) == size(two[0])          # one digit and two digits look the same
    masked, info = apply_mask(crop, [(24, 40, 40, 52)], "digits", "a")
    assert (masked[40:52, 24:40] == FILL).all() and info["grown"] == 0


def test_crops_without_a_number_get_a_decoy_and_strict_hides_the_torso():
    crop = np.full((160, 64, 3), 200, np.uint8)
    masked, info = apply_mask(crop, [], "digits", "seed")
    assert info["decoy"] and (masked == FILL).all(-1).sum() > 0
    again, _ = apply_mask(crop, [], "digits", "seed")
    assert (again == masked).all()                # deterministic per crop
    strict, _ = apply_mask(crop, [], "strict", "seed")
    assert (strict[int(0.2 * 160):int(0.6 * 160), int(0.2 * 64):int(0.8 * 64)] == FILL).all()
    raw, _ = apply_mask(crop, [(1, 1, 5, 5)], "raw", "seed")
    assert (raw == crop).all()


def test_crop_coordinates_move_digit_boxes_into_the_crop():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop, local = crop_with_digits(frame, [50, 10, 90, 90], [(60, 20, 70, 30), (0, 0, 5, 5)])
    assert crop.shape[:2] == (80, 40) and local == [(10, 10, 20, 20)]


# --- queries: leakage and open set -------------------------------------------

def _frag(key, identity, team, start, end, track):
    return {"key": key, "identity": identity, "team": team, "start": start, "end": end,
            "track_id": track, "video": "v"}


def test_queries_are_time_separated_and_never_see_themselves():
    frags = [_frag("a1", "A", 0, 0, 100, 1), _frag("b1", "B", 0, 0, 60, 2),
             _frag("a2", "A", 0, 120, 200, 3),      # within min_sep of a1: a1 not usable
             _frag("a3", "A", 0, 400, 500, 1)]
    crops = {k: [i * 10 + j for j in range(4)] for i, k in enumerate(["a1", "b1", "a2", "a3"])}
    queries = build_queries(frags, crops, min_sep=50)
    q2 = next(q for q in queries if q.fragment == "a2" and not q.synthetic_unknown)
    assert "A" not in q2.galleries and not q2.known       # a natural unknown, not a leak
    q3 = next(q for q in queries if q.fragment == "a3" and not q.synthetic_unknown)
    assert q3.known and set(q3.galleries["A"]) == set(crops["a1"]) | set(crops["a2"])
    assert not set(q3.crops) & {c for ids in q3.galleries.values() for c in ids}
    assert not q3.reentry                                 # a1 shares tracker id 1
    synth = next(q for q in queries if q.fragment == "a3" and q.synthetic_unknown)
    assert "A" not in synth.galleries and not synth.known


def test_abstention_single_candidate_and_unknown_false_matches():
    assert decide({"A": 0.9})[2] == -np.inf               # nothing to beat: margin gate abstains
    rows = [{"known": True, "correct": True, "score": 0.9, "margin": 0.1, "candidates": 2, "duration_s": 4.0},
            {"known": True, "correct": False, "score": 0.8, "margin": 0.01, "candidates": 2, "duration_s": 2.0},
            {"known": False, "correct": False, "score": 0.85, "margin": -np.inf, "candidates": 1,
             "duration_s": 3.0}]
    p = operating_point(rows, tau=0.7, margin=0.05)
    assert p["correct_links"] == 1 and p["accepted"] == 1 and p["precision"] == 1.0
    r = runtime_point(rows)                                  # runtime skips margin for one candidate
    assert r["false_matches_unknown"] == 1 and r["wrong_link_seconds"] == 3.0
    assert operating_point(rows, tau=0.95, margin=0.0)["accepted"] == 0


def test_scoring_is_read_only_proposals_never_feed_back():
    frags = [_frag("a1", "A", 0, 0, 100, 1), _frag("b1", "B", 0, 0, 100, 2),
             _frag("a2", "A", 0, 300, 400, 3)]
    crops = {"a1": [0, 1], "b1": [2, 3], "a2": [4, 5, 6]}
    emb = np.eye(8, dtype=np.float32)[[0, 0, 1, 1, 0, 0, 0]]
    queries = build_queries(frags, crops, min_sep=50)
    before = copy.deepcopy([q.__dict__ for q in queries])
    first = outcomes(queries, emb, "gallery_agg")
    second = outcomes(queries, emb, "gallery_agg")
    assert first == second and [q.__dict__ for q in queries] == before
    assert score_query(queries[0], emb, "ema_single")["A"] == pytest.approx(1.0)


def test_triplet_negatives_are_teammates_only():
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1.0, 0.05], [0, 1.0], [0, 1.0]]), dim=1)
    ident = torch.tensor([0, 0, 1, 2])
    # identities 1 and 2 are in another group: no teammate negative for 0, so no loss
    assert batch_hard_triplet(z, ident, torch.tensor([0, 0, 1, 1])) == 0
    near = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0.01], [0, 1.0]])
    z = torch.nn.functional.normalize(near, dim=1)
    loss = batch_hard_triplet(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
    assert loss > 0


def test_review_exclusions_remove_fragments_and_their_crops(tmp_path):
    import json
    from experiments.teammate_reid.benchmark import load_reviewed
    frags = [{"key": k, "identity": i, "status": s} for k, i, s in
             (("v:t1:0-9", "A", "accepted"), ("v:t2:0-9", "B", "accepted"), ("v:t3:0-9", "C", "excluded"))]
    crops = [{"crop_id": n, "fragment": k} for n, k in enumerate(["v:t1:0-9", "v:t2:0-9", "v:t3:0-9"])]
    (tmp_path / "fragments.json").write_text(json.dumps(frags))
    (tmp_path / "crops.json").write_text(json.dumps(crops))
    (tmp_path / "review.json").write_text(json.dumps({"excluded": {"v:t2:0-9": "two people"}}))
    kept_crops, kept, _ = load_reviewed(tmp_path)
    assert [f["key"] for f in kept] == ["v:t1:0-9"] and [c["crop_id"] for c in kept_crops] == [0]
