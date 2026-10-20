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


def test_after_galleries_keep_the_gap_and_drop_the_query_track():
    frags = [_frag("q", "A", 0, 200, 300, 7), _frag("b0", "B", 0, 0, 100, 2),
             _frag("a_soon", "A", 0, 320, 400, 4),   # starts within min_sep after the query
             _frag("a_same", "A", 0, 500, 600, 7),   # the query's own track continuing
             _frag("a_late", "A", 0, 700, 800, 5)]
    crops = {k: [i * 10 + j for j in range(4)] for i, k in enumerate(["q", "b0", "a_soon", "a_same", "a_late"])}

    def query(**kw):
        return next(x for x in build_queries(frags, crops, min_sep=50, **kw)
                    if x.fragment == "q" and not x.synthetic_unknown)

    assert not query().known                                     # default: earlier fragments only
    both = query(after=True)
    assert set(both.galleries["A"]) == set(crops["a_same"]) | set(crops["a_late"])
    strict = query(after=True, exclude_track=True)
    assert set(strict.galleries["A"]) == set(crops["a_late"]) and strict.known


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


# --- stage 1: transfer to held-out matches at locked thresholds -------------

def _row(identity, known, correct, score, margin, synthetic=False, match="M1", duration=2.0):
    return {"fragment": f"{identity}:{score}", "identity": identity, "known": known,
            "synthetic_unknown": synthetic, "reentry": False, "best": "x", "score": score,
            "margin": margin, "correct": correct, "candidates": 3, "duration_s": duration,
            "match": match}


def test_locked_thresholds_are_never_rewritten(tmp_path, monkeypatch):
    from experiments.teammate_reid import transfer
    monkeypatch.setattr(transfer, "THRESHOLDS", tmp_path / "thresholds.json")
    (tmp_path / "thresholds.json").write_text('{"points": {}}')
    with pytest.raises(SystemExit, match="locked"):
        transfer.lock()
    loaded = transfer.load_thresholds()
    assert loaded["points"] == {} and len(loaded["file_sha256"]) == 64


def test_natural_and_synthetic_unknown_rates_are_reported_apart():
    from experiments.teammate_reid.transfer import point_with_parts
    rows = [_row("a", True, True, 0.95, 0.05),          # accepted, correct
            _row("b", True, False, 0.95, 0.05),         # accepted, wrong known
            _row("c", True, True, 0.80, 0.05),          # below the score floor
            _row("d", False, False, 0.95, 0.05),        # natural unknown, false match
            _row("e", False, False, 0.50, 0.00),        # natural unknown, abstained
            _row("a", False, False, 0.95, 0.001, synthetic=True)]  # synthetic, margin gate holds
    p = point_with_parts(rows, tau=0.9, margin=0.01)
    assert p["coverage"] == pytest.approx(1 / 3)
    assert p["precision"] == pytest.approx(1 / 3)       # 1 correct of 3 accepted
    assert p["unknown_false_match_rate"] == pytest.approx(1 / 3)
    assert p["unknown_false_match_rate_natural"] == pytest.approx(1 / 2)
    assert p["unknown_false_match_rate_synthetic"] == 0.0
    assert (p["known_queries"], p["unknown_natural_queries"], p["unknown_synthetic_queries"]) == (3, 2, 1)


def test_verdict_needs_every_bound_and_flags_natural_unknowns():
    from experiments.teammate_reid.transfer import CRITERION, verdict
    good = {"precision": 0.93, "unknown_false_match_rate": 0.03, "coverage": 0.25,
            "unknown_false_match_rate_natural": 0.10}
    v = verdict(good, CRITERION)
    assert v["pass"] and v["natural_unknown_rate_above_bound"]
    assert not verdict({**good, "coverage": 0.19}, CRITERION)["pass"]
    assert not verdict({**good, "precision": float("nan")}, CRITERION)["pass"]


def test_clustered_ci_resamples_whole_identities_within_matches():
    from experiments.teammate_reid.transfer import clustered_ci
    # one identity always right, one always wrong: a clustered resample can only
    # draw precision 0, 0.5 or 1 -- never a value from splitting an identity's rows
    rows = [_row("good", True, True, 0.95, 0.05) for _ in range(5)] + \
           [_row("bad", True, False, 0.95, 0.05) for _ in range(5)]
    ci = clustered_ci(rows, 0.9, 0.01, reps=300)
    assert ci["precision"][0] == 0.0 and ci["precision"][1] == 1.0


def test_held_out_evaluation_refuses_development_matches(tmp_path, monkeypatch):
    import json
    from experiments.teammate_reid import transfer
    monkeypatch.setattr(transfer, "THRESHOLDS", tmp_path / "thresholds.json")
    (tmp_path / "thresholds.json").write_text('{"points": {}, "primary": "p", "criterion": {}}')
    frag = {"key": "k", "status": "accepted", "identity": "Kiel-Lemgo|v|team0|#1", "match": "Kiel-Lemgo",
            "video": "v", "team": 0}
    (tmp_path / "fragments.json").write_text(json.dumps([frag]))
    (tmp_path / "crops.json").write_text("[]")
    with pytest.raises(SystemExit, match="development matches"):
        transfer.evaluate(tmp_path)


def test_held_out_windows_share_no_team_with_development_data():
    from experiments.teammate_reid.heldout import WINDOWS
    from experiments.teammate_reid.run import HELDOUT, VIDEOS
    development = ("Kiel", "Lemgo", "BHC", "FAG", "Melsungen", "Berlin", "Ber")
    for name, match, source, _ in WINDOWS:
        assert not any(team.lower() in (name + match + source).lower() for team in development), name
    assert not {s.name for s in HELDOUT} & {s.name for s in VIDEOS}


# --- leave-one-match-out calibration check ----------------------------------

def test_calibration_stats_read_scores_only_never_labels():
    from experiments.teammate_reid.calibration import calibrate, video_stats
    rows = [{**_row("a", True, True, 0.9, 0.05), "video": "v", "others": [0.80, 0.70]},
            {**_row("b", False, False, 0.85, 0.02), "video": "v", "others": [0.75, 0.72]},
            {**_row("a", False, False, 0.9, 0.01, synthetic=True), "video": "v", "others": [9.0]}]
    stats = video_stats(rows)
    relabelled = [{**r, "identity": "zz", "known": not r["known"], "correct": False} for r in rows]
    assert video_stats(relabelled) == stats                 # labels cannot move the baseline
    assert stats["v"][0] == pytest.approx(0.735)            # synthetic duplicates are left out
    cal = calibrate(rows, stats)
    assert cal[0]["score"] > cal[1]["score"] and cal[0]["identity"] == "a"


def test_selected_point_respects_both_bounds_on_training_rows():
    from experiments.teammate_reid.calibration import select_point
    rows = [_row(f"k{i}", True, i >= 2, 0.80 + 0.02 * i, 0.05) for i in range(10)] + \
           [_row(f"u{i}", False, False, 0.95, 0.05) for i in range(2)] + \
           [_row(f"n{i}", False, False, 0.70, 0.05) for i in range(40)]
    p = select_point(rows, target=0.9, grid=40)
    assert p is not None and p["precision"] >= 0.9 and p["unknown_false_match_rate"] <= 0.05
    # the two low-scoring wrong links and both confident unknowns are gated out
    assert p["wrong_links_known"] == 0 and p["false_matches_unknown"] == 0
    # nothing meets an impossible bound: abstain rather than return a point
    assert select_point([_row("w", True, False, 0.99, 0.05)], target=0.9, grid=10) is None


def test_candidate_z_uses_gallery_only_never_the_query_label():
    from experiments.teammate_reid.candidate_calibration import query_row
    from experiments.teammate_reid.evaluation import Query
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T
    # three gallery players (A, B, C), two fragments each, all ending before the query at frame 1000
    frags, crops = {}, {}
    for n, ident in enumerate("AABBCC"):
        key = f"g{n}"
        frags[key] = {"key": key, "identity": ident, "team": 0, "start": 100 * n, "end": 100 * n + 50}
        crops[key] = list(range(5 * n, 5 * n + 5))
    galleries = {i: [c for k, f in frags.items() if f["identity"] == i for c in crops[k]] for i in "ABC"}
    q = Query("v", 0, "A", "q", 1000, 1100, list(range(30, 38)), galleries, {}, True, True)
    decision = query_row(q, frags, crops, sim)
    relabelled = query_row(Query(**{**q.__dict__, "identity": "C", "known": False}), frags, crops, sim)
    assert decision == relabelled                          # the query's label never enters
    assert decision["candidates"] == 3 and np.isfinite(decision["score"])
    # a later fragment is not yet in the gallery and must not join the impostor profile
    frags["late"] = {"key": "late", "identity": "B", "team": 0, "start": 1050, "end": 1200}
    crops["late"] = list(range(38, 40))
    assert query_row(q, frags, crops, sim) == decision


def test_sealed_windows_share_no_team_with_any_development_or_test_window():
    from experiments.teammate_reid.heldout import DEV_WINDOWS, SEALED_WINDOWS, WINDOWS
    teams = lambda windows: {t.lower() for _, match, _, _ in windows for t in match.split("-")}
    development = teams(DEV_WINDOWS) | teams(WINDOWS) | {"kiel", "lemgo", "bhc", "fag", "melsungen", "berlin"}
    assert not teams(SEALED_WINDOWS) & development
    names = [w[0] for w in DEV_WINDOWS + SEALED_WINDOWS + WINDOWS]
    assert len(names) == len(set(names))


def test_sealed_errors_and_development_guard(tmp_path):
    import json
    from experiments.teammate_reid import sealed
    gate = (0.9, 0.01, 0)
    base = {"score": 0.95, "margin": 0.05, "best": "a", "identity": "a", "known": True}
    assert not sealed.is_error(base, gate)                                 # correct link
    assert sealed.is_error({**base, "best": "b"}, gate)                    # wrong teammate
    assert sealed.is_error({**base, "known": False}, gate)                 # any link for an unknown
    assert not sealed.is_error({**base, "known": False, "score": 0.5}, gate)   # abstained
    lock = json.loads(sealed.LOCK.read_text())
    frag = {"key": "k", "status": "accepted", "identity": "x", "match": lock["development_matches"][0],
            "video": "v", "team": 0}
    (tmp_path / "fragments.json").write_text(json.dumps([frag]))
    (tmp_path / "crops.json").write_text("[]")
    with pytest.raises(SystemExit, match="development matches"):
        sealed.evaluate(tmp_path, "t")


def test_split_is_disjoint_and_selection_never_sees_test_rows():
    from experiments.teammate_reid import split_test as st
    assert not set(st.TUNING) & set(st.TEST)
    rows = {"f": [{"match": m} for m in st.TUNING + st.TEST]}
    kept = st.tuning_rows(rows)["f"]
    assert {r["match"] for r in kept} == set(st.TUNING)


def test_stage2_v2_galleries_use_both_sides_but_never_the_segment_track():
    from experiments.teammate_reid.stage2 import MIN_SEP, gallery_allowed

    seg = {"start": 1000, "end": 1200, "track_id": 7}
    early = {"start": 0, "end": 1000 - MIN_SEP, "track_id": 3}
    late = {"start": 1200 + MIN_SEP, "end": 1500, "track_id": 3}
    near = {"start": 1200 + MIN_SEP - 1, "end": 1500, "track_id": 3}
    own = {"start": 1200 + MIN_SEP, "end": 1500, "track_id": 7}
    assert gallery_allowed(early, seg, v2=False) and not gallery_allowed(late, seg, v2=False)
    assert gallery_allowed(early, seg, v2=True) and gallery_allowed(late, seg, v2=True)
    assert not gallery_allowed(near, seg, v2=True) and not gallery_allowed(own, seg, v2=True)
    assert gallery_allowed({**early, "track_id": 7}, seg, v2=False)   # v1 kept its own-track galleries


def test_coverage_baseline_rule_is_the_split_v2_lock():
    import json
    from pathlib import Path
    from experiments.teammate_reid.coverage_rules import V2

    lock = json.loads((Path("experiments/teammate_reid/locked/split_v2.json")).read_text())
    assert (V2.tau, V2.margin) == tuple(lock["gate"])
    assert V2.halves == lock["halves_guard"] and V2.single_tau is None and not V2.co_ignore_duplicates
