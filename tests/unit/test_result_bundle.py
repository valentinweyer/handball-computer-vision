"""The result bundle: round trip, version guard, and the tables built from a run."""
import json

import numpy as np
import pandas as pd
import pytest

from handball_cv.court.homography import CourtFit
from handball_cv.pipeline import tables
from handball_cv.pipeline.result import TABLES, load, write_bundle
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.visualization.palette import (
    CONTESTED, GOALKEEPER, NEW_UNCERTAIN, TEAM_A, TEAM_B, TEAM_STATE, color_index_for,
)


@pytest.fixture
def files(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not really a video")
    return {"video": video}


INFO = {"source": {"fps": 25.0, "frames": 3}}


def test_round_trip_keeps_rows_and_types(tmp_path, files):
    tracks = pd.DataFrame({"frame": np.array([1, 2], dtype="int32"), "number": ["7", None]})
    out = write_bundle(tmp_path / "b", {"tracks": tracks}, files=files, info=INFO)
    result = load(out)
    assert result.tables == ["tracks"]
    pd.testing.assert_frame_equal(result.tracks, tracks)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tables"]["tracks"]["version"] == TABLES["tracks"][0]
    assert (out / "video.mp4").read_bytes() == b"not really a video"


def test_unknown_table_is_refused_on_write(tmp_path, files):
    with pytest.raises(ValueError, match="unknown table"):
        write_bundle(tmp_path / "b", {"ball": pd.DataFrame()}, files=files, info=INFO)


def test_a_known_table_at_an_unknown_version_is_refused(tmp_path, files):
    out = write_bundle(tmp_path / "b", {"tracks": pd.DataFrame({"frame": [1]})}, files=files, info=INFO)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["tables"]["tracks"]["version"] = 99
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version 99"):
        load(out)


def test_a_table_this_code_has_never_heard_of_still_loads(tmp_path, files):
    """How a later stage (ball, actions) extends a bundle without a code change here."""
    out = write_bundle(tmp_path / "b", {"tracks": pd.DataFrame({"frame": [1]})}, files=files, info=INFO)
    pd.DataFrame({"frame": [1], "x": [3.0]}).to_parquet(out / "ball.parquet", index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["tables"]["ball"] = {"file": "ball.parquet", "version": 1}
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert load(out).ball["x"].tolist() == [3.0]


def test_no_manifest_is_not_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_team_state_and_palette_index_invert():
    for index in (TEAM_A, TEAM_B, NEW_UNCERTAIN, CONTESTED, GOALKEEPER):
        team = 1 if index == TEAM_B else 0
        assert color_index_for(team, TEAM_STATE[index], index == GOALKEEPER) == index


# ---- tables from a run ----------------------------------------------------

def scaled_fit(scale: float) -> CourtFit:
    """court cm -> image px is a pure scale, so image (x, y) -> court (x/scale, y/scale)."""
    homography = np.diag([scale, scale, 1.0])
    return CourtFit(homography=homography, inverse=np.linalg.inv(homography), inliers=8,
                    source="keypoints")


@pytest.fixture
def geometry(tmp_path):
    cache = GeometryCache(width=40, height=30, total_frames=4)
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:20, 10:20] = True
    for frame in (1, 2, 3):
        cache.add(frame, [7, 9], [[10, 5, 20, 20], [0, 0, 4, 10]], [mask, mask],
                  color_index=[TEAM_A, GOALKEEPER], team_id=[0, 1])
    return GeometryCache.load(cache.save(tmp_path / "geometry.npz"))


RUN = {
    "frame_players": {"1": [7, 9], "2": [7, 9], "3": [7, 9]},
    "number_reads": [{"frame": f, "player_id": 7, "value": "15", "number_row": 0,
                      "box": [1, 2, 3, 4]} for f in (1, 2, 3)],
    "identity_events": [],
    "player_teams": {"7": 0, "9": 1},
    "player_aliases": {},
    "numbers_resolved": {"7": "15"},
    "number_votes_raw": {"7": {"15": 3}},
}


def test_tracks_rows_numbers_and_court_positions(geometry):
    fits = [CourtFit()] + [scaled_fit(0.5)] * 3
    tracks = tables.tracks_table(RUN, geometry, fits, fps=25.0, ocr_every=1)
    assert len(tracks) == 6
    p7 = tracks[tracks.track_id == 7].sort_values("frame")
    # Three agreeing reads resolve on the third: live from then on, backfilled before.
    assert p7.number_live.tolist() == [None, None, "15"]
    assert p7.number.tolist() == ["15", "15", "15"]
    first = p7.iloc[0]
    assert (first.foot_x, first.foot_y) == (15.0, 20.0)
    assert (first.court_x, first.court_y) == (30.0, 40.0)
    assert first.time_s == pytest.approx(1 / 25)
    gk = tracks[tracks.track_id == 9].iloc[0]
    assert bool(gk.is_goalkeeper) and gk.team_state is None
    assert first.team_state == "stable"


def test_mask_row_points_at_that_rows_mask(geometry):
    tracks = tables.tracks_table(RUN, geometry, [scaled_fit(1.0)] * 4, fps=25.0, ocr_every=1)
    row = tracks[(tracks.frame == 2) & (tracks.track_id == 7)].iloc[0]
    mask = geometry.row_mask(int(row.mask_row))
    ys, xs = np.nonzero(mask)
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == (10, 5, 19, 19)


def test_an_abstaining_court_fit_gives_no_position(geometry):
    tracks = tables.tracks_table(RUN, geometry, [CourtFit()] * 4, fps=25.0, ocr_every=1)
    assert tracks.court_x.isna().all() and (tracks.court_fit_source == "none").all()
    court = tables.court_table([CourtFit(), scaled_fit(2.0)])
    assert court.loc[0, ["h0", "inv8"]].isna().all()
    assert court.loc[1, "h0"] == 2.0 and court.loc[1, "inv0"] == 0.5


def test_players_and_reads(geometry):
    tracks = tables.tracks_table(RUN, geometry, [scaled_fit(1.0)] * 4, fps=25.0, ocr_every=1)
    players = tables.players_table(RUN, tracks).set_index("player_id")
    assert players.loc[7, "number"] == "15" and players.loc[7, "reads"] == 3
    assert players.loc[7, "frames_visible"] == 3 and bool(players.loc[9, "is_goalkeeper"])
    reads = tables.reads_table(RUN)
    assert reads.value.tolist() == ["15"] * 3 and reads.confidence.isna().all()


def test_keypoints_and_detections_tables():
    keypoints = tables.keypoints_table([[(0, 1.0, 2.0, 0.9), (36, 3.0, 4.0, 0.1)]])
    assert keypoints.vertex.tolist()[0] == 1 and len(keypoints) == 2
    cache = {"offsets": np.array([0, 2, 2, 3]), "boxes": np.zeros((3, 4)),
             "confidence": np.array([0.9, 0.4, 0.5]), "class_id": np.array([2, 4, 3])}
    detections = tables.detections_table(cache)
    assert detections.frame.tolist() == [0, 0, 2]
    assert detections.class_name.tolist() == ["player", "number", "referee"]


def test_a_linked_fragment_carries_its_identitys_number(tmp_path):
    """Number linking folds fragment 12 into 7: one person, never on screen twice at once."""
    cache = GeometryCache(width=40, height=30, total_frames=8)
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:20, 10:20] = True
    for frame in range(1, 7):
        player = 7 if frame <= 3 else 12
        cache.add(frame, [player], [[10, 5, 20, 20]], [mask], color_index=[TEAM_A], team_id=[0])
    geometry = GeometryCache.load(cache.save(tmp_path / "geometry.npz"))
    run = {
        "frame_players": {str(f): [7 if f <= 3 else 12] for f in range(1, 7)},
        "number_reads": [{"frame": f, "player_id": 7, "value": "15", "number_row": 0,
                          "box": [1, 2, 3, 4]} for f in (1, 2, 3)],
        "identity_events": [{"frame": 4, "type": "link", "player_id": 12, "into_player_id": 7}],
        "player_teams": {"7": 0, "12": 0},
        "player_aliases": {"12": 7},
    }
    tracks = tables.tracks_table(run, geometry, [scaled_fit(1.0)] * 8, fps=25.0, ocr_every=1)
    fragment = tracks[tracks.track_id == 12]
    assert (fragment.player_id == 7).all()
    assert fragment.number.tolist() == ["15", "15", "15"]
    # The live column is what the render drew: from the link on, through the fold.
    assert fragment.number_live.tolist() == ["15", "15", "15"]


def test_one_identity_is_never_on_two_boxes():
    canonical = tables.canonical_resolver({"12": 7, "13": 7})
    assert tables.one_place_per_identity([7, 12, 3], canonical) == [7, 12, 3]
    assert tables.one_place_per_identity([12, 13], canonical) == [12, 13]
    assert tables.one_place_per_identity([12, 3], canonical) == [7, 3]


def test_a_number_twice_in_one_team_is_withheld_from_both():
    numbers = tables.one_number_per_team(["4", "4", "4", "9", "9"], [0, 0, 1, 0, 0],
                                         [False, False, False, False, True])
    assert numbers == [None, None, "4", "9", "9"]
