import numpy as np
import pandas as pd
import pytest

from handball_cv.visualization.minimap_positions import MAX_CORRECTION_CM, smooth_minimap_positions


def table(x, y=None):
    x = np.asarray(x, dtype=float)
    return pd.DataFrame({"frame":np.arange(len(x)), "track_id":1, "player_id":1,
                         "court_x":x, "court_y":np.zeros_like(x) if y is None else y})


def test_stationary_jitter_is_reduced_without_changing_input():
    t = table(100 + 10 * (-1.)**np.arange(41))
    before = t.copy(deep=True)
    result = smooth_minimap_positions(t, None, 25.)
    assert result[3:-3, 0].std() < t.court_x.iloc[3:-3].std() / 4
    pd.testing.assert_frame_equal(t, before)


def test_constant_velocity_has_no_trailing_or_position_bias():
    f = np.arange(50)
    t = table(100+12*f, 200-7*f)
    np.testing.assert_allclose(smooth_minimap_positions(t,None,25.), t[["court_x","court_y"]], atol=1e-10)


def test_short_tracks_and_segment_ends_remain_raw():
    t = table([0, 10, 0, 10, 0])
    np.testing.assert_array_equal(smooth_minimap_positions(t,None,25.), t[["court_x","court_y"]])
    t = table(10*(-1.)**np.arange(31))
    result=smooth_minimap_positions(t,None,25.)
    np.testing.assert_array_equal(result[:3],t[["court_x","court_y"]].iloc[:3])
    np.testing.assert_array_equal(result[-3:],t[["court_x","court_y"]].iloc[-3:])


@pytest.mark.parametrize("kind", ["gap","track","person","missing","jump","reid","suspected_id_switch"])
def test_different_segments_are_never_blended(kind):
    t=table(np.r_[np.zeros(20),np.full(20,40.)])
    events=None
    if kind=="gap":t.loc[20:,"frame"]+=5
    elif kind=="track":t.loc[20:,"track_id"]=2
    elif kind=="person":t.loc[20:,"player_id"]=2
    elif kind=="missing":t.loc[19,["court_x","court_y"]]=np.nan
    elif kind=="jump":t.loc[20:,"court_x"]=200.
    else:events=pd.DataFrame([{"frame":20,"player_id":1,"type":kind}])
    np.testing.assert_allclose(smooth_minimap_positions(t,events,25.),t[["court_x","court_y"]],atol=1e-10,equal_nan=True)


def test_a_team_label_correction_is_not_a_track_break():
    t=table(10*(-1.)**np.arange(41))
    events=pd.DataFrame([{"frame":20,"player_id":1,"type":"team_switch"}])
    np.testing.assert_array_equal(smooth_minimap_positions(t,events,25.),smooth_minimap_positions(t,None,25.))


def test_corrections_are_bounded_even_for_a_large_isolated_spike():
    t=table(np.zeros(31));t.loc[15,"court_x"]=80.
    result=smooth_minimap_positions(t,None,25.)
    assert np.linalg.norm(result-t[["court_x","court_y"]].to_numpy(),axis=1).max() <= MAX_CORRECTION_CM+1e-10


def test_a_sharp_turn_is_not_delayed_or_overshot():
    f=np.arange(-20,21,dtype=float)
    t=table(np.minimum(f,0)*10, np.maximum(f,0)*10)
    result=smooth_minimap_positions(t,None,25.)
    assert np.linalg.norm(result[20]-[0,0]) < 10.  # under 10 cm rounding at the turn
    assert np.argmin(np.linalg.norm(result,axis=1))==20
    assert (result[:,0]<=0).all() and (result[:,1]>=0).all()


def test_unsorted_rows_and_nonstandard_index_keep_their_alignment():
    t=table(10*(-1.)**np.arange(31))
    expected=smooth_minimap_positions(t,None,25.)
    permutation=np.random.default_rng(2).permutation(len(t))
    shuffled=t.iloc[permutation].copy();shuffled.index=np.arange(100,131)
    np.testing.assert_array_equal(smooth_minimap_positions(shuffled,None,25.),expected[permutation])


def test_empty_input_and_invalid_frame_rates():
    assert smooth_minimap_positions(table([]),None,25.).shape==(0,2)
    for fps in (0.,-1.,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="fps"):
            smooth_minimap_positions(table([1,2]),None,fps)
