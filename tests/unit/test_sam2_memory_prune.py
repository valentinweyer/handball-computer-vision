"""Pruning SAM2's stored per-frame outputs keeps exactly what propagation can still read."""
from types import SimpleNamespace

from handball_cv.tracking.sam2_driver import PRUNE_MARGIN_FRAMES, memory_window, prune_frame_outputs


def test_window_is_the_farther_of_mask_memory_and_object_pointers():
    sam21_large = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1,
                                  max_obj_ptrs_in_encoder=16, use_obj_ptrs_in_encoder=True)
    assert memory_window(sam21_large) == 15
    no_pointers = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=3,
                                  max_obj_ptrs_in_encoder=16, use_obj_ptrs_in_encoder=False)
    assert memory_window(no_pointers) == 18
    assert memory_window(object()) is None


def test_old_non_conditioning_outputs_go_and_conditioning_ones_stay():
    state = {
        "output_dict_per_obj": {0: {"cond_frame_outputs": {0: "seed", 3: "reprompt"},
                                    "non_cond_frame_outputs": {t: t for t in range(1, 100)}}},
        "frames_tracked_per_obj": {0: {t: {"reverse": False} for t in range(100)}},
    }
    prune_frame_outputs(state, frame_idx=99, window=15)
    kept = sorted(state["output_dict_per_obj"][0]["non_cond_frame_outputs"])
    assert kept[0] == 99 - 15 - PRUNE_MARGIN_FRAMES and kept[-1] == 99
    assert state["output_dict_per_obj"][0]["cond_frame_outputs"] == {0: "seed", 3: "reprompt"}
    assert min(state["frames_tracked_per_obj"][0]) == kept[0]
