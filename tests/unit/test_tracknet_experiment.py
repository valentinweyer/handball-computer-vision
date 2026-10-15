"""Temporal heatmap experiment: label geometry, absent targets, and scoring."""
import numpy as np
import torch
from experiments.tracknet.experiment import H, W, target_heatmap, heatmap_loss, metrics


def test_heatmap_uses_xy_and_has_one_positive_centre():
    target = target_heatmap([517.2, 81.1])
    assert np.unravel_index(target.argmax(), target.shape) == (81, 517)
    assert (target == 1).sum() == 1
    assert np.isfinite(target).all()


def test_absence_is_an_all_zero_target_with_finite_loss():
    target = torch.from_numpy(target_heatmap(None))[None]
    assert not target.any()
    logits = torch.full((1, 1, H*W), -4.6, requires_grad=True)
    loss = heatmap_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad > 0).all()  # gradient descent suppresses predictions


def test_wrong_location_counts_as_both_false_positive_and_missed_ball():
    rows = [
        {'xy': [10, 20], 'score': .9, 'distance_input_px': 2.},
        {'xy': [30, 40], 'score': .9, 'distance_input_px': 50.},
        {'xy': None, 'score': .9, 'distance_input_px': None},
        {'xy': [80, 90], 'score': .1, 'distance_input_px': 1.},
    ]
    result = metrics(rows, .5)
    assert (result['tp'], result['fp'], result['fn']) == (1, 2, 2)
    assert result['false_positives_on_absent'] == 1
    assert result['precision'] == result['recall'] == 1/3
