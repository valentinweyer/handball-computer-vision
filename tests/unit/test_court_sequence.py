"""Clip-level court helpers: one entry per frame, in order."""
import numpy as np

from handball_cv.court.sequence import fit_court, measure_motion


def textured(shift: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 200), dtype=np.uint8)
    base = np.repeat(np.repeat(base, 4, axis=0), 4, axis=1)[:240, :400]
    return np.stack([np.roll(base, shift, axis=1)] * 3, axis=2)


def test_motion_has_no_entry_for_the_first_frame_and_finds_a_pan():
    motion = measure_motion([textured(0), textured(6)])
    assert motion[0] is None
    matrix = np.asarray(motion[1])
    assert abs(matrix[0, 2] - 6) < 0.5 and abs(matrix[1, 2]) < 0.5


def test_fit_court_returns_one_fit_per_frame_and_abstains_without_keypoints():
    fits = fit_court([[], []], [None, None], (1920, 1080))
    assert len(fits) == 2 and not any(fit.usable for fit in fits)
