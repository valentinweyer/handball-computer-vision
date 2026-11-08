import numpy as np
import pytest

from handball_cv.jersey.identity import small_crop_read_ok
from handball_cv.teams.model import MIN_CROP_AREA


@pytest.mark.parametrize("text, confidence, ok", [
    ("24", 0.95, True),     # two digits clear the lower bar
    ("24", 0.85, False),
    ("4", 0.995, True),     # one digit needs the higher one: 24 read as 4 is the failure
    ("4", 0.98, False),
    ("", 1.0, False),
    ("ab", 1.0, False),     # not a jersey number
    ("24", None, False),    # a reader that reports no confidence cannot qualify
])
def test_small_crop_reads_need_the_stricter_rule(text, confidence, ok):
    assert small_crop_read_ok(text, confidence) is ok


def _frame_with_digit(height, width):
    frame = np.full((200, 200, 3), 40, np.uint8)
    frame[50:50 + height, 50:50 + width] = 40
    # High contrast, sharp edges: a white stroke on dark cloth.
    frame[50 + height // 4:50 + 3 * height // 4, 50 + width // 3:50 + 2 * width // 3] = 230
    return frame, np.array([50, 50, 50 + width, 50 + height], float)


def test_a_crop_small_only_in_area_is_small_and_not_readable():
    from handball_cv.pipeline.track import readable_number_crop, small_number_crop
    frame, box = _frame_with_digit(18, 16)
    assert 18 * 16 < MIN_CROP_AREA
    assert not readable_number_crop(frame, box)
    assert small_number_crop(frame, box)


def test_a_full_size_crop_is_readable_not_small():
    from handball_cv.pipeline.track import readable_number_crop, small_number_crop
    frame, box = _frame_with_digit(30, 24)
    assert readable_number_crop(frame, box)
    assert not small_number_crop(frame, box)


def test_too_thin_or_flat_crops_stay_refused():
    from handball_cv.pipeline.track import small_number_crop
    frame, box = _frame_with_digit(30, 8)              # under the minimum side
    assert not small_number_crop(frame, box)
    flat = np.full((200, 200, 3), 40, np.uint8)        # no contrast at all
    assert not small_number_crop(flat, np.array([50, 50, 66, 68], float))
