import numpy as np
from PIL import Image

from handball_cv.detection import number_sheets as ns


def test_tiles_keep_their_aspect_and_wide_ones_fit_the_square():
    assert ns.tile_size(50, 100, 224) == (112, 224)
    assert ns.tile_size(200, 100, 224) == (224, 112)


def test_two_rows_of_350_fit_one_sheet_and_352_do_not():
    # 2 x 350 + one gap is exactly 704; 352 left a sheet holding one row.
    four = [(300, 350)] * 4
    assert {k for k, _, _ in ns.pack_rows(four, 350)} == {0}
    assert {k for k, _, _ in ns.pack_rows([(300, 352)] * 4, 352)} == {0, 1}


def test_rows_wrap_left_to_right_then_down():
    place = ns.pack_rows([(300, 224)] * 3, 224)
    assert place == [(0, 0, 0), (0, 304, 0), (0, 0, 228)]


def test_a_sheet_detection_maps_back_through_its_tile():
    image = Image.new("RGB", (400, 300))
    crops, context = ns.host_crops(image, np.array([[100.0, 50.0, 150.0, 150.0]]))
    sheets, tiles = ns.render(crops, 200)
    assert tiles == [(0, 0, 0, 100, 200)]   # 50x100 drawn at 2x
    # A number at (40..60, 40..80) on the sheet is (20..30, 20..40) in the crop.
    boxes, scores = ns.from_sheets([(np.array([[40.0, 40.0, 60.0, 80.0]]), np.array([0.9]))],
                                   crops, context, tiles)
    assert np.allclose(boxes, [[120.0, 70.0, 130.0, 90.0]]) and scores == [0.9]


def test_a_number_mostly_off_its_player_is_dropped():
    host = (0.0, 0.0, 10.0, 10.0)
    assert ns._on_host((5.0, 0.0, 15.0, 10.0), host)       # half on it
    assert not ns._on_host((6.0, 0.0, 16.0, 10.0), host)


def test_duplicates_keep_the_stronger_box_in_position_order():
    boxes = [(50.0, 0.0, 60.0, 10.0), (0.0, 0.0, 10.0, 10.0), (0.5, 0.0, 10.5, 10.0)]
    kept, scores = ns.deduplicate(boxes, [0.5, 0.6, 0.9])
    assert kept == [(0.5, 0.0, 10.5, 10.0), (50.0, 0.0, 60.0, 10.0)] and scores == [0.9, 0.5]


class _Detector:
    def __init__(self):
        self.calls = []

    def detect(self, frame_rgb, hosts, threshold):
        self.calls.append((len(hosts), threshold))
        x1, y1, x2, y2 = hosts[0]
        return np.array([[x1, y1, x1 + 5, y1 + 5]]), np.array([0.8])


def test_number_rows_are_replaced_and_other_rows_kept():
    cache = {
        "offsets": np.array([0, 3, 4]),
        "boxes": np.array([[10, 10, 50, 100], [12, 20, 20, 30], [60, 10, 90, 90], [0, 0, 5, 5]], float),
        "confidence": np.array([0.9, 0.7, 0.8, 0.6]),
        "class_id": np.array([2, 4, 3, 4]),   # player, number, referee | number alone
        "threshold": np.asarray(0.3), "detector_id": np.asarray("det.ts"), "stride": np.asarray(1),
    }
    decoded = []
    detector = _Detector()
    out = ns.replace_number_rows(cache, lambda i: decoded.append(i) or np.zeros((120, 120, 3), np.uint8),
                                 detector, detector_id="sheets.ts")
    assert decoded == [0]                       # frame 1 has no player to cut out
    assert detector.calls == [(1, 0.3)]         # the player alone hosts, at the cache threshold
    assert out["offsets"].tolist() == [0, 3, 3]
    assert out["class_id"].tolist() == [2, 3, 4]
    assert np.allclose(out["boxes"][2], [10, 10, 15, 15])
    assert str(out["detector_id"]) == "det.ts + sheet numbers sheets.ts"
    assert int(out["stride"]) == 1
