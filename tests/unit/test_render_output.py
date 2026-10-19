"""Bundle rendering pieces: the x264 writer, the court minimap, court fits from the table."""
import cv2
import numpy as np
import pytest

from handball_cv.court.homography import CourtFit
from handball_cv.pipeline.tables import court_table
from handball_cv.visualization.court_minimap import CourtMinimap
from handball_cv.visualization.video_writer import FfmpegWriter
from scripts.render_result import court_fits

PALETTE = [(255, 190, 0), (0, 120, 255), (0, 220, 255), (0, 0, 220), (180, 180, 180)]


def test_writer_round_trip(tmp_path):
    path = tmp_path / "out.mp4"
    with FfmpegWriter(path, 64, 48, 25) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    capture = cv2.VideoCapture(str(path))
    frames = 0
    while capture.read()[0]:
        frames += 1
    assert frames == 10
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64


def test_writer_refuses_a_wrong_frame_size(tmp_path):
    writer = FfmpegWriter(tmp_path / "out.mp4", 64, 48, 25)
    with pytest.raises(ValueError):
        writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.close()


@pytest.fixture(scope="module")
def minimap():
    pytest.importorskip("sports")
    return CourtMinimap(460, PALETTE)


def identity_fit() -> CourtFit:
    return CourtFit(homography=np.eye(3), inverse=np.eye(3), inliers=8, source="keypoints")


def test_court_corners_land_on_the_drawn_court(minimap):
    corners = minimap.to_pixels([(0, 0), (4000, 2000)])
    pad, scale = minimap.pad, minimap.scale
    assert corners[0] == pytest.approx((pad, 2000 * scale + pad))
    assert corners[1] == pytest.approx((4000 * scale + pad, pad))
    width, height = minimap.size
    assert corners[1][0] < width and corners[0][1] < height


def test_a_player_is_drawn_in_their_palette_colour(minimap):
    image = minimap.draw(identity_fit(), (1920, 1080), [(2000.0, 1000.0, 1, "7")])
    x, y = (int(round(v)) for v in minimap.to_pixels([(2000, 1000)])[0])
    assert tuple(image[y, x]) == PALETTE[1]


def test_no_players_when_the_court_fit_abstains(minimap):
    image = minimap.draw(CourtFit(), (1920, 1080), [(2000.0, 1000.0, 1, "7")])
    x, y = (int(round(v)) for v in minimap.to_pixels([(2000, 1000)])[0])
    assert tuple(image[y, x]) != PALETTE[1]
    assert image.mean() < minimap.background.mean()


def test_players_far_off_the_court_are_not_drawn(minimap):
    far = minimap.draw(identity_fit(), (1920, 1080), [(9000.0, 1000.0, 1, None)])
    none = minimap.draw(identity_fit(), (1920, 1080), [])
    assert np.array_equal(far, none)


@pytest.mark.parametrize("corner", ["top-right", "top-left", "bottom-right", "bottom-left"])
def test_composite_puts_the_minimap_in_the_corner(minimap, corner):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tile = np.full((*minimap.background.shape[:2], 3), 77, dtype=np.uint8)
    minimap.composite(frame, tile, corner=corner, margin=10)
    y = 10 + 5 if corner.startswith("top") else 1080 - 10 - 5
    x = 10 + 5 if corner.endswith("left") else 1920 - 10 - 5
    assert tuple(frame[y, x]) == (77, 77, 77)


def test_court_fits_come_back_from_the_table():
    h = np.array([[2.0, 0, 5], [0, 2.0, 7], [0, 0, 1]])
    fits = [CourtFit(source="none", age=3),
            CourtFit(homography=h, inverse=np.linalg.inv(h), inliers=9, source="keypoints")]
    back = court_fits(court_table(fits))
    assert not back[0].usable and back[0].age == 3
    assert np.allclose(back[1].homography, h) and back[1].inliers == 9
    assert back[1].to_court(np.array([[9.0, 11.0]])) == pytest.approx(np.array([[2.0, 2.0]]))


def test_writer_reports_a_dead_encoder_instead_of_hanging(tmp_path):
    """Frames go to ffmpeg from a thread; its failure must still reach the caller."""
    writer = FfmpegWriter(tmp_path / "out.mp4", 64, 48, 25)
    writer._process.kill()
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        for _ in range(200):  # more than the queue holds: a lost error would block here
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.close()
