"""`LabelDrawer` must draw exactly what `sv.RichLabelAnnotator` drew.

It replaced the annotator for speed alone, and it re-derives the annotator's
layout to do so -- so this pins every pixel against the installed supervision,
and a supervision upgrade that moves its labels fails here, not in a render.
"""
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from handball_cv.visualization.labels import LabelDrawer
from handball_cv.visualization.palette import PALETTE_HEX

REPO_FONT = Path(__file__).resolve().parents[2] / "notebooks/fonts/ConcertOne-Regular.ttf"
FONTS = [font for font in (
    REPO_FONT,
    Path(pytest.importorskip("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf",
) if font.is_file()]


def rich_annotator(font, palette, width):
    return sv.RichLabelAnnotator(
        font_path=str(font), font_size=round(34 * max(width / 1920.0, 1.0)),
        color=palette, text_color=sv.Color.WHITE, text_position=sv.Position.BOTTOM_CENTER,
        text_offset=(0, 10), color_lookup=sv.ColorLookup.INDEX,
    )


@pytest.mark.parametrize("font", FONTS, ids=lambda f: f.stem)
@pytest.mark.parametrize("width", [1920, 3840])  # 3840: big glyphs, accents past the box
def test_matches_rich_label_annotator(font, width):
    height = width * 9 // 16
    rng = np.random.default_rng(width)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    palette = sv.ColorPalette.from_hex(PALETTE_HEX)
    labels = ["#7", "P123", "#24 Ölçü-Ĝåȳ", "#9", "#11", "P4", "#1 Müller", "#88", "#3"]
    boxes = np.array([
        [100.4, 100.0, 160.7, 290.3],                        # ordinary
        [130.0, 120.0, 190.0, 300.5],                        # overlaps the first label
        [600.0, 300.0, 660.0, 480.0],                        # accents above, descenders below
        [-50.0, 200.0, 10.0, 380.0],                         # clipped at the left edge
        [width - 20.0, 200.0, width + 30.0, 380.0],          # clipped at the right edge
        [800.0, height - 60.0, 860.0, height - 5.0],         # label hangs off the bottom
        [900.0, -300.0, 960.0, -40.0],                       # label partly above the top
        [1000.0, height + 50.0, 1060.0, height + 200.0],     # entirely off frame
        [1200.0, 400.0, 1260.0, 580.0],
    ])
    colors = np.arange(len(labels)) % len(PALETTE_HEX)

    expected = rich_annotator(font, palette, width).annotate(
        frame.copy(), sv.Detections(xyxy=boxes, class_id=colors),
        labels=labels, custom_color_lookup=colors)
    got = frame.copy()
    LabelDrawer(font, palette, width).draw(got, boxes, labels, colors)

    assert (got != frame).any()
    np.testing.assert_array_equal(got, expected)


def test_a_missing_font_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        LabelDrawer(tmp_path / "missing.ttf", sv.ColorPalette.from_hex(PALETTE_HEX), 1920)
