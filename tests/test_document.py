"""Step 0: the document is found in the photo and flattened to its real size."""

import numpy as np
import pytest

from scan_quality.scoring.document import (
    EDGE_TRIM_PX,
    PX_PER_MM,
    detect_document,
    order_corners,
    rectify,
    whole_frame_detection,
)
from tests import fixtures


def test_document_on_a_dark_surface_is_found():
    page = fixtures.a4_page()
    detection = detect_document(fixtures.photo_of(page, margin_frac=0.1))
    assert detection is not None
    margin_x, margin_y = int(page.shape[1] * 0.1), int(page.shape[0] * 0.1)
    expected = np.array([
        [margin_x, margin_y],
        [margin_x + page.shape[1] - 1, margin_y],
        [margin_x + page.shape[1] - 1, margin_y + page.shape[0] - 1],
        [margin_x, margin_y + page.shape[0] - 1],
    ])
    # Corners come from a downscaled copy, so a few pixels of error is expected.
    assert np.abs(detection.corners - expected).max() < 6


def test_page_half_in_deep_shadow_is_still_found_whole():
    """A shadowed half is darker than the lit half but brighter than the table."""
    page = fixtures.a4_page()
    detection = detect_document(fixtures.photo_of(fixtures.hard_shadow(page, darkness=0.45)))
    assert detection is not None
    height = detection.corners[3, 1] - detection.corners[0, 1]
    assert height == pytest.approx(page.shape[0], rel=0.01)


@pytest.mark.parametrize(
    "photo",
    [
        pytest.param(lambda: fixtures.photo_of(fixtures.a4_page(), surface_value=240), id="white_on_white"),
        pytest.param(lambda: fixtures.sensor_noise(np.full((1000, 800, 3), 70, np.uint8)), id="no_document"),
    ],
)
def test_nothing_to_find_returns_none(photo):
    assert detect_document(photo()) is None


def test_rectified_size_matches_physical_size():
    detection = detect_document(fixtures.photo_of(fixtures.a4_page()))
    document = rectify(fixtures.photo_of(fixtures.a4_page()), detection, "a4")
    assert document.shape[1] == round(210 * PX_PER_MM) - 2 * EDGE_TRIM_PX
    assert document.shape[0] == round(297 * PX_PER_MM) - 2 * EDGE_TRIM_PX


def test_landscape_cheque_is_rectified_landscape():
    cheque = fixtures.clean_page(width=1400, height=640, line_count=5)
    photo = fixtures.photo_of(cheque)
    document = rectify(photo, detect_document(photo), "cheque")
    assert document.shape[1] > document.shape[0]
    assert document.shape[1] == round(175 * PX_PER_MM) - 2 * EDGE_TRIM_PX


def test_rectified_document_contains_no_surface():
    """The trimmed edge band must be paper, not the table."""
    photo = fixtures.photo_of(fixtures.a4_page(), tilt_px=120)
    document = rectify(photo, detect_document(photo), "a4")
    edges = np.concatenate([document[0].ravel(), document[-1].ravel(), document[:, 0].ravel(), document[:, -1].ravel()])
    assert np.median(edges) > 200


def test_corner_ordering():
    shuffled = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    assert order_corners(shuffled).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_whole_frame_detection_spans_the_image():
    detection = whole_frame_detection(np.zeros((300, 200, 3), np.uint8))
    assert detection.whole_frame
    assert detection.corners.tolist() == [[0, 0], [199, 0], [199, 299], [0, 299]]
