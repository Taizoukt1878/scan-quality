"""Standalone check: contrast reacts to ink legibility, not to how much ink there is."""

import numpy as np
import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.contrast import CHECK_NAME, score_contrast
from tests import fixtures


def _noisy(image):
    return fixtures.sensor_noise(image)


def test_clean_page_scores_well():
    result = score_contrast(_noisy(fixtures.clean_page()))
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_GOOD_MIN


def test_washing_out_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_contrast(_noisy(fixtures.low_contrast(page, f))).score for f in (0.4, 0.1, 0.06)]
    assert scores[0] > scores[1] > scores[2]


def test_severely_washed_out_is_poor():
    result = score_contrast(_noisy(fixtures.low_contrast(fixtures.clean_page(), 0.05)))
    assert result.score < LABEL_FAIR_MIN


def test_sparse_document_is_not_penalized():
    """A cheque is mostly blank paper; that is not low contrast."""
    sparse = _noisy(fixtures.clean_page(line_count=4))
    assert score_contrast(sparse).score >= LABEL_GOOD_MIN


def test_soft_shading_is_flat_fielded_away():
    shaded = _noisy(fixtures.uneven_lighting(fixtures.clean_page(), 0.4))
    assert score_contrast(shaded).score >= LABEL_GOOD_MIN


def test_pale_red_stamp_is_judged_on_its_darkest_channel():
    """Luminance would nearly erase a pale red ink; the min-channel keeps it."""
    page = np.full((800, 800, 3), 230, dtype=np.uint8)
    import cv2

    for row in range(6):
        cv2.putText(page, "VU ET APPROUVE", (40, 110 + row * 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (150, 150, 235), 4, cv2.LINE_AA)  # BGR: pale red
    result = score_contrast(_noisy(page))
    assert result.score >= LABEL_GOOD_MIN
    assert result.details["worst_ink_depth"] > 0.25


def test_blank_page_scores_zero():
    result = score_contrast(_noisy(fixtures.blank_page()))
    assert result.score == 0.0
    assert result.raw_value is None
    assert "reason" in result.details


def test_depth_is_reported():
    result = score_contrast(_noisy(fixtures.clean_page()))
    assert 0.0 < result.details["worst_ink_depth"] <= 1.0
    assert result.details["cells_measured"] > 0


@pytest.mark.parametrize("factor", [0.05])
def test_faint_ink_on_clean_photo_still_fails_on_depth(factor):
    """Low noise can make faint ink separable; depth still rejects it."""
    faint = fixtures.sensor_noise(fixtures.low_contrast(fixtures.clean_page(), factor), sigma=1.0)
    assert score_contrast(faint).score < LABEL_FAIR_MIN
