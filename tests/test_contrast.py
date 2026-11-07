"""Standalone check: contrast reacts to ink/paper separation."""

import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN
from scan_quality.scoring.contrast import CHECK_NAME, rms_contrast, score_contrast
from tests import fixtures


def test_clean_page_has_measurable_contrast():
    result = score_contrast(fixtures.clean_page())
    assert result.name == CHECK_NAME
    assert result.raw_value > 0


def test_washing_out_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_contrast(fixtures.low_contrast(page, f)).score for f in (0.7, 0.4, 0.1)]
    assert score_contrast(page).score > scores[0] > scores[1] > scores[2]


def test_severely_washed_out_is_poor():
    result = score_contrast(fixtures.low_contrast(fixtures.clean_page(), 0.05))
    assert result.score < LABEL_FAIR_MIN
    assert result.label == "poor"


def test_blank_page_scores_zero():
    result = score_contrast(fixtures.blank_page())
    assert result.raw_value == pytest.approx(0.0, abs=1e-6)
    assert result.score == 0.0


def test_dense_black_and_white_scores_high():
    """A half-black/half-white page is the theoretical contrast maximum."""
    import numpy as np

    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:, 200:] = 255
    assert score_contrast(image).score == 100.0
