"""Standalone check: sharpness scoring reacts to focus, not to content."""

import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN
from scan_quality.scoring.sharpness import CHECK_NAME, laplacian_variance, score_sharpness
from tests import fixtures


def test_clean_page_scores_well():
    result = score_sharpness(fixtures.clean_page())
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_FAIR_MIN
    assert result.label in {"fair", "good", "excellent"}


def test_blur_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_sharpness(fixtures.blurred(page, k)).score for k in (3, 9, 21)]
    sharp = score_sharpness(page).score
    assert sharp > scores[0] > scores[1] > scores[2]


def test_heavy_blur_is_poor():
    result = score_sharpness(fixtures.blurred(fixtures.clean_page(), 31))
    assert result.score < LABEL_FAIR_MIN
    assert result.label == "poor"


def test_blank_page_scores_zero():
    """No edges at all is the degenerate worst case, not an error."""
    result = score_sharpness(fixtures.blank_page())
    assert result.score == 0.0
    assert laplacian_variance(fixtures.blank_page()) == pytest.approx(0.0, abs=1e-6)


def test_score_stays_in_range():
    for image in (fixtures.clean_page(), fixtures.blank_page(), fixtures.blurred(fixtures.clean_page(), 15)):
        assert 0.0 <= score_sharpness(image).score <= 100.0


def test_grayscale_input_accepted():
    """Checks must not care whether the loader handed them BGR or gray."""
    import cv2

    page = fixtures.clean_page()
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert score_sharpness(gray).score == pytest.approx(score_sharpness(page).score, abs=0.5)
