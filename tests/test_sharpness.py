"""Standalone check: sharpness reacts to blur, including blur on part of the page."""

import cv2
import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.sharpness import CHECK_NAME, UNMEASURABLE_SCORE, score_sharpness, tile_sharpness
from tests import fixtures


def _page():
    return fixtures.sensor_noise(fixtures.clean_page())


def test_clean_page_scores_well():
    result = score_sharpness(_page())
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_GOOD_MIN
    assert result.retake_message


def test_blur_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_sharpness(fixtures.sensor_noise(fixtures.blurred(page, k))).score for k in (3, 9, 21)]
    assert score_sharpness(_page()).score > scores[0] > scores[1] > scores[2]


def test_heavy_blur_is_poor():
    result = score_sharpness(fixtures.sensor_noise(fixtures.blurred(fixtures.clean_page(), 21)))
    assert result.score < LABEL_FAIR_MIN
    assert result.label == "poor"


def test_blur_on_part_of_the_page_is_caught():
    """Focus falloff: a global measure would average the soft half away."""
    page = fixtures.clean_page()
    page[:, 600:] = fixtures.blurred(page, 15)[:, 600:]
    result = score_sharpness(fixtures.sensor_noise(page))
    assert result.score < LABEL_FAIR_MIN
    # The median tile is still sharp; only the worst-tenth statistic sees it.
    assert result.details["median_ratio"] > result.raw_value


@pytest.mark.parametrize("sigma", [3.0, 8.0])
def test_sensor_noise_does_not_fake_sharpness(sigma):
    """Grain is pixel-level detail too; a blurred page must stay blurred."""
    blurred = fixtures.blurred(fixtures.clean_page(), 21)
    assert score_sharpness(fixtures.sensor_noise(blurred, sigma)).score < LABEL_FAIR_MIN


def test_faint_ink_does_not_change_sharpness():
    """The ratio is normalized by tile contrast: pale ink is not blur."""
    faint = fixtures.sensor_noise(fixtures.low_contrast(fixtures.clean_page(), 0.25))
    assert score_sharpness(faint).score == pytest.approx(score_sharpness(_page()).score, abs=10.0)


def test_blank_page_is_unknown_not_zero():
    """No text means nothing to judge focus on; contrast rejects it instead."""
    result = score_sharpness(fixtures.sensor_noise(fixtures.blank_page()))
    assert result.score == UNMEASURABLE_SCORE
    assert result.label == "unknown"
    assert result.raw_value is None
    assert tile_sharpness(fixtures.blank_page()).size == 0


def test_grayscale_input_accepted():
    page = _page()
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert score_sharpness(gray).score == pytest.approx(score_sharpness(page).score, abs=0.01)
