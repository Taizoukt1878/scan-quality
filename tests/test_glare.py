"""Standalone check: glare is judged by its largest spot."""

import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN
from scan_quality.scoring.glare import CHECK_NAME, largest_glare_blob, score_glare
from tests import fixtures


def test_clean_page_has_no_glare():
    result = score_glare(fixtures.sensor_noise(fixtures.clean_page()))
    assert result.name == CHECK_NAME
    assert result.score == 100.0
    assert result.details["blob_count"] == 0


def test_large_reflection_is_poor():
    result = score_glare(fixtures.glare_spot(fixtures.clean_page(), size_px=(200, 150)))
    assert result.score < LABEL_FAIR_MIN
    assert result.label == "poor"


def test_bigger_spot_scores_lower():
    page = fixtures.clean_page()
    scores = [score_glare(fixtures.glare_spot(page, size_px=(s, s))).score for s in (40, 80, 120)]
    assert scores[0] > scores[1] > scores[2]


def test_scattered_specks_are_ignored():
    """Single saturated pixels hide nothing; only blobs count."""
    page = fixtures.clean_page()
    page[::50, ::50] = 255
    assert score_glare(page).score == 100.0


def test_largest_blob_not_total_area_decides():
    """Two medium spots are less harmful than one spot of their combined size."""
    page = fixtures.clean_page()
    two = fixtures.glare_spot(fixtures.glare_spot(page, (0.3, 0.3), (80, 80)), (0.7, 0.7), (80, 80))
    largest, count = largest_glare_blob(two)
    assert count == 2
    assert score_glare(two).score > score_glare(fixtures.glare_spot(page, size_px=(113, 113))).score


def test_raw_value_is_percent_of_area():
    page = fixtures.glare_spot(fixtures.clean_page(), size_px=(100, 60))
    result = score_glare(page)
    expected = 100.0 * result.details["largest_blob_px"] / (page.shape[0] * page.shape[1])
    assert result.raw_value == pytest.approx(expected, abs=1e-3)
