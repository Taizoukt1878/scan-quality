"""Standalone check: lighting uniformity reacts to illumination gradients."""

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.lighting import CHECK_NAME, cell_brightness_std, score_lighting_uniformity
from tests import fixtures


def test_evenly_lit_page_scores_well():
    result = score_lighting_uniformity(fixtures.clean_page())
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_GOOD_MIN


def test_gradient_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [
        score_lighting_uniformity(fixtures.uneven_lighting(page, s)).score
        for s in (0.1, 0.4, 0.8)
    ]
    assert score_lighting_uniformity(page).score > scores[0] > scores[1] > scores[2]


def test_strong_side_light_is_poor():
    result = score_lighting_uniformity(fixtures.uneven_lighting(fixtures.clean_page(), 0.8))
    assert result.score < LABEL_FAIR_MIN
    assert result.label == "poor"


def test_grid_shape_and_diagnostics():
    result = score_lighting_uniformity(fixtures.clean_page())
    assert result.details["grid"] == "8x8"
    assert result.details["darkest_cell"] <= result.details["brightest_cell"]


def test_cell_grid_dimensions():
    _, cell_means = cell_brightness_std(fixtures.clean_page())
    assert cell_means.shape == (8, 8)


def test_uniform_blank_page_is_perfect():
    """A featureless page has zero spread — the ideal for this check alone."""
    assert score_lighting_uniformity(fixtures.blank_page()).score == 100.0


def test_non_multiple_dimensions_handled():
    """Odd sizes must not crash the grid reshape."""
    page = fixtures.clean_page(width=1003, height=717)
    assert 0.0 <= score_lighting_uniformity(page).score <= 100.0
