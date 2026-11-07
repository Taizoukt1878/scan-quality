"""Standalone check: shadows react to uneven light on the paper, not to the text."""

import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.shadows import CHECK_NAME, score_shadows, shading_ratio
from tests import fixtures


def test_evenly_lit_page_scores_full_marks():
    """Ink is removed before measuring, so text density is not a shadow."""
    result = score_shadows(fixtures.sensor_noise(fixtures.clean_page()))
    assert result.name == CHECK_NAME
    assert result.score == pytest.approx(100.0)


def test_hard_shadow_is_poor():
    result = score_shadows(fixtures.hard_shadow(fixtures.clean_page(), darkness=0.45))
    assert result.score < LABEL_FAIR_MIN
    assert result.raw_value == pytest.approx(0.45, abs=0.03)


def test_soft_shadow_passes():
    """Mild shading is corrected downstream; the anchors are lenient on purpose."""
    result = score_shadows(fixtures.hard_shadow(fixtures.clean_page(), darkness=0.8))
    assert result.score >= LABEL_GOOD_MIN


def test_gradient_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_shadows(fixtures.uneven_lighting(page, s)).score for s in (0.3, 0.45, 0.6)]
    assert scores[0] > scores[1] > scores[2]


def test_dim_but_even_light_is_not_penalized():
    """A ratio, not a difference: a darker exposure is not a shadow."""
    dim = (fixtures.clean_page().astype(float) * 0.6).astype("uint8")
    assert score_shadows(dim).score == pytest.approx(100.0)


def test_ratio_is_reported_with_its_ends():
    ratio, dark, bright = shading_ratio(fixtures.hard_shadow(fixtures.clean_page(), darkness=0.5))
    assert dark < bright
    assert ratio == pytest.approx(dark / bright)
