"""Standalone check: skew detection recovers a known rotation angle."""

import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.skew import (
    CHECK_NAME,
    MIN_CANDIDATE_LINES,
    UNDETECTABLE_SCORE,
    baseline_points,
    detect_skew_angle,
    score_skew,
)
from tests import fixtures

# The estimator is quantized by the Hough theta step, so exact equality is not
# available; half a degree is well inside what the score cares about.
ANGLE_TOLERANCE_DEG = 0.5


def test_straight_page_scores_well():
    result = score_skew(fixtures.clean_page())
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_GOOD_MIN
    assert abs(result.raw_value) < ANGLE_TOLERANCE_DEG


@pytest.mark.parametrize("degrees", [-6.0, -3.0, -1.5, -0.75, 0.75, 1.5, 3.0, 6.0])
def test_known_rotation_is_recovered(degrees):
    """The reported angle must match the applied rotation in sign and size."""
    page = fixtures.rotated(fixtures.clean_page(), degrees)
    angle, line_count, _ = detect_skew_angle(page)
    assert line_count >= MIN_CANDIDATE_LINES
    assert angle == pytest.approx(degrees, abs=ANGLE_TOLERANCE_DEG)


@pytest.mark.parametrize(
    "degrade",
    [
        pytest.param(lambda img: fixtures.blurred(img, 9), id="blurred"),
        pytest.param(lambda img: fixtures.low_contrast(img, 0.3), id="low_contrast"),
        pytest.param(lambda img: fixtures.uneven_lighting(img, 0.8), id="uneven_lighting"),
    ],
)
def test_angle_survives_other_defects(degrade):
    """Skew must stay measurable on pages that fail the other four checks.

    The checks are independent by design: a blurred or badly lit page should
    still report its tilt rather than degrade into "unknown". This is what the
    adaptive binarization in the module buys.
    """
    tilted = fixtures.rotated(fixtures.clean_page(), 2.0)
    angle, _, _ = detect_skew_angle(degrade(tilted))
    assert angle == pytest.approx(2.0, abs=ANGLE_TOLERANCE_DEG)


def test_tilt_lowers_score_monotonically():
    page = fixtures.clean_page()
    scores = [score_skew(fixtures.rotated(page, d)).score for d in (1.0, 2.5, 4.0)]
    assert score_skew(page).score > scores[0] > scores[1] > scores[2]


def test_sign_does_not_affect_score():
    """A page tilted left is exactly as bad as one tilted right."""
    page = fixtures.clean_page()
    assert score_skew(fixtures.rotated(page, 2.0)).score == pytest.approx(
        score_skew(fixtures.rotated(page, -2.0)).score, abs=5.0
    )


def test_badly_tilted_page_is_poor():
    result = score_skew(fixtures.rotated(fixtures.clean_page(), 6.0))
    assert result.score < LABEL_FAIR_MIN


def test_blank_page_is_unknown_not_zero():
    """No baselines means 'could not measure', which must stay neutral."""
    result = score_skew(fixtures.blank_page())
    assert result.score == UNDETECTABLE_SCORE
    assert result.label == "unknown"
    assert result.raw_value is None
    assert "reason" in result.details


def test_baseline_points_are_sparse():
    """One point per character, not per ink pixel — see the module docstring."""
    page = fixtures.clean_page()
    points = baseline_points(page)
    ink_pixels = (page[:, :, 0] < 128).sum()
    assert 0 < (points > 0).sum() < ink_pixels / 10
