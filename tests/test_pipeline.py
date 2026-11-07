"""The pipeline finds the document, runs every check, and gates on the weakest."""

import pytest

from scan_quality.scoring.pipeline import ACCEPT_THRESHOLDS, score_image
from tests import fixtures

EXPECTED_CHECKS = ["framing", "sharpness", "shadows", "glare", "contrast"]


@pytest.fixture(scope="module")
def page():
    return fixtures.a4_page()


def test_clean_capture_is_accepted(page):
    report = score_image(fixtures.photo_of(page))
    assert [check.name for check in report.checks] == EXPECTED_CHECKS
    assert report.accepted
    assert report.failed_checks == []
    assert report.retake_message is None


def test_overall_is_the_minimum(page):
    report = score_image(fixtures.photo_of(page))
    assert report.overall_score == pytest.approx(min(check.score for check in report.checks), abs=0.01)


def test_one_fatal_defect_rejects_despite_good_others(page):
    """A mean of these scores would pass; the gate must not."""
    report = score_image(fixtures.photo_of(fixtures.glare_spot(page, size_px=(260, 180))))
    others = [check.score for check in report.checks if check.name != "glare"]
    assert sum(others) / len(others) > 90
    assert not report.accepted
    assert report.failed_checks == ["glare"]
    assert "reflection" in report.retake_message


@pytest.mark.parametrize(
    "degrade, failing",
    [
        pytest.param(lambda p: fixtures.blurred(p, 21), "sharpness", id="blurred"),
        pytest.param(lambda p: fixtures.hard_shadow(p, darkness=0.45), "shadows", id="shadow"),
        pytest.param(lambda p: fixtures.low_contrast(p, 0.05), "contrast", id="washed_out"),
    ],
)
def test_each_defect_is_blamed_on_its_check(page, degrade, failing):
    report = score_image(fixtures.photo_of(degrade(page)))
    assert not report.accepted
    assert report.failed_checks[0] == failing


def test_missing_document_reports_only_framing():
    """With no document the pixel checks would be measuring the table."""
    report = score_image(fixtures.photo_of(fixtures.a4_page(), surface_value=240))
    assert [check.name for check in report.checks] == ["framing"]
    assert not report.accepted
    assert report.retake_message


def test_rendered_page_skips_detection(page):
    report = score_image(page, whole_frame=True)
    assert report.accepted
    assert report.checks[0].details["in_frame"] is True


def test_timing_and_metadata_are_reported(page):
    report = score_image(fixtures.photo_of(page))
    assert report.processing_time_ms > 0
    assert report.meta["doc_type"] == "a4"
    assert report.meta["document_width"] > 0
    assert all("duration_ms" in check.details for check in report.checks)


def test_every_check_has_a_threshold_and_a_message(page):
    report = score_image(fixtures.photo_of(page))
    for check in report.checks:
        assert check.name in ACCEPT_THRESHOLDS
        assert check.retake_message
