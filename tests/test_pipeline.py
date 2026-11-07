"""The pipeline runs all five checks and averages them without weighting."""

import numpy as np
import pytest

from scan_quality.scoring.ocr import is_tesseract_available
from scan_quality.scoring.pipeline import score_image
from tests import fixtures

EXPECTED_CHECKS = [
    "sharpness",
    "lighting_uniformity",
    "contrast",
    "skew",
    "ocr_confidence",
]

requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(), reason="Tesseract binary not installed"
)


@requires_tesseract
def test_report_contains_all_five_checks_in_order():
    report = score_image(fixtures.clean_page())
    assert [check.name for check in report.checks] == EXPECTED_CHECKS


@requires_tesseract
def test_overall_is_the_unweighted_mean():
    report = score_image(fixtures.clean_page())
    expected = float(np.mean([check.score for check in report.checks]))
    assert report.overall_score == pytest.approx(expected, abs=0.01)


@requires_tesseract
def test_clean_page_beats_thoroughly_degraded_page():
    clean = fixtures.clean_page()
    ruined = fixtures.low_contrast(
        fixtures.uneven_lighting(fixtures.blurred(fixtures.rotated(clean, 4.0), 15), 0.7),
        0.4,
    )
    assert score_image(clean).overall_score > score_image(ruined).overall_score


@requires_tesseract
def test_timing_and_metadata_are_reported():
    page = fixtures.clean_page(width=900, height=1200)
    report = score_image(page)
    assert report.processing_time_ms > 0
    assert report.meta == {"width": 900, "height": 1200}
    assert all("duration_ms" in check.details for check in report.checks)


@requires_tesseract
def test_scores_stay_in_range():
    for image in (fixtures.clean_page(), fixtures.blank_page()):
        report = score_image(image)
        assert 0.0 <= report.overall_score <= 100.0
        assert all(0.0 <= check.score <= 100.0 for check in report.checks)


def test_ocr_failure_propagates(monkeypatch):
    """A missing binary must surface, not be averaged away into a low score."""
    from pytesseract import TesseractNotFoundError

    import scan_quality.scoring.ocr as ocr_module
    from scan_quality.scoring.ocr import OcrUnavailableError

    def explode(*args, **kwargs):
        raise TesseractNotFoundError()

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", explode)
    with pytest.raises(OcrUnavailableError):
        score_image(fixtures.clean_page())
