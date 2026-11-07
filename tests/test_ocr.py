"""Standalone check: OCR confidence tracks how readable the page actually is.

These tests need the Tesseract binary and are skipped when it is absent, so
the suite still runs on a machine that has only the Python dependencies
installed. See README.md for install commands.
"""

import numpy as np
import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN
from scan_quality.scoring.ocr import (
    CHECK_NAME,
    NO_TEXT_SCORE,
    OcrUnavailableError,
    is_tesseract_available,
    mean_word_confidence,
    score_ocr_confidence,
)
from tests import fixtures

requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(), reason="Tesseract binary not installed"
)


@requires_tesseract
def test_clean_page_is_readable():
    result = score_ocr_confidence(fixtures.clean_page())
    assert result.name == CHECK_NAME
    assert result.details["word_count"] > 0
    assert result.score >= LABEL_FAIR_MIN


@requires_tesseract
def test_blur_lowers_confidence():
    page = fixtures.clean_page()
    sharp = score_ocr_confidence(page).score
    blurry = score_ocr_confidence(fixtures.blurred(page, 11)).score
    assert blurry < sharp


@requires_tesseract
def test_blank_page_reports_no_text():
    """No recognizable text is a finding about the document, not an error."""
    result = score_ocr_confidence(fixtures.blank_page())
    assert result.score == NO_TEXT_SCORE
    assert result.raw_value is None
    assert result.details["word_count"] == 0


@requires_tesseract
def test_score_stays_in_range():
    for image in (fixtures.clean_page(), fixtures.blurred(fixtures.clean_page(), 25)):
        assert 0.0 <= score_ocr_confidence(image).score <= 100.0


@requires_tesseract
def test_confidence_is_within_tesseract_scale():
    confidence, words = mean_word_confidence(fixtures.clean_page())
    assert words > 0
    assert 0.0 <= confidence <= 100.0


def test_missing_binary_raises_rather_than_scoring_zero(monkeypatch):
    """A missing binary is a deployment fault and must not look like a bad scan.

    Runs whether or not Tesseract is installed, by forcing the failure.
    """
    from pytesseract import TesseractNotFoundError

    import scan_quality.scoring.ocr as ocr_module

    def explode(*args, **kwargs):
        raise TesseractNotFoundError()

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", explode)
    with pytest.raises(OcrUnavailableError):
        score_ocr_confidence(np.zeros((100, 100, 3), dtype=np.uint8))
