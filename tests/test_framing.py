"""Standalone check: framing rejects cut-off documents and scores resolution."""

import cv2
import pytest

from scan_quality.scoring.base import LABEL_FAIR_MIN, LABEL_GOOD_MIN
from scan_quality.scoring.document import detect_document, whole_frame_detection
from scan_quality.scoring.framing import (
    CHECK_NAME,
    CUT_OFF_MESSAGE,
    NOT_FOUND_MESSAGE,
    TOO_FAR_MESSAGE,
    score_framing,
)
from tests import fixtures


def test_well_framed_capture_at_working_resolution_scores_well():
    result = score_framing(detect_document(fixtures.photo_of(fixtures.a4_page())), "a4")
    assert result.name == CHECK_NAME
    assert result.score >= LABEL_GOOD_MIN
    assert result.raw_value == pytest.approx(8.0, abs=0.1)
    assert result.details["in_frame"] is True


def test_far_away_document_scores_low():
    photo = cv2.resize(fixtures.photo_of(fixtures.a4_page()), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    result = score_framing(detect_document(photo), "a4")
    assert result.raw_value == pytest.approx(4.8, abs=0.1)
    assert result.score < LABEL_FAIR_MIN
    assert result.retake_message == TOO_FAR_MESSAGE


def test_perspective_is_penalized_through_the_far_side():
    """Tilting the phone shrinks the far edge; the worst side decides."""
    photo = fixtures.photo_of(fixtures.a4_page(), tilt_px=200)
    straight = score_framing(detect_document(fixtures.photo_of(fixtures.a4_page())), "a4")
    tilted = score_framing(detect_document(photo), "a4")
    assert tilted.score < straight.score


def test_document_cut_off_by_the_frame_scores_zero():
    photo = fixtures.photo_of(fixtures.a4_page())
    cropped = photo[:, : int(photo.shape[1] * 0.85)]
    result = score_framing(detect_document(cropped), "a4")
    assert result.score == 0.0
    assert result.details["in_frame"] is False
    assert result.retake_message == CUT_OFF_MESSAGE


def test_document_not_found_scores_zero():
    result = score_framing(None, "a4")
    assert result.score == 0.0
    assert result.raw_value is None
    assert result.retake_message == NOT_FOUND_MESSAGE


def test_rendered_page_is_never_cut_off():
    """A PDF render fills its frame by definition; the edge rule does not apply."""
    result = score_framing(whole_frame_detection(fixtures.a4_page()), "a4")
    assert result.details["in_frame"] is True
    assert result.score >= LABEL_GOOD_MIN


def test_wrong_document_type_changes_the_resolution_target():
    """The same pixels over a smaller physical size mean more px/mm."""
    detection = whole_frame_detection(fixtures.clean_page(width=1400, height=640))
    assert score_framing(detection, "cheque").raw_value == pytest.approx(8.0, abs=0.05)
