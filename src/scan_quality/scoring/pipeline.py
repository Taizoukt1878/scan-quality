"""Locates the document, runs every check, and decides accept or reject.

The decision is a weakest-link gate: the capture is accepted only when every
check reaches its threshold, and the overall score is the *minimum* of the
check scores. A mean would let four good scores outvote one fatal defect — a
reflection over a cheque's amount averages out to "good" — and in a capture
gate the only question is whether anything is bad enough to break extraction.

The lowest failing check also chooses the instruction shown to the user, so
every rejection says what to do differently.

Like the checks it calls, this module is pure: an array in, a report out. No
FastAPI, no file I/O.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from scan_quality.scoring.base import LABEL_FAIR_MIN, CheckResult, label_for
from scan_quality.scoring.contrast import score_contrast
from scan_quality.scoring.document import (
    DEFAULT_DOCUMENT_TYPE,
    detect_document,
    rectify,
    whole_frame_detection,
)
from scan_quality.scoring.framing import score_framing
from scan_quality.scoring.glare import score_glare
from scan_quality.scoring.shadows import score_shadows
from scan_quality.scoring.sharpness import score_sharpness

CheckFn = Callable[[np.ndarray], CheckResult]

# Pixel checks run on the rectified document, in reporting order, after the
# framing check. Cheapest-to-explain first: a blurred capture is the most
# common rejection and the clearest message.
DOCUMENT_CHECKS: list[CheckFn] = [score_sharpness, score_shadows, score_glare, score_contrast]

# Minimum score each check must reach for the capture to be accepted. All
# start at the shared "fair" line; tune them per check against labelled
# captures (see docs/scores.md, Calibration).
ACCEPT_THRESHOLDS: dict[str, float] = {
    "framing": LABEL_FAIR_MIN,
    "sharpness": LABEL_FAIR_MIN,
    "shadows": LABEL_FAIR_MIN,
    "glare": LABEL_FAIR_MIN,
    "contrast": LABEL_FAIR_MIN,
}


@dataclass
class QualityReport:
    """Accept/reject decision plus the checks behind it, for one capture.

    Attributes:
        accepted: True when every check reached its threshold.
        overall_score: The lowest check score, 0-100.
        overall_label: The shared quality vocabulary applied to
            ``overall_score``.
        failed_checks: Names of the checks below their threshold, worst first.
        retake_message: What the user should do differently, taken from the
            worst failing check; ``None`` when accepted.
        checks: Per-check results, in reporting order. Only ``framing`` when
            no document was found.
        processing_time_ms: Wall-clock time spent scoring, excluding the
            decode of the uploaded file.
    """

    accepted: bool
    overall_score: float
    overall_label: str
    failed_checks: list[str]
    retake_message: str | None
    checks: list[CheckResult]
    processing_time_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


def _timed(check: Callable[[], CheckResult]) -> CheckResult:
    """Run a check and record its duration in its details."""
    started = time.perf_counter()
    result = check()
    result.details["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return result


def score_image(
    image: np.ndarray, doc_type: str = DEFAULT_DOCUMENT_TYPE, whole_frame: bool = False
) -> QualityReport:
    """Score one capture and decide whether to accept it.

    Args:
        image: Decoded photo as a BGR or grayscale uint8 array.
        doc_type: Key into :data:`~scan_quality.scoring.document.DOCUMENT_SIZES_MM`.
        whole_frame: True when the image *is* the document (a rendered PDF
            page) and no detection is needed.

    Returns:
        A :class:`QualityReport`. When no document is found only the framing
        check is reported: the pixel checks would be measuring the table.
    """
    started = time.perf_counter()

    detection_started = time.perf_counter()
    detection = whole_frame_detection(image) if whole_frame else detect_document(image)
    framing = score_framing(detection, doc_type)
    framing.details["duration_ms"] = round((time.perf_counter() - detection_started) * 1000.0, 2)
    results = [framing]

    document_meta: dict[str, Any] = {}
    if detection is not None:
        document = rectify(image, detection, doc_type)
        document_meta = {"document_width": document.shape[1], "document_height": document.shape[0]}
        results += [_timed(lambda check=check: check(document)) for check in DOCUMENT_CHECKS]

    failed = sorted(
        (result for result in results if result.score < ACCEPT_THRESHOLDS[result.name]),
        key=lambda result: result.score,
    )
    overall = min(result.score for result in results)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    height, width = image.shape[:2]
    return QualityReport(
        accepted=not failed,
        overall_score=round(overall, 2),
        overall_label=label_for(overall),
        failed_checks=[result.name for result in failed],
        retake_message=failed[0].retake_message if failed else None,
        checks=results,
        processing_time_ms=round(elapsed_ms, 2),
        meta={"width": width, "height": height, "doc_type": doc_type, **document_meta},
    )
