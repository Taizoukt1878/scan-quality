"""Runs all five checks over one image and combines them into a report.

The combination rule is deliberately the simplest one that could work: a plain
unweighted mean of the five sub-scores. No check is privileged, because which
defect actually ruins a given document depends on the document — a crooked
cheque still reads fine, a blurred one does not, and the reverse holds for a
dense multi-column quote. Callers that know their own document mix can weight
the individual scores themselves; they are all in the response.

Like the checks it calls, this module is pure: an array in, a report out. No
FastAPI, no file I/O.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from scan_quality.scoring.base import CheckResult, label_for
from scan_quality.scoring.contrast import score_contrast
from scan_quality.scoring.lighting import score_lighting_uniformity
from scan_quality.scoring.ocr import DEFAULT_LANG, score_ocr_confidence
from scan_quality.scoring.sharpness import score_sharpness
from scan_quality.scoring.skew import score_skew

# The five checks, in the order they are reported. OCR runs last because it is
# roughly an order of magnitude slower than the four pixel-statistics checks,
# so a failure in a cheap check surfaces before the expensive one is paid for.
CheckFn = Callable[[np.ndarray], CheckResult]


@dataclass
class QualityReport:
    """Combined result of all five checks over a single page.

    Attributes:
        overall_score: Unweighted mean of the five check scores, 0-100.
        overall_label: The shared quality vocabulary applied to
            ``overall_score``.
        checks: Per-check results, in reporting order.
        processing_time_ms: Wall-clock time spent scoring, excluding the
            decode of the uploaded file.
    """

    overall_score: float
    overall_label: str
    checks: list[CheckResult]
    processing_time_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


def _checks_for(lang: str) -> list[CheckFn]:
    """Build the ordered list of check callables, binding OCR's language."""
    return [
        score_sharpness,
        score_lighting_uniformity,
        score_contrast,
        score_skew,
        lambda image: score_ocr_confidence(image, lang=lang),
    ]


def score_image(image: np.ndarray, lang: str = DEFAULT_LANG) -> QualityReport:
    """Run every quality check on ``image`` and combine the results.

    Args:
        image: Decoded page as a BGR or grayscale uint8 array.
        lang: Tesseract language pack for the OCR check.

    Returns:
        A :class:`QualityReport` whose ``overall_score`` is the unweighted
        mean of the five sub-scores.

    Raises:
        OcrUnavailableError: if the Tesseract binary is missing. Propagated
            rather than swallowed — it is an environment fault, and silently
            scoring 0 would mislabel a perfectly good scan as unreadable.
    """
    started = time.perf_counter()

    results: list[CheckResult] = []
    for check in _checks_for(lang):
        check_started = time.perf_counter()
        result = check(image)
        result.details["duration_ms"] = round((time.perf_counter() - check_started) * 1000.0, 2)
        results.append(result)

    # Simple unweighted average: every check contributes equally.
    overall = float(np.mean([result.score for result in results]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    height, width = image.shape[:2]
    return QualityReport(
        overall_score=round(overall, 2),
        overall_label=label_for(overall),
        checks=results,
        processing_time_ms=round(elapsed_ms, 2),
        meta={"width": width, "height": height},
    )
